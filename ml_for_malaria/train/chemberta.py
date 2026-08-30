from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ml_for_malaria.model.smiles_transformer import (
    ARCHITECTURE,
    DEFAULT_PRETRAINED_NAME,
    MAX_LENGTH,
    SmilesTransformerClassifier,
    _positive_class_proba,
    load_tokenizer_and_model,
)
from ml_for_malaria.report import build_report, compute_test_metrics, write_report
from ml_for_malaria.runs.paths import resolve_run_dir
from ml_for_malaria.schemas import (
    CleanedTrainingData,
    ModelMeta,
    RunConfig,
    TrainingReport,
)
from ml_for_malaria.train.prepare import (
    prepare_training_run,
    scramble_train_labels,
    train_val_indices,
)

DEFAULT_MAX_EPOCHS = 8
DEFAULT_BATCH_SIZE = 8
DEFAULT_PATIENCE = 2
DEFAULT_MIN_DELTA = 0.01


@dataclass
class TransformerTrainResult:
    classifier: SmilesTransformerClassifier
    report: TrainingReport
    outdir: Path


def _require_trainer():
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError(
            "ChemBERTa training requires the optional 'dl' extra "
            "(torch, transformers, accelerate). Install with: uv sync --extra dl"
        ) from exc
    return torch, Dataset, Trainer, TrainingArguments, EarlyStoppingCallback


def _smiles_dataset(tokenizer, smiles: pd.Series, labels: pd.Series):
    _, Dataset, *_ = _require_trainer()

    class _EncodedSmiles(Dataset):
        def __init__(self, encodings, y):
            self.encodings = encodings
            self.labels = y

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: value[idx] for key, value in self.encodings.items()}
            item["labels"] = int(self.labels[idx])
            return item

    smiles_list = list(smiles)
    encodings = tokenizer(
        smiles_list,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    labels_arr = np.asarray(labels, dtype=np.int64)
    n_tokens = int(encodings["input_ids"].shape[0])
    if len(smiles_list) != len(labels_arr) or n_tokens != len(labels_arr):
        raise RuntimeError(
            f"ChemBERTa dataset length mismatch: smiles={len(smiles_list)} "
            f"labels={len(labels_arr)} tokens={n_tokens}"
        )
    return _EncodedSmiles(encodings, labels_arr)


def _training_args(
    output_dir: Path,
    max_epochs: int,
    batch_size: int,
    seed: int,
    accelerator: str,
):
    _, _, _, TrainingArguments, _ = _require_trainer()
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": max_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "seed": seed,
        "report_to": [],
        "logging_strategy": "no",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "dataloader_num_workers": 0,
    }
    if accelerator == "cpu":
        kwargs["use_cpu"] = True
    try:
        return TrainingArguments(**kwargs, eval_strategy="epoch")
    except TypeError:
        return TrainingArguments(**kwargs, evaluation_strategy="epoch")


def _predict_proba(model, tokenizer, smiles: pd.Series) -> np.ndarray:
    torch, *_ = _require_trainer()
    encoded = tokenizer(
        list(smiles),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits
    if logits.shape[0] != len(smiles):
        raise RuntimeError(
            f"ChemBERTa logits batch {logits.shape[0]} != SMILES {len(smiles)}"
        )
    return _positive_class_proba(logits)


def train_smiles_transformer(
    df: pd.DataFrame,
    outdir: str | Path,
    split: str = "random",
    seed: int = 42,
    test_size: float = 0.2,
    force: bool = False,
    pretrained_name: str = DEFAULT_PRETRAINED_NAME,
    freeze_encoder: bool = True,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    patience: int = DEFAULT_PATIENCE,
    accelerator: str = "cpu",
    yscramble: bool = False,
) -> TransformerTrainResult:
    """Fine-tune a SMILES transformer on the shared clean/split protocol.

    ``outdir`` is the parent runs directory; artifacts go in
    ``{outdir}/{arch}_{split}[_yscramble]/seed_{seed}/``.
    ``yscramble=True`` permutes train labels only; test labels stay real.
    """
    _, _, Trainer, _, EarlyStoppingCallback = _require_trainer()
    outdir = resolve_run_dir(
        outdir, ARCHITECTURE, split, seed=seed, yscramble=yscramble
    )
    prepared = prepare_training_run(
        df,
        outdir,
        split=split,
        seed=seed,
        test_size=test_size,
        force=force,
    )
    ckpt = prepared.ckpt
    expected = RunConfig(
        input_hash=prepared.input_hash,
        split=split,
        seed=seed,
        test_size=test_size,
        architecture=ARCHITECTURE,
        cleaned_hash=prepared.cleaned_hash,
        pretrained_name=pretrained_name,
        max_epochs=max_epochs,
        batch_size=batch_size,
        freeze_encoder=freeze_encoder,
        yscramble=True if yscramble else None,
    )
    if ckpt.run_complete(prepared.stored, expected):
        logger.info(f"Reusing completed transformer run in {ckpt.outdir}")
        classifier = SmilesTransformerClassifier.load(ckpt.outdir)
        report = TrainingReport.model_validate(ckpt.load_json(ckpt.report_json_path))
        return TransformerTrainResult(
            classifier=classifier, report=report, outdir=ckpt.outdir
        )

    cleaned = prepared.cleaned
    y_true = cleaned[CleanedTrainingData.LABEL]
    y = (
        scramble_train_labels(y_true, prepared.train_idx, seed)
        if yscramble
        else y_true
    )
    if yscramble:
        logger.info("Y-scramble: permuted train labels; test labels unchanged")
    smiles = cleaned[CleanedTrainingData.SMILES]
    fit_idx, val_idx = train_val_indices(prepared.train_idx, y, seed=seed)
    tokenizer, model = load_tokenizer_and_model(
        pretrained_name, freeze_encoder=freeze_encoder
    )
    train_ds = _smiles_dataset(tokenizer, smiles.iloc[fit_idx], y.iloc[fit_idx])
    val_ds = _smiles_dataset(tokenizer, smiles.iloc[val_idx], y.iloc[val_idx])
    args = _training_args(
        ckpt.outdir / "hf_trainer",
        max_epochs=max_epochs,
        batch_size=batch_size,
        seed=seed,
        accelerator=accelerator,
    )
    logger.info(f"Fine-tuning {pretrained_name} (freeze_encoder={freeze_encoder})")
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=DEFAULT_MIN_DELTA,
            ),
        ],
    }
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)
    trainer.train()
    model = trainer.model
    if trainer.state.best_metric is not None:
        logger.info(f"Restored best eval_loss={trainer.state.best_metric:.4f}")
    ckpt.hf_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt.hf_model_dir)
    tokenizer.save_pretrained(ckpt.hf_model_dir)

    test_smiles = smiles.iloc[prepared.test_idx]
    y_proba = _predict_proba(model, tokenizer, test_smiles)
    y_true_test = y_true.iloc[prepared.test_idx].to_numpy()
    if y_proba.shape[0] != y_true_test.shape[0]:
        raise RuntimeError(
            f"ChemBERTa test scores {y_proba.shape[0]} != labels {y_true_test.shape[0]}"
        )
    y_pred = (y_proba >= 0.5).astype(int)
    test_metrics = compute_test_metrics(y_true_test, y_pred, y_proba)
    logger.info(
        f"Transformer test accuracy={test_metrics.accuracy:.3f} "
        f"roc_auc={test_metrics.roc_auc:.3f}"
    )

    metadata = ModelMeta(
        architecture=ARCHITECTURE,
        pretrained_name=pretrained_name,
        freeze_encoder=freeze_encoder,
    )
    ckpt.save_json(ckpt.meta_path, metadata)
    report = build_report(
        split=split,
        seed=seed,
        test_size=test_size,
        n_train=len(prepared.train_idx),
        n_test=len(prepared.test_idx),
        test_metrics=test_metrics,
        architecture=ARCHITECTURE,
        pretrained_name=pretrained_name,
        yscramble=yscramble,
    )
    write_report(report, ckpt.report_json_path, ckpt.report_md_path)
    ckpt.save_config(expected)

    classifier = SmilesTransformerClassifier(
        model=model, tokenizer=tokenizer, metadata=metadata
    )
    return TransformerTrainResult(
        classifier=classifier, report=report, outdir=ckpt.outdir
    )
