from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ml_for_malaria.chemistry.charges import parse_charge_method
from ml_for_malaria.model.chemprop_classifier import (
    ARCHITECTURE,
    ChempropClassifier,
    build_featurizer,
    extra_atom_fdim,
    molecule_datapoint,
)
from ml_for_malaria.report import build_report, compute_test_metrics, write_report
from ml_for_malaria.runs.paths import resolve_run_dir
from ml_for_malaria.schemas import (
    CleanedTrainingData,
    ModelMeta,
    RunConfig,
    TrainingReport,
)
from ml_for_malaria.train.prepare import prepare_training_run, train_val_indices

DEFAULT_MAX_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_DROPOUT = 0.1
DEFAULT_DEPTH = 3
DEFAULT_PATIENCE = 8
_PARAMS_HIDDEN = "hidden_size"
_PARAMS_DEPTH = "depth"
_PARAMS_DROPOUT = "dropout"
_PARAMS_EXTRA_ATOM_FDIM = "extra_atom_fdim"


@dataclass
class ChempropTrainResult:
    classifier: ChempropClassifier
    report: TrainingReport
    outdir: Path


def _require_lightning_chemprop():
    try:
        import lightning.pytorch as pl
        from chemprop.data import MoleculeDataset, build_dataloader
        from chemprop.models import MPNN
        from chemprop.nn import (
            BinaryClassificationFFN,
            BondMessagePassing,
            NormAggregation,
        )
        from lightning.pytorch.callbacks import EarlyStopping
    except ImportError as exc:
        raise ImportError(
            "Chemprop training requires the optional 'dl' extra "
            "(chemprop, torch, lightning). Install with: uv sync --extra dl"
        ) from exc
    return (
        pl,
        MoleculeDataset,
        build_dataloader,
        MPNN,
        BinaryClassificationFFN,
        BondMessagePassing,
        NormAggregation,
        EarlyStopping,
    )


def _build_mpnn(featurizer, hidden_size: int, dropout: float, depth: int):
    (
        _,
        _,
        _,
        MPNN,
        BinaryClassificationFFN,
        BondMessagePassing,
        NormAggregation,
        _,
    ) = _require_lightning_chemprop()
    mp = BondMessagePassing(
        d_v=featurizer.atom_fdim,
        d_e=featurizer.bond_fdim,
        d_h=hidden_size,
        depth=depth,
        dropout=dropout,
    )
    ffn = BinaryClassificationFFN(
        input_dim=mp.output_dim,
        hidden_dim=hidden_size,
        dropout=dropout,
    )
    return MPNN(mp, NormAggregation(), ffn)


def _datapoints_for_indices(
    cleaned: pd.DataFrame,
    indices: list[int],
    charge_method: str | None,
) -> tuple[list, list[int]]:
    smiles_col = CleanedTrainingData.SMILES
    label_col = CleanedTrainingData.LABEL
    points = []
    kept = []
    for idx in indices:
        smi = cleaned.iloc[idx][smiles_col]
        label = cleaned.iloc[idx][label_col]
        point = molecule_datapoint(smi, y=float(label), charge_method=charge_method)
        if point is None:
            continue
        points.append(point)
        kept.append(idx)
    return points, kept


def _predict_loader(model, loader) -> np.ndarray:
    import torch

    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            bmg, V_d, X_d, *_ = batch
            pred = model(bmg, V_d, X_d)
            chunks.append(pred.detach().cpu().numpy().reshape(-1))
    if not chunks:
        return np.array([], dtype=np.float64)
    return np.clip(np.concatenate(chunks), 0.0, 1.0)


def train_chemprop_classifier(
    df: pd.DataFrame,
    outdir: str | Path,
    split: str = "random",
    seed: int = 42,
    test_size: float = 0.2,
    force: bool = False,
    charge_method: str | None = None,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    dropout: float = DEFAULT_DROPOUT,
    depth: int = DEFAULT_DEPTH,
    patience: int = DEFAULT_PATIENCE,
    accelerator: str = "auto",
) -> ChempropTrainResult:
    """Train a Chemprop D-MPNN on the shared clean/split protocol.

    ``outdir`` is the parent runs directory; artifacts go in
    ``{outdir}/{arch}_{split}`` or ``{outdir}/{arch}_{split}_{charge}``.
    """
    (
        pl,
        MoleculeDataset,
        build_dataloader,
        _,
        _,
        _,
        _,
        EarlyStopping,
    ) = _require_lightning_chemprop()
    charge_method = parse_charge_method(charge_method)
    outdir = resolve_run_dir(outdir, ARCHITECTURE, split, charge_method=charge_method)
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
        charge_method=charge_method,
        max_epochs=max_epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
    )
    if ckpt.run_complete(prepared.stored, expected):
        logger.info(f"Reusing completed Chemprop run in {ckpt.outdir}")
        classifier = ChempropClassifier.load(ckpt.outdir)
        report = TrainingReport.model_validate(ckpt.load_json(ckpt.report_json_path))
        return ChempropTrainResult(
            classifier=classifier, report=report, outdir=ckpt.outdir
        )

    cleaned = prepared.cleaned
    y = cleaned[CleanedTrainingData.LABEL]
    fit_idx, val_idx = train_val_indices(prepared.train_idx, y, seed=seed)
    train_points, train_kept = _datapoints_for_indices(cleaned, fit_idx, charge_method)
    val_points, val_kept = _datapoints_for_indices(cleaned, val_idx, charge_method)
    test_points, test_kept = _datapoints_for_indices(
        cleaned, prepared.test_idx, charge_method
    )
    if not train_points or not test_points:
        raise RuntimeError(
            "Chemprop training dropped every train or test molecule "
            "(parse or charge assignment failed)."
        )
    if not val_points:
        logger.warning("Empty Chemprop val split; reusing a train batch for validation")
        val_points, val_kept = train_points[:1], train_kept[:1]

    featurizer = build_featurizer(charge_method)
    train_dset = MoleculeDataset(train_points, featurizer=featurizer)
    val_dset = MoleculeDataset(val_points, featurizer=featurizer)
    test_dset = MoleculeDataset(test_points, featurizer=featurizer)
    train_loader = build_dataloader(
        train_dset,
        num_workers=0,
        shuffle=True,
        batch_size=batch_size,
        drop_last=False,
    )
    val_loader = build_dataloader(
        val_dset,
        num_workers=0,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
    )
    test_loader = build_dataloader(
        test_dset,
        num_workers=0,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
    )

    model = _build_mpnn(
        featurizer, hidden_size=hidden_size, dropout=dropout, depth=depth
    )
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            verbose=False,
            strict=False,
        )
    ]
    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator=accelerator,
        max_epochs=max_epochs,
        callbacks=callbacks,
        default_root_dir=str(ckpt.outdir / "lightning"),
        num_sanity_val_steps=0,
    )
    logger.info("Training Chemprop D-MPNN")
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(str(ckpt.lightning_ckpt_path))

    y_proba = _predict_loader(model, test_loader)
    y_true = y.iloc[test_kept].to_numpy()
    y_pred = (y_proba >= 0.5).astype(int)
    test_metrics = compute_test_metrics(y_true, y_pred, y_proba)
    logger.info(
        f"Chemprop test accuracy={test_metrics.accuracy:.3f} "
        f"roc_auc={test_metrics.roc_auc:.3f}"
    )

    metadata = ModelMeta(
        architecture=ARCHITECTURE,
        charge_method=charge_method,
        params={
            _PARAMS_HIDDEN: hidden_size,
            _PARAMS_DEPTH: depth,
            _PARAMS_DROPOUT: dropout,
            _PARAMS_EXTRA_ATOM_FDIM: extra_atom_fdim(charge_method),
        },
    )
    ckpt.save_json(ckpt.meta_path, metadata)
    report = build_report(
        split=split,
        seed=seed,
        test_size=test_size,
        n_train=len(set(train_kept) | set(val_kept)),
        n_test=len(test_kept),
        test_metrics=test_metrics,
        architecture=ARCHITECTURE,
        charge_method=charge_method,
    )
    write_report(report, ckpt.report_json_path, ckpt.report_md_path)
    ckpt.save_config(expected)

    classifier = ChempropClassifier(
        model=model, featurizer=featurizer, metadata=metadata
    )
    return ChempropTrainResult(classifier=classifier, report=report, outdir=ckpt.outdir)
