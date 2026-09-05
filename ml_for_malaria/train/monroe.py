"""Train Monroe frozen-encoder + TabPFN on the shared clean/split protocol."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ml_for_malaria.env import load_local_env
from ml_for_malaria.model.monroe_classifier import (
    ARCHITECTURE,
    EMBEDDING_DIM,
    MonroeClassifier,
    embed_smiles_lookup,
    fit_predict_proba,
    load_encoder,
    monroe_checkpoint_dir,
    require_monroe_checkout,
    save_support,
)
from ml_for_malaria.report import build_report, compute_test_metrics, write_report
from ml_for_malaria.runs.paths import resolve_run_dir
from ml_for_malaria.schemas import (
    CleanedTrainingData,
    ModelMeta,
    RunConfig,
    TrainingReport,
)
from ml_for_malaria.train.prepare import prepare_training_run, scramble_train_labels

_PARAMS_CKPT = "checkpoint_dir"
_PARAMS_DIM = "embedding_dim"
_PARAMS_SEED = "seed"
_EMBED_STEM = "monroe_embeddings"


@dataclass
class MonroeTrainResult:
    classifier: MonroeClassifier
    report: TrainingReport
    outdir: Path


def _shared_embed_paths(parent: Path) -> tuple[Path, Path]:
    folder = Path(parent) / "features"
    return folder / f"{_EMBED_STEM}.npz", folder / f"{_EMBED_STEM}.json"


def _load_or_build_embeddings(
    parent: Path,
    cleaned: pd.DataFrame,
    cleaned_hash: str,
    *,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, ok) aligned to ``cleaned`` rows; cache under parent/features/."""
    parquet, meta_path = _shared_embed_paths(parent)
    smiles = cleaned[CleanedTrainingData.SMILES].astype(str).tolist()
    if parquet.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get(RunConfig.cleaned_hash) == cleaned_hash:
            payload = np.load(parquet, allow_pickle=False)
            X, ok, names = payload["X"], payload["ok"], payload["smiles"]
            if len(X) == len(smiles) and list(names) == smiles:
                logger.info(f"Loading shared Monroe embeddings from {parquet}")
                return X, ok

    require_monroe_checkout()
    encoder, device = load_encoder()
    logger.info(
        f"Embedding {len(dict.fromkeys(smiles))} unique SMILES with Monroe on {device}"
    )
    started = time.perf_counter()
    lookup = embed_smiles_lookup(smiles, encoder, device=device, batch_size=batch_size)
    logger.info(f"Monroe embedding finished in {time.perf_counter() - started:.1f}s")

    X = np.full((len(smiles), EMBEDDING_DIM), np.nan, dtype=np.float32)
    ok = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        vector = lookup.get(smi)
        if vector is not None:
            X[i] = vector
            ok[i] = True
    parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp = parquet.parent / f"{parquet.stem}_writing.npz"
    np.savez_compressed(
        tmp,
        X=X,
        ok=ok,
        smiles=np.asarray(smiles),
    )
    os.replace(tmp, parquet)
    meta_path.write_text(
        json.dumps({RunConfig.cleaned_hash: cleaned_hash, "embedding_dim": EMBEDDING_DIM}),
        encoding="utf-8",
    )
    logger.info(f"Wrote shared Monroe embeddings to {parquet}")
    return X, ok


def train_monroe_classifier(
    df: pd.DataFrame,
    outdir: str | Path,
    split: str = "random",
    seed: int = 42,
    test_size: float = 0.2,
    force: bool = False,
    yscramble: bool = False,
    batch_size: int = 32,
) -> MonroeTrainResult:
    """Fit TabPFN in-context on frozen Monroe embeddings for one seed."""
    load_local_env()
    require_monroe_checkout()

    parent = Path(outdir)
    outdir = resolve_run_dir(
        parent,
        ARCHITECTURE,
        split,
        seed=seed,
        yscramble=yscramble,
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
        freeze_encoder=True,
        yscramble=True if yscramble else None,
    )
    if ckpt.run_complete(prepared.stored, expected):
        logger.info(f"Reusing completed Monroe run in {ckpt.outdir}")
        classifier = MonroeClassifier.load(ckpt.outdir)
        report = TrainingReport.model_validate(ckpt.load_json(ckpt.report_json_path))
        return MonroeTrainResult(
            classifier=classifier, report=report, outdir=ckpt.outdir
        )

    cleaned = prepared.cleaned
    X, ok = _load_or_build_embeddings(
        parent, cleaned, prepared.cleaned_hash, batch_size=batch_size
    )
    y_true = cleaned[CleanedTrainingData.LABEL].astype(int)
    y = (
        scramble_train_labels(y_true, prepared.train_idx, seed)
        if yscramble
        else y_true
    )
    if yscramble:
        logger.info("Y-scramble: permuted train labels; test labels unchanged")

    train_idx = [i for i in prepared.train_idx if ok[i]]
    test_idx = [i for i in prepared.test_idx if ok[i]]
    dropped_train = len(prepared.train_idx) - len(train_idx)
    dropped_test = len(prepared.test_idx) - len(test_idx)
    if dropped_train or dropped_test:
        logger.warning(
            "Dropped molecules without Monroe embeddings: "
            f"train={dropped_train}/{len(prepared.train_idx)}, "
            f"test={dropped_test}/{len(prepared.test_idx)}"
        )
    if not train_idx or not test_idx:
        raise RuntimeError(
            "Monroe training dropped every train or test molecule "
            "(embedding failed)."
        )

    X_train = X[train_idx]
    y_train = y.iloc[train_idx].to_numpy(dtype=np.int64)
    X_test = X[test_idx]
    y_true_test = y_true.iloc[test_idx].to_numpy(dtype=np.int64)
    smiles_train = cleaned[CleanedTrainingData.SMILES].iloc[train_idx].astype(str).tolist()

    logger.info(
        f"TabPFN in-context fit "
        f"(train={len(train_idx)}, test={len(test_idx)}, dim={X_train.shape[1]})"
    )
    y_proba = fit_predict_proba(X_train, y_train, X_test, seed=seed)
    y_pred = (y_proba >= 0.5).astype(int)
    test_metrics = compute_test_metrics(y_true_test, y_pred, y_proba)
    logger.info(
        f"Monroe test accuracy={test_metrics.accuracy:.3f} "
        f"roc_auc={test_metrics.roc_auc:.3f}"
    )

    save_support(ckpt.outdir, X_train, y_train, smiles_train)
    metadata = ModelMeta(
        architecture=ARCHITECTURE,
        freeze_encoder=True,
        params={
            _PARAMS_CKPT: str(monroe_checkpoint_dir()),
            _PARAMS_DIM: EMBEDDING_DIM,
            _PARAMS_SEED: seed,
        },
    )
    ckpt.save_json(ckpt.meta_path, metadata)
    report = build_report(
        split=split,
        seed=seed,
        test_size=test_size,
        n_train=len(train_idx),
        n_test=len(test_idx),
        test_metrics=test_metrics,
        architecture=ARCHITECTURE,
        yscramble=yscramble,
    )
    write_report(report, ckpt.report_json_path, ckpt.report_md_path)
    ckpt.save_config(expected)

    classifier = MonroeClassifier(
        X_train=X_train,
        y_train=y_train,
        metadata=metadata,
        outdir=ckpt.outdir,
        seed=seed,
    )
    return MonroeTrainResult(classifier=classifier, report=report, outdir=ckpt.outdir)
