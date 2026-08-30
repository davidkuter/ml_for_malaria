from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ml_for_malaria.chemistry.charges import (
    charge_cache_path,
    parse_charge_method,
    require_charge_backend,
)
from ml_for_malaria.model.chemprop_classifier import (
    ARCHITECTURE,
    ChempropClassifier,
    binary_mol_probabilities,
    build_datapoints,
    build_featurizer,
    extra_atom_fdim,
    graph_batch_to_device,
    n_mols_in_graph_batch,
)
from ml_for_malaria.report import build_report, compute_test_metrics, write_report
from ml_for_malaria.runs.paths import resolve_run_dir
from ml_for_malaria.schemas import (
    ChargeMethod,
    CleanedTrainingData,
    ModelMeta,
    RunConfig,
    TrainingReport,
)
from ml_for_malaria.train.prepare import prepare_training_run, train_val_indices

DEFAULT_MAX_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_DROPOUT = 0.1
DEFAULT_DEPTH = 3
DEFAULT_PATIENCE = 8
DEFAULT_MIN_DELTA = 0.01
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
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
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
        ModelCheckpoint,
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


def _datapoints_for_index_groups(
    cleaned: pd.DataFrame,
    groups: list[list[int]],
    charge_method: str | None,
    n_jobs: int,
    cache_path: str | Path | None = None,
) -> list[tuple[list, list[int]]]:
    smiles_col = CleanedTrainingData.SMILES
    label_col = CleanedTrainingData.LABEL
    flat = [idx for group in groups for idx in group]
    subset = cleaned.iloc[list(flat)]
    built = build_datapoints(
        subset[smiles_col].tolist(),
        subset[label_col].astype(float).tolist(),
        charge_method,
        n_jobs=n_jobs,
        cache_path=cache_path,
    )
    results: list[tuple[list, list[int]]] = []
    start = 0
    for group in groups:
        chunk = built[start : start + len(group)]
        points = []
        kept = []
        for idx, point in zip(group, chunk):
            if point is not None:
                points.append(point)
                kept.append(idx)
        results.append((points, kept))
        start += len(group)
    return results


def _predict_loader(model, loader) -> np.ndarray:
    import torch

    model.eval()
    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []
    n_mols = 0
    with torch.no_grad():
        for batch in loader:
            bmg, V_d, X_d = graph_batch_to_device(batch, device)
            pred = model(bmg, V_d, X_d)
            batch_n = n_mols_in_graph_batch(bmg)
            chunks.append(binary_mol_probabilities(pred, batch_n))
            n_mols += batch_n
    if not chunks:
        return np.array([], dtype=np.float64)
    probs = np.concatenate(chunks)
    if probs.shape[0] != n_mols:
        raise RuntimeError(
            f"Chemprop concatenated {probs.shape[0]} scores for {n_mols} molecules"
        )
    return probs


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
    n_jobs: int | None = None,
) -> ChempropTrainResult:
    """Train a Chemprop D-MPNN on the shared clean/split protocol.

    ``outdir`` is the parent runs directory; artifacts go in
    ``{outdir}/{arch}_{split}[_charge]/seed_{seed}/`` when ``seed`` is passed through
    ``resolve_run_dir``.
    ``n_jobs`` defaults to all cores for NAGL charges and 1 otherwise.
    """
    (
        pl,
        MoleculeDataset,
        build_dataloader,
        MPNN,
        _,
        _,
        _,
        EarlyStopping,
        ModelCheckpoint,
    ) = _require_lightning_chemprop()
    charge_method = parse_charge_method(charge_method)
    if n_jobs is not None:
        workers = n_jobs
    elif charge_method == ChargeMethod.NAGL:
        workers = -1
    else:
        workers = 1
    parent = Path(outdir)
    outdir = resolve_run_dir(
        parent, ARCHITECTURE, split, charge_method=charge_method, seed=seed
    )
    cache_file = (
        charge_cache_path(parent, charge_method) if charge_method is not None else None
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

    require_charge_backend(charge_method)
    cleaned = prepared.cleaned
    y = cleaned[CleanedTrainingData.LABEL]
    fit_idx, val_idx = train_val_indices(prepared.train_idx, y, seed=seed)
    charge_note = f" with {charge_method} charges" if charge_method else ""
    logger.info(
        f"Building Chemprop graphs{charge_note} "
        f"(fit={len(fit_idx)}, val={len(val_idx)}, test={len(prepared.test_idx)}, "
        f"n_jobs={workers})"
    )
    started = time.perf_counter()
    (train_points, train_kept), (val_points, val_kept), (test_points, test_kept) = (
        _datapoints_for_index_groups(
            cleaned,
            [fit_idx, val_idx, prepared.test_idx],
            charge_method,
            workers,
            cache_file,
        )
    )
    logger.info(
        f"Finished Chemprop graphs in {time.perf_counter() - started:.1f}s"
    )
    dropped_fit = len(fit_idx) - len(train_kept)
    dropped_val = len(val_idx) - len(val_kept)
    dropped_test = len(prepared.test_idx) - len(test_kept)
    if dropped_fit or dropped_val or dropped_test:
        logger.warning(
            "Dropped molecules that failed Chemprop parse/charges: "
            f"fit={dropped_fit}/{len(fit_idx)}, val={dropped_val}/{len(val_idx)}, "
            f"test={dropped_test}/{len(prepared.test_idx)}"
        )
    if not train_points or not test_points:
        raise RuntimeError(
            "Chemprop training dropped every train or test molecule "
            "(parse or charge assignment failed)."
        )
    if not val_points:
        raise RuntimeError(
            "Chemprop validation split is empty after parse/charge drops; "
            "refusing to early-stop on a dummy train batch."
        )

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
    lightning_dir = ckpt.outdir / "lightning"
    checkpoint = ModelCheckpoint(
        dirpath=str(lightning_dir / "checkpoints"),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        min_delta=DEFAULT_MIN_DELTA,
        verbose=False,
        strict=True,
    )
    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator=accelerator,
        max_epochs=max_epochs,
        callbacks=[checkpoint, early_stop],
        default_root_dir=str(lightning_dir),
        num_sanity_val_steps=0,
    )
    logger.info("Training Chemprop D-MPNN")
    trainer.fit(model, train_loader, val_loader)
    best_path = checkpoint.best_model_path
    if not best_path:
        raise RuntimeError(
            "Chemprop training logged no val_loss; cannot pick a checkpoint."
        )
    if early_stop.stopped_epoch:
        logger.info(f"Early stopping at epoch {early_stop.stopped_epoch}")
    else:
        logger.info(
            f"Trained all {max_epochs} epochs; restoring best val_loss checkpoint"
        )
    if checkpoint.best_model_score is not None:
        logger.info(f"Best val_loss={float(checkpoint.best_model_score):.4f}")
    shutil.copy2(best_path, ckpt.lightning_ckpt_path)
    model = MPNN.load_from_checkpoint(str(best_path), map_location="cpu")
    model.eval()

    y_proba = _predict_loader(model, test_loader)
    if y_proba.shape[0] != len(test_kept):
        raise RuntimeError(
            f"Chemprop test scores {y_proba.shape[0]} != kept test mols {len(test_kept)}"
        )
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
