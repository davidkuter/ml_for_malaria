from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ml_for_malaria.chemistry.charges import ChargeAssignmentError, atom_charges
from ml_for_malaria.model.predict import prepare_predict_smiles
from ml_for_malaria.runs.checkpoints import RunCheckpointer
from ml_for_malaria.schemas import Architecture, ModelMeta, Predictions

ARCHITECTURE = Architecture.CHEMPROP
_EXTRA_ATOM_FDIM = 1


def _require_chemprop():
    try:
        from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
        from chemprop.models import MPNN
    except ImportError as exc:
        raise ImportError(
            "Chemprop requires the optional 'dl' extra (chemprop, torch). "
            "Install with: uv sync --extra dl"
        ) from exc
    return (
        MoleculeDatapoint,
        MoleculeDataset,
        build_dataloader,
        SimpleMoleculeMolGraphFeaturizer,
        MPNN,
    )


def extra_atom_fdim(charge_method: str | None) -> int:
    return _EXTRA_ATOM_FDIM if charge_method else 0


def n_mols_in_graph_batch(bmg) -> int:
    """Number of molecules in a Chemprop ``BatchMolGraph``."""
    if bmg.batch.numel() == 0:
        return 0
    return int(bmg.batch.max().item()) + 1


def binary_mol_probabilities(pred, n_mols: int) -> np.ndarray:
    """Squeeze Chemprop binary output to shape ``(n_mols,)`` or raise."""
    probs = pred.detach().cpu().numpy()
    if probs.ndim == 2 and probs.shape[-1] == 1:
        probs = probs[:, 0]
    elif probs.ndim != 1:
        raise RuntimeError(
            f"Unexpected Chemprop output shape {probs.shape}; expected (N,) or (N, 1)"
        )
    if probs.shape[0] != n_mols:
        raise RuntimeError(
            f"Chemprop output length {probs.shape[0]} != batch molecules {n_mols}"
        )
    if not np.isfinite(probs).all():
        raise RuntimeError("Chemprop produced non-finite probabilities")
    return np.clip(probs.astype(np.float64), 0.0, 1.0)


def graph_batch_to_device(batch, device):
    """Move a Chemprop dataloader batch onto ``device`` (in-place for the graph)."""
    bmg, V_d, X_d, *_ = batch
    bmg.to(device)
    if V_d is not None:
        V_d = V_d.to(device)
    if X_d is not None:
        X_d = X_d.to(device)
    return bmg, V_d, X_d


def molecule_datapoint(smiles: str, y: float | None, charge_method: str | None):
    """Build a Chemprop datapoint; charges are taken from the same mol Chemprop featurizes."""
    MoleculeDatapoint, *_ = _require_chemprop()
    kwargs: dict = {}
    if y is not None:
        kwargs["y"] = np.array([float(y)], dtype=np.float64)
    try:
        point = MoleculeDatapoint.from_smi(smiles, **kwargs)
    except (TypeError, ValueError, RuntimeError):
        logger.warning(f"Dropping {smiles!r}: Chemprop datapoint failed")
        return None
    if charge_method is None:
        return point
    try:
        charges = atom_charges(point.mol, charge_method)
    except ChargeAssignmentError:
        logger.warning(f"Dropping {smiles!r}: charge assignment failed")
        return None
    expected = (point.mol.GetNumAtoms(), extra_atom_fdim(charge_method))
    if charges.shape != expected:
        logger.warning(
            f"Dropping {smiles!r}: charge shape {charges.shape} != {expected}"
        )
        return None
    point.V_f = charges
    return point


def build_featurizer(charge_method: str | None):
    *_, SimpleMoleculeMolGraphFeaturizer, _ = _require_chemprop()
    fdim = extra_atom_fdim(charge_method)
    if fdim:
        return SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=fdim)
    return SimpleMoleculeMolGraphFeaturizer()


class ChempropClassifier:
    """Chemprop D-MPNN classifier restored from a training run directory."""

    architecture = ARCHITECTURE

    def __init__(self, model, featurizer, metadata: ModelMeta):
        self.model = model
        self.featurizer = featurizer
        self.metadata = metadata

    @classmethod
    def load(cls, outdir: str | Path) -> ChempropClassifier:
        _, _, _, _, MPNN = _require_chemprop()
        ckpt = RunCheckpointer(outdir)
        if not ckpt.meta_path.exists() or not ckpt.lightning_ckpt_path.exists():
            raise FileNotFoundError(
                f"No saved Chemprop model in {ckpt.outdir}. "
                "Expected model.ckpt and model_meta.json"
            )
        metadata = ModelMeta.model_validate(ckpt.load_json(ckpt.meta_path))
        if metadata.architecture != ARCHITECTURE:
            raise ValueError(
                f"Run directory {ckpt.outdir} was trained with "
                f"architecture={metadata.architecture!r}, "
                f"not {ARCHITECTURE!r}."
            )
        model = MPNN.load_from_checkpoint(
            str(ckpt.lightning_ckpt_path), map_location="cpu"
        )
        model.eval()
        return cls(
            model=model,
            featurizer=build_featurizer(metadata.charge_method),
            metadata=metadata,
        )

    def predict(self, smiles: list[str] | pd.Series) -> pd.DataFrame:
        _, MoleculeDataset, build_dataloader, _, _ = _require_chemprop()
        import torch

        output, sanitized, unique = prepare_predict_smiles(smiles)
        probability = Predictions.PROBABILITY
        if unique.empty:
            return Predictions.validate(output.assign(**{probability: np.nan}))

        kept: list[str] = []
        points = []
        for smi in unique:
            point = molecule_datapoint(
                smi, y=None, charge_method=self.metadata.charge_method
            )
            if point is None:
                continue
            points.append(point)
            kept.append(smi)
        if not points:
            return Predictions.validate(output.assign(**{probability: np.nan}))

        dataset = MoleculeDataset(points, featurizer=self.featurizer)
        loader = build_dataloader(
            dataset, num_workers=0, shuffle=False, batch_size=32, drop_last=False
        )
        probs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            device = next(self.model.parameters()).device
            for batch in loader:
                bmg, V_d, X_d = graph_batch_to_device(batch, device)
                pred = self.model(bmg, V_d, X_d)
                probs.append(binary_mol_probabilities(pred, n_mols_in_graph_batch(bmg)))
        if not probs:
            return Predictions.validate(output.assign(**{probability: np.nan}))
        scores = np.concatenate(probs)
        if scores.shape[0] != len(kept):
            raise RuntimeError(
                f"Chemprop predicted {scores.shape[0]} rows for {len(kept)} molecules"
            )
        mapped = pd.Series(
            scores,
            index=kept,
            name=probability,
        )
        output[probability] = sanitized.map(mapped)
        return Predictions.validate(output)
