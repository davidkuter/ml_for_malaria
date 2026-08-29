from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem

from ml_for_malaria.model.predict import prepare_predict_smiles
from ml_for_malaria.schemas import Architecture, ModelMeta, Predictions
from ml_for_malaria.train.charges import ChargeAssignmentError, atom_charges
from ml_for_malaria.train.checkpoints import RunCheckpointer

ARCHITECTURE = Architecture.CHEMPROP
_EXTRA_ATOM_FDIM = 1


def _require_chemprop():
    try:
        from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
        from chemprop.models import MPNN
    except ImportError as exc:
        raise ImportError(
            "Chemprop requires the optional 'dl' extra (chemprop, torch)."
        ) from exc
    return MoleculeDatapoint, MoleculeDataset, build_dataloader, SimpleMoleculeMolGraphFeaturizer, MPNN


def extra_atom_fdim(charge_method: str | None) -> int:
    return _EXTRA_ATOM_FDIM if charge_method else 0


def molecule_datapoint(smiles: str, y: float | None, charge_method: str | None):
    MoleculeDatapoint, *_ = _require_chemprop()
    kwargs: dict = {}
    if y is not None:
        kwargs["y"] = np.array([float(y)], dtype=np.float64)
    if charge_method is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            kwargs["V_f"] = atom_charges(mol, charge_method)
        except ChargeAssignmentError:
            logger.warning(f"Dropping {smiles!r}: charge assignment failed")
            return None
    try:
        return MoleculeDatapoint.from_smi(smiles, **kwargs)
    except (TypeError, ValueError, RuntimeError):
        logger.warning(f"Dropping {smiles!r}: Chemprop datapoint failed")
        return None


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
            point = molecule_datapoint(smi, y=None, charge_method=self.metadata.charge_method)
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
            for batch in loader:
                bmg, V_d, X_d, *_ = batch
                pred = self.model(bmg, V_d, X_d)
                probs.append(pred.detach().cpu().numpy().reshape(-1))
        if not probs:
            return Predictions.validate(output.assign(**{probability: np.nan}))
        mapped = pd.Series(
            np.clip(np.concatenate(probs), 0.0, 1.0),
            index=kept,
            name=probability,
        )
        output[probability] = sanitized.map(mapped)
        return Predictions.validate(output)
