"""Monroe frozen-encoder + TabPFN in-context classifier."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ml_for_malaria.env import load_local_env
from ml_for_malaria.model.predict import prepare_predict_smiles
from ml_for_malaria.runs.checkpoints import RunCheckpointer
from ml_for_malaria.schemas import Architecture, ModelMeta, Predictions

ARCHITECTURE = Architecture.MONROE
EMBEDDING_DIM = 720
_DEFAULT_MONROE_HOME = Path.home() / "software" / "monroe"
_SUPPORT_NPZ = "monroe_support.npz"
_PARAMS_CKPT = "checkpoint_dir"
_PARAMS_DIM = "embedding_dim"


def monroe_home() -> Path:
    load_local_env()
    raw = os.environ.get("MONROE_HOME")
    return Path(raw).expanduser().resolve() if raw else _DEFAULT_MONROE_HOME.resolve()


def monroe_checkpoint_dir(home: Path | None = None) -> Path:
    return (home or monroe_home()) / "checkpoint"


def require_monroe_checkout(home: Path | None = None) -> Path:
    """Ensure Monroe is importable and weights exist; put checkout on ``sys.path``."""
    load_local_env()
    root = home or monroe_home()
    ckpt = monroe_checkpoint_dir(root)
    weights = ckpt / "weights.pt"
    if not weights.exists():
        raise ImportError(
            f"Monroe checkpoint missing at {weights}. "
            "Clone https://github.com/blazejba/monroe, run `git lfs pull`, "
            "and set MONROE_HOME in .local.env (or the environment)."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _require_tabpfn_token() -> None:
    load_local_env()
    token = os.environ.get("TABPFN_TOKEN")
    if not token:
        raise ImportError(
            "TABPFN_TOKEN is not set. Add it to .local.env (see .local.env.example) "
            "or export it in the environment for TabPFN v3."
        )
    # Prefer env / cache over interactive browser login (hangs under non-TTY agents).
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    try:
        from tabpfn.browser_auth import save_token

        save_token(token.strip())
    except Exception as exc:  # noqa: BLE001 — best-effort cache warm-up
        logger.debug(f"Could not cache TabPFN token: {exc}")


def monroe_modules():
    require_monroe_checkout()
    _require_tabpfn_token()
    try:
        from monroe.eval.embed import _featurize_one, to_pyg
        from monroe.eval.tabpfn import default_ensemble_specs, fit_predict_tabpfn
    except ImportError as exc:
        raise ImportError(
            "Monroe + TabPFN requires a Monroe checkout (MONROE_HOME) and the "
            "'monroe' extra (tabpfn, torch-geometric, wandb). "
            "Install with: uv sync --extra monroe"
        ) from exc
    return _featurize_one, to_pyg, fit_predict_tabpfn, default_ensemble_specs


def load_encoder(device: str | None = None):
    """Load the frozen Monroe encoder, mapping CUDA checkpoints onto CPU when needed."""
    import torch

    require_monroe_checkout()
    from monroe.model.ckpt import (
        _build_encoder,
        _load_config_and_weights,
        _resolve_ckpt_dir,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    map_device = torch.device(device)
    ckpt_dir = _resolve_ckpt_dir(str(monroe_checkpoint_dir()))
    hp_dict, state_dict = _load_config_and_weights(ckpt_dir, device=map_device)

    class _Monroe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _build_encoder(hp_dict)

    model = _Monroe()
    model.load_state_dict(state_dict, strict=False)
    encoder = model.encoder.to(map_device).eval()
    return encoder, device


def embed_smiles_lookup(
    smiles: list[str],
    encoder=None,
    *,
    device: str | None = None,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Embed SMILES without Monroe's Linux-only process pool (safe on Windows)."""
    import torch
    from torch_geometric.data import Batch

    if encoder is None:
        encoder, device = load_encoder(device)
    elif device is None:
        device = next(encoder.parameters()).device
    featurize_one, to_pyg, *_ = monroe_modules()
    encoder = encoder.to(device).eval()

    unique = list(dict.fromkeys(smiles))
    built: list[tuple[str, object]] = []
    for smi in unique:
        result = featurize_one(smi)
        if result is None:
            logger.warning(f"Dropping {smi!r}: Monroe graph featurization failed")
            continue
        built.append(result)
    if not built:
        return {}

    graphs = [to_pyg(graph) for _, graph in built]
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start : start + batch_size]).to(device)
            graph_embedding, _ = encoder(batch)
            vectors.append(graph_embedding.cpu().numpy())
    stacked = np.concatenate(vectors, axis=0)
    if stacked.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"Expected {EMBEDDING_DIM}-d Monroe embeddings, got {stacked.shape[1]}"
        )
    return {smi: stacked[i] for i, (smi, _) in enumerate(built)}


def fit_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int = 42,
) -> np.ndarray:
    """TabPFN binary probabilities for the positive class."""
    _, _, fit_predict_tabpfn, default_ensemble_specs = monroe_modules()
    result = fit_predict_tabpfn(
        X_train,
        y_train,
        X_test,
        is_classification=True,
        ensemble_specs=default_ensemble_specs(),
        seed=seed,
    )
    if isinstance(result, tuple):
        _, proba = result
        return np.asarray(proba, dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def support_path(outdir: str | Path) -> Path:
    return Path(outdir) / _SUPPORT_NPZ


def save_support(
    outdir: str | Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    smiles_train: list[str],
) -> Path:
    path = support_path(outdir)
    np.savez_compressed(
        path,
        X=np.asarray(X_train, dtype=np.float32),
        y=np.asarray(y_train, dtype=np.int64),
        smiles=np.asarray(smiles_train, dtype=object),
    )
    return path


def load_support(outdir: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = support_path(outdir)
    if not path.exists():
        raise FileNotFoundError(f"No Monroe support set at {path}")
    payload = np.load(path, allow_pickle=True)
    return payload["X"], payload["y"], payload["smiles"].tolist()


class MonroeClassifier:
    """In-context Monroe + TabPFN classifier restored from a run directory."""

    architecture = ARCHITECTURE

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        metadata: ModelMeta,
        outdir: Path | None = None,
        seed: int = 42,
    ):
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=np.int64)
        self.metadata = metadata
        self.outdir = Path(outdir) if outdir is not None else None
        self.seed = seed

    @classmethod
    def load(cls, outdir: str | Path) -> MonroeClassifier:
        ckpt = RunCheckpointer(outdir)
        if not ckpt.meta_path.exists() or not support_path(ckpt.outdir).exists():
            raise FileNotFoundError(
                f"No saved Monroe model in {ckpt.outdir}. "
                f"Expected model_meta.json and {_SUPPORT_NPZ}"
            )
        metadata = ModelMeta.model_validate(ckpt.load_json(ckpt.meta_path))
        if metadata.architecture != ARCHITECTURE:
            raise ValueError(
                f"Run directory {ckpt.outdir} was trained with "
                f"architecture={metadata.architecture!r}, "
                f"not {ARCHITECTURE!r}."
            )
        X, y, _ = load_support(ckpt.outdir)
        seed = int(metadata.params.get("seed", 42))
        return cls(X_train=X, y_train=y, metadata=metadata, outdir=ckpt.outdir, seed=seed)

    def predict(self, smiles: list[str] | pd.Series) -> pd.DataFrame:
        output, sanitized, unique = prepare_predict_smiles(smiles)
        probability = Predictions.PROBABILITY
        if unique.empty:
            return Predictions.validate(output.assign(**{probability: np.nan}))

        lookup = embed_smiles_lookup(unique.tolist())
        kept: list[str] = []
        rows: list[np.ndarray] = []
        for smi in unique.tolist():
            vector = lookup.get(smi)
            if vector is None:
                continue
            kept.append(smi)
            rows.append(vector)
        if not rows:
            return Predictions.validate(output.assign(**{probability: np.nan}))

        X_test = np.stack(rows, axis=0)
        proba = fit_predict_proba(
            self.X_train, self.y_train, X_test, seed=self.seed
        )
        mapped = pd.Series(proba, index=kept, name=probability)
        output[probability] = sanitized.map(mapped)
        return Predictions.validate(output)
