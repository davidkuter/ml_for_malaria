from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from ml_for_malaria.schemas import FingerprintFeatures, RunConfig


def shared_feature_paths(parent: Path, fp_name: str, fp_size: int) -> tuple[Path, Path]:
    stem = f"{fp_name}_fp{fp_size}"
    folder = Path(parent) / "features"
    return folder / f"{stem}.parquet", folder / f"{stem}.json"


def load_shared_features(
    parent: Path, fp_name: str, fp_size: int, cleaned_hash: str
) -> pd.DataFrame | None:
    parquet, meta_path = shared_feature_paths(parent, fp_name, fp_size)
    if not parquet.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get(RunConfig.cleaned_hash) != cleaned_hash:
        return None
    logger.info(f"Loading shared features from {parquet}")
    features = pd.read_parquet(parquet)
    features.columns = [int(col) for col in features.columns]
    return FingerprintFeatures.validate(features.reset_index(drop=True))


def save_shared_features(
    parent: Path,
    fp_name: str,
    fp_size: int,
    cleaned_hash: str,
    features: pd.DataFrame,
) -> None:
    parquet, meta_path = shared_feature_paths(parent, fp_name, fp_size)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    FingerprintFeatures.validate(features).to_parquet(parquet)
    meta_path.write_text(
        json.dumps({RunConfig.cleaned_hash: cleaned_hash, RunConfig.fp_size: fp_size}),
        encoding="utf-8",
    )
