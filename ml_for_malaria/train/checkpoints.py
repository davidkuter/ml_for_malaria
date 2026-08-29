from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def data_hash(df: pd.DataFrame, columns: list[str]) -> str:
    """Stable hash of selected dataframe columns."""
    payload = df[columns].astype(str).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays so they can be written as JSON."""
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


class RunCheckpointer:
    """Load/save run artifacts under ``outdir`` and decide when to reuse them."""

    def __init__(self, outdir: str | Path, force: bool = False):
        self.outdir = Path(outdir)
        self.force = force
        self.outdir.mkdir(parents=True, exist_ok=True)
        (self.outdir / "splits").mkdir(exist_ok=True)
        (self.outdir / "features").mkdir(exist_ok=True)
        (self.outdir / "hyperopt").mkdir(exist_ok=True)

    def save_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")

    def load_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def config_path(self) -> Path:
        return self.outdir / "config.json"

    def save_config(self, config: dict) -> None:
        self.save_json(self.config_path, config)

    def load_config(self) -> dict | None:
        if not self.config_path.exists():
            return None
        return self.load_json(self.config_path)

    def _cached(self, path: Path) -> bool:
        return (not self.force) and path.exists()

    def should_reuse(self, path: Path, stored: dict | None, expected: dict) -> bool:
        """Reuse ``path`` when it exists, force is off, and ``stored`` matches ``expected``."""
        if self.force or not path.exists() or stored is None:
            return False
        return all(stored.get(key) == value for key, value in expected.items())

    @property
    def cleaned_path(self) -> Path:
        return self.outdir / "cleaned.parquet"

    def load_cleaned(self) -> pd.DataFrame:
        logger.info(f"Loading cleaned data from {self.cleaned_path}")
        return pd.read_parquet(self.cleaned_path)

    def save_cleaned(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.cleaned_path, index=False)

    def split_path(self, split: str, seed: int) -> Path:
        return self.outdir / "splits" / f"{split}_seed{seed}.json"

    def features_path(self, fp_name: str) -> Path:
        return self.outdir / "features" / f"{fp_name}.parquet"

    def load_features(self, fp_name: str) -> pd.DataFrame:
        path = self.features_path(fp_name)
        logger.info(f"Loading features from {path}")
        features = pd.read_parquet(path)
        features.columns = [int(col) for col in features.columns]
        return features.reset_index(drop=True)

    def save_features(self, fp_name: str, features: pd.DataFrame) -> None:
        features.to_parquet(self.features_path(fp_name))

    def hyperopt_path(self, fp_name: str) -> Path:
        return self.outdir / "hyperopt" / f"{fp_name}.json"

    @property
    def model_path(self) -> Path:
        return self.outdir / "model.ubj"

    @property
    def meta_path(self) -> Path:
        return self.outdir / "model_meta.json"

    @property
    def report_json_path(self) -> Path:
        return self.outdir / "report.json"

    @property
    def report_md_path(self) -> Path:
        return self.outdir / "report.md"

    def run_complete(self, stored: dict | None, expected: dict) -> bool:
        if self.force or stored is None:
            return False
        if not all(stored.get(key) == value for key, value in expected.items()):
            return False
        return (
            self.model_path.exists()
            and self.meta_path.exists()
            and self.report_json_path.exists()
        )
