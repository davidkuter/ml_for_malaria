from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel

from ml_for_malaria.schemas import (
    SKLEARN_JOBLIB_ARCHITECTURES,
    Architecture,
    CleanedTrainingData,
    FingerprintFeatures,
    RunConfig,
)


def data_hash(df: pd.DataFrame, columns: list[str]) -> str:
    """Stable hash of selected dataframe columns."""
    payload = df[columns].astype(str).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays so they can be written as JSON."""
    if isinstance(obj, BaseModel):
        return to_jsonable(obj.model_dump())
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


def _as_dict(stored: dict | BaseModel) -> dict:
    return stored.model_dump() if isinstance(stored, BaseModel) else stored


class RunCheckpointer:
    """Load/save run artifacts under ``outdir`` and decide when to reuse them."""

    REPORT_JSON = "report.json"
    REPORT_MD = "report.md"

    def __init__(self, outdir: str | Path, force: bool = False):
        self.outdir = Path(outdir)
        self.force = force
        self.outdir.mkdir(parents=True, exist_ok=True)
        (self.outdir / "splits").mkdir(exist_ok=True)
        (self.outdir / "features").mkdir(exist_ok=True)
        (self.outdir / "hyperopt").mkdir(exist_ok=True)
        (self.outdir / "models").mkdir(exist_ok=True)

    def save_json(self, path: Path, data: dict | BaseModel) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")

    def load_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def config_path(self) -> Path:
        return self.outdir / "config.json"

    def save_config(self, config: RunConfig) -> None:
        self.save_json(self.config_path, config)

    def load_config(self) -> RunConfig | None:
        if not self.config_path.exists():
            return None
        return RunConfig.model_validate(self.load_json(self.config_path))

    def should_reuse(
        self, path: Path, stored: dict | BaseModel | None, expected: dict
    ) -> bool:
        """Reuse ``path`` when it exists, force is off, and ``stored`` matches ``expected``."""
        if self.force or not path.exists() or stored is None:
            return False
        dumped = _as_dict(stored)
        return all(dumped.get(key) == value for key, value in expected.items())

    @property
    def cleaned_path(self) -> Path:
        return self.outdir / "cleaned.parquet"

    def load_cleaned(self) -> pd.DataFrame:
        logger.info(f"Loading cleaned data from {self.cleaned_path}")
        return CleanedTrainingData.validate(pd.read_parquet(self.cleaned_path))

    def save_cleaned(self, df: pd.DataFrame) -> None:
        CleanedTrainingData.validate(df).to_parquet(self.cleaned_path, index=False)

    def split_path(self, split: str, seed: int) -> Path:
        return self.outdir / "splits" / f"{split}_seed{seed}.json"

    def features_path(self, fp_name: str) -> Path:
        return self.outdir / "features" / f"{fp_name}.parquet"

    def load_features(self, fp_name: str) -> pd.DataFrame:
        path = self.features_path(fp_name)
        logger.info(f"Loading features from {path}")
        features = pd.read_parquet(path)
        features.columns = [int(col) for col in features.columns]
        return FingerprintFeatures.validate(features.reset_index(drop=True))

    def save_features(self, fp_name: str, features: pd.DataFrame) -> None:
        FingerprintFeatures.validate(features).to_parquet(self.features_path(fp_name))

    def hyperopt_path(self, fp_name: str) -> Path:
        return self.outdir / "hyperopt" / f"{fp_name}.json"

    def fingerprint_dir(self, fp_name: str) -> Path:
        return self.outdir / "models" / fp_name

    def fingerprint_model_path(self, fp_name: str) -> Path:
        return self.fingerprint_dir(fp_name) / "model.ubj"

    def fingerprint_sklearn_path(self, fp_name: str) -> Path:
        return self.fingerprint_dir(fp_name) / "model.joblib"

    def fingerprint_meta_path(self, fp_name: str) -> Path:
        return self.fingerprint_dir(fp_name) / "model_meta.json"

    @property
    def model_path(self) -> Path:
        return self.outdir / "model.ubj"

    @property
    def sklearn_model_path(self) -> Path:
        return self.outdir / "model.joblib"

    @property
    def lightning_ckpt_path(self) -> Path:
        return self.outdir / "model.ckpt"

    @property
    def monroe_support_path(self) -> Path:
        return self.outdir / "monroe_support.npz"

    @property
    def hf_model_dir(self) -> Path:
        return self.outdir / "hf_model"

    @property
    def meta_path(self) -> Path:
        return self.outdir / "model_meta.json"

    @property
    def report_json_path(self) -> Path:
        return self.outdir / self.REPORT_JSON

    @property
    def report_md_path(self) -> Path:
        return self.outdir / self.REPORT_MD

    def model_artifact_path(self, architecture: str) -> Path:
        if architecture in (Architecture.CHEMPROP, Architecture.CHEMELEON):
            return self.lightning_ckpt_path
        if architecture == Architecture.CHEMBERTA:
            return self.hf_model_dir / "config.json"
        if architecture == Architecture.MONROE:
            return self.monroe_support_path
        if architecture in SKLEARN_JOBLIB_ARCHITECTURES:
            return self.sklearn_model_path
        return self.model_path

    def _fingerprint_models_ready(self, expected: RunConfig) -> bool:
        if not expected.fingerprints:
            return True
        first = expected.fingerprints[0]
        if expected.architecture == Architecture.XGBOOST:
            return self.fingerprint_model_path(first).exists()
        if expected.architecture in SKLEARN_JOBLIB_ARCHITECTURES:
            return self.fingerprint_sklearn_path(first).exists()
        return True

    def run_complete(self, stored: RunConfig | None, expected: RunConfig) -> bool:
        if self.force or stored is None:
            return False
        artifact = self.model_artifact_path(expected.architecture)
        if not self.should_reuse(
            artifact, stored, expected.model_dump(exclude_none=True)
        ):
            return False
        if not (self.meta_path.exists() and self.report_json_path.exists()):
            return False
        return self._fingerprint_models_ready(expected)
