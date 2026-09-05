from pathlib import Path

from ml_for_malaria.runs.checkpoints import RunCheckpointer
from ml_for_malaria.schemas import Architecture, ModelMeta


def load_classifier(outdir: str | Path, fingerprint: str | None = None):
    """Load the classifier matching ``model_meta.json`` architecture."""
    ckpt = RunCheckpointer(outdir)
    if not ckpt.meta_path.exists():
        raise FileNotFoundError(f"No model_meta.json in {ckpt.outdir}")
    metadata = ModelMeta.model_validate(ckpt.load_json(ckpt.meta_path))
    if metadata.architecture == Architecture.XGBOOST:
        from ml_for_malaria.model.xgb_classifier import XGBFingerprintClassifier

        return XGBFingerprintClassifier.load(outdir, fingerprint=fingerprint)
    if metadata.architecture == Architecture.RANDOM_FOREST:
        from ml_for_malaria.model.rf_classifier import RFFingerprintClassifier

        return RFFingerprintClassifier.load(outdir, fingerprint=fingerprint)
    if metadata.architecture == Architecture.KNN:
        from ml_for_malaria.model.knn_classifier import KNNFingerprintClassifier

        return KNNFingerprintClassifier.load(outdir, fingerprint=fingerprint)
    if metadata.architecture == Architecture.LOGISTIC:
        from ml_for_malaria.model.logistic_classifier import (
            LogisticFingerprintClassifier,
        )

        return LogisticFingerprintClassifier.load(outdir, fingerprint=fingerprint)
    if metadata.architecture == Architecture.CHEMPROP:
        from ml_for_malaria.model.chemprop_classifier import ChempropClassifier

        return ChempropClassifier.load(outdir)
    if metadata.architecture == Architecture.CHEMELEON:
        from ml_for_malaria.model.chemprop_classifier import ChempropClassifier

        return ChempropClassifier.load(outdir)
    if metadata.architecture == Architecture.CHEMBERTA:
        from ml_for_malaria.model.smiles_transformer import SmilesTransformerClassifier

        return SmilesTransformerClassifier.load(outdir)
    if metadata.architecture == Architecture.MONROE:
        from ml_for_malaria.model.monroe_classifier import MonroeClassifier

        return MonroeClassifier.load(outdir)
    raise ValueError(f"Unknown architecture {metadata.architecture!r} in {ckpt.outdir}")
