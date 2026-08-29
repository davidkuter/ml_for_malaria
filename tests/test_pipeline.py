from pathlib import Path

import pandas as pd
import pytest

from ml_for_malaria.schemas import (
    CleanedTrainingData,
    EvalMetrics,
    FingerprintComparison,
    FingerprintScore,
    ModelMeta,
    Predictions,
    RunConfig,
)
from ml_for_malaria.train.checkpoints import RunCheckpointer, data_hash
from ml_for_malaria.train.featurization import (
    encode_binary_labels,
    featurize_smiles,
    get_fingerprint_generator,
    sanitize_smiles,
)
from ml_for_malaria.train.report import build_report, compute_test_metrics, report_to_markdown
from ml_for_malaria.train.split import ScaffoldSplitter, get_splitter


def test_get_splitter_unknown():
    with pytest.raises(ValueError, match="Unknown split"):
        get_splitter("cluster")


def test_scaffold_splitter_is_a_stub():
    splitter = get_splitter("scaffold")
    assert isinstance(splitter, ScaffoldSplitter)
    with pytest.raises(NotImplementedError, match="Bemis–Murcko"):
        splitter.split(
            pd.Series(["CCO", "CCC"]),
            pd.Series([0, 1]),
            test_size=0.5,
            seed=0,
        )


def test_random_splitter_is_stratified():
    smiles = pd.Series([f"C{'C' * i}" for i in range(20)])
    labels = pd.Series([0] * 10 + [1] * 10)
    train_idx, test_idx = get_splitter("random").split(
        smiles, labels, test_size=0.2, seed=42
    )
    assert len(train_idx) == 16
    assert len(test_idx) == 4
    assert set(train_idx).isdisjoint(test_idx)
    test_labels = labels.iloc[test_idx]
    assert int((test_labels == 0).sum()) == 2
    assert int((test_labels == 1).sum()) == 2


def test_encode_binary_labels_rejects_unknown():
    series = pd.Series(["Active", "Inactive", "Maybe"])
    with pytest.raises(ValueError, match="Unexpected labels"):
        encode_binary_labels(series)


def test_encode_binary_labels_maps_active_inactive():
    series = pd.Series(["Active", "Inactive", "Active"])
    encoded = encode_binary_labels(series)
    assert encoded.tolist() == [1, 0, 1]


def test_sanitize_smiles_returns_none_on_failure():
    assert sanitize_smiles("not-a-smiles") is None
    assert sanitize_smiles(None) is None  # type: ignore[arg-type]
    assert isinstance(sanitize_smiles("CCO"), str)


def test_featurize_skips_failed_mols_without_misindexing():
    generator = get_fingerprint_generator("AtomPair", fp_size=128)
    features = featurize_smiles(
        ["CCO", "not-a-smiles", "CCC"], fp_generator=generator, sanitize=False
    )
    assert list(features.index) == ["CCO", "CCC"]
    assert len(features) == 2
    assert features.shape[1] == 128


def test_featurize_all_failures_returns_empty_frame():
    generator = get_fingerprint_generator("AtomPair", fp_size=128)
    features = featurize_smiles(
        ["not-a-smiles", "also-bad"], fp_generator=generator, sanitize=False
    )
    assert features is not None
    assert features.empty


def test_xgb_load_rejects_other_architecture(tmp_path: Path):
    import json

    from ml_for_malaria.model import XGBFingerprintClassifier

    (tmp_path / "model.ubj").write_bytes(b"placeholder")
    (tmp_path / "model_meta.json").write_text(
        json.dumps(
            {
                ModelMeta.architecture: "pytorch",
                ModelMeta.fingerprint: "AtomPair",
                ModelMeta.fp_size: 128,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pytorch"):
        XGBFingerprintClassifier.load(tmp_path)


def test_compute_test_metrics_and_report_roundtrip(tmp_path: Path):
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_proba = [0.1, 0.6, 0.9, 0.8]
    metrics = compute_test_metrics(y_true, y_pred, y_proba)
    dumped = metrics.model_dump()
    assert {
        EvalMetrics.threshold,
        EvalMetrics.accuracy,
        EvalMetrics.roc_auc,
        EvalMetrics.per_class,
        EvalMetrics.macro,
        EvalMetrics.weighted,
        EvalMetrics.confusion_matrix,
    } <= set(dumped)
    report = build_report(
        split="random",
        seed=42,
        test_size=0.2,
        n_train=16,
        n_test=4,
        best_fingerprint="AtomPair",
        fingerprint_comparison={
            "AtomPair": FingerprintScore(cv_auc=0.9, n_estimators=50)
        },
        test_metrics=metrics,
    )
    markdown = report_to_markdown(report)
    assert EvalMetrics.roc_auc in markdown
    assert "AtomPair" in markdown
    assert FingerprintComparison.fingerprint in markdown
    assert "true_0" in markdown
    assert "pred_1" in markdown


def test_checkpointer_reuses_matching_config(tmp_path: Path):
    ckpt = RunCheckpointer(tmp_path, force=False)
    path = tmp_path / "item.json"
    ckpt.save_json(path, {"ok": True})
    stored = {RunConfig.seed: 42, RunConfig.split: "random"}
    assert ckpt.should_reuse(path, stored, {RunConfig.seed: 42, RunConfig.split: "random"})
    assert not ckpt.should_reuse(path, stored, {RunConfig.seed: 1, RunConfig.split: "random"})
    forced = RunCheckpointer(tmp_path, force=True)
    assert not forced.should_reuse(
        path, stored, {RunConfig.seed: 42, RunConfig.split: "random"}
    )


def test_data_hash_changes_with_labels():
    columns = [CleanedTrainingData.SMILES, CleanedTrainingData.LABEL]
    df_a = pd.DataFrame(
        {CleanedTrainingData.SMILES: ["CCO"], CleanedTrainingData.LABEL: [1]}
    )
    df_b = pd.DataFrame(
        {CleanedTrainingData.SMILES: ["CCO"], CleanedTrainingData.LABEL: [0]}
    )
    assert data_hash(df_a, columns) != data_hash(df_b, columns)


def test_train_xgb_classifier_smoke_and_checkpoint_reuse(tmp_path: Path):
    from ml_for_malaria.train import train_xgb_classifier

    inactive = [
        "CCO",
        "CCN",
        "CCC",
        "CCCC",
        "CCCCC",
        "CCCCCC",
        "CCOCC",
        "CC(C)O",
        "CC(C)C",
        "CCCCO",
        "C1CCCCC1",
        "CCS",
        "CCBr",
        "CCOC",
        "CCCCCCC",
    ]
    active = [
        "c1ccccc1",
        "c1ccc(C)cc1",
        "c1ccc(O)cc1",
        "c1ccc(N)cc1",
        "c1ccc(F)cc1",
        "c1ccc(Cl)cc1",
        "c1ccncc1",
        "c1cccnc1",
        "CC(=O)O",
        "NCCO",
        "O=C(O)c1ccccc1",
        "COc1ccccc1",
        "CCNc1ccccc1",
        "c1ccc(Br)cc1",
        "c1ccc(CC)cc1",
    ]
    df = pd.DataFrame(
        {
            CleanedTrainingData.SMILES: inactive + active,
            CleanedTrainingData.LABEL: [0] * len(inactive) + [1] * len(active),
        }
    )
    outdir = tmp_path / "run"
    result = train_xgb_classifier(
        df,
        outdir=outdir,
        split="random",
        seed=0,
        test_size=0.25,
        max_evals=1,
        fingerprints=["AtomPair"],
        fp_size=128,
    )
    assert result.report.best_fingerprint == "AtomPair"
    assert result.report.architecture == "xgboost"
    assert (outdir / "report.json").exists()
    assert (outdir / "report.md").exists()
    assert (outdir / "model.ubj").exists()
    assert (outdir / "model_meta.json").exists()
    preds = result.classifier.predict(["CCO", "c1ccncc1"])
    assert Predictions.PROBABILITY in preds.columns
    assert len(preds) == 2

    reused = train_xgb_classifier(
        df,
        outdir=outdir,
        split="random",
        seed=0,
        test_size=0.25,
        max_evals=1,
        fingerprints=["AtomPair"],
        fp_size=128,
    )
    assert reused.report.test_metrics.roc_auc == result.report.test_metrics.roc_auc
