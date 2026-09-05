from pathlib import Path

import pandas as pd
import pytest

from ml_for_malaria.chemistry import (
    encode_binary_labels,
    featurize_smiles,
    get_fingerprint_generator,
    sanitize_smiles,
)
from ml_for_malaria.report import (
    best_fixed_scaffold_identifier,
    build_report,
    compute_test_metrics,
    report_to_markdown,
    write_comparison_report,
    write_report,
)
from ml_for_malaria.runs import (
    RunCheckpointer,
    completed_run_dirs,
    data_hash,
    resolve_run_dir,
    run_dirname,
)
from ml_for_malaria.schemas import (
    Architecture,
    ChargeMethod,
    CleanedTrainingData,
    ComparisonRow,
    ComparisonTable,
    EvalMetrics,
    FingerprintComparison,
    FingerprintScore,
    ModelMeta,
    Predictions,
    RunConfig,
)
from ml_for_malaria.split import ScaffoldSplitter, get_splitter, murcko_group_keys
from ml_for_malaria.train.prepare import scramble_train_labels
from tests.toy_data import toy_binary_df


def test_run_dirname_encodes_architecture_split_and_charge():
    assert run_dirname(Architecture.XGBOOST, "scaffold") == "xgb_scaffold"
    assert run_dirname(Architecture.CHEMPROP, "random") == "chemprop_random"
    assert (
        run_dirname(
            Architecture.CHEMPROP, "scaffold", charge_method=ChargeMethod.GASTEIGER
        )
        == "chemprop_scaffold_gasteiger"
    )
    assert (
        run_dirname(
            Architecture.CHEMPROP, "scaffold", charge_method=ChargeMethod.NAGL
        )
        == "chemprop_scaffold_nagl"
    )
    assert run_dirname(Architecture.CHEMBERTA, "scaffold") == "chemberta_scaffold"
    assert run_dirname(Architecture.CHEMELEON, "scaffold") == "chemeleon_scaffold"
    assert run_dirname(Architecture.MONROE, "scaffold") == "monroe_scaffold"
    assert run_dirname(Architecture.RANDOM_FOREST, "scaffold") == "rf_scaffold"
    assert run_dirname(Architecture.KNN, "scaffold") == "knn_scaffold"
    assert run_dirname(Architecture.LOGISTIC, "scaffold") == "logistic_scaffold"
    assert (
        run_dirname(Architecture.XGBOOST, "scaffold", hpo=True) == "xgb_scaffold_hpo"
    )
    assert (
        run_dirname(Architecture.RANDOM_FOREST, "scaffold", hpo=True)
        == "rf_scaffold_hpo"
    )
    assert (
        run_dirname(Architecture.RANDOM_FOREST, "scaffold", yscramble=True)
        == "rf_scaffold_yscramble"
    )
    assert (
        run_dirname(Architecture.XGBOOST, "scaffold", hpo=True, yscramble=True)
        == "xgb_scaffold_hpo_yscramble"
    )
    parent = Path("runs")
    assert (
        resolve_run_dir(parent, Architecture.XGBOOST, "random") == parent / "xgb_random"
    )
    assert (
        resolve_run_dir(parent, Architecture.XGBOOST, "scaffold", seed=42)
        == parent / "xgb_scaffold" / "seed_42"
    )
    assert (
        resolve_run_dir(
            parent,
            Architecture.XGBOOST,
            "scaffold",
            seed=42,
            hpo=True,
        )
        == parent / "xgb_scaffold_hpo" / "seed_42"
    )
    assert (
        resolve_run_dir(parent, Architecture.RANDOM_FOREST, "random", seed=42)
        == parent / "rf_random" / "seed_42"
    )
    assert (
        resolve_run_dir(
            parent,
            Architecture.CHEMPROP,
            "scaffold",
            charge_method=ChargeMethod.NAGL,
            seed=42,
        )
        == parent / "chemprop_scaffold_nagl" / "seed_42"
    )
    assert (
        resolve_run_dir(
            parent,
            Architecture.RANDOM_FOREST,
            "random",
            seed=0,
            yscramble=True,
        )
        == parent / "rf_random_yscramble" / "seed_0"
    )


def _double_seed(seed: int) -> int:
    return seed * 2


def test_replicate_seeds_consecutive_from_start():
    from ml_for_malaria.runs import replicate_seeds, seed_dir_name

    assert replicate_seeds(10, start=42) == tuple(range(42, 52))
    assert seed_dir_name(42) == "seed_42"
    with pytest.raises(ValueError, match="n_rep"):
        replicate_seeds(0)


def test_replicate_worker_count_serial_gpu_and_auto():
    from ml_for_malaria.runs import replicate_worker_count

    assert replicate_worker_count(10, n_workers=1) == 1
    assert replicate_worker_count(10, serial=True) == 1
    assert replicate_worker_count(1) == 1
    auto = replicate_worker_count(10)
    assert 1 <= auto <= 10
    assert replicate_worker_count(10, n_workers=3) == 3
    assert replicate_worker_count(2, n_workers=8) == 2


def test_map_replicates_preserves_order():
    from ml_for_malaria.runs import map_replicates

    assert map_replicates(_double_seed, [1, 2, 3], n_workers=1) == [2, 4, 6]
    assert map_replicates(_double_seed, [1, 2, 3], n_workers=2) == [2, 4, 6]


def test_completed_run_dirs_lists_report_json(tmp_path: Path):
    done = tmp_path / "xgb_random"
    done.mkdir()
    (done / RunCheckpointer.REPORT_JSON).write_text("{}", encoding="utf-8")
    nested = tmp_path / "xgb_scaffold" / "seed_42"
    nested.mkdir(parents=True)
    (nested / RunCheckpointer.REPORT_JSON).write_text("{}", encoding="utf-8")
    (tmp_path / "empty").mkdir()
    (tmp_path / "xgb_scaffold" / "features").mkdir()
    assert completed_run_dirs(tmp_path) == [done, nested]


def test_get_splitter_unknown():
    with pytest.raises(ValueError, match="Unknown split"):
        get_splitter("cluster")


def test_scaffold_splitter_keeps_murcko_groups_together():
    benzenes = [
        "c1ccccc1",
        "c1ccc(C)cc1",
        "c1ccc(O)cc1",
        "c1ccc(N)cc1",
        "c1ccc(F)cc1",
        "c1ccc(Cl)cc1",
        "c1ccc(Br)cc1",
        "c1ccc(CC)cc1",
    ]
    aliphatics = ["CCO", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCOCC", "CC(C)C", "CCCCO"]
    smiles = pd.Series(benzenes + aliphatics)
    labels = pd.Series([1] * len(benzenes) + [0] * len(aliphatics))
    splitter = get_splitter("scaffold")
    assert isinstance(splitter, ScaffoldSplitter)

    train_idx, test_idx = splitter.split(smiles, labels, test_size=0.25, seed=42)
    n = len(smiles)
    assert set(train_idx).isdisjoint(test_idx)
    assert sorted(train_idx + test_idx) == list(range(n))
    assert 0.1 <= len(test_idx) / n <= 0.45

    keys = murcko_group_keys(smiles)
    shared = ~keys.str.startswith("no_scaffold:")
    train_keys = set(keys.iloc[train_idx][shared.iloc[train_idx]])
    test_keys = set(keys.iloc[test_idx][shared.iloc[test_idx]])
    assert train_keys.isdisjoint(test_keys)

    again_train, again_test = splitter.split(smiles, labels, test_size=0.25, seed=42)
    assert train_idx == again_train
    assert test_idx == again_test
    other_train, other_test = splitter.split(smiles, labels, test_size=0.25, seed=1)
    assert (train_idx, test_idx) != (other_train, other_test)


def test_scramble_train_labels_permutes_train_only():
    labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    train_idx = [0, 1, 2, 3, 4, 5]
    test_idx = [6, 7]
    scrambled = scramble_train_labels(labels, train_idx, seed=0)
    assert scrambled.iloc[test_idx].tolist() == labels.iloc[test_idx].tolist()
    assert sorted(scrambled.iloc[train_idx].tolist()) == sorted(
        labels.iloc[train_idx].tolist()
    )
    assert scrambled.iloc[train_idx].tolist() != labels.iloc[train_idx].tolist()
    assert scramble_train_labels(labels, train_idx, seed=0).equals(scrambled)


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


def test_sanitize_smiles_strips_salts_and_disconnected_fragments():
    assert sanitize_smiles("CCO") == "CCO"
    assert sanitize_smiles("Cl.CCO") == "CCO"
    assert sanitize_smiles("OC(=O)C(F)(F)F.CCO") == "CCO"
    parent = (
        "CC(C)(NC(=O)C1=CC=CN=C1)C1=NC(=C(N1)C1=CC=NC=C1)C1=CC=C(Cl)C(O)=C1"
    )
    salted = f"Cl.{parent}"
    assert sanitize_smiles(salted) == sanitize_smiles(parent)
    assert "." not in sanitize_smiles(salted)
    assert sanitize_smiles("CCCCCCCC.C") == "CCCCCCCC"


def test_sanitize_smiles_neutralizes_charges():
    assert sanitize_smiles("CC(=O)[O-]") == sanitize_smiles("CC(=O)O") == "CC(=O)O"
    assert sanitize_smiles("[Na+].CC(=O)[O-]") == "CC(=O)O"
    assert sanitize_smiles("[NH3+]CC(=O)[O-]") == sanitize_smiles("NCC(=O)O")
    assert sanitize_smiles("C[NH+](C)C") == sanitize_smiles("CN(C)C")


def test_morgan3_feat_bits_is_registered():
    generator = get_fingerprint_generator("Morgan3FeatBits", fp_size=128)
    features = featurize_smiles(["CCO"], fp_generator=generator, sanitize=False)
    assert features.shape == (1, 128)


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
        EvalMetrics.pr_auc,
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
            "AtomPair": FingerprintScore(
                cv_auc=0.9, n_estimators=50, test_metrics=metrics
            )
        },
        test_metrics=metrics,
        architecture=Architecture.XGBOOST,
    )
    markdown = report_to_markdown(report)
    assert EvalMetrics.roc_auc in markdown
    assert EvalMetrics.pr_auc in markdown
    assert 0.0 <= metrics.pr_auc <= 1.0
    with pytest.raises(ValueError, match="same length"):
        compute_test_metrics([0, 1], [0, 1], [0.1])
    assert "AtomPair" in markdown
    assert FingerprintComparison.fingerprint in markdown
    assert FingerprintComparison.f1_0 in markdown
    assert "true_0" in markdown
    assert "pred_1" in markdown
    assert "### AtomPair" in markdown


def test_comparison_report_does_not_warn_on_mixed_seeds(tmp_path: Path):
    metrics = compute_test_metrics([0, 1], [0, 1], [0.1, 0.9])
    left = tmp_path / "xgb"
    right = tmp_path / "chemprop"
    left.mkdir()
    right.mkdir()
    write_report(
        build_report(
            split="scaffold",
            seed=42,
            test_size=0.2,
            n_train=10,
            n_test=4,
            test_metrics=metrics,
            architecture=Architecture.XGBOOST,
            best_fingerprint="AtomPair",
        ),
        left / "report.json",
        left / "report.md",
    )
    write_report(
        build_report(
            split="scaffold",
            seed=1,
            test_size=0.2,
            n_train=10,
            n_test=4,
            test_metrics=metrics,
            architecture=Architecture.CHEMPROP,
            charge_method=None,
        ),
        right / "report.json",
        right / "report.md",
    )
    out = tmp_path / "comparison.md"
    comparison = write_comparison_report([left, right], out)
    assert len(comparison.rows) == 2
    assert comparison.rows[0].architecture == Architecture.XGBOOST
    assert comparison.rows[1].identifier == "none"
    assert not comparison.warnings
    markdown = out.read_text(encoding="utf-8")
    assert ComparisonTable.split in markdown
    assert "±" in markdown or "mean" in markdown
    assert out.exists()
    assert out.with_suffix(".json").exists()


def test_comparison_report_shows_split_without_mismatch_warning(tmp_path: Path):
    metrics = compute_test_metrics([0, 1], [0, 1], [0.1, 0.9])
    left = tmp_path / "xgb_random"
    right = tmp_path / "xgb_scaffold"
    left.mkdir()
    right.mkdir()
    write_report(
        build_report(
            split="random",
            seed=42,
            test_size=0.2,
            n_train=10,
            n_test=4,
            test_metrics=metrics,
            architecture=Architecture.XGBOOST,
            best_fingerprint="AtomPair",
        ),
        left / "report.json",
        left / "report.md",
    )
    write_report(
        build_report(
            split="scaffold",
            seed=42,
            test_size=0.2,
            n_train=10,
            n_test=4,
            test_metrics=metrics,
            architecture=Architecture.XGBOOST,
            best_fingerprint="Morgan",
        ),
        right / "report.json",
        right / "report.md",
    )
    comparison = write_comparison_report([left, right], tmp_path / "comparison.md")
    assert not comparison.warnings
    markdown = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "random" in markdown
    assert "scaffold" in markdown


def test_checkpointer_reuses_matching_config(tmp_path: Path):
    ckpt = RunCheckpointer(tmp_path, force=False)
    path = tmp_path / "item.json"
    ckpt.save_json(path, {"ok": True})
    stored = {RunConfig.seed: 42, RunConfig.split: "random"}
    assert ckpt.should_reuse(
        path, stored, {RunConfig.seed: 42, RunConfig.split: "random"}
    )
    assert not ckpt.should_reuse(
        path, stored, {RunConfig.seed: 1, RunConfig.split: "random"}
    )
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
    from ml_for_malaria.model import load_classifier
    from ml_for_malaria.train import train_xgb_classifier

    df = toy_binary_df()
    result = train_xgb_classifier(
        df,
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        max_evals=1,
        fingerprints=["AtomPair"],
        fp_size=128,
    )
    run = resolve_run_dir(
        tmp_path, Architecture.XGBOOST, "random", seed=0, hpo=True
    )
    assert result.outdir == run
    assert result.report.max_evals == 1
    assert result.report.best_fingerprint == "AtomPair"
    assert result.report.architecture == Architecture.XGBOOST
    assert (run / "report.json").exists()
    assert (run / "report.md").exists()
    assert (run / "model.ubj").exists()
    assert (run / "model_meta.json").exists()
    assert (run / "models" / "AtomPair" / "model.ubj").exists()
    score = result.report.fingerprint_comparison["AtomPair"]
    assert score.test_metrics is not None
    assert score.test_metrics.roc_auc == result.report.test_metrics.roc_auc
    markdown = (run / "report.md").read_text(encoding="utf-8")
    assert FingerprintComparison.f1_1 in markdown
    preds = result.classifier.predict(["CCO", "c1ccncc1"])
    assert Predictions.PROBABILITY in preds.columns
    assert len(preds) == 2
    loaded = load_classifier(run)
    assert loaded.metadata.fingerprint == "AtomPair"
    per_fp = load_classifier(run, fingerprint="AtomPair")
    assert per_fp.metadata.fingerprint == "AtomPair"

    reused = train_xgb_classifier(
        df,
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        max_evals=1,
        fingerprints=["AtomPair"],
        fp_size=128,
    )
    assert reused.outdir == run
    assert reused.report.test_metrics.roc_auc == result.report.test_metrics.roc_auc


def test_train_xgb_classifier_skips_cv_when_max_evals_zero(tmp_path: Path):
    from ml_for_malaria.chemistry.featurization import DEFAULT_FINGERPRINT
    from ml_for_malaria.train import train_xgb_classifier

    df = toy_binary_df()
    result = train_xgb_classifier(
        df,
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        max_evals=0,
        fingerprints=["AtomPair", DEFAULT_FINGERPRINT],
        fp_size=128,
    )
    run = resolve_run_dir(tmp_path, Architecture.XGBOOST, "random", seed=0)
    assert result.outdir == run
    assert result.report.best_fingerprint == DEFAULT_FINGERPRINT
    assert set(result.report.fingerprint_comparison) == {
        "AtomPair",
        DEFAULT_FINGERPRINT,
    }
    assert result.report.fingerprint_comparison["AtomPair"].cv_auc is None
    assert not (run / "hyperopt" / "AtomPair.json").exists()
    assert (run / "model.ubj").exists()
    assert (run / "models" / DEFAULT_FINGERPRINT / "model.ubj").exists()


def test_train_rf_classifier_smoke_and_checkpoint_reuse(tmp_path: Path):
    from ml_for_malaria.chemistry.featurization import DEFAULT_FINGERPRINT
    from ml_for_malaria.model import load_classifier
    from ml_for_malaria.train import train_rf_classifier

    df = toy_binary_df()
    result = train_rf_classifier(
        df,
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        fingerprints=["AtomPair", DEFAULT_FINGERPRINT],
        fp_size=128,
    )
    run = resolve_run_dir(tmp_path, Architecture.RANDOM_FOREST, "random", seed=0)
    assert result.outdir == run
    assert result.report.best_fingerprint == DEFAULT_FINGERPRINT
    assert result.report.architecture == Architecture.RANDOM_FOREST
    assert set(result.report.fingerprint_comparison) == {
        "AtomPair",
        DEFAULT_FINGERPRINT,
    }
    assert result.report.fingerprint_comparison["AtomPair"].cv_auc is None
    assert (run / "report.json").exists()
    assert (run / "model.joblib").exists()
    assert (run / "models" / DEFAULT_FINGERPRINT / "model.joblib").exists()
    preds = result.classifier.predict(["CCO", "c1ccncc1"])
    assert Predictions.PROBABILITY in preds.columns
    assert len(preds) == 2
    loaded = load_classifier(run)
    assert loaded.metadata.fingerprint == DEFAULT_FINGERPRINT
    per_fp = load_classifier(run, fingerprint="AtomPair")
    assert per_fp.metadata.fingerprint == "AtomPair"

    reused = train_rf_classifier(
        df,
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        fingerprints=["AtomPair", DEFAULT_FINGERPRINT],
        fp_size=128,
    )
    assert reused.outdir == run
    assert reused.report.test_metrics.roc_auc == result.report.test_metrics.roc_auc


def test_train_rf_classifier_hpo_writes_hpo_run_dir(tmp_path: Path):
    from ml_for_malaria.train import train_rf_classifier

    df = toy_binary_df()
    result = train_rf_classifier(
        df,
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        fingerprints=["AtomPair"],
        fp_size=128,
        max_evals=1,
    )
    run = resolve_run_dir(
        tmp_path, Architecture.RANDOM_FOREST, "random", seed=0, hpo=True
    )
    assert result.outdir == run
    assert result.report.max_evals == 1
    score = result.report.fingerprint_comparison["AtomPair"]
    assert score.cv_auc is not None
    assert (run / "hyperopt" / "AtomPair.json").exists()
    markdown = (run / "report.md").read_text(encoding="utf-8")
    assert "train folds only" in markdown


def test_train_rf_classifier_yscramble_writes_yscramble_run_dir(tmp_path: Path):
    from ml_for_malaria.chemistry.featurization import (
        DEFAULT_FINGERPRINT,
        clean_training_data,
    )
    from ml_for_malaria.train import train_rf_classifier

    df = toy_binary_df()
    with pytest.raises(ValueError, match="Y-scramble"):
        train_rf_classifier(
            df,
            outdir=tmp_path,
            split="random",
            seed=0,
            test_size=0.25,
            fingerprints=["AtomPair"],
            fp_size=128,
            max_evals=1,
            yscramble=True,
        )
    result = train_rf_classifier(
        df,
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        fingerprints=["AtomPair", DEFAULT_FINGERPRINT],
        fp_size=128,
        yscramble=True,
    )
    run = resolve_run_dir(
        tmp_path, Architecture.RANDOM_FOREST, "random", seed=0, yscramble=True
    )
    assert result.outdir == run
    assert result.report.yscramble is True
    cleaned = pd.read_parquet(run / "cleaned.parquet")
    expected = clean_training_data(df)
    assert (
        cleaned[CleanedTrainingData.LABEL].tolist()
        == expected[CleanedTrainingData.LABEL].tolist()
    )
    markdown = (run / "report.md").read_text(encoding="utf-8")
    assert "y-scrambled" in markdown


@pytest.mark.parametrize(
    ("train_fn_name", "architecture"),
    [
        ("train_knn_classifier", Architecture.KNN),
        ("train_logistic_classifier", Architecture.LOGISTIC),
    ],
)
def test_train_sklearn_baseline_smoke_and_checkpoint_reuse(
    tmp_path: Path, train_fn_name: str, architecture: Architecture
):
    from ml_for_malaria.chemistry.featurization import DEFAULT_FINGERPRINT
    from ml_for_malaria.model import load_classifier
    from ml_for_malaria.train import train_knn_classifier, train_logistic_classifier

    train_fn = {
        "train_knn_classifier": train_knn_classifier,
        "train_logistic_classifier": train_logistic_classifier,
    }[train_fn_name]
    df = toy_binary_df()
    result = train_fn(
        df,
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        fingerprints=["AtomPair", DEFAULT_FINGERPRINT],
        fp_size=128,
    )
    run = resolve_run_dir(tmp_path, architecture, "random", seed=0)
    assert result.outdir == run
    assert result.report.best_fingerprint == DEFAULT_FINGERPRINT
    assert result.report.architecture == architecture
    assert set(result.report.fingerprint_comparison) == {
        "AtomPair",
        DEFAULT_FINGERPRINT,
    }
    assert result.report.fingerprint_comparison["AtomPair"].cv_auc is None
    assert (run / "report.json").exists()
    assert (run / "model.joblib").exists()
    assert (run / "models" / DEFAULT_FINGERPRINT / "model.joblib").exists()
    preds = result.classifier.predict(["CCO", "c1ccncc1"])
    assert Predictions.PROBABILITY in preds.columns
    assert len(preds) == 2
    loaded = load_classifier(run)
    assert loaded.metadata.fingerprint == DEFAULT_FINGERPRINT
    per_fp = load_classifier(run, fingerprint="AtomPair")
    assert per_fp.metadata.fingerprint == "AtomPair"

    reused = train_fn(
        df,
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        fingerprints=["AtomPair", DEFAULT_FINGERPRINT],
        fp_size=128,
    )
    assert reused.outdir == run
    assert reused.report.test_metrics.roc_auc == result.report.test_metrics.roc_auc


def test_train_knn_uses_jaccard_distance(tmp_path: Path):
    from ml_for_malaria.chemistry.featurization import DEFAULT_FINGERPRINT
    from ml_for_malaria.schemas import KNNMetric, KNNParams, KNNWeights
    from ml_for_malaria.train import train_knn_classifier

    df = toy_binary_df()
    result = train_knn_classifier(
        df,
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        fingerprints=[DEFAULT_FINGERPRINT],
        fp_size=128,
    )
    params = result.report.fingerprint_comparison[DEFAULT_FINGERPRINT].params
    assert params[KNNParams.metric] == KNNMetric.JACCARD
    assert params[KNNParams.weights] == KNNWeights.DISTANCE
    assert result.classifier.model.metric == KNNMetric.JACCARD
    assert result.classifier.model.n_neighbors == 5


def test_train_logistic_uses_l2(tmp_path: Path):
    from ml_for_malaria.chemistry.featurization import DEFAULT_FINGERPRINT
    from ml_for_malaria.schemas import (
        LogisticParams,
        LogisticSolver,
        SklearnClassWeight,
    )
    from ml_for_malaria.train import train_logistic_classifier

    df = toy_binary_df()
    result = train_logistic_classifier(
        df,
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        fingerprints=[DEFAULT_FINGERPRINT],
        fp_size=128,
    )
    params = result.report.fingerprint_comparison[DEFAULT_FINGERPRINT].params
    assert params[LogisticParams.C] == 1.0
    assert params[LogisticParams.solver] == LogisticSolver.LBFGS
    assert params[LogisticParams.class_weight] == SklearnClassWeight.BALANCED
    assert result.classifier.model.solver == LogisticSolver.LBFGS
    assert result.classifier.model.C == 1.0


@pytest.mark.parametrize(
    ("train_fn_name", "architecture", "match"),
    [
        ("train_knn_classifier", Architecture.KNN, "Tanimoto k-NN"),
        ("train_logistic_classifier", Architecture.LOGISTIC, "L2-logistic"),
    ],
)
def test_train_sklearn_baseline_rejects_hpo_and_writes_yscramble(
    tmp_path: Path, train_fn_name: str, architecture: Architecture, match: str
):
    from ml_for_malaria.chemistry.featurization import (
        DEFAULT_FINGERPRINT,
        clean_training_data,
    )
    from ml_for_malaria.train import train_knn_classifier, train_logistic_classifier

    train_fn = {
        "train_knn_classifier": train_knn_classifier,
        "train_logistic_classifier": train_logistic_classifier,
    }[train_fn_name]
    df = toy_binary_df()
    with pytest.raises(ValueError, match=match):
        train_fn(
            df,
            outdir=tmp_path,
            split="random",
            seed=0,
            test_size=0.25,
            fingerprints=["AtomPair"],
            fp_size=128,
            max_evals=1,
        )
    result = train_fn(
        df,
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        fingerprints=["AtomPair", DEFAULT_FINGERPRINT],
        fp_size=128,
        yscramble=True,
    )
    run = resolve_run_dir(tmp_path, architecture, "random", seed=0, yscramble=True)
    assert result.outdir == run
    assert result.report.yscramble is True
    cleaned = pd.read_parquet(run / "cleaned.parquet")
    expected = clean_training_data(df)
    assert (
        cleaned[CleanedTrainingData.LABEL].tolist()
        == expected[CleanedTrainingData.LABEL].tolist()
    )
    markdown = (run / "report.md").read_text(encoding="utf-8")
    assert "y-scrambled" in markdown


def test_comparison_report_aggregates_mean_std_across_seeds(tmp_path: Path):
    left = tmp_path / "seed42"
    right = tmp_path / "seed43"
    left.mkdir()
    right.mkdir()
    write_report(
        build_report(
            split="scaffold",
            seed=42,
            test_size=0.2,
            n_train=10,
            n_test=4,
            test_metrics=compute_test_metrics([0, 1], [0, 1], [0.1, 0.9]),
            architecture=Architecture.CHEMPROP,
            charge_method=None,
        ),
        left / "report.json",
        left / "report.md",
    )
    write_report(
        build_report(
            split="scaffold",
            seed=43,
            test_size=0.2,
            n_train=10,
            n_test=4,
            test_metrics=compute_test_metrics([0, 1], [1, 1], [0.4, 0.8]),
            architecture=Architecture.CHEMPROP,
            charge_method=None,
        ),
        right / "report.json",
        right / "report.md",
    )
    comparison = write_comparison_report([left, right], tmp_path / "comparison.md")
    assert not comparison.warnings
    assert len(comparison.aggregates) == 1
    assert comparison.aggregates[0].n_seeds == 2
    assert comparison.aggregates[0].roc_auc_std is not None
    markdown = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "±" in markdown


def test_comparison_report_annotates_hpo_improvement(tmp_path: Path):
    fixed = tmp_path / "fixed"
    weaker = tmp_path / "weaker"
    tuned = tmp_path / "hpo"
    fixed.mkdir()
    weaker.mkdir()
    tuned.mkdir()
    base = compute_test_metrics([0, 1], [0, 1], [0.1, 0.9])
    write_report(
        build_report(
            split="scaffold",
            seed=42,
            test_size=0.2,
            n_train=10,
            n_test=4,
            test_metrics=base.model_copy(
                update={EvalMetrics.roc_auc: 0.80, EvalMetrics.pr_auc: 0.75}
            ),
            architecture=Architecture.XGBOOST,
            best_fingerprint="RDKit",
        ),
        fixed / "report.json",
        fixed / "report.md",
    )
    write_report(
        build_report(
            split="scaffold",
            seed=42,
            test_size=0.2,
            n_train=10,
            n_test=4,
            test_metrics=base.model_copy(
                update={EvalMetrics.roc_auc: 0.70, EvalMetrics.pr_auc: 0.65}
            ),
            architecture=Architecture.XGBOOST,
            best_fingerprint="AtomPair",
        ),
        weaker / "report.json",
        weaker / "report.md",
    )
    write_report(
        build_report(
            split="scaffold",
            seed=42,
            test_size=0.2,
            n_train=10,
            n_test=4,
            test_metrics=base.model_copy(
                update={EvalMetrics.roc_auc: 0.85, EvalMetrics.pr_auc: 0.82}
            ),
            architecture=Architecture.XGBOOST,
            best_fingerprint="RDKit",
            max_evals=50,
        ),
        tuned / "report.json",
        tuned / "report.md",
    )
    comparison = write_comparison_report(
        [fixed, weaker, tuned], tmp_path / "comparison.md"
    )
    assert len(comparison.hpo_deltas) == 1
    delta = comparison.hpo_deltas[0]
    assert delta.identifier == "RDKit"
    assert delta.roc_auc_fixed == pytest.approx(0.80)
    assert delta.roc_auc_hpo == pytest.approx(0.85)
    assert delta.roc_auc_delta == pytest.approx(0.05)
    assert (
        best_fixed_scaffold_identifier(comparison.aggregates, Architecture.XGBOOST)
        == "RDKit"
    )
    markdown = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "HPO improvement vs fixed recipe" in markdown
    assert ComparisonRow.hpo in markdown or "True" in markdown


def test_comparison_report_random_rf_references_best_scaffold(tmp_path: Path):
    random_run = tmp_path / "random"
    scaffold_best = tmp_path / "scaffold_best"
    scaffold_other = tmp_path / "scaffold_other"
    random_run.mkdir()
    scaffold_best.mkdir()
    scaffold_other.mkdir()
    base = compute_test_metrics([0, 1], [0, 1], [0.1, 0.9])
    write_report(
        build_report(
            split="random",
            seed=42,
            test_size=0.2,
            n_train=10,
            n_test=8,
            test_metrics=base.model_copy(
                update={EvalMetrics.roc_auc: 0.90, EvalMetrics.pr_auc: 0.88}
            ),
            architecture=Architecture.RANDOM_FOREST,
            best_fingerprint="AtomPair",
        ),
        random_run / "report.json",
        random_run / "report.md",
    )
    write_report(
        build_report(
            split="scaffold",
            seed=42,
            test_size=0.2,
            n_train=10,
            n_test=4,
            test_metrics=base.model_copy(
                update={EvalMetrics.roc_auc: 0.93, EvalMetrics.pr_auc: 0.92}
            ),
            architecture=Architecture.RANDOM_FOREST,
            best_fingerprint="Morgan2Bits",
        ),
        scaffold_best / "report.json",
        scaffold_best / "report.md",
    )
    write_report(
        build_report(
            split="scaffold",
            seed=42,
            test_size=0.2,
            n_train=10,
            n_test=4,
            test_metrics=base.model_copy(
                update={EvalMetrics.roc_auc: 0.88, EvalMetrics.pr_auc: 0.85}
            ),
            architecture=Architecture.RANDOM_FOREST,
            best_fingerprint="AtomPair",
        ),
        scaffold_other / "report.json",
        scaffold_other / "report.md",
    )
    comparison = write_comparison_report(
        [random_run, scaffold_best, scaffold_other], tmp_path / "comparison.md"
    )
    assert len(comparison.split_references) == 1
    ref = comparison.split_references[0]
    assert ref.identifier == "AtomPair"
    assert ref.reference_identifier == "Morgan2Bits"
    assert ref.roc_auc == pytest.approx(0.90)
    assert ref.roc_auc_reference == pytest.approx(0.93)
    assert ref.roc_auc_delta == pytest.approx(-0.03)
    markdown = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "Random split vs best scaffold (reference)" in markdown
    assert "Morgan2Bits" in markdown


def test_comparison_report_annotates_yscramble_drop(tmp_path: Path):
    real = tmp_path / "real"
    scramble = tmp_path / "yscramble"
    real.mkdir()
    scramble.mkdir()
    base = compute_test_metrics([0, 1], [0, 1], [0.1, 0.9])
    write_report(
        build_report(
            split="scaffold",
            seed=42,
            test_size=0.2,
            n_train=10,
            n_test=4,
            test_metrics=base.model_copy(
                update={EvalMetrics.roc_auc: 0.93, EvalMetrics.pr_auc: 0.92}
            ),
            architecture=Architecture.RANDOM_FOREST,
            best_fingerprint="Morgan2Bits",
        ),
        real / "report.json",
        real / "report.md",
    )
    write_report(
        build_report(
            split="scaffold",
            seed=42,
            test_size=0.2,
            n_train=10,
            n_test=4,
            test_metrics=base.model_copy(
                update={EvalMetrics.roc_auc: 0.52, EvalMetrics.pr_auc: 0.51}
            ),
            architecture=Architecture.RANDOM_FOREST,
            best_fingerprint="Morgan2Bits",
            yscramble=True,
        ),
        scramble / "report.json",
        scramble / "report.md",
    )
    comparison = write_comparison_report(
        [real, scramble], tmp_path / "comparison.md"
    )
    assert len(comparison.yscramble_deltas) == 1
    delta = comparison.yscramble_deltas[0]
    assert delta.identifier == "Morgan2Bits"
    assert delta.roc_auc_real == pytest.approx(0.93)
    assert delta.roc_auc_scramble == pytest.approx(0.52)
    assert delta.roc_auc_delta == pytest.approx(-0.41)
    assert (
        best_fixed_scaffold_identifier(
            comparison.aggregates, Architecture.RANDOM_FOREST
        )
        == "Morgan2Bits"
    )
    markdown = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "Y-scramble (train labels permuted)" in markdown


