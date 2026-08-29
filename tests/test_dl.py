from pathlib import Path

import pytest

from ml_for_malaria.model.smiles_transformer import TINY_TEST_NAME
from ml_for_malaria.runs import resolve_run_dir
from ml_for_malaria.schemas import Architecture, ChargeMethod, Predictions
from tests.toy_data import toy_binary_df


def test_chemprop_smoke_with_gasteiger(tmp_path: Path):
    pytest.importorskip("chemprop")
    pytest.importorskip("lightning")
    from ml_for_malaria.model import load_classifier
    from ml_for_malaria.train.chemprop import train_chemprop_classifier

    result = train_chemprop_classifier(
        toy_binary_df(),
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        charge_method=ChargeMethod.GASTEIGER,
        max_epochs=1,
        batch_size=8,
        hidden_size=16,
        depth=2,
        patience=1,
        accelerator="cpu",
        force=True,
        n_jobs=2,
    )
    run = resolve_run_dir(
        tmp_path,
        Architecture.CHEMPROP,
        "random",
        charge_method=ChargeMethod.GASTEIGER,
    )
    assert result.outdir == run
    assert result.report.architecture == Architecture.CHEMPROP
    assert result.report.charge_method == ChargeMethod.GASTEIGER
    assert (run / "model.ckpt").exists()
    preds = result.classifier.predict(["CCO", "c1ccccc1"])
    assert Predictions.PROBABILITY in preds.columns
    loaded = load_classifier(run)
    again = loaded.predict(["CCO"])
    assert len(again) == 1


def test_chemprop_smoke_without_charges(tmp_path: Path):
    pytest.importorskip("chemprop")
    pytest.importorskip("lightning")
    from ml_for_malaria.train.chemprop import train_chemprop_classifier

    result = train_chemprop_classifier(
        toy_binary_df(),
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        charge_method=None,
        max_epochs=1,
        batch_size=8,
        hidden_size=16,
        depth=2,
        patience=1,
        accelerator="cpu",
        force=True,
    )
    run = resolve_run_dir(tmp_path, Architecture.CHEMPROP, "random")
    assert result.outdir == run
    assert result.report.charge_method is None
    preds = result.classifier.predict(["CCO"])
    assert len(preds) == 1


def test_chemberta_tiny_test_does_not_download(tmp_path: Path):
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from ml_for_malaria.model import load_classifier
    from ml_for_malaria.train.chemberta import train_smiles_transformer

    result = train_smiles_transformer(
        toy_binary_df(),
        outdir=tmp_path,
        split="random",
        seed=0,
        test_size=0.25,
        pretrained_name=TINY_TEST_NAME,
        freeze_encoder=True,
        max_epochs=1,
        batch_size=8,
        accelerator="cpu",
        force=True,
    )
    run = resolve_run_dir(tmp_path, Architecture.CHEMBERTA, "random")
    assert result.outdir == run
    assert result.report.architecture == Architecture.CHEMBERTA
    assert result.report.pretrained_name == TINY_TEST_NAME
    assert (run / "hf_model" / "config.json").exists()
    preds = result.classifier.predict(["CCO", "c1ccccc1"])
    assert Predictions.PROBABILITY in preds.columns
    loaded = load_classifier(run)
    assert loaded.metadata.pretrained_name == TINY_TEST_NAME
    assert len(loaded.predict(["CCO"])) == 1
