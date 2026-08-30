import importlib.util

import numpy as np
import pytest
from rdkit import Chem

from ml_for_malaria.chemistry import (
    ChargeAssignmentError,
    atom_charges,
    charge_cache_path,
    charges_for_smiles,
    parse_charge_method,
    require_charge_backend,
)
from ml_for_malaria.schemas import ChargeMethod


def test_parse_charge_method_accepts_none_and_known():
    assert parse_charge_method(None) is None
    assert parse_charge_method(ChargeMethod.GASTEIGER) == ChargeMethod.GASTEIGER
    assert parse_charge_method(ChargeMethod.NAGL) == ChargeMethod.NAGL
    with pytest.raises(ValueError, match="Unknown charge_method"):
        parse_charge_method("am1bcc")


def test_gasteiger_charges_match_atom_count():
    mol = Chem.MolFromSmiles("CCO")
    charges = atom_charges(mol, ChargeMethod.GASTEIGER)
    assert charges.shape == (mol.GetNumAtoms(), 1)
    assert np.isfinite(charges).all()
    assert abs(float(charges.sum()) - float(Chem.GetFormalCharge(mol))) < 1.5


def test_require_charge_backend_nagl_imports_or_raises():
    require_charge_backend(None)
    require_charge_backend(ChargeMethod.GASTEIGER)
    nagl_installed = (
        importlib.util.find_spec("openff.nagl") is not None
        and importlib.util.find_spec("openff.toolkit") is not None
    )
    if not nagl_installed:
        with pytest.raises(ImportError, match="uv sync --extra"):
            require_charge_backend(ChargeMethod.NAGL)
        return
    require_charge_backend(ChargeMethod.NAGL)


def test_nagl_charges_or_skip():
    pytest.importorskip("openff.toolkit")
    pytest.importorskip("openff.nagl")
    mol = Chem.MolFromSmiles("CCO")
    try:
        charges = atom_charges(mol, ChargeMethod.NAGL)
    except ChargeAssignmentError:
        pytest.skip("NAGL model is not available")
    assert charges.shape == (mol.GetNumAtoms(), 1)
    assert np.isfinite(charges).all()
    again = atom_charges(Chem.MolFromSmiles("CC"), ChargeMethod.NAGL)
    assert again.shape == (Chem.MolFromSmiles("CC").GetNumAtoms(), 1)


def test_gasteiger_charge_cache_roundtrip(tmp_path, monkeypatch):
    calls = {"n": 0}
    real_assign = atom_charges

    def counting_assign(mol, method):
        calls["n"] += 1
        return real_assign(mol, method)

    monkeypatch.setattr(
        "ml_for_malaria.chemistry.charges.atom_charges", counting_assign
    )
    cache = charge_cache_path(tmp_path, ChargeMethod.GASTEIGER)
    smiles = ["CCO", "c1ccccc1"]
    first = charges_for_smiles(smiles, ChargeMethod.GASTEIGER, cache)
    assert cache.exists()
    assert calls["n"] == 2
    assert set(first) == set(smiles)
    second = charges_for_smiles(smiles, ChargeMethod.GASTEIGER, cache)
    assert calls["n"] == 2
    for smi in smiles:
        np.testing.assert_allclose(first[smi], second[smi])
