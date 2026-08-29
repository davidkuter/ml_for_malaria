import numpy as np
import pytest
from rdkit import Chem

from ml_for_malaria.schemas import ChargeMethod
from ml_for_malaria.train.charges import (
    ChargeAssignmentError,
    atom_charges,
    parse_charge_method,
)


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
