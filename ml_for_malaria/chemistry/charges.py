from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

from ml_for_malaria.schemas import ChargeMethod

NAGL_PARTIAL_CHARGE_METHOD = "openff-gnn-am1bcc-1.0.0"


class ChargeAssignmentError(RuntimeError):
    """Raised when a molecule cannot be assigned partial charges."""


def parse_charge_method(method: str | None) -> str | None:
    if method is None:
        return None
    try:
        return ChargeMethod(method)
    except ValueError as exc:
        supported = ", ".join(m.value for m in ChargeMethod)
        raise ValueError(
            f"Unknown charge_method {method!r}. Supported: None, {supported}"
        ) from exc


def atom_charges(mol: Chem.Mol, method: str) -> np.ndarray:
    """Return RDKit-ordered partial charges with shape ``(n_atoms, 1)``."""
    parsed = parse_charge_method(method)
    if parsed is None:
        raise ValueError("charge_method is None; atom features are not requested")
    if parsed == ChargeMethod.GASTEIGER:
        return _gasteiger_charges(mol)
    return _nagl_charges(mol)


def _gasteiger_charges(mol: Chem.Mol) -> np.ndarray:
    charged = Chem.Mol(mol)
    try:
        rdPartialCharges.ComputeGasteigerCharges(charged)
    except (ValueError, RuntimeError) as exc:
        raise ChargeAssignmentError("Gasteiger charge assignment failed") from exc
    values = np.array(
        [atom.GetDoubleProp("_GasteigerCharge") for atom in charged.GetAtoms()],
        dtype=np.float64,
    )
    if values.size != charged.GetNumAtoms() or not np.isfinite(values).all():
        raise ChargeAssignmentError("Gasteiger produced non-finite charges")
    return values.reshape(-1, 1)


def _nagl_charges(mol: Chem.Mol) -> np.ndarray:
    try:
        from openff.toolkit.topology import Molecule
    except ImportError as exc:
        raise ImportError(
            "NAGL charges require OpenFF NAGL (conda-forge: openff-nagl, "
            "openff-nagl-models, and openff-toolkit). It is not a pip-only extra."
        ) from exc
    try:
        offmol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
        offmol.assign_partial_charges(partial_charge_method=NAGL_PARTIAL_CHARGE_METHOD)
        charges = offmol.partial_charges
        if charges is None:
            raise ChargeAssignmentError("NAGL returned no partial charges")
        values = np.asarray(charges.m, dtype=np.float64)
    except ImportError:
        raise
    except (ValueError, RuntimeError, AttributeError, TypeError) as exc:
        raise ChargeAssignmentError("NAGL charge assignment failed") from exc
    if values.size != mol.GetNumAtoms() or not np.isfinite(values).all():
        raise ChargeAssignmentError(
            "NAGL charges do not align with RDKit atom order or are non-finite"
        )
    return values.reshape(-1, 1)
