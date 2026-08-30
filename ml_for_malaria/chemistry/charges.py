from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

from ml_for_malaria.schemas import AtomChargeCache, ChargeMethod, empty_frame

NAGL_PARTIAL_CHARGE_METHOD = "openff-gnn-am1bcc-1.0.0.pt"
NAGL_INSTALL_HINT = (
    "charge_method='nagl' requires OpenFF NAGL. Install the optional extra:\n"
    "  uv sync --extra dl --extra nagl\n"
    "Then retry, or use charge_method='gasteiger' / None."
)
_NAGL_LOCK = threading.Lock()
_NAGL_GNN = None


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


def require_charge_backend(method: str | None) -> None:
    """Fail immediately if ``method`` needs a package that is not installed."""
    parsed = parse_charge_method(method)
    if parsed != ChargeMethod.NAGL:
        return
    try:
        import openff.nagl  # noqa: F401
        from openff.toolkit.topology import Molecule  # noqa: F401

        _nagl_gnn()
    except ImportError as exc:
        raise ImportError(NAGL_INSTALL_HINT) from exc


def _nagl_gnn():
    """Load the NAGL GNN once; ``assign_partial_charges`` reloads it per molecule."""
    global _NAGL_GNN
    with _NAGL_LOCK:
        if _NAGL_GNN is None:
            from openff.nagl import GNNModel
            from openff.nagl_models._dynamic_fetch import get_model

            logger.info(f"Loading NAGL charge model {NAGL_PARTIAL_CHARGE_METHOD}")
            path = get_model(filename=NAGL_PARTIAL_CHARGE_METHOD)
            _NAGL_GNN = GNNModel.load(path, eval_mode=True)
            logger.info("NAGL charge model ready")
        return _NAGL_GNN


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
    require_charge_backend(ChargeMethod.NAGL)
    from openff.toolkit.topology import Molecule
    from openff.units import Quantity, unit

    try:
        import torch

        offmol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
        model = _nagl_gnn()
        with torch.inference_mode():
            raw = model.compute_property(
                offmol,
                as_numpy=True,
                readout_name="am1bcc_charges",
                check_domains=False,
                error_if_unsupported=False,
            )
        offmol.partial_charges = Quantity(
            np.asarray(raw, dtype=np.float64),
            unit.elementary_charge,
        )
        offmol._normalize_partial_charges()
        charges = offmol.partial_charges
        if charges is None:
            raise ChargeAssignmentError("NAGL returned no partial charges")
        values = np.asarray(charges.m, dtype=np.float64)
    except ImportError:
        raise
    except (ValueError, RuntimeError, AttributeError, TypeError) as exc:
        raise ChargeAssignmentError("NAGL charge assignment failed") from exc
    aligned = _align_nagl_charges_to_rdkit(mol, offmol, values)
    if aligned.size != mol.GetNumAtoms() or not np.isfinite(aligned).all():
        raise ChargeAssignmentError(
            "NAGL charges do not align with RDKit atom order or are non-finite"
        )
    return aligned.reshape(-1, 1)


def _align_nagl_charges_to_rdkit(mol: Chem.Mol, offmol, values: np.ndarray) -> np.ndarray:
    """Map all-atom NAGL charges onto the RDKit mol Chemprop featurizes.

    OpenFF adds hydrogens before NAGL; Chemprop defaults to implicit-H graphs.
    Extra hydrogen charges are summed onto the bonded heavy atom.
    """
    n_rdkit = mol.GetNumAtoms()
    if values.size == n_rdkit:
        return values.astype(np.float64, copy=False)
    if values.size != offmol.n_atoms or values.size < n_rdkit:
        raise ChargeAssignmentError(
            "NAGL charges do not align with RDKit atom order or are non-finite"
        )
    for idx, atom in enumerate(mol.GetAtoms()):
        if int(offmol.atom(idx).atomic_number) != atom.GetAtomicNum():
            raise ChargeAssignmentError(
                "NAGL charges do not align with RDKit atom order or are non-finite"
            )
    aligned = np.array(values[:n_rdkit], dtype=np.float64, copy=True)
    for idx in range(n_rdkit, offmol.n_atoms):
        off_atom = offmol.atom(idx)
        if int(off_atom.atomic_number) != 1:
            raise ChargeAssignmentError(
                "NAGL charges do not align with RDKit atom order or are non-finite"
            )
        parents = [bonded.molecule_atom_index for bonded in off_atom.bonded_atoms]
        if len(parents) != 1 or parents[0] >= n_rdkit:
            raise ChargeAssignmentError(
                "NAGL charges do not align with RDKit atom order or are non-finite"
            )
        aligned[parents[0]] += values[idx]
    return aligned


def charge_cache_path(parent: str | Path, method: str) -> Path:
    """Parquet of SMILES-keyed charges under the runs parent, not a seed dir."""
    parsed = parse_charge_method(method)
    if parsed is None:
        raise ValueError("charge cache requires a charge_method")
    return Path(parent) / "charges" / f"{parsed}.parquet"


def load_charge_cache(path: str | Path) -> pd.DataFrame:
    cache_path = Path(path)
    if not cache_path.exists():
        return empty_frame(AtomChargeCache)
    loaded = pd.read_parquet(cache_path)
    return AtomChargeCache.validate(loaded)


def save_charge_cache(path: str | Path, cache: pd.DataFrame) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    AtomChargeCache.validate(cache).to_parquet(cache_path, index=False)


def _charges_from_cache_row(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def charges_for_smiles(
    smiles: list[str] | pd.Series,
    method: str,
    cache_path: str | Path | None = None,
    *,
    mols: list[Chem.Mol | None] | None = None,
    n_jobs: int = 1,
) -> dict[str, np.ndarray]:
    """Return successful charge vectors, filling ``cache_path`` for misses only."""
    parsed = parse_charge_method(method)
    if parsed is None:
        raise ValueError("charges_for_smiles requires a charge_method")
    requested = pd.Series(list(smiles), dtype=str)
    if mols is not None and len(mols) != len(requested):
        raise ValueError("mols must be the same length as smiles")
    cache = (
        load_charge_cache(cache_path)
        if cache_path is not None
        else empty_frame(AtomChargeCache)
    )
    mol_list = (
        list(mols) if mols is not None else [Chem.MolFromSmiles(smi) for smi in requested]
    )
    cached_map: dict[str, np.ndarray] = {}
    if not cache.empty:
        cached_map = {
            smi: _charges_from_cache_row(values)
            for smi, values in zip(
                cache[AtomChargeCache.SMILES], cache[AtomChargeCache.charges]
            )
        }
    found: dict[str, np.ndarray] = {}
    misses: list[tuple[str, Chem.Mol]] = []
    for smi, mol in zip(requested.tolist(), mol_list):
        if mol is None:
            continue
        charges = cached_map.get(smi)
        if charges is not None and charges.shape[0] == mol.GetNumAtoms():
            found[smi] = charges
            continue
        misses.append((smi, mol))

    def _assign(item: tuple[str, Chem.Mol]) -> tuple[str, np.ndarray] | None:
        smi, mol = item
        try:
            return smi, atom_charges(mol, parsed)
        except ChargeAssignmentError:
            logger.warning(f"Dropping {smi!r}: charge assignment failed")
            return None

    unique_misses: list[tuple[str, Chem.Mol]] = []
    seen: set[str] = set()
    for smi, mol in misses:
        if smi in seen:
            continue
        seen.add(smi)
        unique_misses.append((smi, mol))
    misses = unique_misses

    computed: list[tuple[str, np.ndarray]] = []
    if misses:
        if n_jobs != 1 and len(misses) > 1:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_assign)(item) for item in misses
            )
        else:
            results = [_assign(item) for item in misses]
        computed = [row for row in results if row is not None]

    found.update(dict(computed))
    if cache_path is not None and computed:
        added = pd.DataFrame(
            {
                AtomChargeCache.SMILES: [smi for smi, _ in computed],
                AtomChargeCache.charges: [q.reshape(-1).tolist() for _, q in computed],
            }
        )
        combined = pd.concat([cache, added], ignore_index=True)
        combined = combined.drop_duplicates(AtomChargeCache.SMILES, keep="last")
        save_charge_cache(cache_path, combined)
    return found
