# Architecture

This document is the placement contract for `ml_for_malaria`. Existing Cursor rules still apply (leakage-safe splits, schemas over string literals, pandas over row loops, assay context). This file answers **where code goes** and **what may import what**.

Do not collapse chemistry, split, runs, or report back into `train/`. `train/` is fit loops only.

## Package map

```text
ml_for_malaria/
  chemistry/        # molecules: sanitise, fingerprints, charges
  split.py          # compound-level train/test splitters
  runs/             # run directories + checkpoint I/O
  report.py         # held-out metrics, markdown/JSON reports
  train/            # architecture trainers + shared prepare_training_run
  model/            # load / predict classifiers
  interpretation/   # SHAP / atom highlights
  schemas.py        # Pandera tables + Pydantic payloads
contrib/<dataset>/  # ingest CSVs, call library trainers, write run dirs
```

| Package | Put here | Do not put here |
| --- | --- | --- |
| `chemistry` | SMILES sanitise/standardise, fingerprint generators, bit-to-atom maps, partial charges | Split indices, checkpoints, sklearn metrics, torch training loops |
| `split` | Random / scaffold (or later) **compound** splitters returning positional indices | Fingerprint generation, model fitting |
| `runs` | `resolve_run_dir`, `RunCheckpointer`, hashes, `config.json` / parquet paths | Featurization, metric formulas |
| `report` | `compute_test_metrics`, `TrainingReport` markdown, `write_comparison_report` | Training, RDKit |
| `train` | `prepare_training_run`, `train_xgb_classifier`, `train_rf_classifier`, `train_knn_classifier`, `train_logistic_classifier`, `train_chemprop_classifier`, `train_smiles_transformer` | Anything needed at **predict** time |
| `model` | Classifier classes, `load_classifier`, shared predict SMILES alignment | HPO, splitting, writing `report.md` |
| `interpretation` | SHAP and structure highlighting | Training |
| `contrib` | Dataset-specific column names, paths, script entrypoints | Library logic that another dataset would reuse |

`chemistry` is used at **train and predict**. That is why it is not a “preparation” folder: fingerprints and charges must stay aligned with the saved model, not live only in the trainer.

## Dependency direction

Allowed (downstream → upstream):

```text
contrib → train / model / chemistry / runs / report / schemas
train   → chemistry, split, runs, report, model, schemas
model   → chemistry, runs, schemas   (and interpretation from XGB predict helpers)
report  → runs, schemas
runs    → schemas                    (paths may lazy-import checkpoints)
chemistry → schemas, RDKit/datamol
```

Forbidden:

- `chemistry` must not import `train`, `model`, or `report`.
- `split` must not import `train` or `model`.
- `model` must not import `train` (load/predict cannot depend on a trainer).
- `runs` must not import `train` or `chemistry`.

New helpers go in the **lowest** package that matches the table above, not in the caller.

## Training vs inference

1. **Ingest** (contrib only): rename raw columns onto `CleanedTrainingData` / `Predictions`. String literals for vendor CSV headers stay here.
2. **Prepare** (`train.prepare`): `clean_training_data` + `get_splitter` + freeze `SplitIndices` via `RunCheckpointer`. Split **compounds** before comparing fingerprints or architectures. Reuse the checkpointed split when hashes match.
3. **Fit** (`train.xgb` / `rf` / `knn` / `logistic` / `chemprop` / `chemberta`): train only on train indices. Inner val is carved from train when the architecture uses early stopping. Never tune on test. TPE (`max_evals>=1`) is train-fold ROC-AUC only (XGBoost / random forest). Y-scramble (`yscramble=True`) permutes **train** labels only; test labels stay real. Do not combine y-scramble with HPO. Tanimoto k-NN and L2-logistic are fixed-recipe fingerprint baselines (no TPE).
4. **Select** (XGBoost / random forest / k-NN / logistic): when `max_evals>=1`, cross-validation AUC picks the default fingerprint / `model.ubj` or `model.joblib`. When `max_evals=0`, the default artifact is `DEFAULT_FINGERPRINT` (Morgan2FeatBits) if that generator was trained. k-NN and logistic always use the fixed recipe. Held-out test metrics are reported for every fingerprint and are **not** the selector.
5. **Persist**: `resolve_run_dir(parent, architecture, split, charge_method=..., seed=..., hpo=..., yscramble=...)` under the contrib parent (e.g. `data/pfpkg/runs/pfpkg`). Omit `seed` for the legacy experiment folder (`xgb_random`). With a seed, trainers create `{arch}_{split}[_charge][_hpo][_yscramble]/seed_{n}` (e.g. `xgb_scaffold/seed_42`, `rf_scaffold_yscramble/seed_42`, `knn_scaffold/seed_42`). `hpo=True` / `yscramble=True` keep those runs from overwriting the fixed recipe. `replicate_seeds(n_rep, start=…)` is the library helper for consecutive replicate seeds; contrib chooses `n_rep`. The pfpkg suite runs RF / XGB / k-NN / logistic seeds (including HPO) in a process pool (`map_replicates`); Chemprop/ChemBERTa stay sequential on one GPU. TPE trials inside a seed stay serial. Shared Chemprop charges live in `parent/charges/{method}.parquet`; shared fingerprint features (XGB, random forest, k-NN, logistic) in `parent/features/`.
6. **Score**: `compute_test_metrics` on the frozen test split; write `report.md` / `report.json`. Compare architectures with `write_comparison_report`. HPO and y-scramble rows are grouped separately. Comparison markdown annotates HPO Δ vs the matching fixed recipe, y-scramble Δ vs real labels (train permutation only), and random-split RF vs the best scaffold identifier as a **reference** (not a ranking).
7. **Predict**: `load_classifier(outdir)` (or a specific classifier `.load`). Sanitize and featurize with the **same** `chemistry` functions and the metadata in the run dir (`fingerprint`, `fp_size`, `charge_method`, `pretrained_name`).

Predictions are triage hypotheses: persist probability and interpretation with the model, not a potency claim.

## Optional deep learning

Chemprop and ChemBERTa live behind the `dl` extra. Do **not** import them from `ml_for_malaria.train.__init__` — that would pull torch on every `from ml_for_malaria.train import train_xgb_classifier`.

```python
from ml_for_malaria.train import (
    train_rf_classifier,
    train_xgb_classifier,
    train_knn_classifier,
    train_logistic_classifier,
)
from ml_for_malaria.train.chemprop import train_chemprop_classifier
from ml_for_malaria.train.chemberta import train_smiles_transformer
from ml_for_malaria.model import load_classifier
from ml_for_malaria.chemistry import encode_binary_labels, sanitize_smiles, atom_charges
from ml_for_malaria.runs import (
    resolve_run_dir,
    completed_run_dirs,
    RunCheckpointer,
    replicate_seeds,
)
from ml_for_malaria.report import write_comparison_report, compute_test_metrics
from ml_for_malaria.split import get_splitter
```

`charge_method` (`None` / `gasteiger` / `nagl`) is Chemprop-only extra atom features. ChemBERTa has no charge vector. Failed sanitize or charge assignment **drops** the molecule; do not impute `q=0` or a dummy structure. `nagl` is the `nagl` extra (`uv sync --extra nagl`); `require_charge_backend` fails before training if those packages are missing. SMILES-keyed charge vectors are cached in `chemistry` (`charges_for_smiles`) so multi-seed Chemprop does not recompute NAGL; the parquet lives on the contrib runs parent, not inside a seed run dir.

## Runs vs the library package

Artifact directories may be named `runs/` (gitignored under `data/` and repo-root `/runs/`). The Python package is `ml_for_malaria.runs`. Do not “fix” that collision by moving checkpoint code into `train/`.

## Placement checklist

When adding a function, pick the first match:

1. Needed to **predict** from a saved run? → `chemistry` or `model`.
2. Compound **indices** only? → `split`.
3. Writing/reading a run folder? → `runs`.
4. ROC-AUC / F1 / markdown tables? → `report`.
5. Fit loop or HPO for one architecture? → `train/<arch>.py`.
6. Dataset CSV quirks? → `contrib/<dataset>/`.

If it seems to belong in `train/` but predict would need it too, it does not belong in `train/`.
