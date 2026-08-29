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
| `train` | `prepare_training_run`, `train_xgb_classifier`, `train_chemprop_classifier`, `train_smiles_transformer` | Anything needed at **predict** time |
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
3. **Fit** (`train.xgb` / `chemprop` / `chemberta`): train only on train indices. Inner val is carved from train. Never tune on test.
4. **Select** (XGBoost): cross-validation AUC picks the default fingerprint / `model.ubj`. Held-out test metrics are reported for every fingerprint and are **not** the selector.
5. **Persist**: `resolve_run_dir(parent, architecture, split, charge_method=...)` under the contrib parent (e.g. `data/pfpkg/runs/pfpkg`). Callers pass the parent; trainers create `xgb_scaffold`, `chemprop_scaffold_nagl`, `chemberta_random`, …
6. **Score**: `compute_test_metrics` on the frozen test split; write `report.md` / `report.json`. Compare architectures with `write_comparison_report`.
7. **Predict**: `load_classifier(outdir)` (or a specific classifier `.load`). Sanitize and featurize with the **same** `chemistry` functions and the metadata in the run dir (`fingerprint`, `fp_size`, `charge_method`, `pretrained_name`).

Predictions are triage hypotheses: persist probability and interpretation with the model, not a potency claim.

## Optional deep learning

Chemprop and ChemBERTa live behind the `dl` extra. Do **not** import them from `ml_for_malaria.train.__init__` — that would pull torch on every `from ml_for_malaria.train import train_xgb_classifier`.

```python
from ml_for_malaria.train import train_xgb_classifier
from ml_for_malaria.train.chemprop import train_chemprop_classifier
from ml_for_malaria.train.chemberta import train_smiles_transformer
from ml_for_malaria.model import load_classifier
from ml_for_malaria.chemistry import encode_binary_labels, sanitize_smiles, atom_charges
from ml_for_malaria.runs import resolve_run_dir, completed_run_dirs, RunCheckpointer
from ml_for_malaria.report import write_comparison_report, compute_test_metrics
from ml_for_malaria.split import get_splitter
```

`charge_method` (`None` / `gasteiger` / `nagl`) is Chemprop-only extra atom features. ChemBERTa has no charge vector. Failed sanitize or charge assignment **drops** the molecule; do not impute `q=0` or a dummy structure.

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
