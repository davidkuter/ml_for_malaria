from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

from ml_for_malaria.chemistry import encode_binary_labels
from ml_for_malaria.report import write_comparison_report
from ml_for_malaria.runs import replicate_seeds, resolve_run_dir
from ml_for_malaria.schemas import Architecture, ChargeMethod, CleanedTrainingData
from ml_for_malaria.train import train_rf_classifier, train_xgb_classifier

ROOT = Path(__file__).resolve().parents[2]
PFPKG = ROOT / "data" / "pfpkg"
DATASET_PATH = PFPKG / "input" / "100nM_Training_Set.csv"
RUNS = PFPKG / "runs" / "pfpkg"
N_REP = 10
SEED_START = 42
FORCE = False
XGB_MAX_EVALS = 0


@dataclass(frozen=True)
class PfpkgJob:
    architecture: Architecture
    split: str
    charge_method: str | None = None


PFPKG_JOBS = (
    PfpkgJob(Architecture.CHEMBERTA, "scaffold"),
    PfpkgJob(Architecture.CHEMPROP, "scaffold"),
    PfpkgJob(Architecture.CHEMPROP, "scaffold", ChargeMethod.GASTEIGER),
    PfpkgJob(Architecture.CHEMPROP, "scaffold", ChargeMethod.NAGL),
    PfpkgJob(Architecture.RANDOM_FOREST, "random"),
    PfpkgJob(Architecture.RANDOM_FOREST, "scaffold"),
    PfpkgJob(Architecture.XGBOOST, "scaffold"),
)


def load_training_frame(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = raw.rename(columns={"Lable": CleanedTrainingData.LABEL})
    df = df[[CleanedTrainingData.SMILES, CleanedTrainingData.LABEL]]
    df[CleanedTrainingData.LABEL] = encode_binary_labels(
        df[CleanedTrainingData.LABEL],
        active_label="Active",
        inactive_label="Inactive",
    )
    return df


def _run_job(job: PfpkgJob, df: pd.DataFrame, outdir: Path, force: bool, seed: int):
    if job.architecture == Architecture.CHEMBERTA:
        from ml_for_malaria.model.smiles_transformer import DEFAULT_PRETRAINED_NAME
        from ml_for_malaria.train.chemberta import train_smiles_transformer

        return train_smiles_transformer(
            df=df,
            outdir=outdir,
            split=job.split,
            seed=seed,
            pretrained_name=DEFAULT_PRETRAINED_NAME,
            freeze_encoder=True,
            force=force,
        )
    if job.architecture == Architecture.CHEMPROP:
        from ml_for_malaria.train.chemprop import train_chemprop_classifier

        return train_chemprop_classifier(
            df=df,
            outdir=outdir,
            split=job.split,
            seed=seed,
            charge_method=job.charge_method,
            force=force,
        )
    if job.architecture == Architecture.RANDOM_FOREST:
        return train_rf_classifier(
            df=df,
            outdir=outdir,
            split=job.split,
            seed=seed,
            force=force,
        )
    if job.architecture == Architecture.XGBOOST:
        return train_xgb_classifier(
            df=df,
            outdir=outdir,
            split=job.split,
            seed=seed,
            force=force,
            max_evals=XGB_MAX_EVALS,
        )
    raise ValueError(f"Unsupported suite architecture {job.architecture!r}")


def run_pfpkg_suite(
    df: pd.DataFrame,
    outdir: Path,
    *,
    force: bool = False,
    n_rep: int = N_REP,
    seed_start: int = SEED_START,
) -> list[Path]:
    """Fit each pfpkg comparison job for ``n_rep`` seeds; completed runs are reused."""
    seeds = replicate_seeds(n_rep, start=seed_start)
    outdirs: list[Path] = []
    for job in PFPKG_JOBS:
        for seed in seeds:
            run_dir = resolve_run_dir(
                outdir,
                job.architecture,
                job.split,
                charge_method=job.charge_method,
                seed=seed,
            )
            logger.info(f"Suite job {run_dir.relative_to(outdir)}")
            try:
                result = _run_job(job, df, outdir, force=force, seed=seed)
            except ImportError as exc:
                logger.error(f"Skipping {run_dir}: {exc}")
                continue
            logger.info(f"Ready {result.outdir / 'report.md'}")
            outdirs.append(result.outdir)
    return outdirs


def main() -> None:
    logger.info(f"Loading data from: {DATASET_PATH}")
    df = load_training_frame(DATASET_PATH)
    run_dirs = run_pfpkg_suite(df, RUNS, force=FORCE, n_rep=N_REP, seed_start=SEED_START)
    if not run_dirs:
        raise SystemExit(f"No completed runs with report.json under {RUNS}")
    result = write_comparison_report(run_dirs, RUNS / "comparison.md")
    logger.info(
        f"Wrote comparison of {len(result.rows)} rows "
        f"({len(result.aggregates)} aggregates) to {RUNS / 'comparison.md'}"
    )
    for warning in result.warnings:
        logger.warning(warning)


if __name__ == "__main__":
    main()
