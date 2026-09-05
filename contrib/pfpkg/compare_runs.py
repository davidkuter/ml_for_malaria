from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd
from loguru import logger

from ml_for_malaria.chemistry import encode_binary_labels
from ml_for_malaria.env import load_local_env
from ml_for_malaria.report import (
    best_fixed_scaffold_identifier,
    write_comparison_report,
)
from ml_for_malaria.runs import (
    map_replicates,
    replicate_seeds,
    replicate_worker_count,
    resolve_run_dir,
)
from ml_for_malaria.schemas import (
    Architecture,
    ChargeMethod,
    CleanedTrainingData,
    ComparisonAggregate,
)
from ml_for_malaria.train.fit_one import fit_one_run

ROOT = Path(__file__).resolve().parents[2]
PFPKG = ROOT / "data" / "pfpkg"
DATASET_PATH = PFPKG / "input" / "100nM_Training_Set.csv"
RUNS = PFPKG / "runs" / "pfpkg"
N_REP = 10
SEED_START = 42
FORCE = False
XGB_MAX_EVALS = 0
# None = min(n_rep, CPU count) for fingerprint sklearn (RF/XGB/k-NN/logistic).
# Chemprop/ChemBERTa stay sequential (GPU).
N_WORKERS: int | None = None


@dataclass(frozen=True)
class PfpkgJob:
    architecture: Architecture
    split: str
    charge_method: str | None = None
    fingerprints: tuple[str, ...] | None = None
    max_evals: int | None = None
    yscramble: bool = False


PFPKG_JOBS = (
    PfpkgJob(Architecture.CHEMBERTA, "scaffold"),
    PfpkgJob(Architecture.CHEMPROP, "scaffold"),
    PfpkgJob(Architecture.CHEMPROP, "scaffold", ChargeMethod.GASTEIGER),
    PfpkgJob(Architecture.CHEMPROP, "scaffold", ChargeMethod.NAGL),
    PfpkgJob(Architecture.CHEMELEON, "scaffold"),
    PfpkgJob(Architecture.MONROE, "scaffold"),
    PfpkgJob(Architecture.RANDOM_FOREST, "random"),
    PfpkgJob(Architecture.RANDOM_FOREST, "scaffold"),
    PfpkgJob(Architecture.XGBOOST, "scaffold"),
    PfpkgJob(Architecture.KNN, "random"),
    PfpkgJob(Architecture.KNN, "scaffold"),
    PfpkgJob(Architecture.LOGISTIC, "scaffold"),
)
HPO_ARCHITECTURES = (Architecture.RANDOM_FOREST, Architecture.XGBOOST)
HPO_MAX_EVALS = 50


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


def _job_max_evals(job: PfpkgJob) -> int:
    if job.architecture == Architecture.XGBOOST:
        return job.max_evals if job.max_evals is not None else XGB_MAX_EVALS
    return job.max_evals or 0


def _gpu_architecture(architecture: str) -> bool:
    return architecture in (
        Architecture.CHEMPROP,
        Architecture.CHEMELEON,
        Architecture.CHEMBERTA,
        Architecture.MONROE,
    )


def hpo_jobs_from_aggregates(
    aggregates: list[ComparisonAggregate],
) -> tuple[PfpkgJob, ...]:
    """Tune the best fixed-recipe scaffold fingerprint for each tree architecture."""
    jobs = []
    for architecture in HPO_ARCHITECTURES:
        fingerprint = best_fixed_scaffold_identifier(aggregates, architecture)
        logger.info(
            f"HPO {architecture} scaffold fingerprint {fingerprint} "
            f"(max_evals={HPO_MAX_EVALS})"
        )
        jobs.append(
            PfpkgJob(
                architecture,
                "scaffold",
                fingerprints=(fingerprint,),
                max_evals=HPO_MAX_EVALS,
            )
        )
    return tuple(jobs)


def yscramble_jobs() -> tuple[PfpkgJob, ...]:
    """Fixed-recipe y-scramble for every suite architecture (no HPO)."""
    return tuple(
        PfpkgJob(
            job.architecture,
            job.split,
            charge_method=job.charge_method,
            fingerprints=job.fingerprints,
            max_evals=0,
            yscramble=True,
        )
        for job in PFPKG_JOBS
    )


def run_pfpkg_suite(
    df: pd.DataFrame,
    outdir: Path,
    *,
    force: bool = False,
    n_rep: int = N_REP,
    seed_start: int = SEED_START,
    jobs: Sequence[PfpkgJob] = PFPKG_JOBS,
    n_workers: int | None = N_WORKERS,
) -> list[Path]:
    """Fit each pfpkg comparison job for ``n_rep`` seeds; completed runs are reused.

    Fingerprint sklearn seeds (RF, XGBoost, k-NN, logistic, including HPO)
    run in a process pool. Chemprop and ChemBERTa stay sequential so they
    do not share one GPU.
    """
    seeds = replicate_seeds(n_rep, start=seed_start)
    outdirs: list[Path] = []
    for job in jobs:
        hpo = bool(job.max_evals and job.max_evals > 0)
        workers = replicate_worker_count(
            len(seeds),
            n_workers=n_workers,
            serial=_gpu_architecture(job.architecture),
        )
        experiment = resolve_run_dir(
            outdir,
            job.architecture,
            job.split,
            charge_method=job.charge_method,
            hpo=hpo,
            yscramble=job.yscramble,
        )
        logger.info(
            f"Suite job {experiment.relative_to(outdir)} "
            f"seeds={len(seeds)} workers={workers}"
        )
        worker = partial(
            fit_one_run,
            architecture=job.architecture,
            df=df,
            outdir=outdir,
            split=job.split,
            charge_method=job.charge_method,
            fingerprints=list(job.fingerprints) if job.fingerprints else None,
            max_evals=_job_max_evals(job),
            yscramble=job.yscramble,
            force=force,
        )
        for run_dir in map_replicates(worker, seeds, n_workers=workers):
            if run_dir is not None:
                outdirs.append(run_dir)
    return outdirs


def main() -> None:
    load_local_env()
    logger.info(f"Loading data from: {DATASET_PATH}")
    df = load_training_frame(DATASET_PATH)
    run_dirs = run_pfpkg_suite(df, RUNS, force=FORCE, n_rep=N_REP, seed_start=SEED_START)
    if not run_dirs:
        raise SystemExit(f"No completed runs with report.json under {RUNS}")
    baseline = write_comparison_report(run_dirs, RUNS / "comparison.md")
    logger.info(
        f"Fixed-recipe comparison: {len(baseline.rows)} rows "
        f"({len(baseline.aggregates)} aggregates)"
    )
    hpo_jobs = hpo_jobs_from_aggregates(baseline.aggregates)
    hpo_dirs = run_pfpkg_suite(
        df,
        RUNS,
        force=FORCE,
        n_rep=N_REP,
        seed_start=SEED_START,
        jobs=hpo_jobs,
    )
    scramble_dirs = run_pfpkg_suite(
        df,
        RUNS,
        force=FORCE,
        n_rep=N_REP,
        seed_start=SEED_START,
        jobs=yscramble_jobs(),
    )
    run_dirs = run_dirs + hpo_dirs + scramble_dirs
    result = write_comparison_report(run_dirs, RUNS / "comparison.md")
    logger.info(
        f"Wrote comparison of {len(result.rows)} rows "
        f"({len(result.aggregates)} aggregates, {len(result.hpo_deltas)} HPO deltas, "
        f"{len(result.yscramble_deltas)} y-scramble deltas) "
        f"to {RUNS / 'comparison.md'}"
    )
    for warning in result.warnings:
        logger.warning(warning)


if __name__ == "__main__":
    main()
