from pathlib import Path

from loguru import logger

from ml_for_malaria.train import completed_run_dirs, write_comparison_report

ROOT = Path(__file__).resolve().parents[2]
PFPKG = ROOT / "data" / "pfpkg"
RUNS = PFPKG / "runs" / "pfpkg"

run_dirs = completed_run_dirs(RUNS)
if not run_dirs:
    raise SystemExit(f"No completed runs with report.json under {RUNS}")

result = write_comparison_report(run_dirs, RUNS / "comparison.md")
logger.info(f"Wrote comparison of {len(result.rows)} runs to {RUNS / 'comparison.md'}")
for warning in result.warnings:
    logger.warning(warning)
