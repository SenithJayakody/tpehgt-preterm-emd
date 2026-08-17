"""Export compact, deterministic manuscript artifacts from completed outputs."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from config import PLOT_DIR, RESULT_DIR
from validate_final_pipeline import CANONICAL_EXPERIMENTS

PUBLICATION_DIR = Path("reproducibility")
MAIN_EXPERIMENTS = ["annotated_interval_imf1", "fixed_3min_imf1"]
PLOT_ARTIFACTS = [
    "roc_pr_curve_values.csv",
    "roc_pr_curve_metrics.csv",
    "confusion_consensus_record_values.csv",
    "confusion_matrix_values.csv",
    "feature_distribution_record_values.csv",
    "feature_distribution_cohens_d.csv",
    "grouped_permutation_importance_summary.csv",
]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required completed-pipeline output is missing: {path}")
    return path


def main() -> None:
    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for experiment in CANONICAL_EXPERIMENTS:
        frame = pd.read_csv(require_file(RESULT_DIR / experiment / "summary_metrics.csv"))
        frame = frame[["model", *[column for column in frame.columns if column.endswith("_mean")]]]
        frame.insert(0, "experiment", experiment)
        summaries.append(frame)
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(PUBLICATION_DIR / "classification_summary_metrics.csv", index=False)

    combined[combined["experiment"].str.match(r"^(annotated_interval|fixed_3min)_imf[1-4]$")].to_csv(
        PUBLICATION_DIR / "imf_comparison_summary.csv", index=False
    )
    combined[combined["experiment"].isin([
        "annotated_interval_imf1", "annotated_interval_time_domain",
        "fixed_3min_imf1", "fixed_3min_time_domain",
    ])].to_csv(PUBLICATION_DIR / "filtered_vs_imf_summary.csv", index=False)

    predictions = []
    for experiment in MAIN_EXPERIMENTS:
        frame = pd.read_csv(require_file(RESULT_DIR / experiment / "record_predictions.csv"))
        predictions.append(frame)
    pd.concat(predictions, ignore_index=True).to_csv(
        PUBLICATION_DIR / "main_experiments_oof_record_predictions.csv", index=False
    )

    plot_dir = PLOT_DIR / "paper"
    for filename in PLOT_ARTIFACTS:
        shutil.copyfile(require_file(plot_dir / filename), PUBLICATION_DIR / filename)
    print(f"Exported compact publication artifacts to: {PUBLICATION_DIR}")


if __name__ == "__main__":
    main()
