"""
Recompute grouped permutation importance using the CURRENT classification code.

This script intentionally imports the Random Forest pipeline, record-level split
construction, record aggregation, and AP calculation from classify_groupwise_cv.py so
the importance workflow cannot silently drift from the manuscript analysis.

Importance = baseline recording-level AP - permuted recording-level AP.
One joint row permutation is applied to all columns in a feature family, preserving
within-family feature correlations while disrupting its relationship to the outcome.
Within each repetition, scores from all five validation folds are pooled before AP
is calculated on the complete 26-record out-of-fold set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

from config import FEATURE_DIR, FIG_DPI, N_REPEATS, PLOT_DIR, RANDOM_SEED
from classify_groupwise_cv import (
    aggregate_to_record,
    build_outer_splits,
    get_feature_matrix,
    get_models,
    make_meta,
    make_pipeline,
    predict_preterm_scores,
    safe_average_precision,
)
from paper_style import IMPORTANCE_COLORS, save_png_pdf, set_importance_style

FEATURE_GROUP_SUFFIXES = {
    "Burst and peak features": [
        "PEAK_RATE",
        "PEAK_AMP_MEAN",
        "PEAK_AMP_CV",
        "IPI_MEAN",
        "IPI_CV",
        "PW_MEAN",
        "BURST_RATE",
        "PEAKS_PER_BURST_MEAN",
    ],
    "Temporal energy features": ["DASDV", "LOG", "MTKE"],
    "Entropy features": ["SE", "perm_entropy", "sampen"],
}

EXPERIMENTS = [
    (
        "Annotated intervals IMF1",
        FEATURE_DIR / "tpehgt_annotated_interval_imf1_features.csv",
    ),
    ("Fixed 3-min IMF1", FEATURE_DIR / "tpehgt_fixed_3min_imf1_features.csv"),
]


def feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    grouped = {}
    matched = set()
    for group, suffixes in FEATURE_GROUP_SUFFIXES.items():
        cols = [c for c in feature_columns if any(c == suffix or c.endswith("_" + suffix) for suffix in suffixes)]
        grouped[group] = cols
        matched.update(cols)
    unmatched = sorted(set(feature_columns) - matched)
    if unmatched:
        raise ValueError(f"Unmatched feature columns; update feature grouping: {unmatched}")
    return grouped


def aggregate_scores(meta: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Aggregate segment scores and return one score per recording."""
    return aggregate_to_record(meta, scores)[["record", "label", "score"]]


def validate_oof_records(
    frame: pd.DataFrame,
    expected_records: set[str],
    experiment: str,
    repeat: int,
    description: str,
) -> pd.DataFrame:
    """Validate complete, unique recording-level OOF coverage."""
    records = frame["record"].astype(str)
    duplicates = sorted(records[records.duplicated()].unique())
    actual_records = set(records)

    if duplicates or actual_records != expected_records:
        missing = sorted(expected_records - actual_records)
        extra = sorted(actual_records - expected_records)
        raise RuntimeError(
            f"Invalid OOF coverage for experiment={experiment}, repeat={repeat}, "
            f"scores={description}: duplicates={duplicates}, missing={missing}, "
            f"extra={extra}"
        )

    if len(frame) != len(expected_records):
        raise RuntimeError(
            f"Expected {len(expected_records)} OOF recording scores for "
            f"experiment={experiment}, repeat={repeat}, scores={description}; "
            f"found {len(frame)}."
        )

    return frame.sort_values("record", kind="stable").reset_index(drop=True)


def compute_one_experiment(name: str, csv_path: Path, n_repeats: int, permutations_per_fold: int) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    x, y, groups, record_col, feature_cols = get_feature_matrix(df)
    grouped = feature_groups(feature_cols)
    splits_by_repeat, _ = build_outer_splits(x, y, groups, n_repeats=n_repeats)
    expected_records = set(groups.astype(str))

    rf = get_models()["Random Forest"]
    rows = []

    for repeat in range(n_repeats):
        baseline_parts = []
        permuted_parts = {
            (permutation, group_name): []
            for permutation in range(permutations_per_fold)
            for group_name in grouped
        }

        for fold, (train_idx, val_idx) in enumerate(splits_by_repeat[repeat]):
            x_train = x.iloc[train_idx]
            x_val = x.iloc[val_idx]
            y_train = y[train_idx]
            meta_val = make_meta(df.iloc[val_idx], record_col)

            pipe = make_pipeline(clone(rf))
            pipe.fit(x_train, y_train)
            baseline_scores = predict_preterm_scores(pipe, x_val)
            baseline_parts.append(aggregate_scores(meta_val, baseline_scores))

            for permutation in range(permutations_per_fold):
                for group_index, (group_name, group_cols) in enumerate(grouped.items()):
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [RANDOM_SEED, repeat, fold, permutation, group_index]
                        )
                    )
                    perm_index = rng.permutation(len(x_val))
                    x_perm = x_val.copy()
                    x_perm.loc[:, group_cols] = x_val.iloc[perm_index][group_cols].to_numpy()
                    perm_scores = predict_preterm_scores(pipe, x_perm)
                    permuted_parts[(permutation, group_name)].append(
                        aggregate_scores(meta_val, perm_scores)
                    )

        baseline_oof = validate_oof_records(
            pd.concat(baseline_parts, ignore_index=True),
            expected_records,
            name,
            repeat,
            "baseline",
        )
        baseline_ap = safe_average_precision(
            baseline_oof["label"].to_numpy(),
            baseline_oof["score"].to_numpy(),
        )

        if not np.isfinite(baseline_ap):
            raise RuntimeError(
                f"Baseline AP is undefined for experiment={name}, repeat={repeat}."
            )

        for (permutation, group_name), parts in permuted_parts.items():
            permuted_oof = validate_oof_records(
                pd.concat(parts, ignore_index=True),
                expected_records,
                name,
                repeat,
                f"permutation={permutation}, feature_group={group_name}",
            )

            if not np.array_equal(
                baseline_oof["label"].to_numpy(),
                permuted_oof["label"].to_numpy(),
            ):
                raise RuntimeError(
                    f"OOF labels changed after permutation for experiment={name}, "
                    f"repeat={repeat}, permutation={permutation}, "
                    f"feature_group={group_name}."
                )

            permuted_ap = safe_average_precision(
                permuted_oof["label"].to_numpy(),
                permuted_oof["score"].to_numpy(),
            )
            rows.append(
                {
                    "experiment": name,
                    "model": "RF",
                    "repeat": repeat,
                    "permutation": permutation,
                    "feature_group": group_name,
                    "baseline_ap": baseline_ap,
                    "permuted_ap": permuted_ap,
                    "decrease_ap": baseline_ap - permuted_ap,
                    "n_records": len(baseline_oof),
                    "n_features_in_group": len(grouped[group_name]),
                }
            )

    return pd.DataFrame(rows)


def plot_importance(summary: pd.DataFrame, out_dir: Path) -> None:
    set_importance_style(FIG_DPI)
    group_order = list(FEATURE_GROUP_SUFFIXES)
    experiment_order = ["Annotated intervals IMF1", "Fixed 3-min IMF1"]
    x = np.arange(len(group_order))
    width = 0.32
    offsets = {experiment_order[0]: -width / 2, experiment_order[1]: width / 2}

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for exp in experiment_order:
        vals = []
        for group in group_order:
            match = summary[(summary["experiment"] == exp) & (summary["feature_group"] == group)]
            vals.append(float(match["mean_decrease_ap"].iloc[0]) if not match.empty else np.nan)
        bars = ax.bar(
            x + offsets[exp],
            vals,
            width=width,
            label=f"{exp} (Random Forest)",
            color=IMPORTANCE_COLORS[exp],
            edgecolor="black",
            linewidth=0.7,
        )
        for bar, value in zip(bars, vals):
            if np.isfinite(value):
                y = value + (0.002 if value >= 0 else -0.002)
                va = "bottom" if value >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.3f}", ha="center", va=va, fontsize=9)

    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=15, ha="right")
    ax.set_ylabel("Mean decrease in recording-level AP")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_png_pdf(fig, out_dir / "grouped_feature_importance", FIG_DPI)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped permutation importance for current RF pipeline.")
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--permutations-per-fold", type=int, default=1, help="Use 1 to reproduce the current manuscript workflow.")
    parser.add_argument("--out-dir", type=Path, default=PLOT_DIR / "paper")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        compute_one_experiment(name, path, args.n_repeats, args.permutations_per_fold)
        for name, path in EXPERIMENTS
    ]
    all_values = pd.concat(frames, ignore_index=True)
    all_values.to_csv(
        args.out_dir / "grouped_permutation_importance_repeat_values.csv",
        index=False,
        float_format="%.17g",
    )

    summary = all_values.groupby(["experiment", "model", "feature_group"], as_index=False).agg(
        mean_decrease_ap=("decrease_ap", "mean"),
        sd_decrease_ap=("decrease_ap", "std"),
        n_repetitions=("repeat", "nunique"),
        n_permutations=("permutation", "nunique"),
        n_values=("decrease_ap", "size"),
        n_features_in_group=("n_features_in_group", "first"),
    )
    summary.to_csv(
        args.out_dir / "grouped_permutation_importance_summary.csv",
        index=False,
        float_format="%.17g",
    )
    plot_importance(summary, args.out_dir)
    print(summary.to_string(index=False))
    print(f"Saved grouped permutation importance outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
