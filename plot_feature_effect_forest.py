from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FEATURE_DIR, FIG_DPI, PLOT_DIR
from paper_style import PLOS_ONE_HALF, save_figure, set_paper_style


FEATURE_TYPES = [
    "PEAK_RATE",
    "PEAK_AMP_MEAN",
    "PEAK_AMP_CV",
    "IPI_MEAN",
    "IPI_CV",
    "PW_MEAN",
    "BURST_COUNT",
    "PEAKS_PER_BURST_MEAN",
    "DASDV",
    "LOG",
    "MTKE",
    "SE",
    "perm_entropy",
    "sampen",
]


DISPLAY_NAMES = {
    "PEAK_RATE": "Peak rate",
    "PEAK_AMP_MEAN": "Peak amp. mean",
    "PEAK_AMP_CV": "Peak amp. CV",
    "IPI_MEAN": "IPI mean",
    "IPI_CV": "IPI CV",
    "PW_MEAN": "Peak width mean",
    "BURST_COUNT": "Burst count",
    "PEAKS_PER_BURST_MEAN": "Peaks per burst",
    "DASDV": "DASDV",
    "LOG": "Log detector",
    "MTKE": "MTKE",
    "SE": "Shannon entropy",
    "perm_entropy": "Permutation entropy",
    "sampen": "Sample entropy",
}


FEATURE_GROUPS = {
    "Peak and burst features": [
        "PEAK_RATE",
        "PEAK_AMP_MEAN",
        "PEAK_AMP_CV",
        "IPI_MEAN",
        "IPI_CV",
        "PW_MEAN",
        "BURST_COUNT",
        "PEAKS_PER_BURST_MEAN",
    ],
    "Temporal energy features": [
        "DASDV",
        "LOG",
        "MTKE",
    ],
    "Entropy features": [
        "SE",
        "perm_entropy",
        "sampen",
    ],
}


GROUP_COLORS = {
    "Peak and burst features": "#2c7fb8",
    "Temporal energy features": "#e6550d",
    "Entropy features": "#117a65",
}


IMPORTANCE_GROUP_ALIASES = {
    "Peak and burst features": "Burst and peak features",
    "Temporal energy features": "Temporal energy features",
    "Entropy features": "Entropy features",
}


def get_record_column(df: pd.DataFrame) -> str:
    if "record" in df.columns:
        return "record"
    if "name" in df.columns:
        return "name"
    raise ValueError("CSV must contain either 'record' or 'name' column.")


def find_channel_columns(df: pd.DataFrame, feature_type: str) -> list[str]:
    return sorted([c for c in df.columns if c.endswith("_" + feature_type)])


def cohen_d(preterm_values: np.ndarray, term_values: np.ndarray) -> float:
    preterm_values = np.asarray(preterm_values, dtype=float)
    term_values = np.asarray(term_values, dtype=float)
    preterm_values = preterm_values[np.isfinite(preterm_values)]
    term_values = term_values[np.isfinite(term_values)]

    n1 = len(preterm_values)
    n0 = len(term_values)

    if n1 < 2 or n0 < 2:
        return np.nan

    var1 = np.var(preterm_values, ddof=1)
    var0 = np.var(term_values, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2))

    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan

    return float((np.mean(preterm_values) - np.mean(term_values)) / pooled)


def bootstrap_cohen_d_ci(
    preterm_values: np.ndarray,
    term_values: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    preterm_values = np.asarray(preterm_values, dtype=float)
    term_values = np.asarray(term_values, dtype=float)
    preterm_values = preterm_values[np.isfinite(preterm_values)]
    term_values = term_values[np.isfinite(term_values)]

    if len(preterm_values) < 2 or len(term_values) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot = []

    for _ in range(n_bootstrap):
        preterm_sample = rng.choice(preterm_values, size=len(preterm_values), replace=True)
        term_sample = rng.choice(term_values, size=len(term_values), replace=True)
        d = cohen_d(preterm_sample, term_sample)

        if np.isfinite(d):
            boot.append(d)

    if len(boot) == 0:
        return np.nan, np.nan

    lo, hi = np.percentile(np.asarray(boot), [2.5, 97.5])
    return float(lo), float(hi)


def feature_to_group(feature_type: str) -> str:
    for group_name, feature_types in FEATURE_GROUPS.items():
        if feature_type in feature_types:
            return group_name
    raise ValueError(f"No feature group configured for {feature_type}")


def make_record_level_features(df: pd.DataFrame) -> pd.DataFrame:
    record_col = get_record_column(df)

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    segment_df = df[[record_col, "label"]].rename(columns={record_col: "record"}).copy()

    for feature_type in FEATURE_TYPES:
        channel_cols = find_channel_columns(df, feature_type)

        if len(channel_cols) == 0:
            print(f"WARNING: No columns found for feature type: {feature_type}")
            segment_df[feature_type] = np.nan
            continue

        segment_df[feature_type] = (
            df[channel_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        )

    agg = {feature_type: "mean" for feature_type in FEATURE_TYPES}
    agg["label"] = "first"
    return segment_df.groupby("record", as_index=False).agg(agg)


def read_group_importance(
    path: Path,
    experiment_label: str,
    model: str,
) -> dict[str, float]:
    path = Path(path)

    if not path.exists():
        print(f"WARNING: Importance file not found: {path}")
        return {}

    df = pd.read_csv(path)
    required = {"experiment", "model", "feature_group", "mean_decrease_pr_auc"}

    if not required.issubset(df.columns):
        print(f"WARNING: Importance file missing required columns: {path}")
        return {}

    sub = df[
        (df["experiment"].astype(str) == experiment_label)
        & (df["model"].astype(str) == model)
    ].copy()

    if sub.empty:
        print(
            "WARNING: No matching importance rows for "
            f"experiment='{experiment_label}', model='{model}'"
        )
        return {}

    values = {}
    for group_name, importance_name in IMPORTANCE_GROUP_ALIASES.items():
        match = sub[sub["feature_group"].astype(str) == importance_name]

        if not match.empty:
            values[group_name] = float(match.iloc[0]["mean_decrease_pr_auc"])

    return values


def marker_sizes_from_importance(group_importance: dict[str, float]) -> dict[str, float]:
    if not group_importance:
        return {group: 44.0 for group in FEATURE_GROUPS}

    vals = np.array(
        [max(0.0, float(group_importance.get(group, 0.0))) for group in FEATURE_GROUPS],
        dtype=float,
    )

    if np.all(vals == 0):
        return {group: 44.0 for group in FEATURE_GROUPS}

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    sizes = {}
    for group, value in zip(FEATURE_GROUPS, vals):
        if vmax > vmin:
            scaled = (value - vmin) / (vmax - vmin)
        else:
            scaled = 0.5
        sizes[group] = 38.0 + 62.0 * scaled

    return sizes


def compute_effect_table(
    record_df: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    group_importance: dict[str, float],
) -> pd.DataFrame:
    rows = []

    for idx, feature_type in enumerate(FEATURE_TYPES):
        group_name = feature_to_group(feature_type)
        term_values = record_df.loc[record_df["label"] == 0, feature_type].to_numpy()
        preterm_values = record_df.loc[record_df["label"] == 1, feature_type].to_numpy()

        d = cohen_d(preterm_values, term_values)
        ci_low, ci_high = bootstrap_cohen_d_ci(
            preterm_values,
            term_values,
            n_bootstrap=n_bootstrap,
            seed=seed + idx,
        )

        rows.append(
            {
                "feature_type": feature_type,
                "display_name": DISPLAY_NAMES.get(feature_type, feature_type),
                "feature_group": group_name,
                "cohen_d_preterm_minus_term": d,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "term_mean": float(np.nanmean(term_values)),
                "preterm_mean": float(np.nanmean(preterm_values)),
                "n_term_records": int(np.isfinite(term_values).sum()),
                "n_preterm_records": int(np.isfinite(preterm_values).sum()),
                "group_mean_decrease_pr_auc": group_importance.get(group_name, np.nan),
            }
        )

    return pd.DataFrame(rows)


def ordered_plot_rows(effect_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = 0

    for group_name, feature_types in FEATURE_GROUPS.items():
        for feature_type in feature_types:
            row = effect_df[effect_df["feature_type"] == feature_type].iloc[0].to_dict()
            row["y"] = y
            rows.append(row)
            y += 1

        y += 0.8

    return pd.DataFrame(rows)


def plot_forest(
    effect_df: pd.DataFrame,
    out_base: Path,
    title: str,
    group_importance: dict[str, float],
) -> None:
    set_paper_style()

    plot_df = ordered_plot_rows(effect_df)
    sizes = marker_sizes_from_importance(group_importance)

    fig_height = 5.6
    fig, ax = plt.subplots(figsize=(PLOS_ONE_HALF, fig_height))

    xmin = np.nanmin(plot_df["ci_low"].to_numpy())
    xmax = np.nanmax(plot_df["ci_high"].to_numpy())

    if not np.isfinite(xmin) or not np.isfinite(xmax):
        xmin, xmax = -1.0, 1.0

    pad = max(0.35, 0.12 * (xmax - xmin))
    xmin -= pad
    xmax += pad

    ax.axvline(0, color="#333333", lw=0.8, ls="--", zorder=1)

    group_midpoints = []

    for group_name, group_rows in plot_df.groupby("feature_group", sort=False):
        color = GROUP_COLORS[group_name]
        y_values = group_rows["y"].to_numpy(dtype=float)
        group_midpoints.append((group_name, float(np.mean(y_values))))

        ax.axhspan(
            y_values.min() - 0.48,
            y_values.max() + 0.48,
            color=color,
            alpha=0.055,
            zorder=0,
        )

        for _, row in group_rows.iterrows():
            d = row["cohen_d_preterm_minus_term"]
            ci_low = row["ci_low"]
            ci_high = row["ci_high"]
            y = row["y"]

            if np.isfinite(ci_low) and np.isfinite(ci_high):
                ax.plot([ci_low, ci_high], [y, y], color=color, lw=1.5, zorder=2)
                ax.plot([ci_low, ci_low], [y - 0.12, y + 0.12], color=color, lw=1.0)
                ax.plot([ci_high, ci_high], [y - 0.12, y + 0.12], color=color, lw=1.0)

            ax.scatter(
                d,
                y,
                s=sizes[group_name],
                color=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )

    ax.set_yticks(plot_df["y"])
    ax.set_yticklabels(plot_df["display_name"])
    ax.invert_yaxis()
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Cohen's d (preterm minus term)")
    ax.set_title(title, loc="left", pad=8)
    ax.grid(axis="x", color="#d9d9d9", lw=0.5, alpha=0.8)

    for group_name, midpoint in group_midpoints:
        ax.text(
            xmin,
            midpoint,
            group_name.replace(" features", ""),
            color=GROUP_COLORS[group_name],
            fontsize=7,
            fontweight="bold",
            ha="left",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.85},
        )

    legend_handles = []
    for group_name, color in GROUP_COLORS.items():
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=5.5,
                label=group_name,
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        title="Feature group",
        title_fontsize=7,
    )

    if group_importance:
        note = "Marker size reflects grouped permutation importance"
        ax.text(
            1.0,
            -0.095,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#555555",
        )

    fig.tight_layout()
    save_figure(fig, out_base, dpi=FIG_DPI)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a grouped Cohen's d forest plot for the 14 EHG feature types."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=FEATURE_DIR / "tpehgt_fixed_3min_imf1_features.csv",
        help="Feature CSV generated by extract_features.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PLOT_DIR / "paper",
        help="Directory for the figure and effect-size CSV.",
    )
    parser.add_argument(
        "--out-name",
        default="feature_effect_forest_fixed_3min_imf1",
        help="Output basename without extension.",
    )
    parser.add_argument(
        "--importance-csv",
        type=Path,
        default=PLOT_DIR / "paper" / "grouped_permutation_importance_summary.csv",
        help="Grouped permutation importance summary CSV.",
    )
    parser.add_argument(
        "--importance-experiment",
        default="Fixed 3-min IMF1",
        help="Experiment label in the grouped importance CSV.",
    )
    parser.add_argument(
        "--importance-model",
        default="Random Forest",
        help="Model name in the grouped importance CSV.",
    )
    parser.add_argument(
        "--title",
        default="Record-level feature effect sizes",
        help="Figure title.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help="Number of bootstrap resamples for 95% confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap confidence intervals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    record_df = make_record_level_features(df)

    group_importance = read_group_importance(
        args.importance_csv,
        experiment_label=args.importance_experiment,
        model=args.importance_model,
    )

    effect_df = compute_effect_table(
        record_df,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        group_importance=group_importance,
    )

    effect_csv = args.out_dir / f"{args.out_name}_cohens_d.csv"
    effect_df.to_csv(effect_csv, index=False)

    out_base = args.out_dir / args.out_name
    plot_forest(
        effect_df,
        out_base=out_base,
        title=args.title,
        group_importance=group_importance,
    )

    print(f"Saved effect-size table: {effect_csv}")
    print(f"Saved figure: {out_base.with_suffix('.png')}")
    print(f"Saved figure: {out_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
