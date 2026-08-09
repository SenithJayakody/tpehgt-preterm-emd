"""Generate the 14-panel descriptive feature-distribution figure (Fig. 8).

For each feature type, channels are averaged within each segment and the
segment values are then averaged within each recording. This descriptive
aggregation is distinct from MAX aggregation of classifier scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FEATURE_DIR, FIG_DPI, PLOT_DIR
from paper_style import FEATURE_PRETERM_COLOR, FEATURE_TERM_COLOR, save_png_pdf, set_feature_grid_style

FEATURE_TYPES = [
    "PEAK_RATE",
    "PEAK_AMP_MEAN",
    "PEAK_AMP_CV",
    "IPI_MEAN",
    "IPI_CV",
    "PW_MEAN",
    "BURST_RATE",
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
    "PW_MEAN": "Peak width at half prominence",
    "BURST_RATE": "Burst rate",
    "PEAKS_PER_BURST_MEAN": "Peaks per burst",
    "DASDV": "DASDV",
    "LOG": "Log detector",
    "MTKE": "MTKE",
    "SE": "Shannon entropy",
    "perm_entropy": "Permutation entropy",
    "sampen": "Sample entropy",
}


def record_column(df: pd.DataFrame) -> str:
    if "record" in df.columns:
        return "record"
    if "name" in df.columns:
        return "name"
    raise ValueError("CSV must contain 'record' or 'name'.")


def channel_columns(df: pd.DataFrame, feature_type: str) -> list[str]:
    return sorted([c for c in df.columns if c.endswith("_" + feature_type)])


def cohen_d(preterm: np.ndarray, term: np.ndarray) -> float:
    preterm = np.asarray(preterm, dtype=float)
    term = np.asarray(term, dtype=float)
    preterm = preterm[np.isfinite(preterm)]
    term = term[np.isfinite(term)]
    n1, n0 = len(preterm), len(term)
    if n1 < 2 or n0 < 2:
        return np.nan
    v1 = np.var(preterm, ddof=1)
    v0 = np.var(term, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2))
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    return float((np.mean(preterm) - np.mean(term)) / pooled)


def make_record_level_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build descriptive recording means from channel and segment means."""
    rec_col = record_column(df)
    if "label" not in df.columns:
        raise ValueError("Feature CSV must contain label.")

    segment = df[[rec_col, "label"]].rename(columns={rec_col: "record"}).copy()
    used = {}
    for feature in FEATURE_TYPES:
        cols = channel_columns(df, feature)
        used[feature] = cols
        if not cols:
            raise ValueError(f"No channel columns found for current feature: {feature}")
        segment[feature] = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    agg = {f: "mean" for f in FEATURE_TYPES}
    agg["label"] = "first"
    record_df = segment.groupby("record", as_index=False).agg(agg)

    rows = []
    for feature in FEATURE_TYPES:
        term = record_df.loc[record_df["label"] == 0, feature].to_numpy()
        pre = record_df.loc[record_df["label"] == 1, feature].to_numpy()
        rows.append(
            {
                "feature_type": feature,
                "display_name": DISPLAY_NAMES[feature],
                "cohen_d_preterm_minus_term": cohen_d(pre, term),
                "term_mean": float(np.nanmean(term)),
                "preterm_mean": float(np.nanmean(pre)),
                "n_channel_columns_used": len(used[feature]),
                "channel_columns_used": ", ".join(used[feature]),
            }
        )
    return record_df, pd.DataFrame(rows)


def plot(record_df: pd.DataFrame, effects: pd.DataFrame, out_dir: Path) -> None:
    set_feature_grid_style(FIG_DPI)
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(4, 4, figsize=(13.5, 11.0))
    axes = axes.ravel()

    for idx, feature in enumerate(FEATURE_TYPES):
        ax = axes[idx]
        term = record_df.loc[record_df["label"] == 0, feature].dropna().to_numpy()
        pre = record_df.loc[record_df["label"] == 1, feature].dropna().to_numpy()
        data = [term, pre]
        positions = [1, 2]

        parts = ax.violinplot(data, positions=positions, widths=0.75, showmeans=False, showmedians=False, showextrema=False)
        for body, color in zip(parts["bodies"], [FEATURE_TERM_COLOR, FEATURE_PRETERM_COLOR]):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.35)
            body.set_linewidth(0.8)

        for pos, values, color in zip(positions, data, [FEATURE_TERM_COLOR, FEATURE_PRETERM_COLOR]):
            if len(values) == 0:
                continue
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            ax.plot([pos - 0.20, pos + 0.20], [med, med], color="black", linewidth=1.4)
            ax.plot([pos, pos], [q1, q3], color="black", linewidth=1.2)
            jitter = rng.normal(0, 0.045, size=len(values))
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=28,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.85,
                zorder=3,
            )

        d = float(effects.loc[effects["feature_type"] == feature, "cohen_d_preterm_minus_term"].iloc[0])
        d_text = f"{d:.2f}" if np.isfinite(d) else "NA"
        ax.set_title(f"{DISPLAY_NAMES[feature]}\nCohen's d = {d_text}")
        ax.set_xticks(positions)
        ax.set_xticklabels(["Term", "Preterm"])
        ax.set_ylabel("Recording-level value")
        ax.text(-0.16, 1.15, chr(ord("A") + idx), transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", ha="left")
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)

    for ax in axes[len(FEATURE_TYPES) :]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_png_pdf(fig, out_dir / "feature_distributions", FIG_DPI)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript Fig. 8 feature distributions.")
    parser.add_argument("--csv", type=Path, default=FEATURE_DIR / "tpehgt_fixed_3min_imf1_features.csv")
    parser.add_argument("--out-dir", type=Path, default=PLOT_DIR / "paper")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    record_df, effects = make_record_level_table(df)
    record_df.to_csv(args.out_dir / "feature_distribution_record_values.csv", index=False)
    effects.to_csv(args.out_dir / "feature_distribution_cohens_d.csv", index=False)
    plot(record_df, effects, args.out_dir)
    print(f"Saved manuscript feature distribution figure to: {args.out_dir}")


if __name__ == "__main__":
    main()
