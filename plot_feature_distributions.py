"""
plot_feature_distributions.py

Generate paper-ready recording-level feature distribution plots.

The feature CSV contains segment-level features. This script first aggregates
segments to one feature vector per recording, then plots term vs preterm
recording-level distributions for selected features.

Examples
--------
Use default representative features:
    python plot_feature_distributions.py --csv outputs/features/tpehgt_fixed_3min_imf1_features.csv

Automatically choose the top 6 features by standardized class difference:
    python plot_feature_distributions.py --csv outputs/features/tpehgt_fixed_3min_imf1_features.csv --feature_mode auto

Specify your own feature columns:
    python plot_feature_distributions.py --features ehg2_perm_entropy ehg2_DASDV ehg1_sampen
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import FEATURE_DIR, OUTPUT_DIR


# ---------------------------------------------------------------------
# Paper style
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "savefig.dpi": 300,
})

TERM_COLOR = "#6baed6"
PRETERM_COLOR = "#e56b6f"
POINT_COLOR = "#303030"

METADATA_COLUMNS = {
    "mode", "feature_source", "record", "name", "label", "segment_id",
    "start", "end", "start_sample", "end_sample", "start_sec", "end_sec",
    "imf", "mother_id", "group",
}

DEFAULT_FEATURES = [
    "ehg2_perm_entropy",
    "ehg2_DASDV",
    "ehg1_sampen",
    "ehg2_MTKE",
    "ehg2_LOG",
    "ehg2_PEAK_AMP_MEAN",
]

DISPLAY_NAMES = {
    "ehg1": "EHG1",
    "ehg2": "EHG2",
    "ehg3": "EHG3",
    "ch0": "EHG1",
    "ch1": "EHG2",
    "ch2": "EHG3",
    "perm_entropy": "perm entropy",
    "sampen": "sampen",
    "DASDV": "DASDV",
    "LOG": "LOG",
    "MTKE": "MTKE",
    "PEAK_AMP_MEAN": "PEAK AMP MEAN",
    "PEAK_RATE": "PEAK RATE",
    "PEAK_AMP_CV": "PEAK AMP CV",
    "IPI_MEAN": "IPI MEAN",
    "IPI_CV": "IPI CV",
    "PW_MEAN": "PW MEAN",
    "BURST_COUNT": "BURST COUNT",
    "PEAKS_PER_BURST_MEAN": "PEAKS PER BURST",
    "SE": "SE",
}


def panel_label(ax, label: str, x: float = -0.18, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def normalize_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def resolve_feature_name(requested: str, columns: List[str]) -> str | None:
    wanted = normalize_name(requested)
    mapping = {normalize_name(c): c for c in columns}
    return mapping.get(wanted)


def feature_display_name(col: str) -> str:
    parts = col.split("_", 1)
    if len(parts) == 2:
        channel, feat = parts
    else:
        return col.replace("_", " ")
    channel_disp = DISPLAY_NAMES.get(channel, channel.upper())
    feat_disp = DISPLAY_NAMES.get(feat, feat.replace("_", " "))
    return f"{channel_disp} {feat_disp}"


def get_record_column(df: pd.DataFrame) -> str:
    if "record" in df.columns:
        return "record"
    if "name" in df.columns:
        return "name"
    raise ValueError("CSV must contain either 'record' or 'name'.")


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
    return list(numeric.columns)


def aggregate_to_record(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    record_col = get_record_column(df)
    feature_cols = get_feature_columns(df)

    temp = df[[record_col, "label"] + feature_cols].copy()
    temp[feature_cols] = temp[feature_cols].apply(pd.to_numeric, errors="coerce")

    out = temp.groupby(record_col, as_index=False).agg(
        label=("label", "first"),
        **{c: (c, "mean") for c in feature_cols},
    )
    out = out.rename(columns={record_col: "record"})
    return out, feature_cols


def auto_select_features(record_df: pd.DataFrame, feature_cols: List[str], n: int) -> List[str]:
    rows = []
    for c in feature_cols:
        term = record_df.loc[record_df["label"] == 0, c].astype(float)
        pre = record_df.loc[record_df["label"] == 1, c].astype(float)
        if len(term) < 2 or len(pre) < 2:
            continue
        pooled = np.sqrt(0.5 * (np.nanvar(term) + np.nanvar(pre))) + 1e-12
        score = abs(np.nanmean(pre) - np.nanmean(term)) / pooled
        rows.append((c, score))
    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    return [c for c, _ in rows[:n]]


def choose_features(record_df: pd.DataFrame, feature_cols: List[str], mode: str, requested: List[str], n: int) -> List[str]:
    if requested:
        chosen = []
        for r in requested:
            col = resolve_feature_name(r, feature_cols)
            if col is None:
                print(f"Warning: requested feature not found: {r}")
            else:
                chosen.append(col)
        if chosen:
            return chosen[:n]

    if mode == "default":
        chosen = []
        for r in DEFAULT_FEATURES:
            col = resolve_feature_name(r, feature_cols)
            if col is not None:
                chosen.append(col)
        if len(chosen) >= min(n, len(DEFAULT_FEATURES)):
            return chosen[:n]
        print("Default feature list was not fully available. Falling back to auto selection.")

    return auto_select_features(record_df, feature_cols, n=n)


def draw_violin_with_points(ax, term_vals: np.ndarray, preterm_vals: np.ndarray, title: str) -> None:
    data = [term_vals, preterm_vals]
    vp = ax.violinplot(data, positions=[1, 2], widths=0.75, showmeans=False, showmedians=False, showextrema=False)

    for body, color in zip(vp["bodies"], [TERM_COLOR, PRETERM_COLOR]):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.95)
        body.set_linewidth(0.8)

    rng = np.random.default_rng(42)
    for xpos, vals in zip([1, 2], data):
        vals = np.asarray(vals, dtype=float)
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(
            np.full(len(vals), xpos) + jitter,
            vals,
            s=14,
            color=POINT_COLOR,
            alpha=0.75,
            zorder=3,
            linewidths=0,
        )
        med = np.nanmedian(vals)
        ax.plot([xpos - 0.22, xpos + 0.22], [med, med], color="black", lw=1.2, zorder=4)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Term", "Preterm"])
    ax.set_title(title)
    ax.grid(False)


def plot_feature_grid(record_df: pd.DataFrame, features: List[str], out_path: Path) -> None:
    n = len(features)
    ncols = 3 if n > 3 else n
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for i, (ax, col) in enumerate(zip(axes, features)):
        term_vals = record_df.loc[record_df["label"] == 0, col].astype(float).to_numpy()
        pre_vals = record_df.loc[record_df["label"] == 1, col].astype(float).to_numpy()
        draw_violin_with_points(ax, term_vals, pre_vals, feature_display_name(col))
        panel_label(ax, chr(ord("A") + i))

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(FEATURE_DIR / "tpehgt_fixed_3min_imf1_features.csv"))
    parser.add_argument("--out_dir", default=str(OUTPUT_DIR / "plots" / "paper"))
    parser.add_argument("--feature_mode", choices=["default", "auto"], default="default")
    parser.add_argument("--features", nargs="*", default=[])
    parser.add_argument("--n_features", type=int, default=6)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    record_df, feature_cols = aggregate_to_record(df)
    chosen = choose_features(record_df, feature_cols, args.feature_mode, args.features, args.n_features)

    print("Selected features:")
    for c in chosen:
        print(" ", c)

    out_dir = Path(args.out_dir)
    plot_feature_grid(record_df, chosen, out_dir / "feature_distributions_top6")

    record_df[["record", "label"] + chosen].to_csv(out_dir / "feature_distribution_values.csv", index=False)
    print(f"Saved feature distribution plot to: {out_dir}")


if __name__ == "__main__":
    main()
