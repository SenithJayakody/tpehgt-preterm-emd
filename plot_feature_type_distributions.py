# plot_feature_type_distributions.py
# Generate recording-level distributions of the 14 IMF1 feature types.
#
# The classifier uses 42 channel-specific features:
#   14 feature types x 3 EHG channels
#
# This plot averages each feature type across the 3 channels for visualization only.
#
# Run:
#   python plot_feature_type_distributions.py
#
# Optional:
#   python plot_feature_type_distributions.py --csv outputs/features/tpehgt_fixed_3min_imf1_features.csv
#   python plot_feature_type_distributions.py --out_dir outputs/plots/paper

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from config import OUTPUT_DIR, FIG_DPI
except Exception:
    OUTPUT_DIR = Path("outputs")
    FIG_DPI = 300


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


TERM_COLOR = "#4C78A8"
PRETERM_COLOR = "#D64F4F"


def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "savefig.dpi": FIG_DPI,
        }
    )


def get_record_column(df: pd.DataFrame) -> str:
    if "record" in df.columns:
        return "record"
    if "name" in df.columns:
        return "name"
    raise ValueError("CSV must contain either 'record' or 'name' column.")


def find_channel_columns(df: pd.DataFrame, feature_type: str) -> list[str]:
    """
    Finds channel-specific columns for one feature type.

    Supports names like:
      ch0_DASDV, ch1_DASDV, ch2_DASDV
      ehg1_DASDV, ehg2_DASDV, ehg3_DASDV
    """
    cols = []

    for c in df.columns:
        if c.endswith("_" + feature_type):
            cols.append(c)

    # Keep stable order
    cols = sorted(cols)

    return cols


def cohen_d(preterm_values: np.ndarray, term_values: np.ndarray) -> float:
    """
    Cohen's d = (mean_preterm - mean_term) / pooled_sd

    Positive d: higher in preterm.
    Negative d: higher in term.
    """
    preterm_values = np.asarray(preterm_values, dtype=float)
    term_values = np.asarray(term_values, dtype=float)

    preterm_values = preterm_values[np.isfinite(preterm_values)]
    term_values = term_values[np.isfinite(term_values)]

    n1 = len(preterm_values)
    n0 = len(term_values)

    if n1 < 2 or n0 < 2:
        return np.nan

    s1 = np.var(preterm_values, ddof=1)
    s0 = np.var(term_values, ddof=1)

    pooled = np.sqrt(((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2))

    if pooled == 0:
        return np.nan

    return (np.mean(preterm_values) - np.mean(term_values)) / pooled


def make_record_level_feature_types(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    record_col = get_record_column(df)

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    segment_feature_type_df = df[[record_col, "label"]].copy()
    segment_feature_type_df = segment_feature_type_df.rename(
        columns={record_col: "record"}
    )

    used_columns = {}

    for feature_type in FEATURE_TYPES:
        channel_cols = find_channel_columns(df, feature_type)

        if len(channel_cols) == 0:
            print(f"WARNING: No columns found for feature type: {feature_type}")
            segment_feature_type_df[feature_type] = np.nan
            used_columns[feature_type] = []
            continue

        # Convert to numeric and average across channels for visualization.
        segment_feature_type_df[feature_type] = (
            df[channel_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        )
        used_columns[feature_type] = channel_cols

    # Aggregate segments to recording level.
    agg_dict = {feature_type: "mean" for feature_type in FEATURE_TYPES}
    agg_dict["label"] = "first"

    record_df = segment_feature_type_df.groupby("record", as_index=False).agg(agg_dict)

    # Cohen's d for each feature type.
    rows = []
    for feature_type in FEATURE_TYPES:
        term_values = record_df.loc[record_df["label"] == 0, feature_type].to_numpy()
        preterm_values = record_df.loc[record_df["label"] == 1, feature_type].to_numpy()

        d = cohen_d(preterm_values, term_values)

        rows.append(
            {
                "feature_type": feature_type,
                "display_name": DISPLAY_NAMES.get(feature_type, feature_type),
                "cohen_d_preterm_minus_term": d,
                "term_mean": np.nanmean(term_values),
                "preterm_mean": np.nanmean(preterm_values),
                "n_channel_columns_used": len(used_columns[feature_type]),
                "channel_columns_used": ", ".join(used_columns[feature_type]),
            }
        )

    effect_df = pd.DataFrame(rows)

    return record_df, effect_df


def plot_feature_distributions(
    record_df: pd.DataFrame, effect_df: pd.DataFrame, out_dir: Path
):
    rng = np.random.default_rng(42)

    n_features = len(FEATURE_TYPES)
    n_rows = 4
    n_cols = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.5, 11.0))
    axes = axes.ravel()

    panel_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    for idx, feature_type in enumerate(FEATURE_TYPES):
        ax = axes[idx]

        term_values = (
            record_df.loc[record_df["label"] == 0, feature_type].dropna().to_numpy()
        )
        preterm_values = (
            record_df.loc[record_df["label"] == 1, feature_type].dropna().to_numpy()
        )

        data = [term_values, preterm_values]
        positions = [1, 2]

        parts = ax.violinplot(
            data,
            positions=positions,
            widths=0.75,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body, color in zip(parts["bodies"], [TERM_COLOR, PRETERM_COLOR]):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.35)
            body.set_linewidth(0.8)

        # Box-like median and IQR lines
        for pos, values, color in zip(positions, data, [TERM_COLOR, PRETERM_COLOR]):
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

        effect_row = effect_df[effect_df["feature_type"] == feature_type].iloc[0]
        d_value = effect_row["cohen_d_preterm_minus_term"]

        title = DISPLAY_NAMES.get(feature_type, feature_type)

        if np.isfinite(d_value):
            ax.set_title(f"{title}\nCohen's d = {d_value:.2f}")
        else:
            ax.set_title(f"{title}\nCohen's d = NA")

        ax.set_xticks(positions)
        ax.set_xticklabels(["Term", "Preterm"])
        ax.set_ylabel("Recording-level value")

        ax.text(
            -0.16,
            1.15,
            panel_labels[idx],
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="left",
        )

        ax.grid(axis="y", alpha=0.25, linewidth=0.7)

    # Hide unused panels
    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    # fig.suptitle(
    #     "Recording-level distributions of IMF1 feature types",
    #     fontsize=16,
    #     y=0.995,
    # )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = out_dir / "feature_type_distributions_14.png"
    pdf_path = out_dir / "feature_type_distributions_14.pdf"

    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default=str(
            Path(OUTPUT_DIR) / "features" / "tpehgt_fixed_3min_imf1_features.csv"
        ),
        help="Feature CSV file. Default: outputs/features/tpehgt_fixed_3min_imf1_features.csv",
    )

    parser.add_argument(
        "--out_dir",
        default=str(Path(OUTPUT_DIR) / "plots" / "paper"),
        help="Output directory. Default: outputs/plots/paper",
    )

    args = parser.parse_args()

    set_paper_style()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    record_df, effect_df = make_record_level_feature_types(df)

    record_csv = out_dir / "feature_type_record_level_values_14.csv"
    effect_csv = out_dir / "feature_type_cohens_d_14.csv"

    record_df.to_csv(record_csv, index=False)
    effect_df.to_csv(effect_csv, index=False)

    print(f"Saved: {record_csv}")
    print(f"Saved: {effect_csv}")

    print("\nCohen's d values:")
    print(
        effect_df[
            ["display_name", "cohen_d_preterm_minus_term", "term_mean", "preterm_mean"]
        ].to_string(index=False)
    )

    plot_feature_distributions(record_df, effect_df, out_dir)


if __name__ == "__main__":
    main()
