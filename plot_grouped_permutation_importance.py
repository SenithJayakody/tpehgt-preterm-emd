# plot_grouped_permutation_importance.py
# Computes grouped permutation feature importance and generates a paper-ready plot.
#
# Importance = decrease in recording-level PR-AUC after permuting a feature group.
#
# Default experiments:
#   1) fixed_3min_imf1, Random Forest
#   2) contraction_imf1, Random Forest
#
# Run:
#   python plot_grouped_permutation_importance.py
#
# Optional:
#   python plot_grouped_permutation_importance.py --n_repeats 30 --n_splits 5
#
# Outputs:
#   outputs/plots/paper/grouped_permutation_importance_summary.csv
#   outputs/plots/paper/grouped_feature_importance.png
#   outputs/plots/paper/grouped_feature_importance.pdf

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier

try:
    from config import OUTPUT_DIR, FIG_DPI
except Exception:
    OUTPUT_DIR = Path("outputs")
    FIG_DPI = 300


METADATA_COLUMNS = {
    "mode",
    "feature_source",
    "record",
    "name",
    "mother_id",
    "label",
    "segment_id",
    "start_sample",
    "end_sample",
    "start_sec",
    "end_sec",
    "imf",
}


FEATURE_GROUPS = {
    "Burst and peak features": [
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


EXPERIMENTS = [
    {
        "name": "Contraction IMF1",
        "csv": Path(OUTPUT_DIR) / "features" / "tpehgt_contraction_imf1_features.csv",
        "model_name": "Random Forest",
        "plot_color": "#1F4E79",
    },
    {
        "name": "Fixed 3-min IMF1",
        "csv": Path(OUTPUT_DIR) / "features" / "tpehgt_fixed_3min_imf1_features.csv",
        "model_name": "Random Forest",
        "plot_color": "#16836F",
    },
]


def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "savefig.dpi": FIG_DPI,
        }
    )


def get_record_column(df: pd.DataFrame) -> str:
    if "record" in df.columns:
        return "record"
    if "name" in df.columns:
        return "name"
    raise ValueError("CSV must contain either 'record' or 'name' column.")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols = []
    for col in df.columns:
        if col not in METADATA_COLUMNS:
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
    return feature_cols


def assign_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    grouped = {group_name: [] for group_name in FEATURE_GROUPS}
    unmatched = []

    for col in feature_cols:
        matched = False

        for group_name, suffixes in FEATURE_GROUPS.items():
            for suffix in suffixes:
                if col.endswith("_" + suffix) or col == suffix:
                    grouped[group_name].append(col)
                    matched = True
                    break
            if matched:
                break

        if not matched:
            unmatched.append(col)

    if unmatched:
        grouped["Other"] = unmatched

    return grouped


def make_model(seed: int) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
        class_weight=None,
    )

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    return pipe


def predict_preterm_scores(pipe: Pipeline, x: pd.DataFrame) -> np.ndarray:
    model = pipe.named_steps["model"]

    if hasattr(model, "predict_proba"):
        return pipe.predict_proba(x)[:, 1]

    if hasattr(model, "decision_function"):
        scores = pipe.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))

    raise ValueError("Model does not support probability or decision-function scoring.")


def aggregate_record_scores(
    records: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    temp = pd.DataFrame(
        {
            "record": records,
            "label": labels,
            "score": scores,
        }
    )

    # Same aggregation idea as the main classification:
    # maximum segment probability per recording.
    record_df = temp.groupby("record", as_index=False).agg(
        label=("label", "first"),
        score=("score", "max"),
    )

    return record_df


def record_pr_auc(records: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> float:
    record_df = aggregate_record_scores(records, labels, scores)

    if record_df["label"].nunique() < 2:
        return np.nan

    return float(
        average_precision_score(
            record_df["label"].to_numpy(),
            record_df["score"].to_numpy(),
        )
    )


def compute_grouped_importance_for_experiment(
    csv_path: Path,
    experiment_name: str,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:

    if not csv_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'label' column.")

    record_col = get_record_column(df)
    feature_cols = get_feature_columns(df)

    if len(feature_cols) == 0:
        raise ValueError(f"No numeric feature columns found in {csv_path}")

    grouped_features = assign_feature_groups(feature_cols)

    print("\n" + "=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"CSV       : {csv_path}")
    print(f"Rows      : {len(df)}")
    print(f"Features  : {len(feature_cols)}")
    print(f"Records   : {df[record_col].nunique()}")
    print("Feature groups:")

    for group_name, cols in grouped_features.items():
        print(f"  {group_name}: {len(cols)} columns")

    x = df[feature_cols].copy()
    y = df["label"].astype(int).to_numpy()
    groups = df[record_col].astype(str).to_numpy()
    records = df[record_col].astype(str).to_numpy()

    rows = []

    for repeat in range(n_repeats):
        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + repeat,
        )

        for fold, (train_idx, val_idx) in enumerate(cv.split(x, y, groups), start=1):
            x_train = x.iloc[train_idx].copy()
            y_train = y[train_idx]

            x_val = x.iloc[val_idx].copy()
            y_val = y[val_idx]
            records_val = records[val_idx]

            pipe = make_model(seed + repeat * 100 + fold)
            pipe.fit(x_train, y_train)

            base_scores = predict_preterm_scores(pipe, x_val)
            base_pr_auc = record_pr_auc(records_val, y_val, base_scores)

            if not np.isfinite(base_pr_auc):
                continue

            rng = np.random.default_rng(seed + repeat * 1000 + fold)

            for group_name, group_cols in grouped_features.items():
                if len(group_cols) == 0:
                    continue

                x_perm = x_val.copy()

                # Permute the whole feature group together across validation segments.
                # This breaks the relationship between this feature group and the labels
                # while preserving correlations inside the group.
                perm_idx = rng.permutation(len(x_perm))
                x_perm.loc[:, group_cols] = x_perm.iloc[perm_idx][group_cols].to_numpy()

                perm_scores = predict_preterm_scores(pipe, x_perm)
                perm_pr_auc = record_pr_auc(records_val, y_val, perm_scores)

                if not np.isfinite(perm_pr_auc):
                    continue

                rows.append(
                    {
                        "experiment": experiment_name,
                        "model": "Random Forest",
                        "repeat": repeat + 1,
                        "fold": fold,
                        "feature_group": group_name,
                        "base_pr_auc": base_pr_auc,
                        "permuted_pr_auc": perm_pr_auc,
                        "decrease_pr_auc": base_pr_auc - perm_pr_auc,
                        "n_features_in_group": len(group_cols),
                    }
                )

    return pd.DataFrame(rows)


def plot_grouped_importance(summary_df: pd.DataFrame, out_dir: Path):
    group_order = [
        "Burst and peak features",
        "Temporal energy features",
        "Entropy features",
    ]

    if "Other" in summary_df["feature_group"].unique():
        group_order.append("Other")

    experiment_order = [
        "Contraction IMF1",
        "Fixed 3-min IMF1",
    ]

    colors = {
        "Contraction IMF1": "#1F4E79",
        "Fixed 3-min IMF1": "#16836F",
    }

    x = np.arange(len(group_order))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    offsets = {
        "Contraction IMF1": -width / 2,
        "Fixed 3-min IMF1": width / 2,
    }

    for exp in experiment_order:
        vals = []
        for group in group_order:
            row = summary_df[
                (summary_df["experiment"] == exp)
                & (summary_df["feature_group"] == group)
            ]
            if row.empty:
                vals.append(0.0)
            else:
                vals.append(float(row["mean_decrease_pr_auc"].iloc[0]))

        ax.bar(
            x + offsets[exp],
            vals,
            width=width,
            label=f"{exp} (Random Forest)",
            color=colors[exp],
            edgecolor="black",
            linewidth=0.7,
        )

        for xi, val in zip(x + offsets[exp], vals):
            ax.text(
                xi,
                val + 0.002,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=15, ha="right")
    ax.set_ylabel("Mean decrease in recording-level PR-AUC")
    # ax.set_title("Grouped permutation feature importance")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()

    png_path = out_dir / "grouped_feature_importance.png"
    pdf_path = out_dir / "grouped_feature_importance.pdf"

    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--out_dir",
        default=str(Path(OUTPUT_DIR) / "plots" / "paper"),
    )

    parser.add_argument(
        "--fixed_csv",
        default=str(
            Path(OUTPUT_DIR) / "features" / "tpehgt_fixed_3min_imf1_features.csv"
        ),
    )

    parser.add_argument(
        "--contraction_csv",
        default=str(
            Path(OUTPUT_DIR) / "features" / "tpehgt_contraction_imf1_features.csv"
        ),
    )

    args = parser.parse_args()

    set_paper_style()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "name": "Contraction IMF1",
            "csv": Path(args.contraction_csv),
        },
        {
            "name": "Fixed 3-min IMF1",
            "csv": Path(args.fixed_csv),
        },
    ]

    all_rows = []

    for exp in experiments:
        imp_df = compute_grouped_importance_for_experiment(
            csv_path=exp["csv"],
            experiment_name=exp["name"],
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            seed=args.seed,
        )

        all_rows.append(imp_df)

    importance_df = pd.concat(all_rows, ignore_index=True)

    all_csv = out_dir / "grouped_permutation_importance_all_folds.csv"
    importance_df.to_csv(all_csv, index=False)
    print(f"Saved: {all_csv}")

    # Mean only. No SD reported.
    summary_df = importance_df.groupby(
        ["experiment", "model", "feature_group"], as_index=False
    ).agg(
        mean_decrease_pr_auc=("decrease_pr_auc", "mean"),
        n_folds=("decrease_pr_auc", "size"),
        n_features_in_group=("n_features_in_group", "first"),
    )

    summary_csv = out_dir / "grouped_permutation_importance_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    print("\nGrouped permutation importance summary:")
    print(summary_df.to_string(index=False))

    plot_grouped_importance(summary_df, out_dir)


if __name__ == "__main__":
    main()
