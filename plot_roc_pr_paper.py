# plot_roc_pr_paper.py
# Generates side-by-side ROC and Precision-Recall curves for recording-level predictions.
#
# Example:
#   python plot_roc_pr_paper.py
#
# Optional:
#   python plot_roc_pr_paper.py --experiment fixed_3min_imf1
#   python plot_roc_pr_paper.py --curve_source pooled
#   python plot_roc_pr_paper.py --curve_source consensus
#   python plot_roc_pr_paper.py --models "Random Forest" CatBoost "Gradient Boosting"

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

from config import OUTPUT_DIR, FIG_DPI

# Paper-friendly model colors
MODEL_COLORS = {
    "Random Forest": "#1b9e77",
    "CatBoost": "#377eb8",
    "Gradient Boosting": "#e69f00",
    "MLP": "#984ea3",
    "SVM": "#d62728",
    "Logistic Regression": "#4d4d4d",
    "QDA": "#56b4e9",
    "Naive Bayes": "#f781bf",
    "Decision Tree": "#999999",
}


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9.5,
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


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top",
        ha="left",
    )


def find_prediction_file(experiment: str) -> Path:
    result_dir = Path(OUTPUT_DIR) / "results" / experiment

    candidates = [
        result_dir / "record_predictions.csv",
        result_dir / "record_level_predictions.csv",
        result_dir / "predictions.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find prediction file in {result_dir}. "
        "Expected one of: record_predictions.csv, record_level_predictions.csv, predictions.csv"
    )


def find_summary_file(experiment: str) -> Path | None:
    result_dir = Path(OUTPUT_DIR) / "results" / experiment

    candidates = [
        result_dir / "summary_record_metrics.csv",
        result_dir / "summary_metrics.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def standardize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes the script robust to slightly different column names.
    Required final columns:
      model, record, label, score
    """

    rename_map = {}

    if "y_true" in df.columns and "label" not in df.columns:
        rename_map["y_true"] = "label"

    if "true_label" in df.columns and "label" not in df.columns:
        rename_map["true_label"] = "label"

    if "record_score" in df.columns and "score" not in df.columns:
        rename_map["record_score"] = "score"

    if "probability" in df.columns and "score" not in df.columns:
        rename_map["probability"] = "score"

    if "preterm_probability" in df.columns and "score" not in df.columns:
        rename_map["preterm_probability"] = "score"

    if "pred_score" in df.columns and "score" not in df.columns:
        rename_map["pred_score"] = "score"

    df = df.rename(columns=rename_map)

    required = {"model", "record", "label", "score"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Prediction file is missing required columns: {missing}\n"
            f"Available columns are: {list(df.columns)}"
        )

    df["label"] = df["label"].astype(int)
    df["score"] = df["score"].astype(float)

    return df


def choose_models(
    pred_df: pd.DataFrame,
    summary_path: Path | None,
    requested_models: list[str] | None,
    top_n: int,
) -> list[str]:
    if requested_models:
        return requested_models

    if summary_path is not None:
        summary = pd.read_csv(summary_path)

        if "model" in summary.columns:
            if "record_pr_auc_mean" in summary.columns:
                summary = summary.sort_values("record_pr_auc_mean", ascending=False)
            elif "pr_auc_mean" in summary.columns:
                summary = summary.sort_values("pr_auc_mean", ascending=False)
            elif "record_roc_auc_mean" in summary.columns:
                summary = summary.sort_values("record_roc_auc_mean", ascending=False)
            elif "record_balanced_accuracy_mean" in summary.columns:
                summary = summary.sort_values(
                    "record_balanced_accuracy_mean", ascending=False
                )

            return summary["model"].head(top_n).tolist()

    return sorted(pred_df["model"].unique())[:top_n]


def get_curve_data(
    df: pd.DataFrame,
    model: str,
    curve_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    curve_source = "pooled":
        Uses all repeated out-of-fold recording predictions.
        This gives smoother curves and usually matches repeated-CV behavior better.

    curve_source = "consensus":
        Averages repeated scores for each recording, so each of the 26 recordings appears once.
        This is useful if you want the curve to represent one final score per recording.
    """

    g = df[df["model"] == model].copy()

    if g.empty:
        raise ValueError(f"No predictions found for model: {model}")

    if curve_source == "pooled":
        y_true = g["label"].to_numpy()
        scores = g["score"].to_numpy()

    elif curve_source == "consensus":
        record_df = g.groupby("record", as_index=False).agg(
            label=("label", "first"),
            score=("score", "mean"),
        )

        y_true = record_df["label"].to_numpy()
        scores = record_df["score"].to_numpy()

    else:
        raise ValueError("curve_source must be either 'pooled' or 'consensus'.")

    return y_true, scores


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        default="fixed_3min_imf1",
        help="Experiment folder inside outputs/results/",
    )

    parser.add_argument(
        "--curve_source",
        default="pooled",
        choices=["pooled", "consensus"],
        help="Use pooled repeated-CV predictions or one consensus score per recording.",
    )

    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Models to plot. If omitted, top models are selected from summary metrics.",
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of models to plot if --models is not provided.",
    )

    parser.add_argument(
        "--show_auc",
        action="store_true",
        help="Show AUC values in the legend. Leave off if table reports final AUC values.",
    )

    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output folder. Default: outputs/plots/paper",
    )

    args = parser.parse_args()

    set_paper_style()

    pred_path = find_prediction_file(args.experiment)
    summary_path = find_summary_file(args.experiment)

    pred_df = pd.read_csv(pred_path)
    pred_df = standardize_prediction_columns(pred_df)

    models = choose_models(
        pred_df=pred_df,
        summary_path=summary_path,
        requested_models=args.models,
        top_n=args.top_n,
    )

    out_dir = (
        Path(args.out_dir) if args.out_dir else Path(OUTPUT_DIR) / "plots" / "paper"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))

    ax_roc, ax_pr = axes

    auc_rows = []

    for model in models:
        y_true, scores = get_curve_data(pred_df, model, args.curve_source)

        if len(np.unique(y_true)) < 2:
            print(f"Skipping {model}: only one class present.")
            continue

        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)

        roc_auc = roc_auc_score(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)

        color = MODEL_COLORS.get(model, None)

        if args.show_auc:
            roc_label = f"{model} ({roc_auc:.3f})"
            pr_label = f"{model} ({pr_auc:.3f})"
        else:
            roc_label = model
            pr_label = model

        ax_roc.plot(
            fpr,
            tpr,
            linewidth=2.3,
            color=color,
            label=roc_label,
        )

        ax_pr.plot(
            recall,
            precision,
            linewidth=2.3,
            color=color,
            label=pr_label,
        )

        auc_rows.append(
            {
                "experiment": args.experiment,
                "curve_source": args.curve_source,
                "model": model,
                "n_points": len(y_true),
                "roc_auc_for_plotted_curve": roc_auc,
                "pr_auc_for_plotted_curve": pr_auc,
            }
        )

    # ROC formatting
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1.2)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(frameon=False, loc="lower right")

    # PR formatting
    prevalence = pred_df["label"].mean()
    ax_pr.axhline(prevalence, linestyle="--", color="0.55", linewidth=1.2)
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-recall curve")
    ax_pr.legend(frameon=False, loc="lower left")

    add_panel_label(ax_roc, "A")
    add_panel_label(ax_pr, "B")

    fig.tight_layout()

    out_base = out_dir / f"roc_pr_curves_{args.experiment}_{args.curve_source}"

    fig.savefig(out_base.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(out_base.with_suffix(".csv"), index=False)

    print(f"Prediction file used: {pred_path}")
    print(f"Models plotted: {models}")
    print(f"Saved: {out_base.with_suffix('.png')}")
    print(f"Saved: {out_base.with_suffix('.pdf')}")
    print(f"Saved: {out_base.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
