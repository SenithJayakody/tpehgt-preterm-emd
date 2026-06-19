# plot_roc_pr_curves.py
# Generate ROC and precision-recall curves for record-wise classification.
#
# Default:
#   python plot_roc_pr_curves.py
#
# Example:
#   python plot_roc_pr_curves.py --experiment fixed_3min_imf1
#
# Use selected models:
#   python plot_roc_pr_curves.py --models "Random Forest" CatBoost "Gradient Boosting" MLP SVM
#
# Curve source:
#   --curve_source consensus  -> one averaged score per recording, counts = 26 recordings
#   --curve_source pooled     -> all out-of-fold repeated-CV predictions pooled

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

COLORS = {
    "CatBoost": "#1f4e79",
    "Random Forest": "#16836f",
    "Gradient Boosting": "#c57f0a",
    "MLP": "#7b3294",
    "SVM": "#a12a22",
    "Logistic Regression": "#4d4d4d",
    "QDA": "#6baed6",
    "Naive Bayes": "#d95f02",
    "Decision Tree": "#7570b3",
}


def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.4,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "savefig.dpi": FIG_DPI,
        }
    )


def add_panel_label(ax, label):
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        va="top",
        ha="left",
    )


def find_prediction_file(experiment: str):
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
        f"Could not find record_predictions.csv in {result_dir}. "
        f"Run classify_groupwise_cv.py first."
    )


def find_summary_file(experiment: str):
    result_dir = Path(OUTPUT_DIR) / "results" / experiment
    path = result_dir / "summary_metrics.csv"

    if path.exists():
        return path

    return None


def choose_models(
    pred_df: pd.DataFrame, summary_path: Path | None, requested_models, top_n: int
):
    if requested_models:
        return requested_models

    if summary_path is not None:
        summary = pd.read_csv(summary_path)

        if "record_pr_auc_mean" in summary.columns:
            summary = summary.sort_values("record_pr_auc_mean", ascending=False)
        elif "record_roc_auc_mean" in summary.columns:
            summary = summary.sort_values("record_roc_auc_mean", ascending=False)
        elif "record_balanced_accuracy_mean" in summary.columns:
            summary = summary.sort_values(
                "record_balanced_accuracy_mean", ascending=False
            )

        models = summary["model"].head(top_n).tolist()
        return models

    return sorted(pred_df["model"].unique())[:top_n]


def get_curve_data(df: pd.DataFrame, model: str, curve_source: str):
    g = df[df["model"] == model].copy()

    if g.empty:
        raise ValueError(f"No predictions found for model: {model}")

    if curve_source == "consensus":
        # One score per recording: average repeated out-of-fold scores.
        record_df = g.groupby("record", as_index=False).agg(
            label=("label", "first"),
            score=("score", "mean"),
        )

        y_true = record_df["label"].astype(int).to_numpy()
        scores = record_df["score"].astype(float).to_numpy()

    elif curve_source == "pooled":
        # All repeated out-of-fold predictions pooled.
        y_true = g["label"].astype(int).to_numpy()
        scores = g["score"].astype(float).to_numpy()

    else:
        raise ValueError("curve_source must be 'consensus' or 'pooled'")

    return y_true, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="fixed_3min_imf1")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument(
        "--curve_source", default="consensus", choices=["consensus", "pooled"]
    )
    parser.add_argument("--show_auc_in_legend", action="store_true")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    set_paper_style()

    pred_path = find_prediction_file(args.experiment)
    summary_path = find_summary_file(args.experiment)

    pred_df = pd.read_csv(pred_path)

    if "record" not in pred_df.columns:
        raise ValueError("Prediction file must contain a 'record' column.")

    if (
        "label" not in pred_df.columns
        or "score" not in pred_df.columns
        or "model" not in pred_df.columns
    ):
        raise ValueError(
            "Prediction file must contain 'model', 'label', and 'score' columns."
        )

    models = choose_models(pred_df, summary_path, args.models, args.top_n)

    out_dir = (
        Path(args.out_dir) if args.out_dir else Path(OUTPUT_DIR) / "plots" / "paper"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    ax_roc, ax_pr = axes

    for model in models:
        y_true, scores = get_curve_data(pred_df, model, args.curve_source)

        if len(np.unique(y_true)) < 2:
            print(f"Skipping {model}: only one class present.")
            continue

        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)

        roc_auc = roc_auc_score(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)

        color = COLORS.get(model, None)

        if args.show_auc_in_legend:
            roc_label = f"{model} ({roc_auc:.3f})"
            pr_label = f"{model} ({pr_auc:.3f})"
        else:
            roc_label = model
            pr_label = model

        ax_roc.plot(
            fpr,
            tpr,
            linewidth=2.4,
            color=color,
            label=roc_label,
        )

        ax_pr.plot(
            recall,
            precision,
            linewidth=2.4,
            color=color,
            label=pr_label,
        )

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1.4)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC")
    ax_roc.legend(frameon=False, loc="lower right")

    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-recall")
    ax_pr.legend(frameon=False, loc="lower left")

    add_panel_label(ax_roc, "A")
    add_panel_label(ax_pr, "B")

    title = args.experiment.replace("_", " ")
    source_text = (
        "26-record consensus"
        if args.curve_source == "consensus"
        else "pooled out-of-fold"
    )
    fig.suptitle(f"{title} ({source_text})", y=1.03)

    fig.tight_layout()

    out_base = out_dir / f"roc_pr_curves_{args.experiment}_{args.curve_source}"

    fig.savefig(out_base.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_base.with_suffix('.png')}")
    print(f"Saved: {out_base.with_suffix('.pdf')}")

    # Save exact AUC values used in the plotted curves.
    rows = []

    for model in models:
        y_true, scores = get_curve_data(pred_df, model, args.curve_source)

        if len(np.unique(y_true)) < 2:
            continue

        rows.append(
            {
                "experiment": args.experiment,
                "curve_source": args.curve_source,
                "model": model,
                "n_points": len(y_true),
                "roc_auc_for_plotted_curve": roc_auc_score(y_true, scores),
                "pr_auc_for_plotted_curve": average_precision_score(y_true, scores),
            }
        )

    auc_df = pd.DataFrame(rows)
    auc_csv = out_base.with_suffix(".csv")
    auc_df.to_csv(auc_csv, index=False)
    print(f"Saved: {auc_csv}")


if __name__ == "__main__":
    main()
