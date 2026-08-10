"""
Generate manuscript model-performance figures from outputs/results/.

Outputs under outputs/plots/paper/:
  roc_pr_curves.{png,pdf}      # manuscript Fig. 6
  confusion_matrices.{png,pdf} # manuscript Fig. 7

Fig. 6 pools repeated out-of-fold recording scores across all 30 repetitions.
Fig. 7 uses one consensus binary decision per recording: majority vote over the
repeated out-of-fold binary decisions already produced using each fold's training-
selected threshold. Therefore each confusion matrix sums to 26 recordings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from config import FIG_DPI, N_REPEATS, PLOT_DIR, RESULT_DIR, TOP_MODELS_FOR_ROC
from paper_style import (
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    normalize_model_name,
    panel_label,
    save_png_pdf,
    set_curve_style,
)

DEFAULT_CURVE_MODELS = [
    normalize_model_name(name)
    for name in TOP_MODELS_FOR_ROC
]


def read_predictions(experiment: str) -> pd.DataFrame:
    path = RESULT_DIR / experiment / "record_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"model", "record", "label", "score", "prediction", "repeat", "fold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["model"] = df["model"].map(normalize_model_name)
    return df


def read_summary(experiment: str) -> pd.DataFrame:
    path = RESULT_DIR / experiment / "summary_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["model"] = df["model"].map(normalize_model_name)
    return df


def plot_roc_pr(experiment: str, models: list[str], out_dir: Path) -> None:
    set_curve_style(FIG_DPI)
    preds = read_predictions(experiment)
    models = [normalize_model_name(m) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax_roc, ax_pr = axes
    curve_rows = []
    metric_rows = []

    for model in models:
        g = preds[preds["model"] == model]
        if g.empty:
            print(f"WARNING: no predictions for {model}; skipping.")
            continue
        y = g["label"].astype(int).to_numpy()
        score = g["score"].astype(float).to_numpy()
        if np.unique(y).size < 2:
            continue

        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        roc_auc = float(roc_auc_score(y, score))
        ap = float(average_precision_score(y, score))
        color = MODEL_COLORS.get(model)
        display = MODEL_DISPLAY_NAMES.get(model, model)

        ax_roc.plot(fpr, tpr, linewidth=2.4, color=color, label=display)
        ax_pr.plot(recall, precision, linewidth=2.4, color=color, label=display)

        curve_rows.extend(
            {"model": model, "curve": "ROC", "x": float(x), "y": float(v)}
            for x, v in zip(fpr, tpr)
        )
        curve_rows.extend(
            {"model": model, "curve": "precision_recall", "x": float(x), "y": float(v)}
            for x, v in zip(recall, precision)
        )
        metric_rows.append(
            {
                "experiment": experiment,
                "curve_source": "pooled_repeated_out_of_fold_record_scores",
                "model": model,
                "n_record_scores": len(y),
                "pooled_oof_roc_auc": roc_auc,
                "pooled_oof_ap": ap,
            }
        )

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1.4)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(frameon=False, loc="lower right")

    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-recall curve")
    ax_pr.legend(frameon=False, loc="lower left")

    panel_label(ax_roc, "A", fontsize=22)
    panel_label(ax_pr, "B", fontsize=22)
    fig.tight_layout()
    save_png_pdf(fig, out_dir / "roc_pr_curves", FIG_DPI)
    plt.close(fig)

    pd.DataFrame(curve_rows).to_csv(out_dir / "roc_pr_curve_values.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "roc_pr_curve_metrics.csv", index=False)


def best_model_by_balanced_accuracy(experiment: str) -> str:
    summary = read_summary(experiment)
    col = "record_balanced_accuracy_mean"
    if col not in summary.columns:
        raise ValueError(f"{experiment} summary lacks {col}")
    return str(summary.sort_values(col, ascending=False).iloc[0]["model"])


def consensus_predictions(preds: pd.DataFrame, model: str) -> pd.DataFrame:
    model = normalize_model_name(model)
    df = preds[preds["model"] == model].copy()
    if df.empty:
        raise ValueError(f"No predictions for model={model}")
    rows = []
    for record, g in df.groupby("record", sort=True):
        labels = g["label"].astype(int).unique()
        if len(labels) != 1:
            raise ValueError(f"Inconsistent labels for record={record}")

        if len(g) != N_REPEATS or g["repeat"].nunique() != N_REPEATS:
            raise ValueError(
                f"Expected {N_REPEATS} repeated OOF votes for record={record}, "
                f"model={model}; found rows={len(g)}, "
                f"unique_repeats={g['repeat'].nunique()}."
            )

        votes = g["prediction"].astype(int)
        positive_votes = int(votes.sum())

        if positive_votes * 2 == len(votes):
            raise ValueError(
                f"Exact majority-vote tie for record={record}, model={model}: "
                f"{positive_votes}/{len(votes)} preterm votes."
            )

        vote_fraction = float(votes.mean())
        rows.append(
            {
                "record": record,
                "label": int(labels[0]),
                "consensus_prediction": int(positive_votes * 2 > len(votes)),
                "preterm_vote_fraction": vote_fraction,
                "mean_oof_score": float(g["score"].astype(float).mean()),
                "n_repeated_oof_votes": len(g),
                "model": model,
            }
        )
    return pd.DataFrame(rows)


def plot_confusions(
    annotated_model: str | None,
    fixed_model: str | None,
    out_dir: Path,
) -> None:
    set_curve_style(FIG_DPI)
    experiments = [
        (
            "annotated_interval_imf1",
            "Annotated intervals",
            "Oranges",
            annotated_model,
        ),
        ("fixed_3min_imf1", "Fixed 3-minute", "Blues", fixed_model),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.1))
    consensus_rows = []
    matrix_rows = []

    for ax, (experiment, title, cmap, requested_model), letter in zip(axes, experiments, "AB"):
        model = normalize_model_name(requested_model) if requested_model else best_model_by_balanced_accuracy(experiment)
        preds = read_predictions(experiment)
        consensus = consensus_predictions(preds, model)
        consensus["experiment"] = experiment
        consensus_rows.append(consensus)

        cm = confusion_matrix(
            consensus["label"].astype(int),
            consensus["consensus_prediction"].astype(int),
            labels=[0, 1],
        )
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = cm / np.where(row_sums == 0, 1, row_sums)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                text_color = "white" if norm[i, j] > 0.55 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.2f})", ha="center", va="center", color=text_color, fontsize=12)
                matrix_rows.append(
                    {
                        "experiment": experiment,
                        "model": model,
                        "true_label": i,
                        "predicted_label": j,
                        "count": int(cm[i, j]),
                        "row_normalized_fraction": float(norm[i, j]),
                    }
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Term", "Preterm"])
        ax.set_yticklabels(["Term", "Preterm"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title}\n{MODEL_DISPLAY_NAMES.get(model, model)}")
        panel_label(ax, letter, fontsize=18)

    fig.tight_layout()
    save_png_pdf(fig, out_dir / "confusion_matrices", FIG_DPI)
    plt.close(fig)

    pd.concat(consensus_rows, ignore_index=True).to_csv(out_dir / "confusion_consensus_record_values.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(out_dir / "confusion_matrix_values.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript performance figures 6-7.")
    parser.add_argument("--experiment", default="fixed_3min_imf1")
    parser.add_argument("--models", nargs="*", default=DEFAULT_CURVE_MODELS)
    parser.add_argument(
        "--annotated-model",
        default=None,
        help="Model for the annotated-interval panel. Default: best mean balanced-accuracy model.",
    )
    parser.add_argument("--fixed-model", default=None, help="Default: best mean balanced-accuracy model.")
    parser.add_argument("--out-dir", type=Path, default=PLOT_DIR / "paper")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_pr(args.experiment, args.models, args.out_dir)
    plot_confusions(args.annotated_model, args.fixed_model, args.out_dir)
    print(f"Saved manuscript performance figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
