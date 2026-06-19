# plot_confusion_matrices.py
# Generates recording-level consensus confusion matrices for the paper.
#
# Run:
#   python plot_confusion_matrices.py
#
# Expected input files:
#   outputs/results/contraction_imf1/record_predictions.csv
#   outputs/results/fixed_3min_imf1/record_predictions.csv
#
# Output:
#   outputs/plots/paper/confusion_matrices_26_record_consensus.png
#   outputs/plots/paper/confusion_matrices_26_record_consensus.pdf
#   outputs/plots/paper/confusion_consensus_contraction_imf1_Random_Forest.csv
#   outputs/plots/paper/confusion_consensus_fixed_3min_imf1_Random_Forest.csv

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

try:
    from config import OUTPUT_DIR, FIG_DPI
except Exception:
    OUTPUT_DIR = Path("outputs")
    FIG_DPI = 300


def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.3,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "savefig.dpi": FIG_DPI,
        }
    )


def add_panel_label(ax, label):
    ax.text(
        -0.14,
        1.13,
        label,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        va="top",
        ha="left",
    )


def standardize_columns(df):
    rename = {}

    if "true_label" in df.columns and "label" not in df.columns:
        rename["true_label"] = "label"
    if "y_true" in df.columns and "label" not in df.columns:
        rename["y_true"] = "label"

    if "prediction" in df.columns and "pred" not in df.columns:
        rename["prediction"] = "pred"
    if "y_pred" in df.columns and "pred" not in df.columns:
        rename["y_pred"] = "pred"
    if "record_prediction" in df.columns and "pred" not in df.columns:
        rename["record_prediction"] = "pred"

    if "record_score" in df.columns and "score" not in df.columns:
        rename["record_score"] = "score"
    if "preterm_probability" in df.columns and "score" not in df.columns:
        rename["preterm_probability"] = "score"
    if "probability" in df.columns and "score" not in df.columns:
        rename["probability"] = "score"
    if "pred_score" in df.columns and "score" not in df.columns:
        rename["pred_score"] = "score"

    df = df.rename(columns=rename)

    required = {"record", "model", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Prediction file is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["record"] = df["record"].astype(str)
    df["model"] = df["model"].astype(str)
    df["label"] = df["label"].astype(int)

    if "pred" in df.columns:
        df["pred"] = df["pred"].astype(int)

    if "score" in df.columns:
        df["score"] = df["score"].astype(float)

    return df


def find_prediction_file(experiment):
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
        f"Could not find prediction file for experiment '{experiment}' in {result_dir}.\n"
        f"Expected one of: record_predictions.csv, record_level_predictions.csv, predictions.csv"
    )


def make_consensus_predictions(df, model):
    """
    Converts repeated out-of-fold predictions to one final prediction per recording.

    Preferred:
      use majority vote over available binary predictions.

    Fallback:
      if only score is available, average score per record and threshold at 0.5.
      This fallback is less ideal, so the script prints a warning.
    """

    g = df[df["model"] == model].copy()

    if g.empty:
        raise ValueError(f"No predictions found for model: {model}")

    if "pred" in g.columns:
        consensus = g.groupby("record", as_index=False).agg(
            label=("label", "first"),
            pred_mean=("pred", "mean"),
            n_predictions=("pred", "size"),
        )
        consensus["pred"] = (consensus["pred_mean"] >= 0.5).astype(int)

    elif "score" in g.columns:
        print(
            f"WARNING: No binary prediction column found for {model}. "
            "Using average score >= 0.5 as fallback. "
            "For exact paper results, use a record_predictions.csv file with a pred/y_pred column."
        )
        consensus = g.groupby("record", as_index=False).agg(
            label=("label", "first"),
            score_mean=("score", "mean"),
            n_predictions=("score", "size"),
        )
        consensus["pred"] = (consensus["score_mean"] >= 0.5).astype(int)

    else:
        raise ValueError(
            "Prediction file must contain either a binary prediction column "
            "('pred', 'prediction', or 'y_pred') or a score column."
        )

    return consensus


def plot_single_confusion(ax, cm, title, cmap):
    labels = ["Term", "Preterm"]

    ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(1, cm.max()))

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, pad=18)

    row_sums = cm.sum(axis=1, keepdims=True)
    proportions = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({proportions[i, j]:.2f})",
                ha="center",
                va="center",
                color=color,
                fontsize=14,
            )

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)


def main():
    set_paper_style()

    out_dir = Path(OUTPUT_DIR) / "plots" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        {
            "panel": "A",
            "experiment": "contraction_imf1",
            "model": "Random Forest",
            "title": "Contraction IMF1\nRandom Forest",
            "cmap": "Oranges",
        },
        {
            "panel": "B",
            "experiment": "fixed_3min_imf1",
            "model": "Random Forest",
            "title": "Fixed 3-min IMF1\nRandom Forest",
            "cmap": "Blues",
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))

    for ax, panel in zip(axes, panels):
        pred_path = find_prediction_file(panel["experiment"])
        df = pd.read_csv(pred_path)
        df = standardize_columns(df)

        consensus = make_consensus_predictions(df, panel["model"])

        cm = confusion_matrix(
            consensus["label"].to_numpy(),
            consensus["pred"].to_numpy(),
            labels=[0, 1],
        )

        csv_name = (
            f"confusion_consensus_{panel['experiment']}_"
            f"{panel['model'].replace(' ', '_')}.csv"
        )
        consensus.to_csv(out_dir / csv_name, index=False)

        print("\n", panel["experiment"], panel["model"])
        print("Prediction file:", pred_path)
        print("Number of records:", len(consensus))
        print(cm)

        plot_single_confusion(
            ax=ax,
            cm=cm,
            title=panel["title"],
            cmap=panel["cmap"],
        )
        add_panel_label(ax, panel["panel"])

    fig.tight_layout(w_pad=3.0)

    png_path = out_dir / "confusion_matrices_26_record_consensus.png"
    pdf_path = out_dir / "confusion_matrices_26_record_consensus.pdf"

    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
