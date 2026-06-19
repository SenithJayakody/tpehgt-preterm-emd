# plot_best_model_comparison.py
# Generates the best-configuration comparison bar chart for the PLOS ONE paper.
#
# Run:
#   python plot_best_model_comparison.py
#
# Output:
#   outputs/plots/paper/best_model_comparison.png
#   outputs/plots/paper/best_model_comparison.pdf

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.3,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "savefig.dpi": FIG_DPI,
        }
    )


def main():
    set_paper_style()

    out_dir = Path(OUTPUT_DIR) / "plots" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Latest final results, mean values only.
    # Best model in each row selected based on balanced accuracy.
    configurations = [
        {
            "label": "Contraction Time-domain\n(CatBoost)",
            "color": "#E66101",
            "accuracy": 0.7065,
            "f1": 0.5722,
            "balanced_accuracy": 0.7116,
            "mcc": 0.3859,
            "roc_auc": 0.7712,
            "pr_auc": 0.8478,
        },
        {
            "label": "Contraction IMF1\n(Random Forest)",
            "color": "#1F4E79",
            "accuracy": 0.7522,
            "f1": 0.6674,
            "balanced_accuracy": 0.7564,
            "mcc": 0.4594,
            "roc_auc": 0.7848,
            "pr_auc": 0.8659,
        },
        {
            "label": "Fixed 3-min Time-domain\n(Random Forest)",
            "color": "#3288BD",
            "accuracy": 0.7969,
            "f1": 0.7221,
            "balanced_accuracy": 0.8072,
            "mcc": 0.6435,
            "roc_auc": 0.8415,
            "pr_auc": 0.9154,
        },
        {
            "label": "Fixed 3-min IMF1\n(Random Forest)",
            "color": "#16836F",
            "accuracy": 0.8520,
            "f1": 0.8160,
            "balanced_accuracy": 0.8633,
            "mcc": 0.7479,
            "roc_auc": 0.8468,
            "pr_auc": 0.9204,
        },
    ]

    metrics = [
        ("accuracy", "Accuracy"),
        ("f1", "F1"),
        ("balanced_accuracy", "Balanced Acc."),
        ("mcc", "MCC"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
    ]

    x = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12.5, 5.4))

    offsets = np.linspace(
        -width * (len(configurations) - 1) / 2,
        width * (len(configurations) - 1) / 2,
        len(configurations),
    )

    for cfg, offset in zip(configurations, offsets):
        values = [cfg[key] for key, _ in metrics]
        ax.bar(
            x + offset,
            values,
            width=width,
            label=cfg["label"],
            color=cfg["color"],
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in metrics])
    ax.set_ylabel("Mean recording-level value")
    ax.set_ylim(0, 1.05)
    # ax.set_title("Best configurations")
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        columnspacing=1.4,
        handlelength=1.8,
    )

    fig.tight_layout()

    png_path = out_dir / "best_model_comparison.png"
    pdf_path = out_dir / "best_model_comparison.pdf"

    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
