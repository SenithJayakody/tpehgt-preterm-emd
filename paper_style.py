from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Colors used by the figures already present in the manuscript.
# -----------------------------------------------------------------------------
TERM_COLOR = "#1f77b4"
PRETERM_COLOR = "#d62728"

SIGNAL_COLOR = "#4d4d4d"
RESIDUAL_COLOR = "#666666"
IMF_COLORS = {
    1: "#1f4e79",
    2: "#16836f",
    3: "#c07a0a",
    4: "#9d2d25",
}
ACF_IMF_COLOR = "#d62728"

CONTRACTION_SHADE = "#e6550d"
FIXED_BOUNDARY_COLOR = "#2c7fb8"

MODEL_COLORS = {
    "RF": "#16836f",
    "CB": "#1f4e79",
    "GB": "#c57f0a",
    "MLP": "#7b3294",
    "SVM": "#a12a22",
    "LR": "#4d4d4d",
    "QDA": "#6baed6",
    "NB": "#d95f02",
    "DT": "#7570b3",
}

IMPORTANCE_COLORS = {
    "Contraction IMF1": "#1F4E79",
    "Fixed 3-min IMF1": "#16836F",
}

FEATURE_TERM_COLOR = "#4C78A8"
FEATURE_PRETERM_COLOR = "#D64F4F"


MODEL_NAME_ALIASES = {
    "Random Forest": "RF",
    "CatBoost": "CB",
    "Gradient Boosting": "GB",
    "Logistic Regression": "LR",
    "Decision Tree": "DT",
    "Naive Bayes": "NB",
    "QDA": "QDA",
    "SVM": "SVM",
    "MLP": "MLP",
    "RF": "RF",
    "CB": "CB",
    "GB": "GB",
    "LR": "LR",
    "DT": "DT",
    "NB": "NB",
}

MODEL_DISPLAY_NAMES = {
    "RF": "Random Forest",
    "CB": "CatBoost",
    "GB": "Gradient Boosting",
    "MLP": "MLP",
    "SVM": "SVM",
    "LR": "Logistic Regression",
    "QDA": "QDA",
    "NB": "Naive Bayes",
    "DT": "Decision Tree",
}


def normalize_model_name(name: str) -> str:
    return MODEL_NAME_ALIASES.get(str(name), str(name))


def set_curve_style(fig_dpi: int = 300) -> None:
    """Style used by the current ROC/precision-recall and mean-PSD figures."""
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
            "savefig.dpi": fig_dpi,
        }
    )


def set_signal_grid_style(fig_dpi: int = 300) -> None:
    """Style used by the current IMF/time/PSD/ACF figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "savefig.dpi": fig_dpi,
        }
    )


def set_feature_grid_style(fig_dpi: int = 300) -> None:
    """Style used by the current 14-panel feature-distribution figure."""
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
            "savefig.dpi": fig_dpi,
        }
    )


def set_importance_style(fig_dpi: int = 300) -> None:
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
            "savefig.dpi": fig_dpi,
        }
    )


def panel_label(ax, label: str, x: float = -0.16, y: float = 1.08, fontsize: int = 20) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_png_pdf(fig, out_base: Path, fig_dpi: int = 300) -> None:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=fig_dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
