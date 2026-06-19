"""
Shared matplotlib style for PLOS ONE-ready figures.
"""

import matplotlib as mpl

PLOS_SINGLE = 5.2
PLOS_ONE_HALF = 6.83
PLOS_DOUBLE = 7.5

COLORS = {
    "term": "#1f77b4",
    "preterm": "#d62728",
    "fixed": "#2c7fb8",
    "contraction": "#e6550d",
    "imf1": "#1b4f72",
    "imf2": "#117a65",
    "imf3": "#b9770e",
    "imf4": "#922b21",
    "raw": "#555555",
    "filtered": "#555555",
    "residual": "#555555",
    "band": "#cccccc",
    "neutral": "#555555",
    "catboost": "#1b4f72",
    "rf": "#117a65",
    "gb": "#b9770e",
    "mlp": "#6c3483",
    "svm": "#922b21",
}


def set_paper_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.grid": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
    })


def panel_label(ax, text, x=-0.14, y=1.04, weight="bold", size=11):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        va="bottom",
        ha="left",
    )


def save_figure(fig, out_base, dpi=300):
    """
    Save a figure as PNG and PDF using the same basename.

    Parameters
    ----------
    fig : matplotlib Figure
    out_base : str or Path without suffix, or with suffix
    """
    from pathlib import Path

    out_base = Path(out_base)
    if out_base.suffix:
        out_base = out_base.with_suffix("")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_base.with_suffix(".pdf"), dpi=dpi)
