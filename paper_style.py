from __future__ import annotations

from pathlib import Path
from io import BytesIO

import matplotlib.pyplot as plt
from PIL import Image

PLOS_FIGURE_NAMES = {
    "segmentation_comparison": "Fig2.tif",
    "imf_acf_psd": "Fig3.tif",
    "psd_comparison": "Fig4.tif",
    "peak_burst_detection": "Fig5.tif",
    "roc_pr_curves": "Fig6.tif",
    "confusion_matrices": "Fig7.tif",
    "feature_distributions": "Fig8.tif",
    "grouped_feature_importance": "Fig9.tif",
}
PLOS_TIFF_DIR = Path("outputs/plos_figures")
PLOS_DPI = 300
PLOS_MIN_WIDTH_PX = 789
PLOS_MAX_WIDTH_PX = 2250
PLOS_MAX_HEIGHT_PX = 2625
PLOS_MAX_FILE_SIZE_MB = 10.0

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
DUMMY_SHADE = "#8073ac"
BURST_SHADE = "#e6550d"
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
    "Annotated intervals IMF1": "#1F4E79",
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
    manuscript_name = PLOS_FIGURE_NAMES.get(out_base.name)
    if manuscript_name is not None:
        save_plos_tiff(fig, PLOS_TIFF_DIR / manuscript_name)


def save_plos_tiff(fig, path: Path) -> dict[str, object]:
    """Save a flattened, LZW-compressed PLOS TIFF without changing the plot."""
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=PLOS_DPI,
        bbox_inches="tight",
        pad_inches=2 / 72,
        facecolor="white",
        transparent=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        image = rendered.convert("RGB")
        width, height = image.size
        scale = min(1.0, PLOS_MAX_WIDTH_PX / width, PLOS_MAX_HEIGHT_PX / height)
        if scale < 1.0:
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
        if image.width < PLOS_MIN_WIDTH_PX:
            raise RuntimeError(
                f"{path.name}: rendered width {image.width}px is below the "
                f"{PLOS_MIN_WIDTH_PX}px minimum; refusing to upscale"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(
            path,
            format="TIFF",
            compression="tiff_lzw",
            dpi=(PLOS_DPI, PLOS_DPI),
        )
    result = validate_plos_tiff(path)
    print("PLOS TIFF: " + " | ".join(f"{key}={value}" for key, value in result.items()))
    return result


def validate_plos_tiff(path: Path) -> dict[str, object]:
    """Validate one TIFF against the repository's PLOS technical constraints."""
    path = Path(path)
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (0, 0))
        dpi_x, dpi_y = (float(dpi[0]), float(dpi[1])) if len(dpi) == 2 else (0.0, 0.0)
        compression = str(image.info.get("compression", "")).lower()
        has_alpha = "A" in image.getbands()
        size_mb = path.stat().st_size / (1024 * 1024)
        failures = []
        if not PLOS_MIN_WIDTH_PX <= image.width <= PLOS_MAX_WIDTH_PX:
            failures.append("width")
        if image.height > PLOS_MAX_HEIGHT_PX:
            failures.append("height")
        if not (300 <= dpi_x <= 600 and 300 <= dpi_y <= 600):
            failures.append("dpi")
        if image.mode not in {"RGB", "L"}:
            failures.append("color_mode")
        if has_alpha:
            failures.append("alpha")
        if compression not in {"tiff_lzw", "lzw"}:
            failures.append("compression")
        if size_mb >= PLOS_MAX_FILE_SIZE_MB:
            failures.append("file_size")
        result = {
            "filename": path.name,
            "width_px": image.width,
            "height_px": image.height,
            "dpi": f"{dpi_x:g}x{dpi_y:g}",
            "color_mode": image.mode,
            "has_alpha": has_alpha,
            "compression": compression,
            "file_size_MB": round(size_mb, 3),
            "status": "PASS" if not failures else "FAIL: " + ", ".join(failures),
        }
    if failures:
        raise RuntimeError(f"Non-compliant PLOS TIFF {path}: {', '.join(failures)}")
    return result
