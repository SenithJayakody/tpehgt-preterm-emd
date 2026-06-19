"""
plot_imf_psd_acf.py

Generate paper-ready IMF/time-domain, PSD, and autocorrelation plots.

This script is separate from the training code. It reads the TPEHGT records
through io_readers.py and uses the signal channels configured in config.py.
If config.py uses EHG_CHANNELS = [1, 3, 5], the plotted signal is the
TPEHGT dataset-provided filtered EHG signal.

Examples
--------
Generate candidate plots for all records, all channels, fixed 3-min segments:
    python plot_imf_psd_acf.py --mode fixed_3min --channel all --max_segments none

Generate only two segments per record for quick checking:
    python plot_imf_psd_acf.py --mode fixed_3min --channel ehg2 --max_segments 2

Generate one selected record/channel:
    python plot_imf_psd_acf.py --mode fixed_3min --record tpehgt_p002 --channel ehg2 --max_segments 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import welch

from config import DATASET_DIR, FS, MAX_IMFS, OUTPUT_DIR, SIGNAL_VERSION
from io_readers import load_pregnancy_records, fixed_intervals, contraction_intervals
from features import compute_imfs

# ---------------------------------------------------------------------
# Paper style
# ---------------------------------------------------------------------
COLORS = {
    "signal": "#4d4d4d",
    "imf1": "#1f4e79",
    "imf2": "#16836f",
    "imf3": "#c07a0a",
    "imf4": "#9d2d25",
    "residual": "#666666",
    "acf": "#d62728",
}

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
        "savefig.dpi": 300,
    }
)


def panel_label(ax, label: str, x: float = -0.18, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def safe_filename(text: str) -> str:
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
        .replace("--", "-")
    )


# ---------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------
def normalized_acf(
    x: np.ndarray, fs: float, max_lag_sec: float
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    x = x - np.nanmean(x)

    if len(x) < 3 or np.allclose(x, 0):
        lags = np.arange(1)
        return lags / fs, np.ones(1)

    acf = np.correlate(x, x, mode="full")[len(x) - 1 :]
    acf = acf / (acf[0] + 1e-12)

    max_lag = min(int(round(max_lag_sec * fs)), len(acf) - 1)
    lags = np.arange(max_lag + 1) / fs
    return lags, acf[: max_lag + 1]


def compute_psd(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    nperseg = min(1024, len(x))
    if nperseg < 8:
        return np.array([0.0]), np.array([np.nan])
    noverlap = nperseg // 2
    f, pxx = welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    return f, pxx + 1e-18


def get_intervals(record_name: str, mode: str):
    if mode == "fixed_3min":
        return fixed_intervals(record_name, DATASET_DIR)
    if mode == "contraction":
        return contraction_intervals(record_name, DATASET_DIR)
    raise ValueError("mode must be 'fixed_3min' or 'contraction'")


def iter_selected_channels(ehg: dict, channel: str) -> Iterable[Tuple[str, np.ndarray]]:
    if channel == "all":
        yield from ehg.items()
    else:
        if channel not in ehg:
            raise ValueError(f"Channel {channel!r} not found. Available: {list(ehg)}")
        yield channel, ehg[channel]


def signal_display_name() -> str:
    if str(SIGNAL_VERSION).lower().startswith("raw"):
        return "Original EHG"
    return "Filtered EHG"


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_imf_psd_acf_grid(
    segment: np.ndarray,
    fs: float,
    title: str,
    out_path: Path,
    psd_max_freq: float = 5.0,
    acf_max_lag_sec: float = 20.0,
) -> None:
    segment = np.asarray(segment, dtype=float).ravel()
    t = np.arange(len(segment)) / fs

    imfs = compute_imfs(segment, max_imfs=MAX_IMFS)
    residual = segment - np.sum(imfs[:MAX_IMFS], axis=0)

    rows: List[Tuple[str, np.ndarray, str]] = [
        (signal_display_name(), segment, COLORS["signal"])
    ]
    for i in range(MAX_IMFS):
        rows.append((f"IMF{i + 1}", imfs[i], COLORS[f"imf{i + 1}"]))
    rows.append(("Residual", residual, COLORS["residual"]))

    nrows = len(rows)
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(7.5, 9.0), sharex="col")

    column_titles = ["Time domain", "Power spectral density", "Autocorrelation (ACF)"]
    column_labels = ["A", "B", "C"]

    for j, ax in enumerate(axes[0]):
        ax.set_title(column_titles[j], pad=8)
        panel_label(ax, column_labels[j], x=-0.22, y=1.16)

    for i, (name, sig, color) in enumerate(rows):
        # Time domain
        ax = axes[i, 0]
        ax.plot(t, sig, color=color, lw=0.9)
        ax.set_ylabel(f"{name}\n(mV)")
        if i == nrows - 1:
            ax.set_xlabel("Time (s)")
        ax.xaxis.set_major_locator(MaxNLocator(4))

        # PSD
        ax = axes[i, 1]
        f, pxx = compute_psd(sig, fs)
        mask = f <= psd_max_freq
        ax.semilogy(f[mask], pxx[mask], color=color, lw=0.9)
        if i == nrows - 1:
            ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, psd_max_freq)
        ax.xaxis.set_major_locator(MaxNLocator(4))

        # ACF
        ax = axes[i, 2]
        lags, acf = normalized_acf(sig, fs, acf_max_lag_sec)
        acf_color = COLORS["acf"] if name.startswith("IMF") else color
        ax.plot(lags, acf, color=acf_color, lw=0.9)
        ax.axhline(0, color="0.65", lw=0.7)
        ax.set_ylim(-0.55, 1.05)
        if i == nrows - 1:
            ax.set_xlabel("Lag (s)")
        ax.xaxis.set_major_locator(MaxNLocator(4))

    fig.suptitle(title, y=0.995, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["fixed_3min", "contraction"], default="fixed_3min"
    )
    parser.add_argument("--record", default="all", help="Record name or 'all'.")
    parser.add_argument("--channel", default="all", help="ehg1, ehg2, ehg3, or all.")
    parser.add_argument(
        "--max_segments",
        default="none",
        help="Number per record/mode/channel, or 'none'.",
    )
    parser.add_argument("--psd_max_freq", type=float, default=5.0)
    parser.add_argument("--acf_max_lag_sec", type=float, default=20.0)
    parser.add_argument(
        "--out_dir", default=str(OUTPUT_DIR / "signal_plots" / "imf_psd_acf")
    )
    args = parser.parse_args()

    max_segments = (
        None if str(args.max_segments).lower() == "none" else int(args.max_segments)
    )
    out_dir = Path(args.out_dir)

    records = load_pregnancy_records(DATASET_DIR)
    if args.record != "all":
        records = [r for r in records if r["record"] == args.record]

    for rec in records:
        record = rec["record"]
        label_name = "preterm" if int(rec["label"]) == 1 else "term"
        intervals = get_intervals(record, args.mode)
        if max_segments is not None:
            intervals = intervals[:max_segments]

        for ch_name, sig in iter_selected_channels(rec["ehg"], args.channel):
            for seg_id, (start, end) in enumerate(intervals):
                if end <= start:
                    continue
                segment = sig[start:end]
                start_sec = start / rec["fs"]
                end_sec = end / rec["fs"]
                title = (
                    f"{label_name}: {record}, {ch_name}, {args.mode} "
                    f"segment {seg_id} ({start_sec:.0f}--{end_sec:.0f} s)"
                )
                fname = safe_filename(
                    f"{label_name}_{record}_{ch_name}_{args.mode}_seg{seg_id:03d}"
                )
                plot_imf_psd_acf_grid(
                    segment=segment,
                    fs=rec["fs"],
                    title=title,
                    out_path=out_dir / args.mode / ch_name / fname,
                    psd_max_freq=args.psd_max_freq,
                    acf_max_lag_sec=args.acf_max_lag_sec,
                )

    print(f"Saved IMF/PSD/ACF plots to: {out_dir}")


if __name__ == "__main__":
    main()
