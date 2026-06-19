from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from tqdm import tqdm

from config import (
    DATASET_DIR,
    SIGNAL_PLOT_DIR,
    MAX_SIGNAL_SEGMENTS_PER_RECORD_MODE,
    FIXED_WINDOW_SEC,
    BT_FIXED_SEC,
    BT_CONTRACTION_SEC,
    FIG_DPI,
)
from features import compute_imfs, detect_peaks, FeatureConfig, get_imf
from io_readers import (
    contraction_intervals,
    fixed_intervals,
    load_pregnancy_records,
    label_name,
)
from paper_style import (
    COLORS,
    PLOS_DOUBLE,
    PLOS_ONE_HALF,
    panel_label,
    save_figure,
    set_paper_style,
)

set_paper_style()


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace(".", "p")
    )


def autocorrelation(x: np.ndarray, fs: float, max_lag_sec: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    x = x - np.mean(x)

    denom = np.dot(x, x)

    if denom <= 0:
        lags = np.arange(int(max_lag_sec * fs) + 1) / fs
        return lags, np.zeros_like(lags)

    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2:] / denom

    max_lag = min(int(max_lag_sec * fs), len(corr) - 1)
    lags = np.arange(max_lag + 1) / fs

    return lags, corr[:max_lag + 1]


def psd_curve(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = min(1024, len(x))

    if nperseg < 16:
        nperseg = len(x)

    noverlap = nperseg // 2 if nperseg > 1 else 0

    f, p = welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
    )

    mask = f <= 3.0

    return f[mask], p[mask]


def plot_imf_acf_psd_grid(
    filtered_segment: np.ndarray,
    fs: float,
    title: str,
    out_path: Path,
) -> None:
    """
    Paper-style figure:
    rows = filtered EHG, IMF1--IMF4, residual
    columns = time domain, PSD, ACF
    """
    imfs, residual = compute_imfs(filtered_segment, max_imfs=4)

    rows = [("Filtered EHG", filtered_segment, COLORS["filtered"])]
    rows += [(f"IMF{i+1}", imfs[i], COLORS[f"imf{i+1}"]) for i in range(4)]
    rows += [("Residual", residual, COLORS["residual"])]

    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(PLOS_DOUBLE, 8.4),
        gridspec_kw={"hspace": 0.08, "wspace": 0.35},
    )

    for r, (row_name, sig, color) in enumerate(rows):
        sig = np.asarray(sig, dtype=float).ravel()
        t = np.arange(len(sig)) / fs

        ax = axes[r, 0]
        ax.plot(t, sig, color=color, lw=0.7)
        ax.set_ylabel(f"{row_name}\n(mV)")
        if r == 0:
            ax.set_title("Time domain")
        if r == len(rows) - 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticklabels([])

        ax = axes[r, 1]
        f, p = psd_curve(sig, fs)
        ax.semilogy(f, p + 1e-20, color=color, lw=0.8)
        if r == 0:
            ax.set_title("Power spectral density")
        if r == len(rows) - 1:
            ax.set_xlabel("Frequency (Hz)")
        else:
            ax.set_xticklabels([])

        ax = axes[r, 2]
        lag, acf = autocorrelation(sig, fs, max_lag_sec=20)
        acf_color = "#d62728" if row_name.startswith("IMF") else color
        ax.plot(lag, acf, color=acf_color, lw=0.9)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_ylim(-0.6, 1.05)
        if r == 0:
            ax.set_title("Autocorrelation (ACF)")
        if r == len(rows) - 1:
            ax.set_xlabel("Lag (s)")
        else:
            ax.set_xticklabels([])

    axes[0, 0].text(
        0.0,
        1.18,
        title,
        transform=axes[0, 0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    fig.tight_layout()
    save_figure(fig, out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_single_segment_peak_panel(
    filtered_segment: np.ndarray,
    fs: float,
    label: int,
    record: str,
    mode: str,
    segment_id: int,
    channel_name: str,
    out_path: Path,
    burst_threshold_sec: float,
) -> None:
    imf1 = get_imf(filtered_segment, imf_number=1, max_imfs=4)
    cfg = FeatureConfig(fs=fs, burst_tau_sec=burst_threshold_sec)
    peaks, _, peak_signal = detect_peaks(imf1, cfg)

    t = np.arange(len(filtered_segment)) / fs
    color = COLORS["preterm"] if label == 1 else COLORS["term"]
    class_name = label_name(label)

    fig, axes = plt.subplots(1, 3, figsize=(PLOS_DOUBLE, 2.45), sharex=True)

    axes[0].plot(t, filtered_segment, color=color, lw=0.7)
    axes[0].set_title("Filtered EHG")
    axes[0].set_ylabel("EHG (mV)")
    axes[0].set_xlabel("Time (s)")

    axes[1].plot(t, imf1, color=color, lw=0.7)
    axes[1].set_title("IMF1")
    axes[1].set_ylabel("IMF1 (mV)")
    axes[1].set_xlabel("Time (s)")

    env = np.abs(imf1)
    axes[2].plot(t, env, color=color, lw=0.6, alpha=0.50)
    if len(peaks) > 0:
        axes[2].plot(t[peaks], env[peaks], "o", color="black", ms=2.2, mew=0)
    axes[2].set_title(f"Detected peaks (n={len(peaks)})")
    axes[2].set_ylabel("|IMF1| (mV)")
    axes[2].set_xlabel("Time (s)")

    for ax, letter in zip(axes, "ABC"):
        panel_label(ax, letter, x=-0.18, y=1.06)

    fig.suptitle(
        f"{class_name}: {record} | {mode} seg {segment_id} | {channel_name}",
        y=1.06,
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_segmentation_overview(
    signal: np.ndarray,
    imf1: np.ndarray,
    fs: float,
    contractions: List[Tuple[int, int]],
    record: str,
    channel_name: str,
    out_path: Path,
) -> None:
    t = np.arange(len(signal)) / fs
    duration = t[-1]

    fig, axes = plt.subplots(2, 1, figsize=(PLOS_DOUBLE, 3.5), sharex=True)

    for ax, sig, ylabel, letter in [
        (axes[0], signal, "Filtered EHG (mV)", "A"),
        (axes[1], imf1, "IMF1 (mV)", "B"),
    ]:
        ax.plot(t, sig, color=COLORS["neutral"], lw=0.55)

        for s, e in contractions:
            ax.axvspan(s / fs, e / fs, color=COLORS["contraction"], alpha=0.20, lw=0)

        for x in np.arange(FIXED_WINDOW_SEC, duration, FIXED_WINDOW_SEC):
            ax.axvline(x, color=COLORS["fixed"], lw=0.8, ls="--", alpha=0.7)

        ax.set_ylabel(ylabel)
        panel_label(ax, letter, x=-0.16, y=1.03)

    axes[1].set_xlabel("Time (s)")
    axes[0].set_title(f"Segmentation overview: {record}, {channel_name}")

    # Legend.
    axes[0].plot([], [], color=COLORS["neutral"], lw=0.8, label="Signal")
    axes[0].fill_between([], [], [], color=COLORS["contraction"], alpha=0.20,
                         label="Annotated contraction")
    axes[0].plot([], [], color=COLORS["fixed"], lw=0.8, ls="--",
                 label="Fixed 3-minute boundary")
    axes[0].legend(frameon=False, ncol=3, loc="upper right")

    fig.tight_layout()
    save_figure(fig, out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_class_psd(records: List[Dict], out_path: Path) -> None:
    """
    Average PSD of IMF1 by class using full recordings and all selected EHG channels.
    No SD bands are shown.
    """
    psds = {0: [], 1: []}
    f_ref = None

    for rec in tqdm(records, desc="Class PSD"):
        label = rec["label"]
        fs = rec["fs"]

        for signal in rec["ehg"].values():
            imf1 = get_imf(signal, imf_number=1, max_imfs=4)
            f, p = psd_curve(imf1, fs)

            if f_ref is None:
                f_ref = f

            psds[label].append(np.interp(f_ref, f, p))

    if f_ref is None or not psds[0] or not psds[1]:
        return

    term_mean = np.mean(psds[0], axis=0)
    preterm_mean = np.mean(psds[1], axis=0)

    fig, ax = plt.subplots(figsize=(PLOS_ONE_HALF, 3.0))

    ax.semilogy(f_ref, term_mean + 1e-20, color=COLORS["term"], lw=1.3, label="Term")
    ax.semilogy(f_ref, preterm_mean + 1e-20, color=COLORS["preterm"], lw=1.3, label="Preterm")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Average IMF1 power spectral density")
    ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, out_path, dpi=FIG_DPI)
    plt.close(fig)


def iter_segments_for_record(rec: Dict, mode: str) -> List[Tuple[int, int]]:
    if mode == "fixed_3min":
        return fixed_intervals(rec["record"], DATASET_DIR)
    if mode == "contraction":
        return contraction_intervals(rec["record"], DATASET_DIR)
    raise ValueError(mode)


def main() -> None:
    SIGNAL_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_pregnancy_records(DATASET_DIR)

    if not records:
        print("No term/preterm records found. Check DATASET_DIR in config.py.")
        return

    # Class-level PSD.
    plot_class_psd(records, SIGNAL_PLOT_DIR / "paper" / "average_imf1_psd_by_class")

    for rec in tqdm(records, desc="Signal plots"):
        record = rec["record"]
        label = rec["label"]
        fs = rec["fs"]

        contractions = contraction_intervals(record, DATASET_DIR)

        for channel_name, signal in rec["ehg"].items():
            # Full-record segmentation overview.
            full_imf1 = get_imf(signal, imf_number=1, max_imfs=4)
            plot_segmentation_overview(
                signal=signal,
                imf1=full_imf1,
                fs=fs,
                contractions=contractions,
                record=record,
                channel_name=channel_name,
                out_path=SIGNAL_PLOT_DIR / "segmentation_overview" / f"{record}_{channel_name}",
            )

            for mode, bt in [
                ("fixed_3min", BT_FIXED_SEC),
                ("contraction", BT_CONTRACTION_SEC),
            ]:
                intervals = iter_segments_for_record(rec, mode)

                if MAX_SIGNAL_SEGMENTS_PER_RECORD_MODE is not None:
                    intervals = intervals[:MAX_SIGNAL_SEGMENTS_PER_RECORD_MODE]

                for segment_id, (start, end) in enumerate(intervals):
                    if end <= start:
                        continue

                    segment = signal[start:end]
                    start_sec = start / fs
                    end_sec = end / fs
                    class_name = label_name(label)

                    title = (
                        f"{class_name}: {record}, {channel_name}, "
                        f"{mode} segment {segment_id} ({start_sec:.0f}-{end_sec:.0f} s)"
                    )

                    base = f"{class_name}_{record}_{channel_name}_{mode}_seg{segment_id:02d}"

                    plot_imf_acf_psd_grid(
                        filtered_segment=segment,
                        fs=fs,
                        title=title,
                        out_path=SIGNAL_PLOT_DIR / "imf_acf_psd" / mode / base,
                    )

                    plot_single_segment_peak_panel(
                        filtered_segment=segment,
                        fs=fs,
                        label=label,
                        record=record,
                        mode=mode,
                        segment_id=segment_id,
                        channel_name=channel_name,
                        out_path=SIGNAL_PLOT_DIR / "peak_detection" / mode / base,
                        burst_threshold_sec=bt,
                    )

    print(f"Saved signal plots to: {SIGNAL_PLOT_DIR}")


if __name__ == "__main__":
    main()
