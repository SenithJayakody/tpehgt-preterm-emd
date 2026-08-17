"""
Generate the signal-derived figures used in the manuscript.

Outputs under outputs/plots/paper/:
  segmentation_comparison.{png,pdf}          # manuscript Fig. 2
  imf_acf_psd.{png,pdf}                      # manuscript Fig. 3
  psd_comparison.{png,pdf}                    # manuscript Fig. 4
  peak_burst_detection.{png,pdf}             # manuscript Fig. 5

Machine-readable values underlying each figure are saved next to the figures.

The current Fig. 2 style is preserved. Panel B deliberately decomposes the complete
30-minute signal ONLY for that visualization. This is not the representation supplied
to the classifier, where EMD is applied after segmentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.signal import welch

from config import (
    BT_FIXED_SEC,
    DATASET_DIR,
    EHG_CHANNEL_NAMES,
    FIG_DPI,
    FIXED_WINDOW_SEC,
    MAX_IMFS,
    PLOT_DIR,
)
from features import FeatureConfig, compute_imfs, detect_peaks
from io_readers import (
    contraction_intervals,
    dummy_intervals,
    fixed_intervals,
    load_pregnancy_records,
)
from paper_style import (
    ACF_IMF_COLOR,
    BURST_SHADE,
    CONTRACTION_SHADE,
    DUMMY_SHADE,
    FIXED_BOUNDARY_COLOR,
    IMF_COLORS,
    PRETERM_COLOR,
    RESIDUAL_COLOR,
    SIGNAL_COLOR,
    TERM_COLOR,
    panel_label,
    save_png_pdf,
    set_curve_style,
    set_signal_grid_style,
)


def normalized_acf(x: np.ndarray, fs: float, max_lag_sec: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    x = x - np.nanmean(x)
    if x.size < 3 or not np.all(np.isfinite(x)) or np.allclose(x, 0):
        return np.array([0.0]), np.array([1.0])
    acf = np.correlate(x, x, mode="full")[x.size - 1 :]
    acf = acf / (acf[0] + 1e-12)
    max_lag = min(int(round(max_lag_sec * fs)), len(acf) - 1)
    lags = np.arange(max_lag + 1) / fs
    return lags, acf[: max_lag + 1]


def compute_psd(x: np.ndarray, fs: float, nperseg: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    use_nperseg = min(nperseg, len(x))
    if use_nperseg < 8:
        return np.array([0.0]), np.array([np.nan])
    f, pxx = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=use_nperseg,
        noverlap=use_nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    return np.asarray(f, dtype=float), np.asarray(pxx, dtype=float)


def find_record(records: List[dict], record_name: str | None) -> dict:
    if record_name is not None:
        matches = [r for r in records if r["record"] == record_name]
        if not matches:
            raise ValueError(f"Record not found: {record_name}")
        return matches[0]

    # Deterministic fallback: first preterm record. The script prints the choice so
    # it can be replaced by the exact record used in the manuscript if needed.
    preterm = [r for r in records if int(r["label"]) == 1]
    return (preterm or records)[0]


def validate_channel(rec: dict, channel: str) -> np.ndarray:
    if channel not in rec["ehg"]:
        raise ValueError(f"Unknown channel {channel!r}; available: {list(rec['ehg'])}")
    return np.asarray(rec["ehg"][channel], dtype=float)


def plot_segmentation_comparison(rec: dict, channel: str, out_dir: Path) -> None:
    """Reproduce the visual design of current manuscript Fig. 2."""
    set_signal_grid_style(FIG_DPI)
    signal = validate_channel(rec, channel)
    fs = float(rec["fs"])
    t = np.arange(signal.size) / fs
    annotated_contractions = contraction_intervals(rec["record"], DATASET_DIR)
    annotated_dummies = dummy_intervals(rec["record"], DATASET_DIR)

    # Explicitly whole-record EMD for visualization only, matching the current figure.
    imfs, _ = compute_imfs(signal, max_imfs=MAX_IMFS)
    if imfs.shape[0] < 1:
        raise RuntimeError("No IMF1 returned for the selected full recording.")
    imf1 = imfs[0]

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 3.7), sharex=True)

    for ax, y, ylabel, letter in [
        (axes[0], signal, "Filtered EHG (mV)", "A"),
        (axes[1], imf1, "IMF1 (mV)", "B"),
    ]:
        for start, end in annotated_contractions:
            ax.axvspan(
                start / fs,
                end / fs,
                color=CONTRACTION_SHADE,
                alpha=0.18,
                lw=0,
                zorder=0,
            )
        for start, end in annotated_dummies:
            ax.axvspan(
                start / fs,
                end / fs,
                color=DUMMY_SHADE,
                alpha=0.16,
                lw=0,
                zorder=0,
            )
        ax.plot(t, y, color=SIGNAL_COLOR, lw=0.75, zorder=2)
        for boundary in np.arange(FIXED_WINDOW_SEC, t[-1] + 1e-9, FIXED_WINDOW_SEC):
            ax.axvline(
                boundary,
                color=FIXED_BOUNDARY_COLOR,
                ls="--",
                lw=0.8,
                alpha=0.60,
                zorder=1,
            )
        ax.set_ylabel(ylabel)
        panel_label(
            ax,
            letter,
            x=0.01,
            y=0.96,
            fontsize=14,
        )

    axes[1].set_xlabel("Time (s)")
    fig.legend(
        handles=[
            Line2D([0], [0], color=SIGNAL_COLOR, lw=0.9, label="Signal"),
            Patch(facecolor=CONTRACTION_SHADE, alpha=0.18, label="Annotated contraction"),
            Patch(
                facecolor=DUMMY_SHADE,
                alpha=0.16,
                label="Annotated dummy (non-contraction)",
            ),
            Line2D([0], [0], color=FIXED_BOUNDARY_COLOR, ls="--", lw=0.8, label="Fixed 3-minute boundary"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_png_pdf(fig, out_dir / "segmentation_comparison", FIG_DPI)
    plt.close(fig)

    contraction_mask = np.zeros(signal.size, dtype=int)
    for start, end in annotated_contractions:
        contraction_mask[max(0, start) : min(signal.size, end)] = 1
    dummy_mask = np.zeros(signal.size, dtype=int)
    for start, end in annotated_dummies:
        dummy_mask[max(0, start) : min(signal.size, end)] = 1
    pd.DataFrame(
        {
            "time_sec": t,
            "filtered_ehg_mv": signal,
            "whole_record_imf1_visualization_only_mv": imf1,
            "inside_annotated_contraction": contraction_mask,
            "inside_annotated_dummy": dummy_mask,
        }
    ).to_csv(out_dir / "segmentation_comparison_values.csv", index=False)


def plot_imf_acf_psd(rec: dict, channel: str, segment_id: int, out_dir: Path) -> None:
    """Reproduce the current 6-row x 3-column EMD/PSD/ACF figure."""
    set_signal_grid_style(FIG_DPI)
    signal = validate_channel(rec, channel)
    intervals = fixed_intervals(rec["record"], DATASET_DIR)
    if not (0 <= segment_id < len(intervals)):
        raise ValueError(f"segment_id must be 0..{len(intervals)-1}")
    start, end = intervals[segment_id]
    segment = signal[start:end]
    fs = float(rec["fs"])
    imfs, residual = compute_imfs(segment, max_imfs=MAX_IMFS)
    if imfs.shape[0] < 4:
        raise RuntimeError(
            f"Selected representative segment returned only {imfs.shape[0]} genuine IMFs. "
            "Choose another representative segment for the four-IMF manuscript figure."
        )

    rows = [("Filtered EHG", segment, SIGNAL_COLOR)]
    rows += [(f"IMF{i}", imfs[i - 1], IMF_COLORS[i]) for i in range(1, 5)]
    rows += [("Residual", residual, RESIDUAL_COLOR)]

    fig, axes = plt.subplots(6, 3, figsize=(7.5, 9.0), sharex="col")
    for j, title in enumerate(["Time domain", "Power spectral density", "Autocorrelation (ACF)"]):
        axes[0, j].set_title(title, pad=8)

    time_csv = {"time_sec": np.arange(segment.size) / fs}
    psd_rows = []
    acf_rows = []

    for row_idx, (name, y, color) in enumerate(rows):
        t = np.arange(y.size) / fs
        axes[row_idx, 0].plot(t, y, color=color, lw=0.9)
        axes[row_idx, 0].set_ylabel(f"{name}\n(mV)")
        axes[row_idx, 0].xaxis.set_major_locator(MaxNLocator(4))
        time_csv[name.replace(" ", "_").lower()] = y

        f, pxx = compute_psd(y, fs)
        mask = f <= 3.0
        axes[row_idx, 1].semilogy(f[mask], pxx[mask] + 1e-18, color=color, lw=0.9)
        axes[row_idx, 1].set_xlim(0, 3.0)
        axes[row_idx, 1].xaxis.set_major_locator(MaxNLocator(4))
        psd_rows.extend(
            {"component": name, "frequency_hz": float(ff), "psd": float(pp)}
            for ff, pp in zip(f[mask], pxx[mask])
        )

        lag, acf = normalized_acf(y, fs, 20.0)
        acf_color = ACF_IMF_COLOR if name.startswith("IMF") else color
        axes[row_idx, 2].plot(lag, acf, color=acf_color, lw=0.9)
        axes[row_idx, 2].axhline(0, color="0.65", lw=0.7)
        axes[row_idx, 2].set_ylim(-0.55, 1.05)
        axes[row_idx, 2].xaxis.set_major_locator(MaxNLocator(4))
        acf_rows.extend(
            {"component": name, "lag_sec": float(ll), "acf": float(aa)}
            for ll, aa in zip(lag, acf)
        )

    for j, xlabel in enumerate(["Time (s)", "Frequency (Hz)", "Lag (s)"]):
        axes[-1, j].set_xlabel(xlabel)

    fig.tight_layout()
    save_png_pdf(fig, out_dir / "imf_acf_psd", FIG_DPI)
    plt.close(fig)

    pd.DataFrame(time_csv).to_csv(out_dir / "imf_acf_psd_time_values.csv", index=False)
    pd.DataFrame(psd_rows).to_csv(out_dir / "imf_acf_psd_psd_values.csv", index=False)
    pd.DataFrame(acf_rows).to_csv(out_dir / "imf_acf_psd_acf_values.csv", index=False)


def compute_record_imf1_psd(rec: dict, nperseg: int = 1024, max_freq: float = 5.0) -> Tuple[np.ndarray, np.ndarray] | None:
    """Average segment PSD -> channel PSD -> one PSD per recording, exactly as Fig. 4 caption."""
    intervals = fixed_intervals(rec["record"], DATASET_DIR)
    channel_psds = []
    freq_ref = None

    for channel in EHG_CHANNEL_NAMES:
        signal = np.asarray(rec["ehg"][channel], dtype=float)
        segment_psds = []
        for start, end in intervals:
            segment = signal[start:end]
            imfs, _ = compute_imfs(segment, max_imfs=MAX_IMFS)
            if imfs.shape[0] < 1:
                continue
            f, pxx = compute_psd(imfs[0], float(rec["fs"]), nperseg=nperseg)
            mask = f <= max_freq
            f = f[mask]
            pxx = pxx[mask]
            if freq_ref is None:
                freq_ref = f
            if len(f) != len(freq_ref) or not np.allclose(f, freq_ref):
                pxx = np.interp(freq_ref, f, pxx)
            segment_psds.append(pxx)
        if segment_psds:
            channel_psds.append(np.mean(segment_psds, axis=0))

    if freq_ref is None or not channel_psds:
        return None
    return freq_ref, np.mean(channel_psds, axis=0)


def plot_class_mean_psd(records: List[dict], out_dir: Path) -> None:
    set_curve_style(FIG_DPI)
    class_curves = {0: [], 1: []}
    record_rows = []
    freq_ref = None

    for rec in records:
        result = compute_record_imf1_psd(rec)
        if result is None:
            continue
        f, pxx = result
        if freq_ref is None:
            freq_ref = f
        if len(f) != len(freq_ref) or not np.allclose(f, freq_ref):
            pxx = np.interp(freq_ref, f, pxx)
        class_curves[int(rec["label"])].append(pxx)
        for ff, pp in zip(freq_ref, pxx):
            record_rows.append(
                {
                    "record": rec["record"],
                    "label": int(rec["label"]),
                    "frequency_hz": float(ff),
                    "record_mean_psd": float(pp),
                }
            )

    if freq_ref is None or not class_curves[0] or not class_curves[1]:
        raise RuntimeError("Could not generate class-average IMF1 PSDs.")

    term_mean = np.mean(class_curves[0], axis=0)
    preterm_mean = np.mean(class_curves[1], axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(freq_ref, term_mean, linewidth=2.6, color=TERM_COLOR, label="Term")
    ax.plot(freq_ref, preterm_mean, linewidth=2.6, color=PRETERM_COLOR, label="Preterm")
    ax.set_yscale("log")
    ax.set_xlim(0, 5.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_png_pdf(fig, out_dir / "psd_comparison", FIG_DPI)
    plt.close(fig)

    pd.DataFrame(
        {
            "frequency_hz": freq_ref,
            "term_mean_psd": term_mean,
            "preterm_mean_psd": preterm_mean,
        }
    ).to_csv(out_dir / "psd_comparison_class_means.csv", index=False)
    pd.DataFrame(record_rows).to_csv(out_dir / "psd_comparison_record_values.csv", index=False)


def burst_groups_from_peaks(peaks: np.ndarray, fs: float, burst_threshold_sec: float) -> List[np.ndarray]:
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size == 0:
        return []
    if peaks.size == 1:
        return [peaks.copy()]
    ipi = np.diff(peaks) / fs
    breaks = np.where(ipi > burst_threshold_sec)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, peaks.size - 1]
    return [peaks[s : e + 1] for s, e in zip(starts, ends)]


def plot_peak_burst_detection(rec: dict, channel: str, segment_id: int, out_dir: Path) -> None:
    set_signal_grid_style(FIG_DPI)
    signal = validate_channel(rec, channel)
    intervals = fixed_intervals(rec["record"], DATASET_DIR)
    if not (0 <= segment_id < len(intervals)):
        raise ValueError(f"segment_id must be 0..{len(intervals)-1}")
    start, end = intervals[segment_id]
    segment = signal[start:end]
    fs = float(rec["fs"])
    imfs, _ = compute_imfs(segment, max_imfs=MAX_IMFS)
    if imfs.shape[0] < 1:
        raise RuntimeError("No IMF1 returned for selected representative segment.")
    imf1 = imfs[0]

    cfg = FeatureConfig(fs=fs, burst_tau_sec=BT_FIXED_SEC)
    peaks, _, peak_signal = detect_peaks(imf1, cfg)
    threshold = float(np.median(peak_signal) + cfg.thresh_k * (np.median(np.abs(peak_signal - np.median(peak_signal))) + 1e-12))
    bursts = burst_groups_from_peaks(peaks, fs, BT_FIXED_SEC)
    t = np.arange(imf1.size) / fs

    fig, ax = plt.subplots(figsize=(10.0, 3.5))
    ax.plot(t, imf1, color=IMF_COLORS[1], lw=0.9, label="IMF1")
    if cfg.peak_mode == "abs":
        ax.axhline(threshold, color=PRETERM_COLOR, ls="--", lw=0.9, label="Peak threshold")
        ax.axhline(-threshold, color=PRETERM_COLOR, ls="--", lw=0.9)
    else:
        ax.axhline(threshold, color=PRETERM_COLOR, ls="--", lw=0.9, label="Peak threshold")

    if peaks.size:
        ax.scatter(
            t[peaks],
            imf1[peaks],
            s=18,
            marker="x",
            linewidths=0.8,
            color="black",
            zorder=4,
            label="Detected peak",
        )

    for burst_id, burst_peaks in enumerate(bursts, start=1):
        left = max(0.0, t[burst_peaks[0]] - 0.10)
        right = min(t[-1], t[burst_peaks[-1]] + 0.10)
        ax.axvspan(left, right, color=BURST_SHADE, alpha=0.16, lw=0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("IMF1 (mV)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=BURST_SHADE, alpha=0.16, label="Burst interval"))
    labels.append("Burst interval")
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_png_pdf(fig, out_dir / "peak_burst_detection", FIG_DPI)
    plt.close(fig)

    burst_id_by_sample = np.zeros(imf1.size, dtype=int)
    for burst_id, burst_peaks in enumerate(bursts, start=1):
        if burst_peaks.size:
            burst_id_by_sample[burst_peaks[0] : burst_peaks[-1] + 1] = burst_id
    is_peak = np.zeros(imf1.size, dtype=int)
    is_peak[peaks] = 1
    pd.DataFrame(
        {
            "time_sec": t,
            "imf1_mv": imf1,
            "abs_imf1_mv": np.abs(imf1),
            "threshold_abs_mv": threshold,
            "is_detected_peak": is_peak,
            "burst_id_between_first_last_peak": burst_id_by_sample,
        }
    ).to_csv(out_dir / "peak_burst_detection_values.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript signal figures 2-5.")
    parser.add_argument("--record", default=None, help="Representative record. Default: first preterm record.")
    parser.add_argument("--channel", default="ehg2", choices=EHG_CHANNEL_NAMES)
    parser.add_argument("--segment-id", type=int, default=0, help="Fixed 3-min segment used for Figs. 3 and 5.")
    parser.add_argument("--out-dir", type=Path, default=PLOT_DIR / "paper")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = load_pregnancy_records(DATASET_DIR)
    if not records:
        raise RuntimeError("No term/preterm records found. Check DATASET_DIR.")
    rec = find_record(records, args.record)
    print(f"Representative signal figures: record={rec['record']}, channel={args.channel}, fixed segment={args.segment_id}")

    plot_segmentation_comparison(rec, args.channel, args.out_dir)
    plot_imf_acf_psd(rec, args.channel, args.segment_id, args.out_dir)
    plot_class_mean_psd(records, args.out_dir)
    plot_peak_burst_detection(rec, args.channel, args.segment_id, args.out_dir)
    print(f"Saved manuscript signal figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
