# plot_mean_imf1_psd.py
# Generate average IMF1 PSD by class for the paper.
#
# Default:
#   python plot_mean_imf1_psd.py
#
# Optional examples:
#   python plot_mean_imf1_psd.py --mode fixed_3min --channel all --psd_max_freq 5
#   python plot_mean_imf1_psd.py --mode fixed_3min --channel ehg2 --psd_max_freq 5
#   python plot_mean_imf1_psd.py --mode contraction --channel all --psd_max_freq 5

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from config import DATASET_DIR, OUTPUT_DIR, FS, EHG_CHANNEL_NAMES, FIG_DPI
from io_readers import (
    load_pregnancy_records,
    fixed_intervals,
    contraction_intervals,
)
from features import compute_imfs

# Paper-style colors
COLORS = {
    "term": "#1f77b4",
    "preterm": "#d62728",
}

LABEL_NAMES = {
    0: "Term",
    1: "Preterm",
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
            "legend.fontsize": 12,
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


def get_intervals(record_name: str, mode: str):
    if mode == "fixed_3min":
        return fixed_intervals(record_name, DATASET_DIR)

    if mode == "contraction":
        return contraction_intervals(record_name, DATASET_DIR)

    raise ValueError("mode must be 'fixed_3min' or 'contraction'")


def selected_channels(channel: str):
    if channel == "all":
        return EHG_CHANNEL_NAMES

    if channel not in EHG_CHANNEL_NAMES:
        raise ValueError(
            f"Unknown channel '{channel}'. Use one of {EHG_CHANNEL_NAMES} or 'all'."
        )

    return [channel]


def compute_record_psd(
    rec: dict,
    mode: str,
    channels: list[str],
    nperseg: int,
    psd_max_freq: float,
):
    """
    Returns one PSD curve per recording by averaging:
      segment PSDs -> channel PSDs -> one recording PSD.

    This version is robust to compute_imfs() returning either:
      imfs
    or:
      imfs, residual
    """
    intervals = get_intervals(rec["record"], mode)

    if not intervals:
        return None, None

    record_psds = []
    freq_ref = None

    for ch_name in channels:
        sig = rec["ehg"][ch_name]
        channel_psds = []

        for start, end in intervals:
            segment = sig[start:end]

            if len(segment) < 16:
                continue

            imf_result = compute_imfs(segment, max_imfs=4)

            # Handle both possible return styles:
            # 1) imfs
            # 2) (imfs, residual)
            if isinstance(imf_result, tuple):
                imfs = imf_result[0]
            else:
                imfs = imf_result

            imfs = np.asarray(imfs, dtype=float)
            print(rec["record"], ch_name, start, end, imfs.shape)

            if imfs.ndim == 1:
                imf1 = imfs
            elif imfs.ndim == 2:
                imf1 = imfs[0, :]
            else:
                continue

            if len(imf1) < 16:
                continue

            use_nperseg = min(nperseg, len(imf1))

            freqs, psd = welch(
                imf1,
                fs=FS,
                window="hann",
                nperseg=use_nperseg,
                noverlap=use_nperseg // 2,
                detrend="constant",
                scaling="density",
            )

            freqs = np.asarray(freqs, dtype=float)
            psd = np.asarray(psd, dtype=float).squeeze()

            # Make sure PSD is 1D.
            if psd.ndim != 1:
                continue

            mask = freqs <= psd_max_freq

            freqs_sel = freqs[mask]
            psd_sel = psd[mask]

            if len(freqs_sel) == 0:
                continue

            if freq_ref is None:
                freq_ref = freqs_sel

            # Align all PSD curves to the same frequency grid.
            if (
                len(freqs_sel) != len(freq_ref)
                or np.max(np.abs(freqs_sel - freq_ref)) > 1e-12
            ):
                psd_sel = np.interp(freq_ref, freqs_sel, psd_sel)

            channel_psds.append(psd_sel)

        if channel_psds:
            record_psds.append(np.mean(channel_psds, axis=0))

    if not record_psds or freq_ref is None:
        return None, None

    return freq_ref, np.mean(record_psds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="fixed_3min", choices=["fixed_3min", "contraction"]
    )
    parser.add_argument("--channel", default="all", help="all, ehg1, ehg2, or ehg3")
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--psd_max_freq", type=float, default=5.0)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    set_paper_style()

    out_dir = (
        Path(args.out_dir) if args.out_dir else Path(OUTPUT_DIR) / "plots" / "paper"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = selected_channels(args.channel)
    records = load_pregnancy_records(DATASET_DIR)

    class_psds = {0: [], 1: []}
    freq_ref = None

    for rec in records:
        freqs, psd = compute_record_psd(
            rec=rec,
            mode=args.mode,
            channels=channels,
            nperseg=args.nperseg,
            psd_max_freq=args.psd_max_freq,
        )

        if psd is None:
            continue

        if freq_ref is None:
            freq_ref = freqs

        class_psds[int(rec["label"])].append(psd)

    if freq_ref is None:
        raise RuntimeError(
            "No PSD curves were generated. Check dataset path and segmentation mode."
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    for label in [0, 1]:
        if not class_psds[label]:
            continue

        mean_psd = np.mean(class_psds[label], axis=0)

        color = COLORS["term"] if label == 0 else COLORS["preterm"]

        ax.plot(
            freq_ref,
            mean_psd,
            linewidth=2.6,
            color=color,
            label=LABEL_NAMES[label],
        )

    ax.set_yscale("log")
    ax.set_xlim(0, args.psd_max_freq)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Average IMF1 power spectral density")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()

    channel_tag = args.channel.replace("all", "all_channels")
    out_base = out_dir / f"average_imf1_psd_by_class_{args.mode}_{channel_tag}"

    fig.savefig(out_base.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_base.with_suffix('.png')}")
    print(f"Saved: {out_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
