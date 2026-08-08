from __future__ import annotations

from dataclasses import dataclass
from math import factorial, log
from typing import Dict, Tuple

import numpy as np
from PyEMD import EMD
from scipy.signal import find_peaks, peak_widths
from scipy.spatial import cKDTree

from config import (
    FS,
    MAX_IMFS,
    PEAK_MODE,
    THK,
    MD_SEC,
    WIDTH_REL_HEIGHT,
    SHANNON_BINS,
    SAMPEN_M,
    SAMPEN_R,
    SAMPEN_TAU,
    PERM_M,
    PERM_TAU,
    PERM_NORMALIZE,
)

FEATURE_NAMES = [
    "PEAK_RATE",
    "PEAK_AMP_MEAN",
    "PEAK_AMP_CV",
    "IPI_MEAN",
    "IPI_CV",
    "PW_MEAN",
    "BURST_COUNT",
    "PEAKS_PER_BURST_MEAN",
    "DASDV",
    "LOG",
    "MTKE",
    "SE",
    "perm_entropy",
    "sampen",
]


@dataclass
class FeatureConfig:
    fs: float = FS
    max_imfs: int = MAX_IMFS
    imf_number: int = 1

    peak_mode: str = PEAK_MODE
    thresh_k: float = THK
    min_distance_sec: float = MD_SEC
    width_rel_height: float = WIDTH_REL_HEIGHT
    burst_tau_sec: float = 2.0

    shannon_bins: int = SHANNON_BINS

    sampen_m: int = SAMPEN_M
    sampen_r: float = SAMPEN_R
    sampen_tau: int = SAMPEN_TAU

    perm_m: int = PERM_M
    perm_tau: int = PERM_TAU
    perm_normalize: bool = PERM_NORMALIZE


def empty_features() -> Dict[str, float]:
    return {name: 0.0 for name in FEATURE_NAMES}


def missing_features() -> Dict[str, float]:
    """Return NaN for every feature when the requested signal representation is unavailable."""
    return {name: np.nan for name in FEATURE_NAMES}


def imf_number_to_index(imf_number: int) -> int:
    if imf_number < 1:
        raise ValueError("Use manuscript-style IMF numbers: IMF1, IMF2, ...")
    return imf_number - 1


def compute_imfs(
    x: np.ndarray,
    max_imfs: int = MAX_IMFS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute EMD and return (imfs, residual).

    The returned IMF array contains only genuine IMFs produced by PyEMD;
    it is not zero-padded when fewer than ``max_imfs`` IMFs are available.
    The residual is obtained separately through ``get_imfs_and_residue()``.

    EMD failures are raised explicitly rather than being converted into
    artificial zero-valued IMFs.
    """
    x = np.asarray(x, dtype=float).ravel()

    if x.size < 5:
        raise ValueError(
            f"Signal is too short for EMD: {x.size} samples."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Input signal contains NaN or infinite values."
        )

    emd = EMD()

    try:
        # Run EMD while requesting no more than the number of modes needed
        # by this analysis.
        emd.emd(x, max_imf=max_imfs)

        # PyEMD's emd() return value can contain a residual as its last row.
        # Use the dedicated accessor to separate genuine IMFs and residual.
        imfs_raw, residual = emd.get_imfs_and_residue()

    except Exception as exc:
        raise RuntimeError("EMD decomposition failed.") from exc

    imfs_raw = np.asarray(imfs_raw, dtype=float)
    residual = np.asarray(residual, dtype=float).ravel()

    if imfs_raw.ndim == 1:
        imfs_raw = imfs_raw[None, :]

    if imfs_raw.ndim != 2:
        raise RuntimeError(
            f"Unexpected IMF array shape returned by PyEMD: {imfs_raw.shape}"
        )

    if imfs_raw.shape[1] != x.size:
        raise RuntimeError(
            "EMD returned IMF length inconsistent with the input signal."
        )

    if residual.size != x.size:
        raise RuntimeError(
            "EMD returned residual length inconsistent with the input signal."
        )

    # Defensive slice only. No artificial zero-padding is performed.
    imfs = imfs_raw[:max_imfs].copy()

    return imfs, residual


def get_imf(
    x: np.ndarray,
    imf_number: int = 1,
    max_imfs: int = MAX_IMFS,
) -> np.ndarray:
    """
    Return the requested genuine IMF.

    If EMD succeeds but does not produce the requested IMF, return a NaN
    signal so that downstream feature values are represented as missing
    rather than as a physiological zero signal.
    """
    x = np.asarray(x, dtype=float).ravel()

    imfs, _ = compute_imfs(x, max_imfs=max_imfs)
    idx = imf_number_to_index(imf_number)

    if idx >= imfs.shape[0]:
        return np.full(x.shape, np.nan, dtype=float)

    return imfs[idx].copy()


def safe_cv(mean_value: float, std_value: float) -> float:
    return float(std_value / (mean_value + 1e-12))


def dasdv(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()

    if x.size < 2:
        return 0.0

    return float(np.sqrt(np.mean(np.diff(x) ** 2)))


def log_detector(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()

    if x.size == 0:
        return 0.0

    return float(np.exp(np.mean(np.log(np.abs(x) + 1e-12))))


def mtke(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()

    if x.size < 3:
        return 0.0

    energy = x[1:-1] ** 2 - x[:-2] * x[2:]

    return float(np.mean(energy))


def shannon_entropy(x: np.ndarray, bins: int = SHANNON_BINS) -> float:
    x = np.asarray(x, dtype=float).ravel()

    if x.size < 2:
        return np.nan

    counts, _ = np.histogram(x, bins=bins, density=False)
    total = counts.sum()

    if total == 0:
        return np.nan

    p = counts.astype(float) / total
    p = p[p > 0]

    return float(-np.sum(p * np.log(p)))


def detect_peaks(
    x: np.ndarray, cfg: FeatureConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect peaks using:
        threshold = median(y) + THK * MAD(y)

    If peak_mode='abs', y = abs(x). If peak_mode='pos', y = x.
    """
    x = np.asarray(x, dtype=float).ravel()

    if x.size < 5:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    if cfg.peak_mode == "abs":
        y = np.abs(x)
    elif cfg.peak_mode == "pos":
        y = x.copy()
    else:
        raise ValueError("peak_mode must be 'abs' or 'pos'")

    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-12
    threshold = med + cfg.thresh_k * mad

    min_distance = max(1, int(round(cfg.min_distance_sec * cfg.fs)))

    peaks, props = find_peaks(
        y,
        height=threshold,
        distance=min_distance,
    )

    amplitudes = props["peak_heights"] if peaks.size else np.array([], dtype=float)

    return peaks.astype(int), amplitudes.astype(float), y.astype(float)


def peak_burst_features(x: np.ndarray, cfg: FeatureConfig) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).ravel()

    out = {
        "PEAK_RATE": 0.0,
        "PEAK_AMP_MEAN": 0.0,
        "PEAK_AMP_CV": 0.0,
        "IPI_MEAN": 0.0,
        "IPI_CV": 0.0,
        "PW_MEAN": 0.0,
        "BURST_COUNT": 0.0,
        "PEAKS_PER_BURST_MEAN": 0.0,
    }

    peaks, amplitudes, peak_signal = detect_peaks(x, cfg)

    duration_sec = x.size / (cfg.fs + 1e-12)
    out["PEAK_RATE"] = float(peaks.size / (duration_sec + 1e-12))

    if peaks.size == 0:
        return out

    amp_mean = float(np.mean(amplitudes)) if amplitudes.size else 0.0
    amp_std = float(np.std(amplitudes)) if amplitudes.size else 0.0

    out["PEAK_AMP_MEAN"] = amp_mean
    out["PEAK_AMP_CV"] = safe_cv(amp_mean, amp_std)

    try:
        widths_samples = peak_widths(
            peak_signal,
            peaks,
            rel_height=cfg.width_rel_height,
        )[0]
        widths_sec = widths_samples / (cfg.fs + 1e-12)
        out["PW_MEAN"] = float(np.mean(widths_sec)) if widths_sec.size else 0.0
    except Exception:
        out["PW_MEAN"] = 0.0

    if peaks.size < 2:
        out["BURST_COUNT"] = 1.0
        out["PEAKS_PER_BURST_MEAN"] = float(peaks.size)
        return out

    ipi = np.diff(peaks) / (cfg.fs + 1e-12)

    ipi_mean = float(np.mean(ipi))
    ipi_std = float(np.std(ipi))

    out["IPI_MEAN"] = ipi_mean
    out["IPI_CV"] = safe_cv(ipi_mean, ipi_std)

    burst_breaks = np.where(ipi > cfg.burst_tau_sec)[0]

    burst_starts = np.r_[0, burst_breaks + 1]
    burst_ends = np.r_[burst_breaks, peaks.size - 1]

    out["BURST_COUNT"] = float(len(burst_starts))

    peaks_per_burst = [
        int(end - start + 1) for start, end in zip(burst_starts, burst_ends)
    ]

    out["PEAKS_PER_BURST_MEAN"] = (
        float(np.mean(peaks_per_burst)) if peaks_per_burst else 0.0
    )

    return out


def embed_signal(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()

    if m < 1 or tau < 1:
        raise ValueError("m and tau must be positive integers")

    n = x.size - (m - 1) * tau

    if n <= 1:
        return np.empty((0, m), dtype=float)

    return np.column_stack([x[j * tau : j * tau + n] for j in range(m)])


def permutation_entropy(
    x: np.ndarray,
    m: int = PERM_M,
    tau: int = PERM_TAU,
    normalize: bool = PERM_NORMALIZE,
) -> float:
    x = np.asarray(x, dtype=float).ravel()
    n = x.size - (m - 1) * tau

    if n <= 1:
        return 0.0

    patterns = {}
    eps = np.finfo(float).eps

    for i in range(n):
        window = x[i : i + m * tau : tau]
        order = np.argsort(window + eps * np.arange(m), kind="mergesort")
        key = tuple(order.tolist())
        patterns[key] = patterns.get(key, 0) + 1

    counts = np.array(list(patterns.values()), dtype=float)
    p = counts / (counts.sum() + 1e-12)

    h = -np.sum(p * np.log(p + 1e-12))

    if normalize:
        h = h / log(factorial(m))

    return float(h)


def sample_entropy(
    x: np.ndarray,
    m: int = SAMPEN_M,
    r: float = SAMPEN_R,
    tau: int = SAMPEN_TAU,
) -> float:
    x = np.asarray(x, dtype=float).ravel()

    if x.size < (m + 2) * tau:
        return np.nan

    tolerance = r * np.std(x, ddof=0)

    if tolerance <= 0 or not np.isfinite(tolerance):
        return np.nan

    xm = embed_signal(x, m, tau)
    xm1 = embed_signal(x, m + 1, tau)

    def match_probability(vectors: np.ndarray) -> float:
        n = vectors.shape[0]

        if n < 2:
            return np.nan

        tree = cKDTree(vectors)
        counts = np.array(
            [len(tree.query_ball_point(v, tolerance, p=np.inf)) - 1 for v in vectors],
            dtype=float,
        )

        return float(counts.sum() / (n * (n - 1)))

    b = match_probability(xm)
    a = match_probability(xm1)

    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return np.nan

    return float(-np.log(a / b))


def extract_features_from_signal(
    signal: np.ndarray, cfg: FeatureConfig
) -> Dict[str, float]:
    """
    Extract all 14 features from a single signal representation.

    For the IMF1 pipeline, this signal is IMF1.
    For the time-domain baseline, this signal is the filtered EHG segment.
    """
    x = np.asarray(signal, dtype=float).ravel()

    if x.size < 5:
        return empty_features()

    # A requested IMF that was not produced by EMD is represented as an
    # all-NaN signal by get_imf(). Preserve that state as missing features
    # so the classifier's training-fold imputer can handle it without
    # confusing IMF absence with a true zero-valued physiological signal.
    if not np.all(np.isfinite(x)):
        return missing_features()

    feats = peak_burst_features(x, cfg)
    feats["DASDV"] = dasdv(x)
    feats["LOG"] = log_detector(x)
    feats["MTKE"] = mtke(x)
    feats["SE"] = shannon_entropy(x, bins=cfg.shannon_bins)
    feats["perm_entropy"] = permutation_entropy(
        x,
        m=cfg.perm_m,
        tau=cfg.perm_tau,
        normalize=cfg.perm_normalize,
    )
    feats["sampen"] = sample_entropy(
        x,
        m=cfg.sampen_m,
        r=cfg.sampen_r,
        tau=cfg.sampen_tau,
    )

    return feats


def extract_imf_features_from_segment(
    segment: np.ndarray,
    fs: float,
    burst_threshold_sec: float,
    imf_number: int = 1,
) -> Dict[str, float]:
    """
    Extract all 14 features from a selected IMF.
    imf_number=1 means IMF1 in manuscript terminology.
    """
    x = np.asarray(segment, dtype=float).ravel()

    cfg = FeatureConfig(
        fs=fs,
        imf_number=imf_number,
        burst_tau_sec=burst_threshold_sec,
    )

    imf_signal = get_imf(x, imf_number=imf_number, max_imfs=cfg.max_imfs)

    return extract_features_from_signal(imf_signal, cfg)


def extract_time_domain_features_from_segment(
    segment: np.ndarray,
    fs: float,
    burst_threshold_sec: float,
) -> Dict[str, float]:
    """
    Extract all 14 features directly from the filtered EHG time-domain segment.
    This is the time-domain baseline without EMD.
    """
    x = np.asarray(segment, dtype=float).ravel()

    cfg = FeatureConfig(
        fs=fs,
        burst_tau_sec=burst_threshold_sec,
    )

    return extract_features_from_signal(x, cfg)


# Backward-compatible alias.
extract_raw_time_features_from_segment = extract_time_domain_features_from_segment


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sig = rng.normal(size=int(FS * 180))
    print(
        extract_imf_features_from_segment(
            sig, fs=FS, burst_threshold_sec=2.0, imf_number=1
        )
    )
    print(
        extract_time_domain_features_from_segment(sig, fs=FS, burst_threshold_sec=2.0)
    )