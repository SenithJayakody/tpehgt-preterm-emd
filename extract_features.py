from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

from config import (
    DATASET_DIR,
    FEATURE_DIR,
    BT_ANNOTATED_INTERVAL_SEC,
    BT_FIXED_SEC,
    FINAL_IMF_NUMBER,
    IMF_SELECTION_NUMBERS,
    MAX_IMFS,
    N_JOBS,
)
from features import (
    compute_imfs,
    extract_imf_features_from_decomposition,
    extract_time_domain_features_from_segment,
)
from io_readers import (
    annotated_intervals,
    fixed_intervals,
    load_pregnancy_records,
    save_record_report,
)


def build_rows_for_record(
    rec: Dict,
    mode: str,
    intervals: List[Tuple[int, int]],
    burst_threshold_sec: float,
    imf_numbers: List[int],
) -> Dict[str, List[Dict]]:
    """Extract every requested representation for one recording.

    A segment/channel decomposition is shared by all requested IMF numbers.
    The returned row lists retain segment and source order deterministically.
    """
    sources = ["time_domain", *(f"imf{number}" for number in imf_numbers)]
    rows_by_source: Dict[str, List[Dict]] = {source: [] for source in sources}
    record = rec["record"]
    fs = rec["fs"]

    for segment_id, (start, end) in enumerate(intervals):
        if end <= start:
            continue

        source_rows = {
            source: {
                "mode": mode,
                "feature_source": source,
                "record": record,
                "label": rec["label"],
                "segment_id": segment_id,
                "start_sample": int(start),
                "end_sample": int(end),
                "start_sec": float(start / fs),
                "end_sec": float(end / fs),
                "imf": "none" if source == "time_domain" else source.upper(),
            }
            for source in sources
        }

        for channel_name, signal in rec["ehg"].items():
            segment = signal[start:end]

            time_features = extract_time_domain_features_from_segment(
                segment,
                fs=fs,
                burst_threshold_sec=burst_threshold_sec,
            )
            for feature_name, value in time_features.items():
                source_rows["time_domain"][f"{channel_name}_{feature_name}"] = value

            # Reuse one decomposition for every requested IMF from this
            # segment and channel.
            imfs, _ = compute_imfs(segment, max_imfs=MAX_IMFS)
            for imf_number in imf_numbers:
                source = f"imf{imf_number}"
                imf_features = extract_imf_features_from_decomposition(
                    imfs=imfs,
                    signal_length=len(segment),
                    fs=fs,
                    burst_threshold_sec=burst_threshold_sec,
                    imf_number=imf_number,
                )
                for feature_name, value in imf_features.items():
                    source_rows[source][f"{channel_name}_{feature_name}"] = value

        for source in sources:
            rows_by_source[source].append(source_rows[source])

    return rows_by_source


def build_feature_tables_for_mode(
    records: List[Dict],
    mode: str,
    intervals_by_record: Dict[str, List[Tuple[int, int]]],
    burst_threshold_sec: float,
    imf_numbers: List[int],
    n_jobs: int,
) -> Dict[str, pd.DataFrame]:
    """Build ordered feature tables, parallelizing independent recordings."""
    with parallel_config(backend="loky", inner_max_num_threads=1):
        record_results = Parallel(n_jobs=n_jobs)(
            delayed(build_rows_for_record)(
                rec=rec,
                mode=mode,
                intervals=intervals_by_record[rec["record"]],
                burst_threshold_sec=burst_threshold_sec,
                imf_numbers=imf_numbers,
            )
            for rec in tqdm(records, desc=f"Extracting {mode}")
        )

    sources = ["time_domain", *(f"imf{number}" for number in imf_numbers)]
    return {
        source: pd.DataFrame(
            row
            for record_result in record_results
            for row in record_result[source]
        )
        for source in sources
    }


def save_feature_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path} | shape={df.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract TPEHGT feature tables.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help="Parallel recording workers (default: config.N_JOBS).",
    )
    return parser.parse_args()


def main(n_jobs: int = N_JOBS) -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    save_record_report(DATASET_DIR, out_csv="outputs/record_report.csv")

    records = load_pregnancy_records(DATASET_DIR)
    print(f"\nLoaded {len(records)} term/preterm recordings")

    annotated_by_record = {
        rec["record"]: annotated_intervals(rec["record"], DATASET_DIR)
        for rec in records
    }
    fixed_by_record = {
        rec["record"]: fixed_intervals(rec["record"], DATASET_DIR)
        for rec in records
    }

    # Include the manuscript IMF even if a future selection list omits it.
    imf_numbers = list(dict.fromkeys([FINAL_IMF_NUMBER, *IMF_SELECTION_NUMBERS]))
    tables_by_mode = {
        "annotated_interval": build_feature_tables_for_mode(
            records=records,
            mode="annotated_interval",
            intervals_by_record=annotated_by_record,
            burst_threshold_sec=BT_ANNOTATED_INTERVAL_SEC,
            imf_numbers=imf_numbers,
            n_jobs=n_jobs,
        ),
        "fixed_3min": build_feature_tables_for_mode(
            records=records,
            mode="fixed_3min",
            intervals_by_record=fixed_by_record,
            burst_threshold_sec=BT_FIXED_SEC,
            imf_numbers=imf_numbers,
            n_jobs=n_jobs,
        ),
    }

    for mode, tables in tables_by_mode.items():
        # Main IMF1 and time-domain experiments.
        save_feature_csv(
            tables[f"imf{FINAL_IMF_NUMBER}"],
            FEATURE_DIR / f"tpehgt_{mode}_imf{FINAL_IMF_NUMBER}_features.csv",
        )
        save_feature_csv(
            tables["time_domain"],
            FEATURE_DIR / f"tpehgt_{mode}_time_domain_features.csv",
        )

        # IMF-selection IMF1 reuses the already-built main table. IMF2--IMF4
        # likewise reuse the single decomposition computed above.
        for imf_number in IMF_SELECTION_NUMBERS:
            save_feature_csv(
                tables[f"imf{imf_number}"],
                FEATURE_DIR
                / "imf_selection"
                / f"tpehgt_{mode}_imf{imf_number}_features.csv",
            )


if __name__ == "__main__":
    args = parse_args()
    main(n_jobs=args.n_jobs)
