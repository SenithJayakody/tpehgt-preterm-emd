from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from config import (
    DATASET_DIR,
    FEATURE_DIR,
    BT_CONTRACTION_SEC,
    BT_FIXED_SEC,
    FINAL_IMF_NUMBER,
    IMF_SELECTION_NUMBERS,
)
from features import (
    extract_imf_features_from_segment,
    extract_time_domain_features_from_segment,
)
from io_readers import (
    contraction_intervals,
    fixed_intervals,
    load_pregnancy_records,
    save_record_report,
)


def build_rows_for_mode(
    records: List[Dict],
    mode: str,
    intervals_by_record: Dict[str, List[Tuple[int, int]]],
    burst_threshold_sec: float,
    feature_source: str,
    imf_number: int | None = None,
) -> pd.DataFrame:
    rows = []

    for rec in tqdm(records, desc=f"Extracting {mode}, {feature_source}"):
        record = rec["record"]
        fs = rec["fs"]
        intervals = intervals_by_record[record]

        for segment_id, (start, end) in enumerate(intervals):
            if end <= start:
                continue

            row = {
                "mode": mode,
                "feature_source": feature_source,
                "record": record,
                "label": rec["label"],
                "segment_id": segment_id,
                "start_sample": int(start),
                "end_sample": int(end),
                "start_sec": float(start / fs),
                "end_sec": float(end / fs),
                "imf": "none" if imf_number is None else f"IMF{imf_number}",
            }

            for channel_name, signal in rec["ehg"].items():
                segment = signal[start:end]

                if feature_source == "time_domain":
                    feats = extract_time_domain_features_from_segment(
                        segment,
                        fs=fs,
                        burst_threshold_sec=burst_threshold_sec,
                    )

                elif feature_source.startswith("imf"):
                    if imf_number is None:
                        raise ValueError("imf_number is required for IMF features")

                    feats = extract_imf_features_from_segment(
                        segment,
                        fs=fs,
                        burst_threshold_sec=burst_threshold_sec,
                        imf_number=imf_number,
                    )

                else:
                    raise ValueError(f"Unknown feature_source: {feature_source}")

                for feature_name, value in feats.items():
                    row[f"{channel_name}_{feature_name}"] = value

            rows.append(row)

    return pd.DataFrame(rows)


def save_feature_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path} | shape={df.shape}")


def main() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    save_record_report(DATASET_DIR, out_csv="outputs/record_report.csv")

    records = load_pregnancy_records(DATASET_DIR)
    print(f"\nLoaded {len(records)} term/preterm recordings")

    contraction_by_record = {
        rec["record"]: contraction_intervals(rec["record"], DATASET_DIR)
        for rec in records
    }

    fixed_by_record = {
        rec["record"]: fixed_intervals(rec["record"], DATASET_DIR)
        for rec in records
    }

    # Main method: IMF1 features.
    contraction_imf1_df = build_rows_for_mode(
        records=records,
        mode="contraction",
        intervals_by_record=contraction_by_record,
        burst_threshold_sec=BT_CONTRACTION_SEC,
        feature_source="imf1",
        imf_number=FINAL_IMF_NUMBER,
    )
    save_feature_csv(
        contraction_imf1_df,
        FEATURE_DIR / "tpehgt_contraction_imf1_features.csv",
    )

    fixed_imf1_df = build_rows_for_mode(
        records=records,
        mode="fixed_3min",
        intervals_by_record=fixed_by_record,
        burst_threshold_sec=BT_FIXED_SEC,
        feature_source="imf1",
        imf_number=FINAL_IMF_NUMBER,
    )
    save_feature_csv(
        fixed_imf1_df,
        FEATURE_DIR / "tpehgt_fixed_3min_imf1_features.csv",
    )

    # Time-domain baseline: same filtered EHG segments, no EMD.
    contraction_time_df = build_rows_for_mode(
        records=records,
        mode="contraction",
        intervals_by_record=contraction_by_record,
        burst_threshold_sec=BT_CONTRACTION_SEC,
        feature_source="time_domain",
        imf_number=None,
    )
    save_feature_csv(
        contraction_time_df,
        FEATURE_DIR / "tpehgt_contraction_time_domain_features.csv",
    )

    fixed_time_df = build_rows_for_mode(
        records=records,
        mode="fixed_3min",
        intervals_by_record=fixed_by_record,
        burst_threshold_sec=BT_FIXED_SEC,
        feature_source="time_domain",
        imf_number=None,
    )
    save_feature_csv(
        fixed_time_df,
        FEATURE_DIR / "tpehgt_fixed_3min_time_domain_features.csv",
    )

    # IMF selection: IMF1--IMF4 for both segmentation strategies.
    for mode, intervals_by_record, bt in [
        ("contraction", contraction_by_record, BT_CONTRACTION_SEC),
        ("fixed_3min", fixed_by_record, BT_FIXED_SEC),
    ]:
        for imf_number in IMF_SELECTION_NUMBERS:
            imf_df = build_rows_for_mode(
                records=records,
                mode=mode,
                intervals_by_record=intervals_by_record,
                burst_threshold_sec=bt,
                feature_source=f"imf{imf_number}",
                imf_number=imf_number,
            )

            save_feature_csv(
                imf_df,
                FEATURE_DIR
                / "imf_selection"
                / f"tpehgt_{mode}_imf{imf_number}_features.csv",
            )


if __name__ == "__main__":
    main()
