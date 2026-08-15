"""Fast final-analysis validation; does not fit classifiers or run EMD."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from config import DATASET_DIR, FEATURE_DIR
from io_readers import (
    annotated_intervals,
    contraction_intervals,
    dummy_intervals,
    fixed_intervals,
    list_record_names,
    parse_label,
)

CANONICAL_EXPERIMENTS = [
    "annotated_interval_imf1",
    "annotated_interval_imf2",
    "annotated_interval_imf3",
    "annotated_interval_imf4",
    "annotated_interval_time_domain",
    "fixed_3min_imf1",
    "fixed_3min_imf2",
    "fixed_3min_imf3",
    "fixed_3min_imf4",
    "fixed_3min_time_domain",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def feature_path(experiment: str) -> Path:
    filename = f"tpehgt_{experiment}_features.csv"
    if experiment.endswith(("imf2", "imf3", "imf4")):
        return FEATURE_DIR / "imf_selection" / filename
    return FEATURE_DIR / filename


def main() -> None:
    records = [name for name in list_record_names(DATASET_DIR) if parse_label(name) in (0, 1)]
    require(len(records) == 26, f"Expected 26 pregnancy recordings, found {len(records)}")

    counts = {0: Counter(), 1: Counter()}
    fixed_counts = []
    for record in records:
        label = parse_label(record)
        contractions = contraction_intervals(record, DATASET_DIR)
        dummies = dummy_intervals(record, DATASET_DIR)
        annotated = annotated_intervals(record, DATASET_DIR)
        require(
            annotated == sorted(set(contractions) | set(dummies)),
            f"Annotated interval union mismatch for {record}",
        )
        counts[label].update(contraction=len(contractions), dummy=len(dummies))
        fixed_counts.append(len(fixed_intervals(record, DATASET_DIR)))

    actual = {
        "preterm_contraction": counts[1]["contraction"],
        "preterm_dummy": counts[1]["dummy"],
        "term_contraction": counts[0]["contraction"],
        "term_dummy": counts[0]["dummy"],
    }
    expected = {
        "preterm_contraction": 47,
        "preterm_dummy": 47,
        "term_contraction": 53,
        "term_dummy": 53,
    }
    require(actual == expected, f"Annotation count mismatch: expected={expected}, actual={actual}")
    require(sum(actual.values()) == 200, f"Expected 200 annotated intervals, found {sum(actual.values())}")
    distribution = Counter(fixed_counts)
    require(sum(fixed_counts) == 249, f"Expected 249 fixed windows, found {sum(fixed_counts)}")
    require(distribution == Counter({10: 15, 9: 11}), f"Fixed-window distribution mismatch: {dict(distribution)}")

    from classify_groupwise_cv import FEATURE_FILES

    actual_experiments = set(FEATURE_FILES)
    expected_experiments = set(CANONICAL_EXPERIMENTS)
    require(
        actual_experiments == expected_experiments,
        "Classifier experiment names are not canonical: "
        f"missing={sorted(expected_experiments - actual_experiments)}, "
        f"extra={sorted(actual_experiments - expected_experiments)}",
    )
    expected_records = set(records)
    checked = 0
    for experiment in CANONICAL_EXPERIMENTS:
        path = feature_path(experiment)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        expected_rows = 200 if experiment.startswith("annotated_interval_") else 249
        require(len(frame) == expected_rows, f"{path}: expected {expected_rows} rows, found {len(frame)}")
        require(set(frame["record"].astype(str)) == expected_records, f"{path}: recording cohort mismatch")
        require(frame.groupby("record")["label"].nunique().eq(1).all(), f"{path}: inconsistent labels")
        expected_mode = "annotated_interval" if expected_rows == 200 else "fixed_3min"
        require(set(frame["mode"].astype(str)) == {expected_mode}, f"{path}: mode is not {expected_mode}")
        checked += 1

    print("Final pipeline validation passed")
    print("  pregnancy recordings: 26")
    print("  contractions: 100 (preterm 47, term 53)")
    print("  dummies: 100 (preterm 47, term 53)")
    print("  annotated intervals: 200")
    print("  fixed 3-minute windows: 249 (15 x 10, 11 x 9)")
    print(f"  canonical feature tables checked: {checked}/{len(CANONICAL_EXPERIMENTS)}")


if __name__ == "__main__":
    main()
