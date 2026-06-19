from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import wfdb

from config import DATASET_DIR, EHG_CHANNELS, EHG_CHANNEL_NAMES, FS, FIXED_WINDOW_SEC


def parse_label(record_name: str) -> int:
    """
    Return 1 for preterm, 0 for term, and -1 for non-pregnant/unknown.
    TPEHGT names are like tpehgt_p001, tpehgt_t001, tpehgt_n001.
    """
    parts = record_name.split("_")
    if len(parts) < 2:
        return -1

    group = parts[1].lower()

    if group.startswith("p"):
        return 1

    if group.startswith("t"):
        return 0

    return -1


def label_name(label: int) -> str:
    return "preterm" if int(label) == 1 else "term"


def list_record_names(dataset_dir: Path = DATASET_DIR) -> List[str]:
    dataset_dir = Path(dataset_dir)
    return sorted(p.stem for p in dataset_dir.glob("*.hea"))


def read_header_comments(record_name: str, dataset_dir: Path = DATASET_DIR) -> Dict[str, str]:
    hea_path = Path(dataset_dir) / f"{record_name}.hea"

    if not hea_path.exists():
        raise FileNotFoundError(hea_path)

    comments: Dict[str, str] = {}

    with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line.startswith("#"):
                continue

            line = line.lstrip("#").strip()

            if not line or line.lower() == "comments:":
                continue

            parts = line.replace("\t", " ").split(None, 1)

            if len(parts) == 2:
                key, value = parts
                comments[key.strip()] = value.strip()

    return comments


def read_record(record_name: str, dataset_dir: Path = DATASET_DIR) -> Dict:
    """
    Read one TPEHGT record and return the selected three filtered EHG channels.
    Channel selection is controlled by config.EHG_CHANNELS.
    """
    base = str(Path(dataset_dir) / record_name)
    signals, meta = wfdb.rdsamp(base)

    fs = float(meta["fs"])

    if abs(fs - FS) > 1e-6:
        print(f"Warning: {record_name} fs={fs}, expected {FS}")

    ehg = {}

    for channel_index, channel_name in zip(EHG_CHANNELS, EHG_CHANNEL_NAMES):
        ehg[channel_name] = signals[:, channel_index].astype(float)

    return {
        "record": record_name,
        "label": parse_label(record_name),
        "fs": fs,
        "ehg": ehg,
        "comments": read_header_comments(record_name, dataset_dir),
    }


def contraction_intervals(record_name: str, dataset_dir: Path = DATASET_DIR) -> List[Tuple[int, int]]:
    """
    Return annotated contraction intervals using BC/EC labels.
    Some TPEHGT files also use BD/ED; these are accepted as backup.
    """
    base = str(Path(dataset_dir) / record_name)

    try:
        ann = wfdb.rdann(base, "atr")
    except Exception:
        return []

    intervals: List[Tuple[int, int]] = []
    start = None

    for sample, note in zip(ann.sample, ann.aux_note):
        note = str(note).strip()

        if note in {"BC", "BD"}:
            start = int(sample)

        elif note in {"EC", "ED"} and start is not None:
            end = int(sample)

            if end > start:
                intervals.append((start, end))

            start = None

    return intervals


def fixed_intervals(record_name: str, dataset_dir: Path = DATASET_DIR) -> List[Tuple[int, int]]:
    """Return non-overlapping 3-minute windows."""
    rec = read_record(record_name, dataset_dir)
    n = len(next(iter(rec["ehg"].values())))
    win = int(FIXED_WINDOW_SEC * rec["fs"])

    return [(s, s + win) for s in range(0, n - win + 1, win)]


def load_pregnancy_records(dataset_dir: Path = DATASET_DIR) -> List[Dict]:
    """
    Load only the 26 term/preterm pregnancy recordings.
    Non-pregnant records are excluded.
    """
    records = []

    for record_name in list_record_names(dataset_dir):
        label = parse_label(record_name)

        if label not in (0, 1):
            continue

        records.append(read_record(record_name, dataset_dir))

    return records


def save_record_report(dataset_dir: Path = DATASET_DIR, out_csv: Path | None = None) -> pd.DataFrame:
    """Save a report of the included term/preterm recordings."""
    rows = []

    for record_name in list_record_names(dataset_dir):
        label = parse_label(record_name)

        if label not in (0, 1):
            continue

        comments = read_header_comments(record_name, dataset_dir)

        rows.append({
            "record": record_name,
            "label": label,
            "class": label_name(label),
            "RecType": comments.get("RecType", ""),
            "Gestation": comments.get("Gestation", ""),
            "Rectime": comments.get("Rectime", ""),
            "Age": comments.get("Age", ""),
            "Parity": comments.get("Parity", ""),
            "Abortions": comments.get("Abortions", ""),
            "Weight": comments.get("Weight", ""),
            "Placental_position": comments.get("Placental_position", ""),
            "Smoker": comments.get("Smoker", ""),
        })

    df = pd.DataFrame(rows)

    print("\nIncluded term/preterm recordings:")
    if not df.empty:
        print(df[["record", "class", "Gestation", "Rectime"]].to_string(index=False))

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved record report: {out_csv}")

    return df


if __name__ == "__main__":
    save_record_report(DATASET_DIR, out_csv="outputs/record_report.csv")
