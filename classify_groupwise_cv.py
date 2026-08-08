from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from sklearn.base import clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

from config import (
    FEATURE_DIR,
    RESULT_DIR,
    N_REPEATS,
    N_SPLITS,
    RANDOM_SEED,
    RECORD_AGGREGATION,
    THRESHOLD_METRIC,
    N_JOBS,
    DEBUG_N_REPEATS,
    DEBUG_EXPERIMENTS,
    DEBUG_MODELS,
    CLASSIFICATION_RESUME,
)

OUTPUT_DECIMALS = 4
CHECKPOINT_VERSION = 1

METADATA_COLUMNS = {
    "mode",
    "feature_source",
    "record",
    "label",
    "segment_id",
    "start_sample",
    "end_sample",
    "start_sec",
    "end_sec",
    "imf",
    # Compatibility with old/intermediate CSVs.
    "name",
    "start",
    "end",
    "mother_id",
    "group",
}


FEATURE_FILES = {
    "contraction_imf1": FEATURE_DIR / "tpehgt_contraction_imf1_features.csv",
    "fixed_3min_imf1": FEATURE_DIR / "tpehgt_fixed_3min_imf1_features.csv",
    "contraction_time_domain": FEATURE_DIR
    / "tpehgt_contraction_time_domain_features.csv",
    "fixed_3min_time_domain": FEATURE_DIR
    / "tpehgt_fixed_3min_time_domain_features.csv",
    "contraction_imf2_selection": FEATURE_DIR
    / "imf_selection"
    / "tpehgt_contraction_imf2_features.csv",
    "contraction_imf3_selection": FEATURE_DIR
    / "imf_selection"
    / "tpehgt_contraction_imf3_features.csv",
    "contraction_imf4_selection": FEATURE_DIR
    / "imf_selection"
    / "tpehgt_contraction_imf4_features.csv",
    "fixed_3min_imf2_selection": FEATURE_DIR
    / "imf_selection"
    / "tpehgt_fixed_3min_imf2_features.csv",
    "fixed_3min_imf3_selection": FEATURE_DIR
    / "imf_selection"
    / "tpehgt_fixed_3min_imf3_features.csv",
    "fixed_3min_imf4_selection": FEATURE_DIR
    / "imf_selection"
    / "tpehgt_fixed_3min_imf4_features.csv",
}

# IMF1-selection comparisons are numerically identical to the manuscript IMF1
# experiments. Their result directories are retained for downstream plotting,
# but the expensive classification is performed only once.
RESULT_ALIASES = {
    "contraction_imf1_selection": "contraction_imf1",
    "fixed_3min_imf1_selection": "fixed_3min_imf1",
}


def get_models() -> Dict[str, object]:
    models: Dict[str, object] = {
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.5),
        "Logistic Regression": LogisticRegression(
            solver="lbfgs",
            C=0.5,
            max_iter=50000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "SVM": SVC(
            kernel="linear",
            C=2.0,
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_SEED,
        ),
        "Naive Bayes": GaussianNB(),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            alpha=0.0005,
            max_iter=3000,
            # Do not use sklearn's internal early-stopping validation split.
            # That split is sample/segment-level and is not aware of recording
            # groups. With early_stopping=False, stopping is based on training
            # loss rather than a separately held-out internal validation subset.
            early_stopping=False,
            n_iter_no_change=25,
            random_state=RANDOM_SEED,
        ),
    }

    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            verbose=0,
            random_seed=RANDOM_SEED,
            loss_function="Logloss",
            depth=6,
            learning_rate=0.1,
            allow_writing_files=False,
            thread_count=1,
        )
    else:
        print("CatBoost is not installed. CatBoost will be skipped.")

    return models


def make_pipeline(model: object) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def get_record_column(df: pd.DataFrame) -> str:
    if "record" in df.columns:
        return "record"
    if "name" in df.columns:
        return "name"
    raise ValueError("CSV must contain either 'record' or 'name' column.")


def get_feature_matrix(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, str, List[str]]:
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    record_col = get_record_column(df)

    feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    x = x.select_dtypes(include=[np.number])

    y = df["label"].astype(int).to_numpy()

    # Record-wise grouping.
    groups = df[record_col].astype(str).to_numpy()

    return x, y, groups, record_col, list(x.columns)


def predict_preterm_scores(pipe: Pipeline, x: pd.DataFrame) -> np.ndarray:
    proba = pipe.predict_proba(x)

    if proba.shape[1] == 2:
        return proba[:, 1].astype(float)

    return np.zeros(len(x), dtype=float)


def aggregate_to_record(meta: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    temp = meta.copy()
    temp["score"] = scores.astype(float)

    if RECORD_AGGREGATION == "max":
        return temp.groupby("record", as_index=False).agg(
            label=("label", "first"),
            score=("score", "max"),
        )

    if RECORD_AGGREGATION == "mean":
        return temp.groupby("record", as_index=False).agg(
            label=("label", "first"),
            score=("score", "mean"),
        )

    raise ValueError(f"Unknown RECORD_AGGREGATION: {RECORD_AGGREGATION}")


def metric_for_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if THRESHOLD_METRIC == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))

    if THRESHOLD_METRIC == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))

    if THRESHOLD_METRIC == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))

    raise ValueError(f"Unknown THRESHOLD_METRIC: {THRESHOLD_METRIC}")


def choose_threshold_from_training_records(
    y_true: np.ndarray, scores: np.ndarray
) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if len(np.unique(y_true)) < 2:
        return 0.5

    unique_scores = np.unique(scores[np.isfinite(scores)])

    if len(unique_scores) == 0:
        return 0.5

    # Keep old-compatible behavior: threshold candidates are unique scores.
    candidates = sorted(set(float(x) for x in unique_scores))

    best_threshold = 0.5
    best_value = -np.inf

    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        value = metric_for_threshold(y_true, pred)

        if value > best_value:
            best_value = value
            best_threshold = threshold

    return float(best_threshold)


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan

        return float(roc_auc_score(y_true, scores))
    except Exception:
        return np.nan


def safe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute trapezoidal area under the precision-recall curve."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan

        precision, recall, _ = precision_recall_curve(y_true, scores)
        return float(auc(recall, precision))
    except Exception:
        return np.nan


def calculate_metrics_from_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate recording-level metrics from already determined binary predictions.

    This is needed when pooling the five outer validation folds within one repeat,
    because each fold can have its own threshold selected from its own outer-training
    data. Therefore, there is no single threshold to re-apply to all 26 records in a
    repetition; the fold-specific binary predictions are pooled directly.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = np.asarray(pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "roc_auc": safe_roc_auc(y_true, scores),
        "pr_auc": safe_pr_auc(y_true, scores),
    }


def evaluate_predictions(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    pred = (scores >= threshold).astype(int)
    return calculate_metrics_from_predictions(y_true, scores, pred)


def make_meta(df_part: pd.DataFrame, record_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record": df_part[record_col].astype(str).values,
            "label": df_part["label"].astype(int).values,
        }
    )


def choose_threshold_with_inner_group_cv(
    model: object,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    meta_train: pd.DataFrame,
    repeat: int,
    fold: int,
) -> float:
    """
    Select the recording-level decision threshold using group-aware inner
    cross-validation performed only within the current outer-training set.

    Each outer-training segment receives an out-of-fold score from a model
    that was not fitted using that segment's recording. The segment-level
    out-of-fold scores are then aggregated to the recording level, and the
    threshold is selected from those recording-level scores.
    """
    inner_cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED + 10000 + repeat * N_SPLITS + fold,
    )

    inner_oof_scores = np.full(len(x_train), np.nan, dtype=float)

    for inner_train_idx, inner_val_idx in inner_cv.split(
        x_train, y_train, groups_train
    ):
        inner_train_records = set(groups_train[inner_train_idx])
        inner_val_records = set(groups_train[inner_val_idx])
        inner_overlap = inner_train_records.intersection(inner_val_records)
        if inner_overlap:
            raise RuntimeError(
                "Record leakage found in inner threshold CV for "
                f"repeat={repeat}, fold={fold}: {sorted(inner_overlap)}"
            )

        inner_pipe = make_pipeline(clone(model))
        inner_pipe.fit(
            x_train.iloc[inner_train_idx],
            y_train[inner_train_idx],
        )

        inner_oof_scores[inner_val_idx] = predict_preterm_scores(
            inner_pipe,
            x_train.iloc[inner_val_idx],
        )

    if np.isnan(inner_oof_scores).any():
        raise RuntimeError(
            "Inner grouped CV did not produce an out-of-fold score "
            "for every outer-training segment."
        )

    inner_records_df = aggregate_to_record(meta_train, inner_oof_scores)

    return choose_threshold_from_training_records(
        inner_records_df["label"].to_numpy(),
        inner_records_df["score"].to_numpy(),
    )


def round_numeric_for_output(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with floating-point columns rounded to OUTPUT_DECIMALS."""
    out = df.copy()
    float_cols = out.select_dtypes(include=["float", "float32", "float64"]).columns
    out[float_cols] = out[float_cols].round(OUTPUT_DECIMALS)
    return out


def build_repeat_level_metrics(
    record_predictions: pd.DataFrame,
    expected_records: set[str],
    experiment_name: str,
) -> pd.DataFrame:
    """
    Pool the five outer validation folds within each repeat so that every recording
    contributes exactly once, calculate metrics on the complete out-of-fold set,
    and return one metric row per model per repeat.
    """
    repeat_rows: List[Dict] = []

    for (model_name, repeat), rep_df in record_predictions.groupby(
        ["model", "repeat"], sort=True
    ):
        rep_df = rep_df.copy()
        rep_records = rep_df["record"].astype(str)

        if rep_records.duplicated().any():
            duplicates = sorted(rep_records[rep_records.duplicated()].unique())
            raise RuntimeError(
                f"Duplicate out-of-fold record predictions found for "
                f"experiment={experiment_name}, model={model_name}, "
                f"repeat={repeat}: {duplicates}"
            )

        actual_records = set(rep_records)
        if actual_records != expected_records:
            missing = sorted(expected_records - actual_records)
            extra = sorted(actual_records - expected_records)
            raise RuntimeError(
                f"Incomplete repeat-level OOF predictions for "
                f"experiment={experiment_name}, model={model_name}, "
                f"repeat={repeat}. Missing={missing}, extra={extra}"
            )

        metrics = calculate_metrics_from_predictions(
            rep_df["label"].to_numpy(),
            rep_df["score"].to_numpy(),
            rep_df["prediction"].to_numpy(),
        )

        # The requested cross-validation metric values are stored to 4 decimals.
        metrics = {
            key: (
                round(float(value), OUTPUT_DECIMALS) if np.isfinite(value) else np.nan
            )
            for key, value in metrics.items()
        }

        row: Dict = {
            "experiment": experiment_name,
            "model": model_name,
            "repeat": int(repeat),
            "n_records": len(rep_df),
        }
        for key, value in metrics.items():
            row[f"record_{key}"] = value

        repeat_rows.append(row)

    return pd.DataFrame(repeat_rows)


def build_outer_splits(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_repeats: int,
) -> Tuple[Dict[int, List[Tuple[np.ndarray, np.ndarray]]], List[Dict]]:
    """Create and audit the unchanged repeated grouped outer CV splits."""
    expected_records = set(groups.astype(str))
    splits_by_repeat: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    split_rows: List[Dict] = []

    for repeat in range(n_repeats):
        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED + repeat,
        )
        repeat_splits = list(cv.split(x, y, groups))
        validation_records: List[str] = []

        for fold, (train_idx, val_idx) in enumerate(repeat_splits):
            train_records = set(groups[train_idx].astype(str))
            val_records = set(groups[val_idx].astype(str))
            overlap = train_records.intersection(val_records)
            if overlap:
                raise RuntimeError(
                    f"Record leakage found in repeat={repeat}, fold={fold}: "
                    f"{sorted(overlap)}"
                )

            validation_records.extend(val_records)
            split_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_records": ",".join(sorted(train_records)),
                    "val_records": ",".join(sorted(val_records)),
                    "n_train_records": len(train_records),
                    "n_val_records": len(val_records),
                }
            )

        if len(validation_records) != len(set(validation_records)):
            raise RuntimeError(f"A record occurs in multiple folds for repeat={repeat}.")
        if set(validation_records) != expected_records:
            raise RuntimeError(f"Outer OOF coverage is incomplete for repeat={repeat}.")

        splits_by_repeat[repeat] = repeat_splits

    return splits_by_repeat, split_rows


def evaluate_model_repeat(
    experiment_name: str,
    model_name: str,
    model: object,
    repeat: int,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    df: pd.DataFrame,
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    record_col: str,
    fingerprint: str,
) -> Dict:
    """Evaluate all outer folds for one model/repeat as one atomic unit."""
    fold_rows: List[Dict] = []
    record_pred_rows: List[pd.DataFrame] = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        x_train = x.iloc[train_idx]
        x_val = x.iloc[val_idx]
        y_train = y[train_idx]
        groups_train = groups[train_idx]
        meta_train = make_meta(df.iloc[train_idx], record_col)
        meta_val = make_meta(df.iloc[val_idx], record_col)

        train_records = set(meta_train["record"])
        val_records = set(meta_val["record"])
        overlap = train_records.intersection(val_records)
        if overlap:
            raise RuntimeError(
                f"Record leakage found in repeat={repeat}, fold={fold}: "
                f"{sorted(overlap)}"
            )

        threshold = choose_threshold_with_inner_group_cv(
            model=model,
            x_train=x_train,
            y_train=y_train,
            groups_train=groups_train,
            meta_train=meta_train,
            repeat=repeat,
            fold=fold,
        )

        pipe = make_pipeline(clone(model))
        pipe.fit(x_train, y_train)
        val_scores = predict_preterm_scores(pipe, x_val)
        val_records_df = aggregate_to_record(meta_val, val_scores)
        record_metrics = evaluate_predictions(
            val_records_df["label"].to_numpy(),
            val_records_df["score"].to_numpy(),
            threshold,
        )

        row = {
            "experiment": experiment_name,
            "model": model_name,
            "repeat": repeat,
            "fold": fold,
            "threshold": round(float(threshold), OUTPUT_DECIMALS),
            "n_train_segments": len(train_idx),
            "n_val_segments": len(val_idx),
            "n_train_records": len(train_records),
            "n_val_records": len(val_records_df),
        }
        for key, value in record_metrics.items():
            row[f"record_{key}"] = (
                round(float(value), OUTPUT_DECIMALS)
                if np.isfinite(value)
                else np.nan
            )
        fold_rows.append(row)

        rec_pred = val_records_df.copy()
        rec_pred["experiment"] = experiment_name
        rec_pred["model"] = model_name
        rec_pred["repeat"] = repeat
        rec_pred["fold"] = fold
        rec_pred["threshold"] = threshold
        rec_pred["prediction"] = (
            rec_pred["score"].to_numpy() >= threshold
        ).astype(int)
        record_pred_rows.append(rec_pred)

    return {
        "status": "complete",
        "fingerprint": fingerprint,
        "experiment": experiment_name,
        "model": model_name,
        "repeat": repeat,
        "fold_rows": fold_rows,
        "record_predictions": pd.concat(record_pred_rows, ignore_index=True),
    }


def evaluate_model_repeat_safely(**kwargs: object) -> Dict:
    """Return task failures to the parent without saving partial fold results."""
    try:
        return evaluate_model_repeat(**kwargs)
    except Exception as exc:
        return {
            "status": "failed",
            "experiment": kwargs["experiment_name"],
            "model": kwargs["model_name"],
            "repeat": kwargs["repeat"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_task_fingerprint(
    csv_digest: str,
    feature_cols: List[str],
    model_name: str,
    model: object,
) -> str:
    state = repr(
        (
            CHECKPOINT_VERSION,
            csv_digest,
            feature_cols,
            model_name,
            sorted(model.get_params(deep=True).items()),
            N_SPLITS,
            RANDOM_SEED,
            RECORD_AGGREGATION,
            THRESHOLD_METRIC,
        )
    )
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


def checkpoint_path(out_dir: Path, model_name: str, repeat: int) -> Path:
    model_slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    return out_dir / ".checkpoints" / model_slug / f"repeat_{repeat:03d}.joblib"


def validate_task_result(
    result: Dict,
    fingerprint: str,
    model_name: str,
    repeat: int,
    expected_records: set[str],
) -> bool:
    if result.get("status") != "complete":
        return False
    if result.get("fingerprint") != fingerprint:
        return False
    if result.get("model") != model_name or result.get("repeat") != repeat:
        return False
    if len(result.get("fold_rows", [])) != N_SPLITS:
        return False

    predictions = result.get("record_predictions")
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return False
    records = predictions["record"].astype(str)
    if records.duplicated().any():
        return False
    return set(records) == expected_records


def load_valid_checkpoint(
    path: Path,
    fingerprint: str,
    model_name: str,
    repeat: int,
    expected_records: set[str],
) -> Dict | None:
    if not path.exists():
        return None
    try:
        result = joblib.load(path)
    except Exception as exc:
        print(f"Ignoring unreadable checkpoint {path}: {exc}")
        return None
    if validate_task_result(
        result, fingerprint, model_name, repeat, expected_records
    ):
        return result
    print(f"Ignoring stale or incomplete checkpoint: {path}")
    return None


def save_checkpoint_atomic(path: Path, result: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + f".{os.getpid()}.tmp")
    joblib.dump(result, temp_path, compress=3)
    os.replace(temp_path, path)


def run_one_feature_file(
    name: str,
    csv_path: Path,
    n_repeats: int,
    n_jobs: int,
    selected_models: List[str] | None,
    resume: bool,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing feature table for {name}: {csv_path}")

    print("\n==============================")
    print(f"Running: {name}")
    print(f"CSV    : {csv_path}")
    print("==============================")

    df = pd.read_csv(csv_path)
    x, y, groups, record_col, feature_cols = get_feature_matrix(df)
    expected_records = set(df[record_col].astype(str).unique())
    print(
        f"Rows: {len(df)} | Features: {x.shape[1]} | "
        f"Records: {len(expected_records)}"
    )

    all_models = get_models()
    if selected_models is None:
        models = all_models
    else:
        unknown = sorted(set(selected_models) - set(all_models))
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}")
        models = {model_name: all_models[model_name] for model_name in selected_models}

    splits_by_repeat, split_rows = build_outer_splits(x, y, groups, n_repeats)
    out_dir = RESULT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_digest = sha256_file(csv_path)

    completed: Dict[Tuple[str, int], Dict] = {}
    pending: List[Tuple[str, object, int, str]] = []
    for repeat in range(n_repeats):
        for model_name, model in models.items():
            fingerprint = make_task_fingerprint(
                csv_digest, feature_cols, model_name, model
            )
            path = checkpoint_path(out_dir, model_name, repeat)
            result = (
                load_valid_checkpoint(
                    path, fingerprint, model_name, repeat, expected_records
                )
                if resume
                else None
            )
            if result is not None:
                completed[(model_name, repeat)] = result
            else:
                pending.append((model_name, model, repeat, fingerprint))

    if completed:
        print(f"Resuming from {len(completed)} completed model/repeat checkpoint(s).")
    print(f"Pending complete model/repeat jobs: {len(pending)}")

    if pending:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            generated_results = Parallel(
                n_jobs=n_jobs,
                return_as="generator_unordered",
            )(
                delayed(evaluate_model_repeat_safely)(
                    experiment_name=name,
                    model_name=model_name,
                    model=model,
                    repeat=repeat,
                    splits=splits_by_repeat[repeat],
                    df=df,
                    x=x,
                    y=y,
                    groups=groups,
                    record_col=record_col,
                    fingerprint=fingerprint,
                )
                for model_name, model, repeat, fingerprint in pending
            )

            for result in generated_results:
                model_name = result["model"]
                repeat = int(result["repeat"])
                if result.get("status") != "complete":
                    print(
                        f"Failed: experiment={name}, model={model_name}, "
                        f"repeat={repeat}: {result.get('error')}"
                    )
                    continue

                fingerprint = result["fingerprint"]
                if not validate_task_result(
                    result, fingerprint, model_name, repeat, expected_records
                ):
                    print(
                        f"Rejected incomplete result: experiment={name}, "
                        f"model={model_name}, repeat={repeat}"
                    )
                    continue

                save_checkpoint_atomic(
                    checkpoint_path(out_dir, model_name, repeat), result
                )
                completed[(model_name, repeat)] = result
                print(
                    f"Checkpointed: {name} | {model_name} | "
                    f"repeat {repeat + 1}/{n_repeats}"
                )

    expected_tasks = {
        (model_name, repeat)
        for repeat in range(n_repeats)
        for model_name in models
    }
    missing_tasks = sorted(expected_tasks - set(completed))
    if missing_tasks:
        raise RuntimeError(
            "Classification is incomplete; final summaries were not written. "
            f"Rerun to resume. Missing jobs: {missing_tasks}"
        )

    ordered_results = [
        completed[(model_name, repeat)]
        for repeat in range(n_repeats)
        for model_name in models
    ]
    fold_results = pd.DataFrame(
        row for result in ordered_results for row in result["fold_rows"]
    ).sort_values(["repeat", "fold", "model"], kind="stable")
    record_predictions = pd.concat(
        [result["record_predictions"] for result in ordered_results],
        ignore_index=True,
    ).sort_values(["repeat", "fold", "model", "record"], kind="stable")
    splits = pd.DataFrame(split_rows).sort_values(["repeat", "fold"], kind="stable")

    repeat_metrics = build_repeat_level_metrics(
        record_predictions=record_predictions,
        expected_records=expected_records,
        experiment_name=name,
    ).sort_values(["model", "repeat"], kind="stable")

    record_metric_cols = [
        "record_accuracy",
        "record_precision",
        "record_recall",
        "record_specificity",
        "record_sensitivity",
        "record_f1",
        "record_balanced_accuracy",
        "record_mcc",
        "record_roc_auc",
        "record_pr_auc",
    ]
    summary = repeat_metrics.groupby("model")[record_metric_cols].agg(["mean", "std"])
    summary.columns = [f"{first}_{second}" for first, second in summary.columns]
    summary = round_numeric_for_output(summary.reset_index())

    round_numeric_for_output(fold_results).to_csv(
        out_dir / "fold_metrics.csv",
        index=False,
        float_format=f"%.{OUTPUT_DECIMALS}f",
    )
    round_numeric_for_output(record_predictions).to_csv(
        out_dir / "record_predictions.csv",
        index=False,
        float_format=f"%.{OUTPUT_DECIMALS}f",
    )
    round_numeric_for_output(repeat_metrics).to_csv(
        out_dir / "repeat_metrics.csv",
        index=False,
        float_format=f"%.{OUTPUT_DECIMALS}f",
    )
    summary.to_csv(
        out_dir / "summary_metrics.csv",
        index=False,
        float_format=f"%.{OUTPUT_DECIMALS}f",
    )
    splits.to_csv(out_dir / "splits.csv", index=False)

    with (out_dir / "feature_columns.txt").open("w", encoding="utf-8") as stream:
        for column in feature_cols:
            stream.write(column + "\n")

    display_cols = [
        "model",
        "record_accuracy_mean",
        "record_f1_mean",
        "record_balanced_accuracy_mean",
        "record_mcc_mean",
        "record_roc_auc_mean",
        "record_pr_auc_mean",
    ]
    print(f"\nRecord-level summary ({n_repeats} complete OOF repetitions):")
    print(
        summary[display_cols].to_string(
            index=False,
            float_format=lambda value: f"{value:.{OUTPUT_DECIMALS}f}",
        )
    )
    print(f"\nSaved outputs to: {out_dir}")


def materialize_result_alias(alias_name: str, source_name: str) -> None:
    """Create IMF1-selection outputs from the already-classified main result."""
    source_dir = RESULT_DIR / source_name
    alias_dir = RESULT_DIR / alias_name
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Cannot create {alias_name}; source results are missing: {source_dir}"
        )
    alias_dir.mkdir(parents=True, exist_ok=True)

    for filename in ["fold_metrics.csv", "record_predictions.csv", "repeat_metrics.csv"]:
        frame = pd.read_csv(source_dir / filename)
        frame["experiment"] = alias_name
        frame.to_csv(
            alias_dir / filename,
            index=False,
            float_format=f"%.{OUTPUT_DECIMALS}f",
        )

    for filename in ["summary_metrics.csv", "splits.csv", "feature_columns.txt"]:
        shutil.copyfile(source_dir / filename, alias_dir / filename)
    print(f"Reused {source_name} results as {alias_name} (no reclassification).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated grouped-CV classification for TPEHGT features."
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=DEBUG_N_REPEATS,
        help=f"Development repetition count; default is the final N_REPEATS={N_REPEATS}.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEBUG_EXPERIMENTS,
        help="Run only these experiment names.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEBUG_MODELS,
        help="Run only these model names (quote names containing spaces).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help="Parallel model/repeat workers (default: config.N_JOBS).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints for this invocation.",
    )
    return parser.parse_args()


def main(
    n_repeats: int | None = None,
    experiments: List[str] | None = None,
    models: List[str] | None = None,
    n_jobs: int = N_JOBS,
    resume: bool = CLASSIFICATION_RESUME,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    effective_repeats = N_REPEATS if n_repeats is None else n_repeats
    if not 1 <= effective_repeats <= N_REPEATS:
        raise ValueError(f"n_repeats must be between 1 and {N_REPEATS}.")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be zero.")

    available_experiments = [*FEATURE_FILES, *RESULT_ALIASES]
    requested = available_experiments if experiments is None else experiments
    unknown = sorted(set(requested) - set(available_experiments))
    if unknown:
        raise ValueError(f"Unknown experiment(s): {unknown}")

    canonical_requested = {
        RESULT_ALIASES.get(experiment_name, experiment_name)
        for experiment_name in requested
    }
    for name, csv_path in FEATURE_FILES.items():
        if name in canonical_requested:
            run_one_feature_file(
                name=name,
                csv_path=csv_path,
                n_repeats=effective_repeats,
                n_jobs=n_jobs,
                selected_models=models,
                resume=resume,
            )

    for alias_name, source_name in RESULT_ALIASES.items():
        if alias_name in requested:
            materialize_result_alias(alias_name, source_name)


if __name__ == "__main__":
    args = parse_args()
    main(
        n_repeats=args.n_repeats,
        experiments=args.experiments,
        models=args.models,
        n_jobs=args.n_jobs,
        resume=CLASSIFICATION_RESUME and not args.no_resume,
    )
