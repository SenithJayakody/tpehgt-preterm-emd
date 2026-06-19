from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
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
)

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
    "contraction_imf1_selection": FEATURE_DIR
    / "imf_selection"
    / "tpehgt_contraction_imf1_features.csv",
    "contraction_imf2_selection": FEATURE_DIR
    / "imf_selection"
    / "tpehgt_contraction_imf2_features.csv",
    "contraction_imf3_selection": FEATURE_DIR
    / "imf_selection"
    / "tpehgt_contraction_imf3_features.csv",
    "contraction_imf4_selection": FEATURE_DIR
    / "imf_selection"
    / "tpehgt_contraction_imf4_features.csv",
    "fixed_3min_imf1_selection": FEATURE_DIR
    / "imf_selection"
    / "tpehgt_fixed_3min_imf1_features.csv",
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
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_SEED,
        ),
        "Naive Bayes": GaussianNB(),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            alpha=0.0005,
            max_iter=3000,
            early_stopping=True,
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
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan

        return float(average_precision_score(y_true, scores))
    except Exception:
        return np.nan


def evaluate_predictions(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    pred = (scores >= threshold).astype(int)

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


def make_meta(df_part: pd.DataFrame, record_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record": df_part[record_col].astype(str).values,
            "label": df_part["label"].astype(int).values,
        }
    )


def run_one_feature_file(name: str, csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"\nSkipping {name}: missing {csv_path}")
        return

    print("\n==============================")
    print(f"Running: {name}")
    print(f"CSV    : {csv_path}")
    print("==============================")

    df = pd.read_csv(csv_path)
    x, y, groups, record_col, feature_cols = get_feature_matrix(df)

    print(
        f"Rows: {len(df)} | Features: {x.shape[1]} | Records: {len(np.unique(groups))}"
    )
    print(f"Record column: {record_col}")

    print("Feature columns:")
    for c in feature_cols:
        print(f"  {c}")

    print("\nRecord-wise grouping check:")
    for record_id, g in df.groupby(record_col):
        print(f"  {record_id}: {len(g)} segments")

    models = get_models()

    fold_rows: List[Dict] = []
    segment_pred_rows: List[pd.DataFrame] = []
    record_pred_rows: List[pd.DataFrame] = []
    split_rows: List[Dict] = []

    for repeat in range(N_REPEATS):
        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED + repeat,
        )

        for fold, (train_idx, val_idx) in enumerate(cv.split(x, y, groups)):
            x_train = x.iloc[train_idx]
            x_val = x.iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            meta_train = make_meta(df.iloc[train_idx], record_col)
            meta_val = make_meta(df.iloc[val_idx], record_col)

            train_records = set(meta_train["record"])
            val_records = set(meta_val["record"])
            overlap = train_records.intersection(val_records)

            if overlap:
                raise RuntimeError(
                    f"Record leakage found in repeat={repeat}, fold={fold}: {overlap}"
                )

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

            for model_name, model in models.items():
                pipe = make_pipeline(model)

                try:
                    pipe.fit(x_train, y_train)

                    train_scores = predict_preterm_scores(pipe, x_train)
                    val_scores = predict_preterm_scores(pipe, x_val)

                    train_records_df = aggregate_to_record(meta_train, train_scores)
                    val_records_df = aggregate_to_record(meta_val, val_scores)

                    threshold = choose_threshold_from_training_records(
                        train_records_df["label"].to_numpy(),
                        train_records_df["score"].to_numpy(),
                    )

                    record_metrics = evaluate_predictions(
                        val_records_df["label"].to_numpy(),
                        val_records_df["score"].to_numpy(),
                        threshold,
                    )

                    segment_metrics = evaluate_predictions(y_val, val_scores, threshold)

                    row = {
                        "experiment": name,
                        "model": model_name,
                        "repeat": repeat,
                        "fold": fold,
                        "threshold": threshold,
                        "n_train_segments": len(train_idx),
                        "n_val_segments": len(val_idx),
                        "n_train_records": len(train_records_df),
                        "n_val_records": len(val_records_df),
                    }

                    for k, v in record_metrics.items():
                        row[f"record_{k}"] = v

                    for k, v in segment_metrics.items():
                        row[f"segment_{k}"] = v

                    fold_rows.append(row)

                    seg_pred = meta_val.copy()
                    seg_pred["experiment"] = name
                    seg_pred["model"] = model_name
                    seg_pred["repeat"] = repeat
                    seg_pred["fold"] = fold
                    seg_pred["score"] = val_scores
                    seg_pred["threshold"] = threshold
                    seg_pred["prediction"] = (val_scores >= threshold).astype(int)
                    segment_pred_rows.append(seg_pred)

                    rec_pred = val_records_df.copy()
                    rec_pred["experiment"] = name
                    rec_pred["model"] = model_name
                    rec_pred["repeat"] = repeat
                    rec_pred["fold"] = fold
                    rec_pred["threshold"] = threshold
                    rec_pred["prediction"] = (rec_pred["score"] >= threshold).astype(
                        int
                    )
                    record_pred_rows.append(rec_pred)

                except Exception as e:
                    print(
                        f"Failed: experiment={name}, repeat={repeat}, "
                        f"fold={fold}, model={model_name}: {e}"
                    )

    if not fold_rows:
        print(f"No results produced for {name}")
        return

    out_dir = RESULT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_results = pd.DataFrame(fold_rows)
    segment_predictions = pd.concat(segment_pred_rows, ignore_index=True)
    record_predictions = pd.concat(record_pred_rows, ignore_index=True)
    splits = pd.DataFrame(split_rows)

    fold_results.to_csv(out_dir / "fold_metrics.csv", index=False)
    segment_predictions.to_csv(out_dir / "segment_predictions.csv", index=False)
    record_predictions.to_csv(out_dir / "record_predictions.csv", index=False)
    splits.to_csv(out_dir / "splits.csv", index=False)

    with open(out_dir / "feature_columns.txt", "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(col + "\n")

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

    segment_metric_cols = [
        "segment_accuracy",
        "segment_precision",
        "segment_recall",
        "segment_specificity",
        "segment_sensitivity",
        "segment_f1",
        "segment_balanced_accuracy",
        "segment_mcc",
        "segment_roc_auc",
        "segment_pr_auc",
    ]

    summary_cols = record_metric_cols + segment_metric_cols

    summary = fold_results.groupby("model")[summary_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()

    summary.to_csv(out_dir / "summary_metrics.csv", index=False)

    print("\nRecord-level summary:")
    display_cols = [
        "model",
        "record_accuracy_mean",
        "record_f1_mean",
        "record_balanced_accuracy_mean",
        "record_mcc_mean",
        "record_roc_auc_mean",
        "record_pr_auc_mean",
    ]

    print(
        summary[display_cols].to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}",
        )
    )

    print(f"\nSaved outputs to: {out_dir}")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for name, csv_path in FEATURE_FILES.items():
        run_one_feature_file(name, csv_path)


if __name__ == "__main__":
    main()
