from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from config import FEATURE_DIR, PLOT_DIR, RESULT_DIR, TOP_MODELS_FOR_ROC, FIG_DPI
from paper_style import (
    COLORS,
    PLOS_DOUBLE,
    PLOS_ONE_HALF,
    PLOS_SINGLE,
    panel_label,
    save_figure,
    set_paper_style,
)

set_paper_style()


METRICS_FOR_TABLES = [
    "accuracy",
    "f1",
    "balanced_accuracy",
    "mcc",
    "roc_auc",
    "pr_auc",
]


MODEL_COLORS = {
    "CatBoost": COLORS["catboost"],
    "Random Forest": COLORS["rf"],
    "Gradient Boosting": COLORS["gb"],
    "MLP": COLORS["mlp"],
    "SVM": COLORS["svm"],
    "QDA": "#555555",
    "Logistic Regression": "#777777",
    "Decision Tree": "#999999",
    "Naive Bayes": "#333333",
}


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


def find_result_dirs() -> List[Path]:
    return sorted([p for p in Path(RESULT_DIR).glob("*") if p.is_dir()])


def experiment_label(experiment: str) -> str:
    label = experiment
    label = label.replace("fixed_3min", "Fixed 3-min")
    label = label.replace("contraction", "Contraction")
    label = label.replace("time_domain", "Time-domain")
    label = label.replace("imf", "IMF")
    label = label.replace("_selection", " selection")
    label = label.replace("_", " ")
    return label


def read_summary(result_dir: Path) -> pd.DataFrame | None:
    path = result_dir / "summary_metrics.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


def read_record_predictions(result_dir: Path) -> pd.DataFrame | None:
    path = result_dir / "record_predictions.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


def read_feature_file_for_experiment(experiment: str) -> pd.DataFrame | None:
    mapping = {
        "contraction_imf1": FEATURE_DIR / "tpehgt_contraction_imf1_features.csv",
        "fixed_3min_imf1": FEATURE_DIR / "tpehgt_fixed_3min_imf1_features.csv",
        "contraction_time_domain": FEATURE_DIR / "tpehgt_contraction_time_domain_features.csv",
        "fixed_3min_time_domain": FEATURE_DIR / "tpehgt_fixed_3min_time_domain_features.csv",
    }

    path = mapping.get(experiment)

    if path is None or not path.exists():
        return None

    return pd.read_csv(path)


def best_model_from_summary(summary: pd.DataFrame, metric: str = "record_accuracy_mean") -> str:
    return str(summary.sort_values(metric, ascending=False).iloc[0]["model"])


def make_consensus_predictions(record_predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Produce one prediction per recording across repeated CV.

    The prediction is majority vote over repeated out-of-fold predictions.
    The score is mean out-of-fold score. This makes the confusion matrix sum to
    the number of recordings, usually 26.
    """
    df = record_predictions[record_predictions["model"] == model].copy()

    if df.empty:
        return df

    rows = []

    for record, g in df.groupby("record"):
        label = int(g["label"].iloc[0])
        pred_mean = float(g["prediction"].mean())
        pred = int(pred_mean >= 0.5)
        score = float(g["score"].mean())

        rows.append({
            "record": record,
            "label": label,
            "prediction": pred,
            "score": score,
            "n_votes": len(g),
        })

    return pd.DataFrame(rows)


def plot_summary_metric(summary: pd.DataFrame, experiment: str, out_dir: Path, metric: str) -> None:
    mean_col = f"record_{metric}_mean"

    if mean_col not in summary.columns:
        return

    df = summary.sort_values(mean_col, ascending=False).copy()

    fig, ax = plt.subplots(figsize=(PLOS_DOUBLE, 3.6))

    x = np.arange(len(df))
    colors = [MODEL_COLORS.get(m, "#555555") for m in df["model"]]
    vals = df[mean_col].to_numpy()

    ax.bar(x, vals, color=colors, edgecolor="0.25", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(experiment_label(experiment))

    for xi, yi in zip(x, vals):
        ax.text(xi, yi + 0.015, f"{yi:.2f}", ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    save_figure(fig, out_dir / f"summary_{metric}", dpi=FIG_DPI)
    plt.close(fig)


def plot_roc_pr_for_experiment(record_predictions: pd.DataFrame, experiment: str, out_dir: Path) -> None:
    """
    Pooled out-of-fold ROC/PR curves.
    No SD bands are shown.
    """
    models = [m for m in TOP_MODELS_FOR_ROC if m in set(record_predictions["model"])]
    if not models:
        models = sorted(record_predictions["model"].unique())[:5]

    fig, axes = plt.subplots(1, 2, figsize=(PLOS_ONE_HALF, 3.0))

    panel_label(axes[0], "A")
    panel_label(axes[1], "B")

    for model in models:
        g = record_predictions[record_predictions["model"] == model]
        y_true = g["label"].astype(int).to_numpy()
        scores = g["score"].astype(float).to_numpy()

        if len(np.unique(y_true)) < 2:
            continue

        color = MODEL_COLORS.get(model, "#555555")

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)

        axes[0].plot(fpr, tpr, color=color, lw=1.2, label=f"{model} ({roc_auc:.3f})")
        axes[1].plot(recall, precision, color=color, lw=1.2, label=f"{model} ({pr_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], color="0.6", lw=0.7, ls="--")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC")
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall")
    axes[1].legend(frameon=False, loc="lower left")

    fig.suptitle(experiment_label(experiment), y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir / "roc_pr_curves", dpi=FIG_DPI)
    plt.close(fig)


def plot_consensus_confusion_matrix(
    consensus: pd.DataFrame,
    title: str,
    out_path: Path,
    cmap: str = "Blues",
) -> None:
    if consensus.empty:
        return

    y_true = consensus["label"].astype(int).to_numpy()
    y_pred = consensus["prediction"].astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(PLOS_SINGLE * 0.75, 3.1))

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    norm = cm / row_sums

    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            color = "white" if norm[i, j] > 0.55 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({norm[i, j]:.2f})",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Term", "Preterm"])
    ax.set_yticklabels(["Term", "Preterm"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, out_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_paper_confusion_matrices() -> None:
    """
    Create a paper-ready two-panel confusion matrix using one consensus
    prediction per recording. Counts sum to 26 for each panel.
    """
    experiments = ["contraction_imf1", "fixed_3min_imf1"]
    result_dirs = {p.name: p for p in find_result_dirs()}

    if not all(e in result_dirs for e in experiments):
        return

    data = []

    for exp in experiments:
        summary = read_summary(result_dirs[exp])
        preds = read_record_predictions(result_dirs[exp])

        if summary is None or preds is None:
            return

        model = best_model_from_summary(summary, metric="record_accuracy_mean")
        consensus = make_consensus_predictions(preds, model)
        cm = confusion_matrix(
            consensus["label"].astype(int),
            consensus["prediction"].astype(int),
            labels=[0, 1],
        )
        data.append((exp, model, cm))

    fig, axes = plt.subplots(1, 2, figsize=(PLOS_ONE_HALF, 3.0))
    cmaps = ["Oranges", "Blues"]

    for ax, (exp, model, cm), cmap, letter in zip(axes, data, cmaps, ["A", "B"]):
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        norm = cm / row_sums

        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)

        for i in range(2):
            for j in range(2):
                color = "white" if norm[i, j] > 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({norm[i, j]:.2f})",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Term", "Preterm"])
        ax.set_yticklabels(["Term", "Preterm"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{experiment_label(exp)}\n{model}")
        panel_label(ax, letter)

    fig.tight_layout()
    out_dir = PLOT_DIR / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir / "confusion_matrices_26_record_consensus", dpi=FIG_DPI)
    plt.close(fig)


def plot_best_model_comparison() -> None:
    """
    Compare best model for:
    contraction time-domain, contraction IMF1,
    fixed time-domain, fixed IMF1.
    """
    experiments = [
        "contraction_time_domain",
        "contraction_imf1",
        "fixed_3min_time_domain",
        "fixed_3min_imf1",
    ]

    result_dirs = {p.name: p for p in find_result_dirs()}
    rows = []

    for exp in experiments:
        if exp not in result_dirs:
            continue

        summary = read_summary(result_dirs[exp])
        if summary is None:
            continue

        best = summary.sort_values("record_accuracy_mean", ascending=False).iloc[0]
        rows.append({
            "experiment": exp,
            "label": experiment_label(exp),
            "model": best["model"],
            "Accuracy": best["record_accuracy_mean"],
            "F1": best["record_f1_mean"],
            "Balanced Acc.": best["record_balanced_accuracy_mean"],
            "ROC-AUC": best["record_roc_auc_mean"],
            "PR-AUC": best["record_pr_auc_mean"],
        })

    if len(rows) < 2:
        return

    df = pd.DataFrame(rows)
    metrics = ["Accuracy", "F1", "Balanced Acc.", "ROC-AUC", "PR-AUC"]

    fig, ax = plt.subplots(figsize=(PLOS_DOUBLE, 3.6))

    x = np.arange(len(metrics))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(df))
    colors = [COLORS["contraction"], COLORS["imf1"], COLORS["fixed"], COLORS["imf2"]]

    for i, (_, row) in enumerate(df.iterrows()):
        vals = [float(row[m]) for m in metrics]
        ax.bar(
            x + offsets[i],
            vals,
            width=width,
            color=colors[i % len(colors)],
            edgecolor="0.25",
            linewidth=0.5,
            label=f"{row['label']} ({row['model']})",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean recording-level value")
    ax.set_title("Best configurations")
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.20))

    fig.tight_layout()
    out_dir = PLOT_DIR / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir / "best_model_comparison", dpi=FIG_DPI)
    plt.close(fig)


def plot_imf_selection_summary() -> None:
    """
    Plot best accuracy/F1/ROC/PR for IMF1--IMF4 for contraction and fixed windows.
    """
    result_dirs = {p.name: p for p in find_result_dirs()}
    modes = ["contraction", "fixed_3min"]
    metrics = [
        ("record_accuracy_mean", "Accuracy"),
        ("record_f1_mean", "F1"),
        ("record_roc_auc_mean", "ROC-AUC"),
        ("record_pr_auc_mean", "PR-AUC"),
    ]

    for mode in modes:
        rows = []

        for i in range(1, 5):
            exp = f"{mode}_imf{i}_selection"
            if exp not in result_dirs:
                continue

            summary = read_summary(result_dirs[exp])
            if summary is None:
                continue

            best = summary.sort_values("record_accuracy_mean", ascending=False).iloc[0]
            row = {"IMF": f"IMF{i}", "model": best["model"]}
            for col, label in metrics:
                row[label] = best[col]
            rows.append(row)

        if not rows:
            continue

        df = pd.DataFrame(rows)

        fig, axes = plt.subplots(1, len(metrics), figsize=(PLOS_DOUBLE, 2.5), sharey=True)

        for ax, (col_label, metric_label), letter in zip(axes, metrics, "ABCD"):
            vals = df[metric_label].astype(float).to_numpy()
            bars = ax.bar(
                np.arange(len(df)),
                vals,
                color=[COLORS["imf1"], COLORS["imf2"], COLORS["imf3"], COLORS["imf4"]][:len(df)],
                edgecolor="0.25",
                linewidth=0.5,
            )
            ax.set_xticks(np.arange(len(df)))
            ax.set_xticklabels(df["IMF"])
            ax.set_ylim(0, 1.05)
            ax.set_title(metric_label)
            panel_label(ax, letter)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6)

        axes[0].set_ylabel("Mean recording-level value")
        fig.suptitle(experiment_label(mode) + " IMF selection", y=1.02)
        fig.tight_layout()

        out_dir = PLOT_DIR / "paper"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_dir / f"{mode}_imf_selection", dpi=FIG_DPI)
        plt.close(fig)


def plot_feature_distributions() -> None:
    """
    Generate feature distribution candidates from fixed_3min_imf1 features.
    A paper-ready 6-panel figure uses the six largest absolute Cohen's d features.
    Individual feature candidates are saved for all features.
    """
    df = read_feature_file_for_experiment("fixed_3min_imf1")

    if df is None:
        return

    feature_cols = [
        c for c in df.columns
        if c not in {
            "mode", "feature_source", "record", "label", "segment_id",
            "start_sample", "end_sample", "start_sec", "end_sec", "imf",
            "name", "start", "end", "mother_id",
        }
    ]

    # Record-level mean feature values.
    rec = df.groupby(["record", "label"], as_index=False)[feature_cols].mean()

    term = rec[rec["label"] == 0]
    preterm = rec[rec["label"] == 1]

    scores = []
    for col in feature_cols:
        a = term[col].dropna().to_numpy()
        b = preterm[col].dropna().to_numpy()

        if len(a) < 2 or len(b) < 2:
            continue

        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = (b.mean() - a.mean()) / (pooled + 1e-12)
        scores.append((col, abs(d), d))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_cols = [s[0] for s in scores[:6]]

    out_dir = PLOT_DIR / "paper"
    cand_dir = PLOT_DIR / "feature_distribution_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)

    def friendly(col: str) -> str:
        return col.replace("ehg", "EHG").replace("_", " ")

    # Main 6-panel figure.
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(PLOS_DOUBLE, 4.4))
    axes = axes.ravel()

    for ax, col, letter in zip(axes, top_cols, "ABCDEF"):
        a = term[col].dropna().to_numpy()
        b = preterm[col].dropna().to_numpy()

        parts = ax.violinplot(
            [a, b],
            positions=[0, 1],
            widths=0.72,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for patch, color in zip(parts["bodies"], [COLORS["term"], COLORS["preterm"]]):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.4)
            patch.set_alpha(0.65)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(0.8)

        rng = np.random.default_rng(0)
        ax.scatter(rng.normal(0, 0.04, len(a)), a, s=8, color="black", alpha=0.55, lw=0)
        ax.scatter(rng.normal(1, 0.04, len(b)), b, s=8, color="black", alpha=0.55, lw=0)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Term", "Preterm"])
        ax.set_title(friendly(col))
        panel_label(ax, letter)

    fig.tight_layout()
    save_figure(fig, out_dir / "feature_distributions_top6", dpi=FIG_DPI)
    plt.close(fig)

    # Individual candidate plots for all features.
    for col, _, _ in scores:
        a = term[col].dropna().to_numpy()
        b = preterm[col].dropna().to_numpy()
        fig, ax = plt.subplots(figsize=(PLOS_SINGLE * 0.75, 3.0))
        parts = ax.violinplot(
            [a, b],
            positions=[0, 1],
            widths=0.72,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for patch, color in zip(parts["bodies"], [COLORS["term"], COLORS["preterm"]]):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.4)
            patch.set_alpha(0.65)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(0.8)
        rng = np.random.default_rng(0)
        ax.scatter(rng.normal(0, 0.04, len(a)), a, s=8, color="black", alpha=0.55, lw=0)
        ax.scatter(rng.normal(1, 0.04, len(b)), b, s=8, color="black", alpha=0.55, lw=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Term", "Preterm"])
        ax.set_title(friendly(col))
        fig.tight_layout()
        save_figure(fig, cand_dir / safe_name(col), dpi=FIG_DPI)
        plt.close(fig)


def make_plots_for_experiment(result_dir: Path) -> None:
    experiment = result_dir.name
    out_dir = PLOT_DIR / experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = read_summary(result_dir)
    preds = read_record_predictions(result_dir)

    if summary is None or preds is None:
        return

    for metric in ["accuracy", "f1", "balanced_accuracy", "mcc", "roc_auc", "pr_auc"]:
        plot_summary_metric(summary, experiment, out_dir, metric)

    plot_roc_pr_for_experiment(preds, experiment, out_dir)

    # Consensus 26-record confusion matrix for the best accuracy model.
    model = best_model_from_summary(summary, metric="record_accuracy_mean")
    consensus = make_consensus_predictions(preds, model)

    plot_consensus_confusion_matrix(
        consensus,
        title=f"{experiment_label(experiment)}\n{model}",
        out_path=out_dir / f"confusion_consensus_{safe_name(model)}",
        cmap="Blues" if "fixed" in experiment else "Oranges",
    )


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    result_dirs = find_result_dirs()

    if not result_dirs:
        print("No result folders found. Run classify_groupwise_cv.py first.")
        return

    for result_dir in result_dirs:
        make_plots_for_experiment(result_dir)

    plot_paper_confusion_matrices()
    plot_best_model_comparison()
    plot_imf_selection_summary()
    plot_feature_distributions()

    print(f"Saved plots to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
