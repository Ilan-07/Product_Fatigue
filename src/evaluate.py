"""
evaluate.py — Rigorous, leakage-aware evaluation on the held-out temporal test set.

Design decisions
----------------
1. All metrics are computed exclusively on the temporal test set (never training data).

2. SHAP values are computed on the underlying estimator extracted from the
   imblearn Pipeline via pipeline.named_steps['clf'].  TreeExplainer cannot
   traverse Pipeline wrappers directly.

3. A "> 98% accuracy" warning is printed — not raised as an exception — so the
   pipeline still completes and you can inspect which features drove it.

4. A CV-vs-test F1 gap table is printed to surface overfitting separately from
   the leakage check.  A large positive gap (CV >> test) means overfit on CV;
   a large negative gap (test >> CV) is unusual and may signal test-set inflation.

5. ROC-AUC for multiclass uses OvR (one-vs-rest) macro averaging, which is the
   standard approach for imbalanced multiclass problems.
"""

import json
import logging
import os
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")          # non-interactive backend; safe for headless servers
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.preprocessing import label_binarize

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn(
        "shap not installed — SHAP plots will be skipped.  "
        "Install with: pip install shap", stacklevel=2
    )

logger = logging.getLogger(__name__)

# Any test accuracy above this triggers a printed leakage warning.
LEAKAGE_ACCURACY_THRESHOLD = 0.98


def _multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    n_classes = y_proba.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


def _expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        mask = (confidences >= lo) & (confidences < hi if hi < 1.0 else confidences <= hi)
        if not np.any(mask):
            continue
        acc_bin = correctness[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += (mask.mean()) * abs(acc_bin - conf_bin)
    return float(ece)


def _mean_abs_shap_vector(shap_vals: Any) -> np.ndarray:
    """
    Normalize SHAP outputs from different explainers / SHAP versions to a
    single 1-D mean-|SHAP| vector aligned with feature columns.

    Common formats:
      - list[n_classes] of (n_samples, n_features)
      - (n_samples, n_features)
      - (n_samples, n_features, n_classes)
      - (n_samples, n_classes, n_features)
    """
    if isinstance(shap_vals, list):
        arr = np.stack([np.asarray(v) for v in shap_vals], axis=0)
        # -> (n_classes, n_samples, n_features)
        return np.abs(arr).mean(axis=(0, 1))

    arr = np.asarray(shap_vals)

    if arr.ndim == 1:
        return np.abs(arr)

    if arr.ndim == 2:
        return np.abs(arr).mean(axis=0)

    if arr.ndim == 3:
        # Prefer the axis matching feature count semantics:
        # SHAP commonly returns either (samples, features, classes) or
        # (samples, classes, features). Collapse non-feature axes.
        if arr.shape[1] <= arr.shape[2]:
            # likely (samples, classes, features)
            return np.abs(arr).mean(axis=(0, 1))
        # likely (samples, features, classes)
        return np.abs(arr).mean(axis=(0, 2))

    raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")


def _apply_decision_policy(
    y_proba: np.ndarray,
    label_classes: np.ndarray,
    decision_threshold: float = 0.5,
    class_weights: dict[str, float] | None = None,
) -> np.ndarray:
    class_weights = class_weights or {}
    if y_proba.shape[1] == 2:
        return (y_proba[:, 1] >= decision_threshold).astype(int)

    weights = np.array(
        [float(class_weights.get(str(cls), 1.0)) for cls in label_classes],
        dtype=float,
    )
    adjusted = y_proba * weights.reshape(1, -1)
    return adjusted.argmax(axis=1)


# ---------------------------------------------------------------------------
# Pipeline introspection helper
# ---------------------------------------------------------------------------

def _extract_clf(pipeline: Any) -> Any:
    """
    Extract the underlying classifier from an imblearn/sklearn Pipeline.
    If the object is already a plain estimator, return it unchanged.
    """
    if hasattr(pipeline, "named_steps") and "clf" in pipeline.named_steps:
        return pipeline.named_steps["clf"]
    return pipeline


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_classifier(
    name: str,
    pipeline: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_classes: np.ndarray,
    cv_f1: float,
    feature_names: list[str],
    calibrated_clf: Any | None = None,
    decision_threshold: float = 0.5,
    class_weights: dict[str, float] | None = None,
    prefix: str = "",
    output_dir: str = "outputs",
) -> dict[str, Any]:
    """
    Evaluate a single classifier on the held-out test set and produce:
      - scalar metrics dict (returned)
      - confusion matrix PNG
      - ROC curve PNG (OvR, one per class)
      - SHAP summary bar PNG (if shap is installed and model is tree/linear)

    Parameters
    ----------
    name          : model name used in filenames and logs
    pipeline      : fitted imblearn Pipeline (SMOTE is bypassed at predict time)
    X_test        : preprocessed test features (already scaled by data_loader)
    y_test        : integer-encoded test labels
    label_classes : array of original class names from LabelEncoder.classes_
    cv_f1         : best CV f1_macro from GridSearchCV (for gap reporting)
    feature_names : list of feature column names matching X_test columns
    prefix        : prepended to all output filenames (e.g. "reviews_")
    output_dir    : directory for PNG and JSON output
    """
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(label_classes)

    raw_y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    y_proba = calibrated_clf.predict_proba(X_test) if calibrated_clf is not None else raw_y_proba
    if y_proba is not None:
        y_pred = _apply_decision_policy(
            y_proba,
            label_classes,
            decision_threshold=decision_threshold,
            class_weights=class_weights,
        )
    else:
        y_pred = pipeline.predict(X_test)

    acc      = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro",  zero_division=0)
    f1_per   = f1_score(y_test, y_pred, average=None,     zero_division=0).tolist()
    prec     = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec      = recall_score(y_test, y_pred, average="macro",  zero_division=0)
    bal_acc  = balanced_accuracy_score(y_test, y_pred)

    # ── ROC-AUC (OvR macro) ────────────────────────────────────────────────
    roc_auc: float | None = None
    raw_brier: float | None = None
    raw_ece: float | None = None
    calibrated_brier: float | None = None
    calibrated_ece: float | None = None
    if y_proba is not None:
        try:
            if n_classes == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                y_bin   = label_binarize(y_test, classes=list(range(n_classes)))
                roc_auc = roc_auc_score(
                    y_bin, y_proba, multi_class="ovr", average="macro"
                )
        except ValueError as exc:
            logger.warning(f"ROC-AUC failed for {name}: {exc}")
        if raw_y_proba is not None:
            raw_brier = _multiclass_brier_score(y_test, raw_y_proba)
            raw_ece = _expected_calibration_error(y_test, raw_y_proba)
        if calibrated_clf is not None:
            cal_proba = y_proba
            calibrated_brier = _multiclass_brier_score(y_test, cal_proba)
            calibrated_ece = _expected_calibration_error(y_test, cal_proba)

    # ── Leakage warning ────────────────────────────────────────────────────
    leakage_flag = bool(acc > LEAKAGE_ACCURACY_THRESHOLD)
    if leakage_flag:
        logger.warning(
            f"[*** LEAKAGE WARNING ***] {prefix}{name}: "
            f"test accuracy = {acc:.4f} > {LEAKAGE_ACCURACY_THRESHOLD:.0%}. "
            "Check for target-derived features or global normalisation leakage."
        )

    # CV-vs-test gap: positive means overfit on CV; negative means test > CV
    gap = cv_f1 - f1_macro
    true_dist = np.bincount(y_test, minlength=n_classes) / len(y_test)
    pred_dist = np.bincount(y_pred, minlength=n_classes) / len(y_pred)
    drift = float(np.abs(true_dist - pred_dist).sum() / 2.0)
    logger.info(
        f"{prefix}{name}  acc={acc:.4f}  f1_macro={f1_macro:.4f}  "
        f"cv_f1={cv_f1:.4f}  gap={gap:+.4f}  roc_auc={roc_auc}"
    )

    # ── Confusion matrix ────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_classes, yticklabels=label_classes, ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}confusion_matrix_{name}.png", dpi=100)
    plt.close(fig)

    # ── ROC curves ─────────────────────────────────────────────────────────
    if y_proba is not None:
        _plot_roc(y_test, y_proba, label_classes, name, prefix, output_dir)

    # ── SHAP summary ────────────────────────────────────────────────────────
    shap_top5: dict[str, float] | None = None
    if SHAP_AVAILABLE:
        clf = _extract_clf(pipeline)
        shap_top5 = _plot_shap(clf, X_test, feature_names, name, prefix, output_dir)

    return {
        "accuracy":         round(acc, 6),
        "f1_macro":         round(f1_macro, 6),
        "f1_per_class":     {
            str(label_classes[i]): round(v, 6) for i, v in enumerate(f1_per)
        },
        "precision_macro":  round(prec, 6),
        "recall_macro":     round(rec, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "roc_auc_ovr_macro": round(roc_auc, 6) if roc_auc is not None else None,
        "cv_f1_macro":      round(cv_f1, 6),
        "cv_test_gap":      round(gap, 6),
        "prediction_distribution": {
            str(label_classes[i]): round(float(pred_dist[i]), 6) for i in range(n_classes)
        },
        "true_distribution": {
            str(label_classes[i]): round(float(true_dist[i]), 6) for i in range(n_classes)
        },
        "prediction_distribution_drift_l1": round(drift, 6),
        "raw_brier_score": round(raw_brier, 6) if raw_brier is not None else None,
        "raw_ece": round(raw_ece, 6) if raw_ece is not None else None,
        "calibrated_brier_score": round(calibrated_brier, 6) if calibrated_brier is not None else None,
        "calibrated_ece": round(calibrated_ece, 6) if calibrated_ece is not None else None,
        "leakage_warning":  leakage_flag,
        "shap_top5":        shap_top5,
    }


# ---------------------------------------------------------------------------
# K-Means evaluation
# ---------------------------------------------------------------------------

def evaluate_kmeans(
    kmeans: KMeans,
    best_k: int,
    silhouette: float,
    X_test: np.ndarray,
    prefix: str = "",
    output_dir: str = "outputs",
) -> dict[str, Any]:
    """
    Visualise K-Means clusters on the test set using a PCA 2D projection.
    Also recomputes silhouette on the test set for a train/test consistency check.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_sample = min(5_000, X_test.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_test.shape[0], n_sample, replace=False)
    X_sample = X_test[idx]

    preds = kmeans.predict(X_sample)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], c=preds,
        cmap="tab10", s=8, alpha=0.5,
    )
    ax.set_title(f"K-Means (k={best_k}) — PCA projection (test set)")
    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}cluster_visualization.png", dpi=100)
    plt.close(fig)

    # Silhouette on test set — compare with training silhouette to detect
    # cluster collapse when the model sees unseen data
    test_sil: float | None = None
    if len(np.unique(preds)) > 1:
        try:
            test_sil = float(
                silhouette_score(
                    X_sample, preds,
                    sample_size=min(5_000, n_sample),
                    random_state=42,
                )
            )
        except Exception as exc:
            logger.warning(f"Test silhouette failed: {exc}")

    return {
        "best_k":            best_k,
        "train_silhouette":  round(silhouette, 6),
        "test_silhouette":   round(test_sil, 6) if test_sil is not None else None,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(summary_rows: list[dict[str, Any]]) -> None:
    """
    Print the final CV-vs-test gap table with a "Leakage Fixed?" column.

    Leakage Fixed? = YES  → test accuracy ≤ 0.98 (no deterministic reconstruction)
    Leakage Fixed? = NO   → test accuracy > 0.98 (model is still memorising the label)
    """
    header = (
        f"{'Modality':<12} {'Model':<24} "
        f"{'CV F1':>8} {'Test F1':>8} {'Gap':>8} {'Leakage Fixed?':>16}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for row in summary_rows:
        # leakage_warning=True  → acc > 0.98 → leakage NOT fixed → "NO  ***"
        # leakage_warning=False → acc ≤ 0.98 → leakage fixed     → "YES"
        fixed = "NO  ***" if row["leakage_warning"] else "YES"
        print(
            f"{row['modality']:<12} {row['model']:<24} "
            f"{row['cv_f1']:>8.4f} {row['test_f1']:>8.4f} "
            f"{row['gap']:>+8.4f} {fixed:>16}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict[str, Any], filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info(f"Metrics saved → {filepath}")


# ---------------------------------------------------------------------------
# Private plot helpers
# ---------------------------------------------------------------------------

def _plot_roc(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    label_classes: np.ndarray,
    name: str,
    prefix: str,
    output_dir: str,
) -> None:
    n_classes = len(label_classes)
    fig, ax = plt.subplots(figsize=(8, 6))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    else:
        y_bin = label_binarize(y_test, classes=list(range(n_classes)))
        for i, cls in enumerate(label_classes):
            if y_bin[:, i].sum() == 0:
                continue     # skip classes absent from the test set
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (OvR) — {name}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}roc_{name}.png", dpi=100)
    plt.close(fig)


def _plot_shap(
    clf: Any,
    X_test: np.ndarray,
    feature_names: list[str],
    name: str,
    prefix: str,
    output_dir: str,
) -> dict[str, float] | None:
    """
    Compute SHAP values and save a mean-|SHAP| bar chart.
    Sub-samples to 500 rows for speed.
    Returns a dict of {feature_name: mean_abs_shap} for the top 5 features,
    or None if SHAP computation fails.
    """
    try:
        n_shap = min(500, X_test.shape[0])
        rng = np.random.default_rng(42)
        idx = rng.choice(X_test.shape[0], n_shap, replace=False)
        X_shap = X_test[idx]

        if hasattr(clf, "feature_importances_"):
            # Tree models: exact, fast TreeExplainer
            explainer  = shap.TreeExplainer(clf)
            shap_vals  = explainer.shap_values(X_shap)
        elif hasattr(clf, "coef_"):
            # Linear models: LinearExplainer
            explainer  = shap.LinearExplainer(clf, X_shap)
            shap_vals  = explainer.shap_values(X_shap)
        else:
            # Fallback: slow KernelExplainer; limit to 100 rows
            X_bg = shap.sample(X_shap, 50, random_state=42)
            explainer = shap.KernelExplainer(clf.predict_proba, X_bg)
            shap_vals = explainer.shap_values(X_shap[:100])

        mean_abs = _mean_abs_shap_vector(shap_vals)

        # Top-5 for the returned dict
        top5_idx = np.argsort(mean_abs)[::-1][:5]
        top5 = {feature_names[i]: round(float(mean_abs[i]), 6) for i in top5_idx}

        # Bar chart: top 20 features
        top20_idx = np.argsort(mean_abs)[::-1][:20]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            [feature_names[i] for i in top20_idx][::-1],
            mean_abs[top20_idx][::-1],
        )
        ax.set_title(f"SHAP Mean |value| — {name} (top 20 features)")
        ax.set_xlabel("Mean |SHAP value|")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefix}shap_{name}.png", dpi=100)
        plt.close(fig)

        return top5

    except Exception as exc:
        logger.warning(f"SHAP failed for {name}: {exc}")
        return None
