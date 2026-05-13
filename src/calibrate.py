"""
calibrate.py — Post-hoc probability calibration for trained classifiers.

Why calibration?
----------------
Tree-based models (XGBoost, Random Forest) produce overconfident probabilities.
Even with correct class predictions and low CV-test gap, the raw probability
for the winning class often exceeds 0.97 for most samples.

Calibration maps raw probability scores to better reflect empirical
frequencies — if the model says p=0.9, roughly 90% of such cases should
actually belong to that class.

Method choice
-------------
Platt Scaling (method="sigmoid"):
  Parametric — fits σ(ax + b) on the raw score.
  Fast, works well when miscalibration is monotonic.
  Preferred for small calibration sets (< 500 samples per class).

Isotonic Regression (method="isotonic"):
  Non-parametric, more flexible.
  Can handle non-monotonic miscalibration.
  Requires more data (≥ 500 samples per class recommended).
  Falls back to sigmoid automatically when the calibration set is too small.

cv="prefit" design
------------------
We use cv="prefit" throughout: the base estimator is already fitted;
CalibratedClassifierCV only fits the calibration mapping (sigmoid or isotonic)
on the held-out calibration set (X_cal, y_cal).

main.py carves 20% of X_train as X_cal *before* calling train_all(), so X_cal
is strictly held-out relative to every fitted classifier.  This is the correct
approach — calibrating on training data would under-regularise the mapping.
"""

import os
import logging
import numpy as np
import joblib
from typing import Any, Dict, Optional, Tuple

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator          # sklearn 1.6+ replacement for cv="prefit"
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Minimum calibration-set samples per class for isotonic to be reliable.
# Below this we silently fall back to sigmoid.
_ISOTONIC_MIN_PER_CLASS = 50

OVERCONFIDENCE_THRESHOLD = 0.97

SEED = 42


# ---------------------------------------------------------------------------
# Core calibration
# ---------------------------------------------------------------------------

def calibrate_model(
    clf: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """
    Wrap an already-fitted classifier with post-hoc probability calibration.

    Parameters
    ----------
    clf    : fitted sklearn-compatible estimator extracted from an imblearn Pipeline
    X_cal  : calibration features — must NOT have been used to train clf
    y_cal  : calibration labels
    method : "sigmoid" (Platt scaling) or "isotonic"

    Returns
    -------
    Fitted CalibratedClassifierCV.

    Raises
    ------
    TypeError if clf does not implement predict_proba.
    """
    if not hasattr(clf, "predict_proba"):
        raise TypeError(
            f"{type(clf).__name__} does not support predict_proba. "
            "Calibration requires a probabilistic classifier."
        )

    n_samples = len(y_cal)
    n_classes = len(np.unique(y_cal))
    effective_method = method

    # Isotonic needs enough samples to fit a reliable non-parametric mapping
    min_needed = _ISOTONIC_MIN_PER_CLASS * n_classes
    if method == "isotonic" and n_samples < min_needed:
        logger.warning(
            f"  Only {n_samples} calibration samples for {n_classes} class(es) "
            f"(need ≥ {min_needed} for isotonic) — falling back to sigmoid."
        )
        effective_method = "sigmoid"

    logger.info(
        f"  Fitting calibration (method='{effective_method}') "
        f"on {n_samples} samples, {n_classes} classes ..."
    )
    # FrozenEstimator prevents CalibratedClassifierCV from cloning/refitting clf.
    # This is the sklearn 1.6+ replacement for the removed cv="prefit" string.
    cal = CalibratedClassifierCV(
        estimator=FrozenEstimator(clf), method=effective_method
    )
    cal.fit(X_cal, y_cal)
    return cal


def calibrate_all(
    results: Dict[str, Any],
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    Calibrate xgboost, random_forest, and logistic_regression from a train_all()
    results dict.  For each model, two variants are produced:
      "sigmoid"  — Platt scaling
      "isotonic" — isotonic regression (falls back to sigmoid if X_cal is small)

    Parameters
    ----------
    results : dict returned by train_all() — keys map to {"pipeline": ..., "cv_f1": ...}
    X_cal   : held-out calibration features
    y_cal   : held-out calibration labels

    Returns
    -------
    {
      "xgboost":             {"sigmoid": CalibratedClassifierCV, "isotonic": ...},
      "random_forest":       {"sigmoid": ...,                    "isotonic": ...},
      "logistic_regression": {"sigmoid": ...,                    "isotonic": ...},
    }
    None values indicate a failed calibration attempt for that variant.
    """
    calibrated: Dict[str, Dict[str, Any]] = {}

    for name in ("xgboost", "random_forest", "logistic_regression"):
        if name not in results:
            logger.warning(f"'{name}' not found in results — skipping calibration.")
            continue

        pipeline = results[name]["pipeline"]

        # Extract the raw classifier; SMOTE step is a no-op at predict time
        # but CalibratedClassifierCV expects a plain estimator, not a Pipeline.
        clf = (
            pipeline.named_steps["clf"]
            if hasattr(pipeline, "named_steps") and "clf" in pipeline.named_steps
            else pipeline
        )

        logger.info(f"Calibrating {name} ...")
        entry: Dict[str, Any] = {}
        for method in ("sigmoid", "isotonic"):
            try:
                entry[method] = calibrate_model(clf, X_cal, y_cal, method=method)
            except Exception as exc:
                logger.error(f"  {name}/{method} calibration failed: {exc}")
                entry[method] = None

        calibrated[name] = entry

    return calibrated


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_calibrated_models(
    calibrated: Dict[str, Dict[str, Any]],
    output_dir: str = "models",
    prefix: str = "",
) -> None:
    """Save each calibrated variant to {output_dir}/{prefix}{name}_calibrated_{method}.pkl"""
    os.makedirs(output_dir, exist_ok=True)
    for name, methods in calibrated.items():
        for method, cal_clf in methods.items():
            if cal_clf is None:
                continue
            path = os.path.join(
                output_dir, f"{prefix}{name}_calibrated_{method}.pkl"
            )
            joblib.dump(cal_clf, path)
            logger.info(f"Saved calibrated {name}/{method} → {path}")


def load_calibrated_model(
    modality: str,
    model_name: str,
    method: str = "isotonic",
    models_dir: str = "models",
) -> Optional[Any]:
    """
    Load a calibrated model from disk.  Returns None if the file does not exist
    (e.g. main.py has not been run yet after this change).

    Preference order: the requested method first, then "sigmoid" as fallback.
    """
    for m in (method, "sigmoid"):
        path = os.path.join(
            models_dir, f"{modality}_{model_name}_calibrated_{m}.pkl"
        )
        if os.path.exists(path):
            logger.info(f"Loading calibrated model (method={m}) from {path}")
            return joblib.load(path)
    return None


# ---------------------------------------------------------------------------
# Decision threshold optimisation
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    clf: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> Tuple[float, float]:
    """
    Find the classification threshold that maximises F1 macro on the
    calibration set.

    For binary classifiers the default 0.5 threshold is optimal only when
    classes are balanced.  With an 87.7%/12.3% split the optimal threshold
    for F1 macro is typically much lower — predicting "fatigue" at a lower
    probability raises minority-class recall without catastrophically
    reducing precision.

    For multiclass problems argmax is already optimal (returns 0.5 sentinel).

    Parameters
    ----------
    clf   : fitted probabilistic estimator (pipeline or calibrated)
    X_cal : calibration features (held-out from training)
    y_cal : calibration labels

    Returns
    -------
    (best_threshold, best_f1_macro) — both floats.
    threshold=0.5 is returned for multiclass (no-op at inference).
    """
    probas = clf.predict_proba(X_cal)
    n_classes = probas.shape[1]

    if n_classes != 2:
        default_f1 = float(f1_score(y_cal, probas.argmax(axis=1),
                                    average="macro", zero_division=0))
        return 0.5, default_f1

    # Binary: sweep P(positive class) threshold over [0.05, 0.95]
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (probas[:, 1] >= t).astype(int)
        score = float(f1_score(y_cal, preds, average="macro", zero_division=0))
        if score > best_f1:
            best_f1, best_t = score, float(t)

    logger.info(
        f"  Optimal threshold: {best_t:.3f}  (cal F1 macro: {best_f1:.4f}, "
        f"vs default-0.5 F1: "
        f"{f1_score(y_cal, (probas[:,1]>=0.5).astype(int), average='macro', zero_division=0):.4f})"
    )
    return best_t, best_f1


def find_optimal_class_weights(
    clf: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    label_classes: np.ndarray,
) -> Tuple[Dict[str, float], float]:
    """
    For multiclass problems, search simple per-class probability multipliers
    that improve macro F1 on the calibration set. Healthy stays at 1.0 and the
    non-healthy classes get tuned. For binary, returns identity weights.
    """
    probas = clf.predict_proba(X_cal)
    n_classes = probas.shape[1]
    if n_classes <= 2:
        return {str(c): 1.0 for c in label_classes}, float(
            f1_score(y_cal, probas.argmax(axis=1), average="macro", zero_division=0)
        )

    base_preds = probas.argmax(axis=1)
    best_f1 = float(f1_score(y_cal, base_preds, average="macro", zero_division=0))
    best_weights = {str(c): 1.0 for c in label_classes}
    healthy_idx = next(
        (idx for idx, cls in enumerate(label_classes) if str(cls).lower() == "healthy"),
        None,
    )
    candidate_indices = [
        idx for idx, cls in enumerate(label_classes)
        if str(cls).lower() != "healthy"
    ]
    if healthy_idx is None or len(candidate_indices) < 1:
        return best_weights, best_f1

    healthy_grid = np.array([0.35, 0.5, 0.65, 0.8, 1.0])
    fatigue_grid = np.array([1.0, 1.25, 1.5, 2.0, 2.5, 3.0])

    for healthy_weight in healthy_grid:
        for w1 in fatigue_grid:
            for w2 in fatigue_grid:
                weights = np.ones(n_classes, dtype=float)
                weights[healthy_idx] = healthy_weight
                weights[candidate_indices[0]] = w1
                if len(candidate_indices) > 1:
                    weights[candidate_indices[1]] = w2
                adjusted = probas * weights
                preds = adjusted.argmax(axis=1)
                score = float(f1_score(y_cal, preds, average="macro", zero_division=0))
                if score > best_f1:
                    best_f1 = score
                    best_weights = {
                        str(label_classes[idx]): float(weights[idx]) for idx in range(n_classes)
                    }

    logger.info(f"  Optimal class weights: {best_weights}  (cal F1 macro: {best_f1:.4f})")
    return best_weights, best_f1


# ---------------------------------------------------------------------------
# Uncertainty / review-needed flag (Problem 9 fix)
# ---------------------------------------------------------------------------

# Default confidence threshold below which predictions are flagged as uncertain
UNCERTAINTY_THRESHOLD = 0.60

# Margin threshold: if top two classes are within this margin, flag as uncertain
MARGIN_THRESHOLD = 0.15


def compute_uncertainty_flag(
    probabilities: np.ndarray,
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> Dict[str, Any]:
    """
    Compute uncertainty/review-needed flags for predictions.

    A prediction is flagged as uncertain when:
    1. Max probability is below uncertainty_threshold, OR
    2. The margin between top-2 classes is below margin_threshold

    Parameters
    ----------
    probabilities        : (n_samples, n_classes) probability array
    uncertainty_threshold : max-prob threshold for uncertainty
    margin_threshold      : margin between top-2 classes

    Returns
    -------
    {
      "uncertainty_flag": bool array,
      "confidence_band": str array ("high", "medium", "low"),
      "review_needed": bool array,
      "max_probability": float array,
      "margin": float array,
    }
    """
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)

    n_samples = probabilities.shape[0]
    max_prob = probabilities.max(axis=1)

    # Sort probabilities descending for margin computation
    sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1] if probabilities.shape[1] > 1 else np.ones(n_samples)

    # Uncertainty flags
    low_confidence = max_prob < uncertainty_threshold
    narrow_margin = margin < margin_threshold
    uncertainty_flag = low_confidence | narrow_margin

    # Confidence bands
    confidence_band = np.where(
        max_prob >= 0.80, "high",
        np.where(max_prob >= 0.60, "medium", "low")
    )

    # Review needed: uncertain predictions on non-healthy classes
    review_needed = uncertainty_flag.copy()

    return {
        "uncertainty_flag": uncertainty_flag,
        "confidence_band": confidence_band,
        "review_needed": review_needed,
        "max_probability": max_prob,
        "margin": margin,
    }


def compute_reliability_plot_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute data for a reliability (calibration) plot.

    Returns
    -------
    {
      "bin_edges": array of bin boundaries,
      "mean_predicted": mean predicted probability per bin,
      "fraction_positive": actual fraction of positives per bin,
      "bin_counts": number of samples per bin,
    }
    """
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_predicted = []
    fraction_positive = []
    bin_counts = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi < 1.0:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)

        if mask.sum() == 0:
            mean_predicted.append(np.nan)
            fraction_positive.append(np.nan)
            bin_counts.append(0)
        else:
            mean_predicted.append(float(confidences[mask].mean()))
            fraction_positive.append(float(correctness[mask].mean()))
            bin_counts.append(int(mask.sum()))

    return {
        "bin_edges": bins,
        "mean_predicted": np.array(mean_predicted),
        "fraction_positive": np.array(fraction_positive),
        "bin_counts": np.array(bin_counts),
    }


# ---------------------------------------------------------------------------
# Calibration set splitting utility
# ---------------------------------------------------------------------------

def split_calibration_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cal_frac: float = 0.20,
) -> tuple:
    """
    Carve a stratified calibration set from X_train.

    Returns (X_train_fit, X_cal, y_train_fit, y_cal).

    The returned X_train_fit is used to train the model; X_cal is held
    out exclusively for calibration.  Stratification preserves class ratios.
    """
    return train_test_split(
        X_train, y_train,
        test_size=cal_frac,
        random_state=SEED,
        stratify=y_train,
    )


# ---------------------------------------------------------------------------
# Prediction comparison
# ---------------------------------------------------------------------------

def predict_with_calibration(
    pipeline: Any,
    calibrated_clf: Optional[Any],
    X_scaled: np.ndarray,
    label_classes: np.ndarray,
    threshold: float = OVERCONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Run one prediction through the raw pipeline AND the calibrated classifier,
    returning a structured comparison dict.

    Parameters
    ----------
    pipeline        : fitted imblearn Pipeline (provides raw probabilities)
    calibrated_clf  : fitted CalibratedClassifierCV; None → calibration skipped
    X_scaled        : (1 × n_features) scaled input array
    label_classes   : class name strings from LabelEncoder.classes_
    threshold       : overconfidence warning level (default 0.97)

    Returns
    -------
    {
      "raw_proba":             {class_name: prob},
      "raw_confidence":        float,
      "raw_pred_class":        str,
      "calibrated_proba":      {class_name: prob} | None,
      "calibrated_confidence": float | None,
      "calibrated_pred_class": str | None,
      "adjustment":            float | None,   # calibrated_conf - raw_conf
      "prediction_changed":    bool | None,
      "warnings":              [str],
    }
    """
    result_warnings = []

    # ── Raw prediction ──────────────────────────────────────────────────────
    raw_proba = pipeline.predict_proba(X_scaled)[0]
    raw_idx   = int(np.argmax(raw_proba))
    raw_conf  = float(raw_proba[raw_idx])
    raw_class = str(label_classes[raw_idx])

    out: Dict[str, Any] = {
        "raw_proba":             {str(label_classes[i]): round(float(p), 6)
                                  for i, p in enumerate(raw_proba)},
        "raw_confidence":        round(raw_conf, 6),
        "raw_pred_class":        raw_class,
        "calibrated_proba":      None,
        "calibrated_confidence": None,
        "calibrated_pred_class": None,
        "adjustment":            None,
        "prediction_changed":    None,
        "warnings":              result_warnings,
    }

    if calibrated_clf is None:
        if raw_conf > threshold:
            result_warnings.append(
                "⚠️ Raw confidence still high — check for leakage or overfitting"
            )
        return out

    # ── Calibrated prediction ───────────────────────────────────────────────
    cal_proba = calibrated_clf.predict_proba(X_scaled)[0]
    cal_idx   = int(np.argmax(cal_proba))
    cal_conf  = float(cal_proba[cal_idx])
    cal_class = str(label_classes[cal_idx])

    adjustment = cal_conf - raw_conf

    out.update({
        "calibrated_proba":      {str(label_classes[i]): round(float(p), 6)
                                   for i, p in enumerate(cal_proba)},
        "calibrated_confidence": round(cal_conf, 6),
        "calibrated_pred_class": cal_class,
        "adjustment":            round(adjustment, 6),
        "prediction_changed":    (raw_class != cal_class),
    })

    # ── Safety checks ────────────────────────────────────────────────────────
    if raw_class != cal_class:
        result_warnings.append(
            "⚠️ Calibration changed class prediction — check dataset consistency"
        )

    if cal_conf > threshold:
        result_warnings.append(
            "⚠️ Calibrated confidence still high — check for leakage or overfitting"
        )

    return out
