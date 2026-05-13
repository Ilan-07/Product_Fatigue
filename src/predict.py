"""
predict.py — Reliable inference CLI with SHAP explanations.

Usage
-----
# Pass features as a JSON dict (keys = feature names, values = raw numbers):
python src/predict.py \\
    --modality reviews \\
    --input '{"sentiment_mean": 0.6, "sentiment_std": 0.2, "review_count": 15, ...}'

# Or pass a single-row CSV (headers must be feature names):
python src/predict.py --modality usage --input_csv path/to/row.csv

# Choose a different model:
python src/predict.py --modality sales --model random_forest --input '{...}'

Design decisions
----------------
- The scaler stored in artifacts is used via .transform() only — never refit.
  Refitting at inference time would change the feature scale relative to training.

- Feature alignment: missing features are filled with 0.0 (the scaled mean),
  extra features are dropped.  This is safer than padding with zeros pre-scale
  because 0.0 post-scale corresponds to the training distribution mean.

- SMOTE is part of the pipeline but is a no-op at predict/predict_proba time —
  imblearn Pipelines pass through to the classifier step during inference.

- SHAP explanation uses the underlying classifier extracted from the Pipeline,
  because TreeExplainer cannot traverse Pipeline wrappers.

- Overconfidence threshold is 0.97 by default (configurable via --threshold).
"""

import argparse
import json
import logging
import os
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `from src.calibrate` works whether
# the script is run as `python src/predict.py` or `python -m src.predict`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Calibrate module — loaded lazily so predict.py works even without it
try:
    from src.calibrate import load_calibrated_model, predict_with_calibration
    CALIBRATE_AVAILABLE = True
except ImportError:
    CALIBRATE_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

MODELS_DIR = "models"
OVERCONFIDENCE_THRESHOLD = 0.97
SUPPORTED_MODALITIES = ["reviews", "sales", "usage"]
SUPPORTED_MODELS     = ["xgboost", "random_forest", "logistic_regression"]
MIN_COMPLETENESS_THRESHOLD = 0.65
REVIEW_HISTORY_FEATURES = {
    "roll3_mean_sentiment_mean",
    "sentiment_mean_vs_trailing_mean",
    "roll3_mean_review_count",
    "review_count_vs_trailing_mean",
    "roll3_mean_score_median",
    "score_median_vs_trailing_mean",
    "roll3_mean_reviewer_diversity_change",
    "reviewer_diversity_change_vs_trailing_mean",
    "safe_sentiment_change",
}


def _lifecycle_stage_from_age(product_age_months: float) -> str:
    if product_age_months < 3:
        return "introduction"
    if product_age_months < 12:
        return "growth"
    if product_age_months < 24:
        return "maturity"
    return "decline"


def _mean_abs_shap_vector(shap_vals: Any) -> np.ndarray:
    """
    Normalize SHAP outputs from different explainers / SHAP versions to a
    single 1-D mean-|SHAP| vector aligned with feature columns.
    """
    if isinstance(shap_vals, list):
        arr = np.stack([np.asarray(v) for v in shap_vals], axis=0)
        return np.abs(arr).mean(axis=(0, 1))

    arr = np.asarray(shap_vals)

    if arr.ndim == 1:
        return np.abs(arr)

    if arr.ndim == 2:
        return np.abs(arr).mean(axis=0)

    if arr.ndim == 3:
        if arr.shape[1] <= arr.shape[2]:
            return np.abs(arr).mean(axis=(0, 1))
        return np.abs(arr).mean(axis=(0, 2))

    raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_artifacts(modality: str, model_name: str = "xgboost"):
    """
    Load (pipeline, artifacts_dict, calibrated_clf_or_None).

    pipeline        : full imblearn Pipeline saved by train.py
    artifacts_dict  : scaler, label_encoder, label_classes, feature_names, train_medians
    calibrated_clf  : CalibratedClassifierCV if available (isotonic preferred),
                      else None (raw pipeline probabilities are used)
    """
    pipeline_path  = os.path.join(MODELS_DIR, f"{modality}_{model_name}_model.pkl")
    artifacts_path = os.path.join(MODELS_DIR, f"{modality}_artifacts.pkl")

    for path in (pipeline_path, artifacts_path):
        if not os.path.exists(path):
            logger.error(f"File not found: {path}  — run main.py first.")
            sys.exit(1)

    pipeline  = joblib.load(pipeline_path)
    artifacts = joblib.load(artifacts_path)

    calibrated_clf = None
    if CALIBRATE_AVAILABLE:
        calibrated_clf = load_calibrated_model(
            modality, model_name, method="isotonic", models_dir=MODELS_DIR
        )
        if calibrated_clf is None:
            logger.info(
                "No calibrated model found — using raw pipeline probabilities. "
                "Run main.py to generate calibrated models."
            )

    return pipeline, artifacts, calibrated_clf


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------

def align_features(
    raw: dict[str, Any],
    feature_names: list[str],
    scaler,
    train_medians: pd.Series | None = None,
    warn_missing: bool = True,
) -> np.ndarray:
    """
    Convert a raw dict {feature_name: value} to a scaled (1 × n_features)
    numpy array matching the exact column order used during training.

    Missing features:
      - If the feature was numerical and train_medians is available, use the
        training median (pre-scale) so that after scaling it maps to its
        training-distribution value.
      - Otherwise default to 0.0 post-scale (the training mean).

    Extra features in raw that are not in feature_names are silently dropped.
    """
    raw = dict(raw)
    lifecycle_stage = raw.pop("lifecycle_stage", None)
    if lifecycle_stage is None and any(
        feature.startswith("lifecycle_stage_") for feature in feature_names
    ):
        age = raw.get("product_age_months")
        if age is not None:
            try:
                lifecycle_stage = _lifecycle_stage_from_age(float(age))
            except (TypeError, ValueError):
                lifecycle_stage = None
    if lifecycle_stage:
        stage = str(lifecycle_stage).strip().lower()
        for feature in feature_names:
            if feature.startswith("lifecycle_stage_"):
                raw[feature] = 1.0 if feature == f"lifecycle_stage_{stage}" else 0.0

    extra   = set(raw.keys()) - set(feature_names)
    missing = set(feature_names) - set(raw.keys())

    if extra and warn_missing:
        logger.warning(
            f"Ignoring {len(extra)} unexpected feature(s): {sorted(extra)}"
        )
    if missing and warn_missing:
        logger.warning(
            f"{len(missing)} feature(s) missing, will use training median or 0.0: "
            f"{sorted(missing)}"
        )

    # Build raw row in the exact training column order (pre-scale)
    median_lookup = (
        train_medians
        if isinstance(train_medians, dict)
        else train_medians.to_dict() if train_medians is not None else {}
    )

    row_pre_scale = []
    for f in feature_names:
        if f in raw:
            row_pre_scale.append(float(raw[f]))
        elif f in median_lookup:
            row_pre_scale.append(float(median_lookup[f]))
        else:
            # Post-scale default of 0 is achieved by using the scaler's mean_
            # which varies per feature.  We approximate by using the scaler mean
            # for that feature, so that transform gives ≈ 0.
            idx = feature_names.index(f)
            if hasattr(scaler, "mean_") and idx < len(scaler.mean_):
                row_pre_scale.append(float(scaler.mean_[idx]))
            else:
                row_pre_scale.append(0.0)

    row = pd.DataFrame([row_pre_scale], columns=feature_names, dtype=np.float64)

    # Apply the saved scaler — transform only, never fit
    try:
        row_scaled = scaler.transform(row)
    except Exception as exc:
        logger.error(f"Scaler transform failed: {exc}")
        sys.exit(1)

    return row_scaled


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------

def _shap_top5(
    clf: Any,
    X_scaled: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Return the top-5 contributing features by mean |SHAP value| for a single
    prediction row.  Returns an empty dict if SHAP is unavailable or fails.
    """
    if not SHAP_AVAILABLE:
        return {}
    try:
        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_scaled)
        elif hasattr(clf, "coef_"):
            explainer = shap.LinearExplainer(clf, X_scaled)
            sv = explainer.shap_values(X_scaled)
        else:
            return {}

        abs_sum = _mean_abs_shap_vector(sv)

        top5_idx = np.argsort(abs_sum)[::-1][:5]
        return {feature_names[i]: round(float(abs_sum[i]), 6) for i in top5_idx}

    except Exception as exc:
        logger.warning(f"SHAP explanation failed: {exc}")
        return {}


def _prediction_completeness(raw_features: dict[str, Any], required_keys: list[str]) -> float:
    if not required_keys:
        return 1.0
    present = sum(1 for key in required_keys if key in raw_features and raw_features[key] is not None)
    return present / len(required_keys)


def _apply_decision_policy(
    proba: np.ndarray,
    label_classes: np.ndarray,
    threshold: float = 0.5,
    class_weights: dict[str, float] | None = None,
) -> int:
    class_weights = class_weights or {}
    if len(label_classes) == 2:
        return 1 if proba[1] >= threshold else 0

    weights = np.array([float(class_weights.get(str(c), 1.0)) for c in label_classes], dtype=float)
    adjusted = proba * weights
    return int(np.argmax(adjusted))


def _review_snapshot_override(
    raw_features: dict[str, Any],
    predicted_class: str,
    confidence: float,
) -> dict[str, Any] | None:
    if predicted_class != "healthy":
        return None

    if any(key in raw_features for key in REVIEW_HISTORY_FEATURES):
        return None

    try:
        sentiment_mean = float(raw_features.get("sentiment_mean", 0.0))
        score_median = float(raw_features.get("score_median", 0.0))
        diversity_change = float(raw_features.get("reviewer_diversity_change", 0.0))
        product_age = float(raw_features.get("product_age_months", 0.0))
    except (TypeError, ValueError):
        return None

    if (
        score_median <= 2.5
        and sentiment_mean <= 0.25
        and diversity_change <= -20.0
        and product_age >= 18.0
    ):
        return {
            "predicted_class": "high_fatigue",
            "confidence": max(0.72, min(0.93, 1.0 - confidence * 0.15)),
            "warning": "Review snapshot fallback applied because strong fatigue signals were present without history features.",
        }

    if (
        score_median <= 3.2
        and sentiment_mean <= 0.6
        and diversity_change <= -8.0
        and product_age >= 12.0
    ):
        return {
            "predicted_class": "moderate_fatigue",
            "confidence": max(0.62, min(0.84, 1.0 - confidence * 0.2)),
            "warning": "Review snapshot fallback applied because moderate fatigue signals were present without history features.",
        }

    return None


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def predict(
    modality: str,
    raw_features: dict[str, Any],
    model_name: str = "xgboost",
    threshold: float = OVERCONFIDENCE_THRESHOLD,
    strict: bool = False,
) -> dict[str, Any]:
    """
    End-to-end prediction for one input row.

    Returns
    -------
    {
      "modality":           str,
      "model":              str,
      "predicted_class":    str,
      "confidence":         float,
      "all_probabilities":  {class_name: prob},
      "shap_top5_features": {feature_name: mean_abs_shap},
      "warnings":           [str],
    }
    """
    pipeline, artifacts, calibrated_clf = load_artifacts(modality, model_name)

    feature_names: list[str]      = artifacts["feature_names"]
    scaler                        = artifacts["scaler"]
    label_classes: np.ndarray     = artifacts["label_classes"]
    train_medians: Any | None  = artifacts.get("train_medians")
    optimal_thresholds: dict      = artifacts.get("optimal_thresholds", {})
    predict_threshold: float      = optimal_thresholds.get(model_name, 0.5)
    class_weight_policies: dict   = artifacts.get("class_weight_policies", {})
    class_weights: dict           = class_weight_policies.get(model_name, {})
    raw_required_features: list[str] = artifacts.get("raw_required_features", [])

    completeness = _prediction_completeness(raw_features, raw_required_features)
    if strict:
        missing_required = [
            key for key in raw_required_features
            if key not in raw_features or raw_features[key] is None
        ]
        if missing_required:
            raise ValueError(
                "Missing required input fields: "
                f"{missing_required}. Provide the full dashboard input payload."
            )
        if completeness < MIN_COMPLETENESS_THRESHOLD:
            raise ValueError(
                f"Input completeness {completeness:.2%} below required "
                f"{MIN_COMPLETENESS_THRESHOLD:.0%}. Provide the full dashboard input payload."
            )

    X_scaled = align_features(raw_features, feature_names, scaler, train_medians)

    # ── Probabilities (raw and, if available, calibrated) ──────────────────
    if CALIBRATE_AVAILABLE and calibrated_clf is not None:
        cal_result = predict_with_calibration(
            pipeline, calibrated_clf, X_scaled, label_classes, threshold
        )
        proba      = np.array([cal_result["calibrated_proba"][str(c)]
                                for c in label_classes])
        pred_idx = _apply_decision_policy(
            proba,
            label_classes,
            threshold=predict_threshold,
            class_weights=class_weights,
        )
        pred_class = str(label_classes[pred_idx])
        confidence = float(proba[pred_idx])
        result_warnings = cal_result["warnings"]
        calibration_info = {
            "raw_confidence":        cal_result["raw_confidence"],
            "calibrated_confidence": cal_result["calibrated_confidence"],
            "adjustment":            cal_result["adjustment"],
            "prediction_changed":    cal_result["prediction_changed"],
            "raw_proba":             cal_result["raw_proba"],
            "decision_threshold":    predict_threshold,
            "class_weights":         class_weights,
        }
    else:
        raw_proba  = pipeline.predict_proba(X_scaled)[0]
        pred_idx = _apply_decision_policy(
            raw_proba,
            label_classes,
            threshold=predict_threshold,
            class_weights=class_weights,
        )
        proba      = raw_proba
        confidence = float(proba[pred_idx])
        pred_class = str(label_classes[pred_idx])
        result_warnings = []
        calibration_info = None
        if confidence > threshold:
            msg = (
                f"Confidence {confidence:.2%} exceeds threshold {threshold:.0%}. "
                "This may indicate an unrealistic input, residual overfitting, "
                "or a feature that directly encodes the label."
            )
            result_warnings.append(msg)
            logger.warning(msg)

    # ── K-Means cluster assignment (supplementary context) ─────────────────
    cluster_id: int | None = None
    km_path = os.path.join(MODELS_DIR, f"{modality}_kmeans_model.pkl")
    if os.path.exists(km_path):
        try:
            km = joblib.load(km_path)
            cluster_id = int(km.predict(X_scaled)[0])
        except Exception as km_exc:
            logger.debug(f"K-Means cluster lookup failed (non-critical): {km_exc}")

    # Extract the classifier from the Pipeline for SHAP
    clf = (
        pipeline.named_steps["clf"]
        if hasattr(pipeline, "named_steps") and "clf" in pipeline.named_steps
        else pipeline
    )
    shap_top5 = _shap_top5(clf, X_scaled, feature_names)

    if modality == "reviews":
        override = _review_snapshot_override(raw_features, pred_class, confidence)
        if override is not None:
            pred_class = override["predicted_class"]
            confidence = float(override["confidence"])
            result_warnings = result_warnings + [override["warning"]]

    # ── Uncertainty flag (new in redesign) ───────────────────────────────────
    max_prob = float(np.max(proba))
    sorted_probs = np.sort(proba)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
    uncertainty_flag = max_prob < 0.60 or margin < 0.15

    if max_prob >= 0.80:
        confidence_band = "high"
    elif max_prob >= 0.60:
        confidence_band = "medium"
    else:
        confidence_band = "low"

    if uncertainty_flag:
        result_warnings.append(
            "Prediction flagged as uncertain — consider manual review"
        )

    # ── Forward-label info ────────────────────────────────────────────────────
    use_forward = artifacts.get("use_forward_labels", False)
    forward_horizon = artifacts.get("forward_horizon")

    return {
        "modality":           modality,
        "model":              model_name,
        "predicted_class":    pred_class,
        "confidence":         round(confidence, 6),
        "all_probabilities":  {
            str(label_classes[i]): round(float(p), 6)
            for i, p in enumerate(proba)
        },
        "calibration":        calibration_info,
        "cluster_id":         cluster_id,
        "shap_top5_features": shap_top5,
        "warnings":           result_warnings,
        "completeness":       round(completeness, 6),
        "uncertainty_flag":   uncertainty_flag,
        "confidence_band":    confidence_band,
        "margin":             round(margin, 6),
        "forward_prediction": use_forward,
        "prediction_horizon": f"next_{forward_horizon}_periods" if forward_horizon else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Product fatigue inference with SHAP explanation."
    )
    parser.add_argument(
        "--modality", required=True, choices=SUPPORTED_MODALITIES,
        help="Which modality's model to use.",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help='JSON dict of {feature_name: value}, e.g. \'{"sentiment_mean": 0.5}\'.',
    )
    parser.add_argument(
        "--input_csv", type=str, default=None,
        help="Path to a single-row CSV (column headers = feature names).",
    )
    parser.add_argument(
        "--model", type=str, default="xgboost", choices=SUPPORTED_MODELS,
        help="Which saved model to use (default: xgboost).",
    )
    parser.add_argument(
        "--threshold", type=float, default=OVERCONFIDENCE_THRESHOLD,
        help=f"Overconfidence warning threshold (default: {OVERCONFIDENCE_THRESHOLD}).",
    )
    args = parser.parse_args()

    if args.input is None and args.input_csv is None:
        parser.error("Provide either --input (JSON) or --input_csv (CSV path).")

    if args.input_csv:
        try:
            row_df = pd.read_csv(args.input_csv).iloc[0]
            raw_features = row_df.to_dict()
        except Exception as exc:
            logger.error(f"Failed to read CSV: {exc}")
            sys.exit(1)
    else:
        try:
            raw_features = json.loads(args.input)
        except json.JSONDecodeError as exc:
            logger.error(f"Invalid JSON input: {exc}")
            sys.exit(1)

    result = predict(args.modality, raw_features, args.model, args.threshold)

    # Pretty-print
    width = 60
    print("\n" + "=" * width)
    print("  PRODUCT FATIGUE PREDICTION")
    print("=" * width)
    print(f"  Modality  : {result['modality']}")
    print(f"  Model     : {result['model']}")
    print(f"  Prediction: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.2%}")

    # ── Calibration comparison block ──────────────────────────────────────
    cal = result.get("calibration")
    if cal is not None:
        raw_c = cal["raw_confidence"]
        adj   = cal["adjustment"]
        sign  = "+" if adj >= 0 else ""
        print(f"\n  {'─'*56}")
        print("  Probability Calibration (isotonic regression)")
        print(f"  {'─'*56}")
        print(f"  Raw Confidence        : {raw_c:.2%}")
        print(f"  Calibrated Confidence : {result['confidence']:.2%}")
        print(f"  Adjustment            : {sign}{adj*100:.2f}%")
        if cal["prediction_changed"]:
            print("  *** Calibration changed class prediction ***")
    else:
        print("  (No calibrated model — raw probabilities shown)")

    print(f"\n  {'─'*56}")
    label = "Calibrated probabilities:" if cal else "Class probabilities:"
    print(f"  {label}")
    for cls, p in result["all_probabilities"].items():
        bar = "█" * int(p * 20)
        print(f"    {cls:<24} {p:.4f}  {bar}")

    # ── Uncertainty & forward prediction info ────────────────────────────────
    if result.get("forward_prediction"):
        print(f"\n  {'─'*56}")
        print("  Forward Prediction (temporal forecasting)")
        print(f"  {'─'*56}")
        print(f"  Prediction Horizon : {result.get('prediction_horizon', 'N/A')}")
        print(f"  Confidence Band    : {result.get('confidence_band', 'N/A')}")
        print(f"  Uncertainty Flag   : {'YES' if result.get('uncertainty_flag') else 'NO'}")
        print(f"  Decision Margin    : {result.get('margin', 0.0):.4f}")

    if result["shap_top5_features"]:
        print("\n  Top-5 SHAP contributions (mean |value|):")
        for feat, val in result["shap_top5_features"].items():
            print(f"    {feat:<40} {val:.6f}")

    if result["warnings"]:
        print("\n  WARNINGS:")
        for w in result["warnings"]:
            print(f"    {w}")

    print("=" * width + "\n")


if __name__ == "__main__":
    main()
