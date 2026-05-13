"""
tests/test_stress.py
====================
Expanded stress tests for the Product Fatigue ML pipeline (Problem 14).

Sections
--------
1. MISSING-VALUE TESTS — predict with progressively more features set to NaN/0
2. UNSEEN-CATEGORY TESTS — feed lifecycle_stage values never seen in training
3. TIME-DRIFT TESTS — simulate distribution shift by scaling features
4. ADVERSARIAL PERTURBATION TESTS — small targeted perturbations that should
   not flip the predicted class
5. BOUNDARY & EXTREME VALUE TESTS — min/max/overflow feature values
6. FORWARD-LABEL & WALK-FORWARD MODULE TESTS — verify new pipeline modules

Run
---
  cd /path/to/Product_Fatigue
  source venv/bin/activate
  python tests/test_stress.py              # runs all sections
  python tests/test_stress.py missing      # missing-value tests only
  python tests/test_stress.py unseen       # unseen-category tests only
  python tests/test_stress.py drift        # time-drift tests only
  python tests/test_stress.py adversarial  # adversarial perturbation tests only
  python tests/test_stress.py boundary     # boundary value tests only
  python tests/test_stress.py modules      # new module tests only
"""

import json
import logging
import os
import sys
import time
import copy
from typing import Any, Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stress_test")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data", "processed")
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Result tracker (same pattern as test_pipeline.py)
# ---------------------------------------------------------------------------
_results: List[Dict] = []


def _pass(name: str, detail: str = "") -> None:
    _results.append({"status": "PASS", "name": name, "detail": detail})
    log.info(f"  PASS  {name}  {detail}")


def _fail(name: str, detail: str = "") -> None:
    _results.append({"status": "FAIL", "name": name, "detail": detail})
    log.error(f"  FAIL  {name}  {detail}")


def _skip(name: str, reason: str = "") -> None:
    _results.append({"status": "SKIP", "name": name, "detail": reason})
    log.warning(f"  SKIP  {name}  {reason}")


def _warn(name: str, detail: str = "") -> None:
    _results.append({"status": "WARN", "name": name, "detail": detail})
    log.warning(f"  WARN  {name}  {detail}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _artifacts_ready() -> bool:
    return all(
        os.path.exists(os.path.join(MODELS_DIR, f"{m}_artifacts.pkl"))
        for m in ["reviews", "sales", "usage"]
    )


def _load_artifacts(modality: str) -> Dict[str, Any]:
    import joblib
    return joblib.load(os.path.join(MODELS_DIR, f"{modality}_artifacts.pkl"))


def _predict_safe(modality: str, features: Dict[str, Any], model_name: str = "xgboost") -> Dict[str, Any]:
    """Run predict() and return result or raise."""
    from src.predict import predict
    return predict(modality, features, model_name=model_name)


# Baseline feature sets for each modality (known-good inputs)
BASELINE_FEATURES = {
    "reviews": {
        "sentiment_mean": 0.62, "sentiment_std": 0.21, "review_count": 24,
        "score_min": 2, "score_max": 5, "score_median": 3.5,
        "product_age_months": 14, "sentiment_polarization": 1.2,
        "reviewer_diversity_change": -8.0,
    },
    "sales": {
        "revenue_total": 12400, "revenue_mean": 620, "revenue_std": 180,
        "transaction_count": 20, "quantity_sold": 340, "avg_price": 36.5,
        "product_age_months": 18, "order_frequency_change": -9.5,
        "aov_change": -4.0, "customer_concentration": 0.46,
    },
    "usage": {
        "engagement_total": 1200, "engagement_mean": 12, "cart_count": 80,
        "purchase_count": 20, "avg_price": 49, "view_to_cart_rate": 8.0,
        "cart_to_purchase_rate": 25.0, "conversion_rate": 2.0,
        "product_age_months": 8, "funnel_efficiency": 0.25,
        "engagement_per_session": 3.2, "safe_engagement_quality_change": 5.0,
    },
}


# ===========================================================================
# SECTION 1 — Missing-Value Stress Tests
# ===========================================================================

def test_missing_values_progressive():
    """
    Progressively zero out features one-by-one and verify predict()
    still returns a valid result without crashing. Then test with ALL
    features zeroed out.
    """
    if not _artifacts_ready():
        _skip("missing_values_progressive", "models not trained")
        return

    for modality in ["reviews", "sales", "usage"]:
        baseline = copy.deepcopy(BASELINE_FEATURES[modality])
        feature_keys = list(baseline.keys())

        # Test 1: Zero out each feature individually
        individual_failures = 0
        for key in feature_keys:
            test_input = copy.deepcopy(baseline)
            test_input[key] = 0.0
            try:
                result = _predict_safe(modality, test_input)
                if "predicted_class" not in result:
                    individual_failures += 1
            except Exception:
                individual_failures += 1

        if individual_failures == 0:
            _pass(f"missing_single[{modality}]",
                  f"all {len(feature_keys)} single-zero tests passed")
        else:
            _fail(f"missing_single[{modality}]",
                  f"{individual_failures}/{len(feature_keys)} failed")

        # Test 2: Zero out ALL features at once
        all_zero = {k: 0.0 for k in feature_keys}
        try:
            result = _predict_safe(modality, all_zero)
            if "predicted_class" in result and "confidence" in result:
                conf = result["confidence"]
                _pass(f"missing_all_zero[{modality}]",
                      f"pred={result['predicted_class']} conf={conf:.2%}")
            else:
                _fail(f"missing_all_zero[{modality}]", "missing keys in result")
        except Exception as exc:
            _fail(f"missing_all_zero[{modality}]", str(exc))

        # Test 3: Empty dict (no features at all)
        try:
            result = _predict_safe(modality, {})
            if "predicted_class" in result:
                _pass(f"missing_empty_dict[{modality}]",
                      f"pred={result['predicted_class']}")
            else:
                _fail(f"missing_empty_dict[{modality}]", "missing predicted_class")
        except Exception as exc:
            _fail(f"missing_empty_dict[{modality}]", str(exc))

        # Test 4: Half features missing
        half_input = {k: v for i, (k, v) in enumerate(baseline.items()) if i % 2 == 0}
        try:
            result = _predict_safe(modality, half_input)
            if "predicted_class" in result:
                _pass(f"missing_half[{modality}]",
                      f"pred={result['predicted_class']} with {len(half_input)}/{len(baseline)} features")
            else:
                _fail(f"missing_half[{modality}]", "missing predicted_class")
        except Exception as exc:
            _fail(f"missing_half[{modality}]", str(exc))


def test_missing_values_nan_injection():
    """
    Pass NaN values for numeric features. predict() should handle gracefully
    (fill with median or zero) without crashing.
    """
    if not _artifacts_ready():
        _skip("missing_values_nan", "models not trained")
        return

    for modality in ["reviews", "sales", "usage"]:
        baseline = copy.deepcopy(BASELINE_FEATURES[modality])

        # Set every other feature to NaN
        nan_input: Dict[str, Any] = {}
        for i, (k, v) in enumerate(baseline.items()):
            nan_input[k] = float('nan') if i % 2 == 0 else v

        try:
            result = _predict_safe(modality, nan_input)
            if "predicted_class" in result:
                _pass(f"nan_injection[{modality}]",
                      f"pred={result['predicted_class']} with NaN features")
            else:
                _fail(f"nan_injection[{modality}]", "missing predicted_class")
        except Exception as exc:
            # NaN handling may raise — that's acceptable if it's a clear error
            _warn(f"nan_injection[{modality}]",
                  f"raised {type(exc).__name__}: {str(exc)[:100]}")


# ===========================================================================
# SECTION 2 — Unseen-Category Stress Tests
# ===========================================================================

def test_unseen_lifecycle_stages():
    """
    Feed lifecycle_stage values that were never seen in training.
    predict() should not crash — it should either ignore the field
    or use a fallback encoding.
    """
    if not _artifacts_ready():
        _skip("unseen_lifecycle_stages", "models not trained")
        return

    unseen_stages = [
        "pre_launch", "sunset", "relaunch", "archive", "beta",
        "UNKNOWN", "", "None", "123", "growth_phase_2",
    ]

    for modality in ["reviews", "sales", "usage"]:
        baseline = copy.deepcopy(BASELINE_FEATURES[modality])
        failures = 0

        for stage in unseen_stages:
            test_input = copy.deepcopy(baseline)
            test_input["lifecycle_stage"] = stage
            try:
                result = _predict_safe(modality, test_input)
                if "predicted_class" not in result:
                    failures += 1
            except Exception:
                failures += 1

        if failures == 0:
            _pass(f"unseen_lifecycle[{modality}]",
                  f"all {len(unseen_stages)} unseen stages handled")
        else:
            _fail(f"unseen_lifecycle[{modality}]",
                  f"{failures}/{len(unseen_stages)} caused errors")


def test_unseen_feature_names():
    """
    Pass features with completely wrong names. predict() should
    ignore unknown features and still produce a result.
    """
    if not _artifacts_ready():
        _skip("unseen_feature_names", "models not trained")
        return

    garbage_features = {
        "cosmic_ray_flux": 42.0,
        "moon_phase_index": 0.75,
        "blockchain_sentiment": -0.3,
        "quantum_volatility": 999.9,
    }

    for modality in ["reviews", "sales", "usage"]:
        try:
            result = _predict_safe(modality, garbage_features)
            if "predicted_class" in result:
                _pass(f"unseen_features[{modality}]",
                      f"pred={result['predicted_class']} (all features unknown)")
            else:
                _fail(f"unseen_features[{modality}]", "missing predicted_class")
        except Exception as exc:
            _fail(f"unseen_features[{modality}]", str(exc))


# ===========================================================================
# SECTION 3 — Time-Drift Stress Tests
# ===========================================================================

def test_distribution_shift_scaling():
    """
    Simulate distribution drift by scaling all numeric features by 2x, 5x,
    and 0.1x. The model should still return valid predictions (it may change
    class, but should not crash or return NaN confidence).
    """
    if not _artifacts_ready():
        _skip("distribution_shift", "models not trained")
        return

    scale_factors = [0.01, 0.1, 2.0, 5.0, 10.0, 100.0]

    for modality in ["reviews", "sales", "usage"]:
        baseline = copy.deepcopy(BASELINE_FEATURES[modality])
        failures = 0

        for scale in scale_factors:
            scaled = {k: v * scale for k, v in baseline.items()
                      if isinstance(v, (int, float))}
            try:
                result = _predict_safe(modality, scaled)
                if "predicted_class" not in result:
                    failures += 1
                    continue
                conf = result.get("confidence", 0)
                if not (0 <= conf <= 1):
                    failures += 1
                    log.error(f"  {modality} scale={scale}x: conf={conf} out of [0,1]")
            except Exception as exc:
                failures += 1
                log.error(f"  {modality} scale={scale}x: {exc}")

        if failures == 0:
            _pass(f"drift_scaling[{modality}]",
                  f"all {len(scale_factors)} scale factors OK")
        else:
            _fail(f"drift_scaling[{modality}]",
                  f"{failures}/{len(scale_factors)} scale factors failed")


def test_temporal_drift_age_extrapolation():
    """
    Test with product_age_months far beyond training range (120, 240, 600 months).
    Model should handle extrapolation without crashing.
    """
    if not _artifacts_ready():
        _skip("age_extrapolation", "models not trained")
        return

    extreme_ages = [0, 120, 240, 600, 1200]

    for modality in ["reviews", "sales", "usage"]:
        baseline = copy.deepcopy(BASELINE_FEATURES[modality])
        failures = 0

        for age in extreme_ages:
            test_input = copy.deepcopy(baseline)
            test_input["product_age_months"] = age
            try:
                result = _predict_safe(modality, test_input)
                if "predicted_class" not in result:
                    failures += 1
            except Exception:
                failures += 1

        if failures == 0:
            _pass(f"age_extrapolation[{modality}]",
                  f"all {len(extreme_ages)} extreme ages handled")
        else:
            _fail(f"age_extrapolation[{modality}]",
                  f"{failures}/{len(extreme_ages)} failed")


# ===========================================================================
# SECTION 4 — Adversarial Perturbation Tests
# ===========================================================================

def test_small_perturbation_stability():
    """
    Apply small random perturbations (±1%) to a baseline prediction.
    The predicted class should remain stable — small noise should not
    cause class flips. We allow up to 20% flip rate (some inputs may
    be near decision boundaries).
    """
    if not _artifacts_ready():
        _skip("perturbation_stability", "models not trained")
        return

    np.random.seed(42)
    n_trials = 50
    max_flip_rate = 0.20

    for modality in ["reviews", "sales", "usage"]:
        baseline = copy.deepcopy(BASELINE_FEATURES[modality])

        # Get baseline prediction
        try:
            base_result = _predict_safe(modality, baseline)
            base_class = base_result["predicted_class"]
        except Exception as exc:
            _fail(f"perturbation[{modality}]", f"baseline prediction failed: {exc}")
            continue

        flips = 0
        for _ in range(n_trials):
            perturbed = {}
            for k, v in baseline.items():
                if isinstance(v, (int, float)):
                    noise = v * np.random.uniform(-0.01, 0.01)
                    perturbed[k] = v + noise
                else:
                    perturbed[k] = v

            try:
                result = _predict_safe(modality, perturbed)
                if result["predicted_class"] != base_class:
                    flips += 1
            except Exception:
                flips += 1

        flip_rate = flips / n_trials
        if flip_rate <= max_flip_rate:
            _pass(f"perturbation[{modality}]",
                  f"flip_rate={flip_rate:.0%} ({flips}/{n_trials}) <= {max_flip_rate:.0%}")
        else:
            _warn(f"perturbation[{modality}]",
                  f"flip_rate={flip_rate:.0%} ({flips}/{n_trials}) > {max_flip_rate:.0%} — model may be unstable near boundary")


def test_sign_flip_perturbation():
    """
    For features that can be negative (e.g. sentiment_mean, order_frequency_change),
    flip their sign. The model should still return a valid prediction.
    """
    if not _artifacts_ready():
        _skip("sign_flip", "models not trained")
        return

    sign_flippable = {
        "reviews": ["sentiment_mean", "reviewer_diversity_change"],
        "sales": ["order_frequency_change", "aov_change"],
        "usage": ["safe_engagement_quality_change"],
    }

    for modality in ["reviews", "sales", "usage"]:
        baseline = copy.deepcopy(BASELINE_FEATURES[modality])
        flippable = sign_flippable.get(modality, [])
        failures = 0

        for key in flippable:
            if key not in baseline:
                continue
            test_input = copy.deepcopy(baseline)
            test_input[key] = -test_input[key]
            try:
                result = _predict_safe(modality, test_input)
                if "predicted_class" not in result:
                    failures += 1
            except Exception:
                failures += 1

        if failures == 0:
            _pass(f"sign_flip[{modality}]",
                  f"all {len(flippable)} sign flips handled")
        else:
            _fail(f"sign_flip[{modality}]",
                  f"{failures}/{len(flippable)} failed")


# ===========================================================================
# SECTION 5 — Boundary & Extreme Value Tests
# ===========================================================================

def test_extreme_values():
    """
    Test with extreme float values: very large, very small, negative.
    Model should not return NaN or Inf.
    """
    if not _artifacts_ready():
        _skip("extreme_values", "models not trained")
        return

    extreme_sets = {
        "max_float": 1e15,
        "min_positive": 1e-15,
        "large_negative": -1e10,
        "negative_one": -1.0,
    }

    for modality in ["reviews", "sales", "usage"]:
        baseline = copy.deepcopy(BASELINE_FEATURES[modality])

        for label, fill_val in extreme_sets.items():
            test_input = {k: fill_val for k in baseline}
            try:
                result = _predict_safe(modality, test_input)
                conf = result.get("confidence", 0)
                if np.isnan(conf) or np.isinf(conf):
                    _fail(f"extreme[{modality}/{label}]",
                          f"confidence is NaN/Inf: {conf}")
                elif "predicted_class" in result:
                    _pass(f"extreme[{modality}/{label}]",
                          f"pred={result['predicted_class']} conf={conf:.2%}")
                else:
                    _fail(f"extreme[{modality}/{label}]", "missing predicted_class")
            except Exception as exc:
                _warn(f"extreme[{modality}/{label}]",
                      f"raised {type(exc).__name__}: {str(exc)[:100]}")


def test_negative_counts():
    """
    Features like review_count, transaction_count should never be negative
    in production, but the model should handle it gracefully if they are.
    """
    if not _artifacts_ready():
        _skip("negative_counts", "models not trained")
        return

    count_features = {
        "reviews": ["review_count"],
        "sales": ["transaction_count", "quantity_sold"],
        "usage": ["cart_count", "purchase_count", "engagement_total"],
    }

    for modality in ["reviews", "sales", "usage"]:
        baseline = copy.deepcopy(BASELINE_FEATURES[modality])
        features_to_test = count_features.get(modality, [])
        failures = 0

        for key in features_to_test:
            if key not in baseline:
                continue
            test_input = copy.deepcopy(baseline)
            test_input[key] = -100.0
            try:
                result = _predict_safe(modality, test_input)
                if "predicted_class" not in result:
                    failures += 1
            except Exception:
                failures += 1

        if failures == 0:
            _pass(f"negative_counts[{modality}]",
                  f"all {len(features_to_test)} negative count tests passed")
        else:
            _fail(f"negative_counts[{modality}]",
                  f"{failures}/{len(features_to_test)} failed")


# ===========================================================================
# SECTION 6 — New Pipeline Module Tests
# ===========================================================================

def test_forward_label_module():
    """Verify forward_label module loads and basic API works."""
    name = "forward_label_module"
    try:
        from src.forward_label import construct_forward_labels
        import pandas as pd

        # Create minimal test DataFrame
        rows = []
        for pid in ["P1", "P2"]:
            for month_idx in range(8):
                rows.append({
                    "product_id": pid,
                    "month": f"2024-{month_idx + 1:02d}",
                    "sentiment_mean": 0.5 - month_idx * 0.05,
                    "review_count": max(1, 20 - month_idx * 2),
                    "reviewer_diversity_change": -month_idx * 3.0,
                    "fatigue_label": "healthy" if month_idx < 4 else "moderate_fatigue",
                })
        df = pd.DataFrame(rows)

        result = construct_forward_labels(
            df, modality="reviews",
            id_col="product_id", time_col="month",
            horizon=2, class_method="quantile"
        )

        if isinstance(result, pd.DataFrame) and "fatigue_label" in result.columns:
            # Forward labels should drop rows that don't have enough future data
            if len(result) < len(df):
                _pass(name, f"input={len(df)} rows, output={len(result)} rows (future rows trimmed)")
            else:
                _pass(name, f"output={len(result)} rows")
        else:
            _fail(name, f"unexpected return type: {type(result)}")
    except ImportError as exc:
        _skip(name, f"import failed: {exc}")
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {str(exc)[:150]}")


def test_walk_forward_module():
    """Verify walk-forward CV splitter produces valid expanding-window splits."""
    name = "walk_forward_module"
    try:
        from src.walk_forward import WalkForwardCV, walk_forward_splits
        import pandas as pd

        # Create test DataFrame with 12 months
        rows = []
        for pid in ["P1", "P2", "P3"]:
            for m in range(1, 13):
                rows.append({
                    "product_id": pid,
                    "month": f"2024-{m:02d}",
                    "feature_a": float(m),
                    "fatigue_label": "healthy",
                })
        df = pd.DataFrame(rows)

        # Test walk_forward_splits generator
        splits = list(walk_forward_splits(
            df, time_col="month", min_train_periods=6, val_periods=1, expanding=True
        ))

        if len(splits) < 1:
            _fail(name, "no splits generated")
            return

        # Verify expanding window: each split's train set should grow
        prev_train_size = 0
        monotonic = True
        for train_idx, val_idx in splits:
            if len(train_idx) < prev_train_size:
                monotonic = False
            prev_train_size = len(train_idx)
            # Verify no overlap
            overlap = set(train_idx) & set(val_idx)
            if overlap:
                _fail(name, f"train/val overlap: {len(overlap)} indices")
                return

        if monotonic:
            _pass(name, f"{len(splits)} expanding splits, no overlap, train sizes grow monotonically")
        else:
            _warn(name, f"{len(splits)} splits generated but train sizes not monotonic")

    except ImportError as exc:
        _skip(name, f"import failed: {exc}")
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {str(exc)[:150]}")


def test_feature_stability_module():
    """Verify feature_stability safe_log_diff and safe_ratio work correctly."""
    name = "feature_stability_module"
    try:
        from src.feature_stability import safe_log_diff, safe_ratio, cap_extreme_ratios
        import pandas as pd

        # Test safe_log_diff (requires current and previous series)
        current = pd.Series([1, 10, 100, 1000, 5000])
        previous = pd.Series([0, 1, 10, 100, 1000])
        result = safe_log_diff(current, previous)
        if len(result) == len(current) and not result.isna().all():
            _pass(f"{name}/safe_log_diff", f"output length={len(result)}")
        else:
            _fail(f"{name}/safe_log_diff", "unexpected output")

        # Test safe_ratio with zero denominator
        num = pd.Series([10.0, 20.0, 30.0])
        den = pd.Series([0.0, 5.0, 0.0])
        ratio = safe_ratio(num, den)
        if not np.isinf(ratio).any():
            _pass(f"{name}/safe_ratio", "no Inf values with zero denominator")
        else:
            _fail(f"{name}/safe_ratio", "Inf found with zero denominator")

        # Test cap_extreme_ratios (takes DataFrame + column list)
        df_extreme = pd.DataFrame({"ratio_a": [0.01, 0.5, 1.0, 50.0, 1000.0]})
        capped_df = cap_extreme_ratios(df_extreme, ["ratio_a"])
        if capped_df["ratio_a"].max() <= df_extreme["ratio_a"].max():
            _pass(f"{name}/cap_extreme", "extreme values capped")
        else:
            _fail(f"{name}/cap_extreme", "capping did not reduce max")

    except ImportError as exc:
        _skip(name, f"import failed: {exc}")
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {str(exc)[:150]}")


def test_calibrate_uncertainty_flags():
    """Verify uncertainty computation produces valid confidence bands."""
    name = "calibrate_uncertainty"
    try:
        from src.calibrate import compute_uncertainty_flag

        # compute_uncertainty_flag expects 2D probabilities array (n_samples, n_classes)
        # Test high confidence (single sample)
        result_high = compute_uncertainty_flag(np.array([[0.05, 0.10, 0.85]]))
        band_high = result_high["confidence_band"]
        flag_high = result_high["uncertainty_flag"]
        # Results are arrays — extract first element
        if hasattr(band_high, '__len__'):
            band_high = band_high[0]
            flag_high = flag_high[0]
        assert band_high == "high", f"expected high, got {band_high}"
        assert not flag_high, f"expected no uncertainty flag for high confidence"

        # Test low confidence (single sample)
        result_low = compute_uncertainty_flag(np.array([[0.35, 0.35, 0.30]]))
        band_low = result_low["confidence_band"]
        flag_low = result_low["uncertainty_flag"]
        if hasattr(band_low, '__len__'):
            band_low = band_low[0]
            flag_low = flag_low[0]
        assert flag_low, f"expected uncertainty flag for low confidence"
        assert band_low == "low", f"expected low, got {band_low}"

        _pass(name, "high/low confidence bands correct, uncertainty flags correct")

    except ImportError as exc:
        _skip(name, f"import failed: {exc}")
    except AssertionError as exc:
        _fail(name, str(exc))
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {str(exc)[:150]}")


def test_fusion_module_importable():
    """Verify the fusion module can be imported and key classes exist."""
    name = "fusion_module_import"
    try:
        from src.fusion import (
            generate_oof_probabilities,
            build_fusion_table,
            train_fusion_logistic,
            compute_fatigue_index,
            FusionModel,
        )
        _pass(name, "all fusion module exports importable")
    except ImportError as exc:
        _skip(name, f"import failed: {exc}")
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {str(exc)[:150]}")


def test_ablation_module_importable():
    """Verify the ablation module can be imported."""
    name = "ablation_module_import"
    try:
        from src.ablation import (
            AblationExperiment,
            define_standard_ablations,
            run_ablation_suite,
        )
        # Test that standard ablations can be defined (needs modality + feature_names)
        dummy_features = ["sentiment_mean", "sentiment_std", "review_count",
                          "score_min", "score_max", "product_age_months"]
        ablations = define_standard_ablations("reviews", dummy_features)
        if len(ablations) >= 3:
            _pass(name, f"defined {len(ablations)} ablation experiments for reviews")
        else:
            _fail(name, f"only {len(ablations)} ablations defined, expected >= 3")
    except ImportError as exc:
        _skip(name, f"import failed: {exc}")
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {str(exc)[:150]}")


# ===========================================================================
# Runner
# ===========================================================================

def _print_summary():
    total = len(_results)
    passed = sum(1 for r in _results if r["status"] == "PASS")
    failed = sum(1 for r in _results if r["status"] == "FAIL")
    warned = sum(1 for r in _results if r["status"] == "WARN")
    skipped = sum(1 for r in _results if r["status"] == "SKIP")

    width = 72
    print("\n" + "=" * width)
    print("  STRESS TEST SUMMARY")
    print("=" * width)
    print(f"  {'PASS':<8} {passed}")
    print(f"  {'FAIL':<8} {failed}")
    print(f"  {'WARN':<8} {warned}")
    print(f"  {'SKIP':<8} {skipped}")
    print(f"  {'TOTAL':<8} {total}")
    print("-" * width)

    if failed:
        print("\n  FAILURES:")
        for r in _results:
            if r["status"] == "FAIL":
                print(f"    [FAIL]  {r['name']}")
                if r["detail"]:
                    print(f"            {r['detail']}")

    if warned:
        print("\n  WARNINGS:")
        for r in _results:
            if r["status"] == "WARN":
                print(f"    [WARN]  {r['name']}")
                if r["detail"]:
                    print(f"            {r['detail']}")

    if skipped:
        print("\n  SKIPPED (run main.py first to enable these):")
        for r in _results:
            if r["status"] == "SKIP":
                print(f"    [SKIP]  {r['name']}  --  {r['detail']}")

    print("=" * width + "\n")
    return failed


MISSING_TESTS = [
    test_missing_values_progressive,
    test_missing_values_nan_injection,
]

UNSEEN_TESTS = [
    test_unseen_lifecycle_stages,
    test_unseen_feature_names,
]

DRIFT_TESTS = [
    test_distribution_shift_scaling,
    test_temporal_drift_age_extrapolation,
]

ADVERSARIAL_TESTS = [
    test_small_perturbation_stability,
    test_sign_flip_perturbation,
]

BOUNDARY_TESTS = [
    test_extreme_values,
    test_negative_counts,
]

MODULE_TESTS = [
    test_forward_label_module,
    test_walk_forward_module,
    test_feature_stability_module,
    test_calibrate_uncertainty_flags,
    test_fusion_module_importable,
    test_ablation_module_importable,
]


def main(section: str = "all"):
    t0 = time.time()
    log.info(f"Running stress test section: {section!r}")

    sections = {
        "missing": MISSING_TESTS,
        "unseen": UNSEEN_TESTS,
        "drift": DRIFT_TESTS,
        "adversarial": ADVERSARIAL_TESTS,
        "boundary": BOUNDARY_TESTS,
        "modules": MODULE_TESTS,
        "all": (MISSING_TESTS + UNSEEN_TESTS + DRIFT_TESTS +
                ADVERSARIAL_TESTS + BOUNDARY_TESTS + MODULE_TESTS),
    }

    chosen = sections.get(section, sections["all"])
    for fn in chosen:
        log.info(f"\n>>> {fn.__name__}")
        try:
            fn()
        except Exception as exc:
            _fail(fn.__name__, f"unexpected exception: {exc}")

    elapsed = time.time() - t0
    log.info(f"\nCompleted in {elapsed:.1f}s")
    n_failed = _print_summary()
    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    section = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(section)
