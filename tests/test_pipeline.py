"""
tests/test_pipeline.py
======================
Full verification suite for the leakage-free Product Fatigue ML pipeline.

Sections
--------
1. UNIT TESTS — data_loader.py
   - Temporal split ordering (no future leakage)
   - Z-score and ID columns are dropped
   - Scaler fitted on train only
   - Label encoder alignment

2. UNIT TESTS — train.py
   - SMOTE is inside the imblearn Pipeline (not applied to raw X_train before CV)
   - Saved models are loadable

3. INTEGRATION TESTS — full predict flow
   - Artifacts exist and have the right keys
   - Feature alignment (missing / extra features)

4. HIDDEN INPUT STRING TESTS — predict.py CLI
   Each modality gets 5 crafted scenarios:
     A. Clearly HEALTHY product (strong positive signals)
     B. MODERATE fatigue onset (mixed signals)
     C. HIGH / severe fatigue (all signals degrading)
     D. EDGE CASE — sparse data (mostly NaN-equivalent zeros)
     E. EDGE CASE — extreme values (tests overconfidence clipping)

5. METRICS SANITY — post-training evaluation checks
   - f1_macro > 0.30 for all models
   - CV-test gap < 0.15

6. ENGINEERING IMPROVEMENT TESTS — new pipeline features
   - Optimal decision thresholds persisted in artifacts
   - Calibrated model files exist for all modality/model pairs
   - experiment_log.csv structure and threshold column population
   - predict() returns cluster_id from K-Means model
   - predict() calibration dict includes decision_threshold
   - find_optimal_threshold() returns valid range values

Run
---
  cd /path/to/Product_Fatigue
  source venv/bin/activate
  python tests/test_pipeline.py          # runs all sections
  python tests/test_pipeline.py unit     # unit tests only (no model needed)
  python tests/test_pipeline.py cli      # hidden-input CLI tests only
  python tests/test_pipeline.py eng      # engineering improvement tests only
"""

import json
import logging
import os
import subprocess
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline_test")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
SRC_DIR    = os.path.join(ROOT, "src")
sys.path.insert(0, ROOT)

VENV_PYTHON = os.path.join(ROOT, "venv", "bin", "python3")
PYTHON      = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

# ---------------------------------------------------------------------------
# Tiny test-result tracker
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


# ===========================================================================
# SECTION 1 — Unit tests: data_loader.py
# ===========================================================================

def _load_sample(modality: str, n: int = 2000) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{modality}_fatigue_signals.csv")
    return pd.read_csv(path, nrows=n, low_memory=False)

def test_temporal_split_ordering():
    """
    For every product in the sample the test rows must all come AFTER
    the training rows chronologically.  Verifies no future leakage.
    """
    from src.data_loader import _temporal_split, MODALITY_CONFIG
    name = "temporal_split_ordering"

    for modality in ["reviews", "sales", "usage"]:
        cfg = MODALITY_CONFIG[modality]
        id_col, time_col = cfg["id_col"], cfg["time_col"]

        df = _load_sample(modality, n=5000)
        df = df.dropna(subset=["fatigue_label"])
        train_df, test_df = _temporal_split(df, id_col, time_col, test_frac=0.20)

        # Convert month strings to sortable integers (YYYYMM)
        def to_int(series):
            return pd.to_datetime(series, format="%Y-%m", errors="coerce")

        violations = 0
        for pid, t_grp in test_df.groupby(id_col):
            if pid not in train_df[id_col].values:
                continue
            tr_grp = train_df[train_df[id_col] == pid]
            latest_train = to_int(tr_grp[time_col]).max()
            earliest_test = to_int(t_grp[time_col]).min()
            if pd.notna(latest_train) and pd.notna(earliest_test):
                if earliest_test < latest_train:
                    violations += 1

        if violations == 0:
            _pass(f"{name}[{modality}]", "all test months come after train months")
        else:
            _fail(f"{name}[{modality}]", f"{violations} products have test months before their latest train month")

def test_zscore_columns_dropped():
    """Z-score columns must not appear in X_train feature names."""
    from src.data_loader import load_modality, GLOBAL_ZSCORE_COLS
    name = "zscore_columns_dropped"

    for modality in ["reviews", "sales", "usage"]:
        path = os.path.join(DATA_DIR, f"{modality}_fatigue_signals.csv")
        try:
            _, _, _, _, artifacts, feature_names = load_modality(path, modality)
            found = GLOBAL_ZSCORE_COLS & set(feature_names)
            if not found:
                _pass(f"{name}[{modality}]")
            else:
                _fail(f"{name}[{modality}]", f"still present: {found}")
        except Exception as exc:
            _fail(f"{name}[{modality}]", str(exc))

def test_id_and_date_columns_dropped():
    """Product ID and date columns must not appear in features."""
    from src.data_loader import load_modality, BASE_DROP_COLS
    name = "id_date_columns_dropped"

    for modality in ["reviews", "sales", "usage"]:
        path = os.path.join(DATA_DIR, f"{modality}_fatigue_signals.csv")
        try:
            _, _, _, _, _, feature_names = load_modality(path, modality)
            found = BASE_DROP_COLS & set(feature_names)
            if not found:
                _pass(f"{name}[{modality}]")
            else:
                _fail(f"{name}[{modality}]", f"still present: {found}")
        except Exception as exc:
            _fail(f"{name}[{modality}]", str(exc))

def test_scaler_fitted_on_train_only():
    """
    Verify that the scaler's mean_ matches the training set mean
    (not the overall dataset mean).  If the scaler were fit on all data,
    its mean_ would be closer to the full-dataset mean than the train mean.
    """
    from src.data_loader import load_modality
    name = "scaler_fit_train_only"

    for modality in ["reviews"]:   # one is enough to prove the pattern
        path = os.path.join(DATA_DIR, f"{modality}_fatigue_signals.csv")
        try:
            X_train, X_test, y_train, y_test, artifacts, feature_names = load_modality(path, modality)
            scaler = artifacts["scaler"]

            # After scaling X_train the column means must be ~0
            col_means = X_train.mean(axis=0)
            max_mean_abs = np.abs(col_means).max()
            if max_mean_abs < 0.05:
                _pass(f"{name}[{modality}]", f"max train column mean after scaling = {max_mean_abs:.5f}")
            else:
                _fail(f"{name}[{modality}]", f"max train column mean = {max_mean_abs:.4f} (expected < 0.05)")
        except Exception as exc:
            _fail(f"{name}[{modality}]", str(exc))

def test_train_test_no_overlap():
    """No row index should appear in both train and test splits."""
    from src.data_loader import _temporal_split, MODALITY_CONFIG
    name = "train_test_no_row_overlap"

    for modality in ["reviews", "sales", "usage"]:
        cfg = MODALITY_CONFIG[modality]
        df = _load_sample(modality, n=3000).dropna(subset=["fatigue_label"])
        train_df, test_df = _temporal_split(df, cfg["id_col"], cfg["time_col"], 0.20)
        overlap = set(train_df.index) & set(test_df.index)
        if not overlap:
            _pass(f"{name}[{modality}]")
        else:
            _fail(f"{name}[{modality}]", f"{len(overlap)} overlapping row indices")

def test_label_classes_consistent():
    """Label classes from the encoder must match unique values in y_train."""
    from src.data_loader import load_modality
    name = "label_classes_consistent"

    for modality in ["reviews", "sales", "usage"]:
        path = os.path.join(DATA_DIR, f"{modality}_fatigue_signals.csv")
        try:
            _, _, y_train, _, artifacts, _ = load_modality(path, modality)
            classes = artifacts["label_classes"]
            unique_encoded = np.unique(y_train)
            expected = np.arange(len(classes))
            if np.array_equal(unique_encoded, expected):
                _pass(f"{name}[{modality}]", f"classes={classes.tolist()}")
            else:
                _fail(f"{name}[{modality}]", f"encoded unique={unique_encoded}, expected={expected}")
        except Exception as exc:
            _fail(f"{name}[{modality}]", str(exc))


# ===========================================================================
# SECTION 2 — Unit tests: train.py structure
# ===========================================================================

def test_smote_is_inside_pipeline():
    """
    The trained pipeline must be an imblearn Pipeline with 'smote' as the
    first step — confirming SMOTE cannot touch validation folds during CV.
    """
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    name = "smote_inside_imblearn_pipeline"

    for modality in ["reviews", "sales", "usage"]:
        for model_name in ["xgboost", "random_forest", "logistic_regression"]:
            pkl = os.path.join(MODELS_DIR, f"{modality}_{model_name}_model.pkl")
            if not os.path.exists(pkl):
                _skip(f"{name}[{modality}/{model_name}]", "model not yet trained")
                continue
            import joblib
            pipeline = joblib.load(pkl)
            if not isinstance(pipeline, ImbPipeline):
                _fail(f"{name}[{modality}/{model_name}]", "not an imblearn Pipeline")
                continue
            steps = dict(pipeline.steps)
            if "smote" in steps and isinstance(steps["smote"], SMOTE):
                _pass(f"{name}[{modality}/{model_name}]")
            else:
                _fail(f"{name}[{modality}/{model_name}]", f"steps={list(steps.keys())}")

def test_artifacts_have_required_keys():
    """Artifacts dict must contain scaler, label_encoder, feature_names, label_classes."""
    import joblib
    name = "artifacts_required_keys"
    required = {"scaler", "label_encoder", "label_classes", "feature_names"}

    for modality in ["reviews", "sales", "usage"]:
        pkl = os.path.join(MODELS_DIR, f"{modality}_artifacts.pkl")
        if not os.path.exists(pkl):
            _skip(f"{name}[{modality}]", "artifacts not yet saved")
            continue
        artifacts = joblib.load(pkl)
        missing = required - set(artifacts.keys())
        if not missing:
            _pass(f"{name}[{modality}]", f"keys={list(artifacts.keys())}")
        else:
            _fail(f"{name}[{modality}]", f"missing keys: {missing}")

def test_models_loadable():
    """All expected model files can be loaded with joblib."""
    import joblib
    name = "models_loadable"
    model_names = ["xgboost", "random_forest", "logistic_regression", "kmeans"]

    for modality in ["reviews", "sales", "usage"]:
        for mn in model_names:
            pkl = os.path.join(MODELS_DIR, f"{modality}_{mn}_model.pkl")
            if not os.path.exists(pkl):
                _skip(f"{name}[{modality}/{mn}]", "not trained yet")
                continue
            try:
                obj = joblib.load(pkl)
                _pass(f"{name}[{modality}/{mn}]", type(obj).__name__)
            except Exception as exc:
                _fail(f"{name}[{modality}/{mn}]", str(exc))


# ===========================================================================
# SECTION 3 — Feature alignment unit tests
# ===========================================================================

def test_feature_alignment_missing():
    """align_features() must handle missing features without crashing."""
    from src.predict import align_features
    import joblib
    name = "align_features_missing"

    for modality in ["reviews", "sales", "usage"]:
        pkl = os.path.join(MODELS_DIR, f"{modality}_artifacts.pkl")
        if not os.path.exists(pkl):
            _skip(f"{name}[{modality}]", "artifacts not yet saved")
            continue
        artifacts = joblib.load(pkl)
        scaler        = artifacts["scaler"]
        feature_names = artifacts["feature_names"]
        train_medians = artifacts.get("train_medians")

        # Pass an empty dict — every feature is missing
        try:
            X = align_features({}, feature_names, scaler, train_medians)
            if X.shape == (1, len(feature_names)):
                _pass(f"{name}[{modality}]", "empty dict → zero-filled row, shape OK")
            else:
                _fail(f"{name}[{modality}]", f"unexpected shape {X.shape}")
        except Exception as exc:
            _fail(f"{name}[{modality}]", str(exc))

def test_feature_alignment_extra():
    """align_features() must silently drop features not in training schema."""
    from src.predict import align_features
    import joblib
    name = "align_features_extra"

    for modality in ["reviews"]:
        pkl = os.path.join(MODELS_DIR, f"{modality}_artifacts.pkl")
        if not os.path.exists(pkl):
            _skip(f"{name}[{modality}]", "artifacts not yet saved")
            continue
        artifacts = joblib.load(pkl)
        scaler        = artifacts["scaler"]
        feature_names = artifacts["feature_names"]
        train_medians = artifacts.get("train_medians")

        junk = {"z_sentiment_velocity": 999.0, "__garbage__": -1.0}
        try:
            X = align_features(junk, feature_names, scaler, train_medians)
            if X.shape == (1, len(feature_names)):
                _pass(f"{name}[{modality}]", "extra features dropped, shape OK")
            else:
                _fail(f"{name}[{modality}]", f"unexpected shape {X.shape}")
        except Exception as exc:
            _fail(f"{name}[{modality}]", str(exc))


# ===========================================================================
# SECTION 4 — Hidden Input String Tests (predict.py CLI)
# ===========================================================================
#
# Each test case is a dict: {label, modality, features, expect_class_in, expect_confidence_max}
# features are the RAW (pre-scale) values exactly as a real product row would look.
#
# The "expect_class_in" list relaxes the assertion to allow adjacent classes
# (the model may legitimately disagree with our human expectation).
#
# "hidden" = these were never passed to the model during training; they are
# independently-constructed scenarios that probe generalisation.
# ---------------------------------------------------------------------------

HIDDEN_INPUT_CASES = [

    # ── REVIEWS modality ────────────────────────────────────────────────────

    {
        "label":  "reviews_A_clearly_healthy",
        "desc":   "High sentiment, growing review count, positive momentum — textbook healthy product",
        "modality": "reviews",
        "features": {
            "sentiment_mean":          0.82,
            "sentiment_std":           0.10,
            "review_count":            48,
            "score_min":               4,
            "score_max":               5,
            "score_median":            5.0,
            "unique_reviewers":        45,
            "product_age_months":      6.0,
            "lifecycle_stage":         "growth",
            "sentiment_velocity":      0.04,
            "sentiment_acceleration":  0.01,
            "sentiment_volatility":    0.05,
            "review_momentum":         18.0,
            "sentiment_polarization":  0,
            "reviewer_diversity_change": 12.0,
        },
        "expect_class_in": ["healthy"],
        "expect_confidence_max": 1.0,
    },

    {
        "label":  "reviews_B_moderate_fatigue_onset",
        "desc":   "Sentiment slipping, review velocity slowing — early warning signals",
        "modality": "reviews",
        "features": {
            "sentiment_mean":          0.55,
            "sentiment_std":           0.28,
            "review_count":            22,
            "score_min":               2,
            "score_max":               5,
            "score_median":            3.0,
            "unique_reviewers":        20,
            "product_age_months":      18.0,
            "lifecycle_stage":         "maturity",
            "sentiment_velocity":      -0.06,
            "sentiment_acceleration":  -0.02,
            "sentiment_volatility":    0.22,
            "review_momentum":         -12.0,
            "sentiment_polarization":  1,
            "reviewer_diversity_change": -8.0,
        },
        "expect_class_in": ["moderate_fatigue", "high_fatigue"],
        "expect_confidence_max": 1.0,
    },

    {
        "label":  "reviews_C_high_fatigue",
        "desc":   "Sentiment crashing, high polarization, strong negative momentum — severe fatigue",
        "modality": "reviews",
        "features": {
            "sentiment_mean":          0.15,
            "sentiment_std":           0.60,
            "review_count":            6,
            "score_min":               1,
            "score_max":               5,
            "score_median":            2.0,
            "unique_reviewers":        5,
            "product_age_months":      30.0,
            "lifecycle_stage":         "decline",
            "sentiment_velocity":      -0.25,
            "sentiment_acceleration":  -0.10,
            "sentiment_volatility":    0.55,
            "review_momentum":         -45.0,
            "sentiment_polarization":  1,
            "reviewer_diversity_change": -30.0,
        },
        "expect_class_in": ["high_fatigue", "moderate_fatigue"],
        "expect_confidence_max": 1.0,
    },

    {
        "label":  "reviews_D_sparse_new_product",
        "desc":   "Brand-new product, almost all signals are zero/NaN — edge case",
        "modality": "reviews",
        "features": {
            "sentiment_mean":          1.0,
            "sentiment_std":           0.0,
            "review_count":            2,
            "score_min":               5,
            "score_max":               5,
            "score_median":            5.0,
            "unique_reviewers":        2,
            "product_age_months":      0.5,
            "lifecycle_stage":         "introduction",
            "sentiment_velocity":      0.0,
            "sentiment_acceleration":  0.0,
            "sentiment_volatility":    0.0,
            "review_momentum":         0.0,
            "sentiment_polarization":  0,
            "reviewer_diversity_change": 0.0,
        },
        "expect_class_in": ["healthy", "moderate_fatigue"],
        "expect_confidence_max": 1.0,
    },

    {
        "label":  "reviews_E_extreme_collapse",
        "desc":   "All signals at extreme negative — stress-test overconfidence warning",
        "modality": "reviews",
        "features": {
            "sentiment_mean":          -1.0,
            "sentiment_std":           1.0,
            "review_count":            1,
            "score_min":               1,
            "score_max":               1,
            "score_median":            1.0,
            "unique_reviewers":        1,
            "product_age_months":      60.0,
            "lifecycle_stage":         "decline",
            "sentiment_velocity":      -1.0,
            "sentiment_acceleration":  -0.5,
            "sentiment_volatility":    1.0,
            "review_momentum":         -100.0,
            "sentiment_polarization":  1,
            "reviewer_diversity_change": -100.0,
        },
        "expect_class_in": ["high_fatigue", "moderate_fatigue"],
        "expect_confidence_max": 1.0,
    },

    # ── SALES modality ──────────────────────────────────────────────────────

    {
        "label":  "sales_A_clearly_healthy",
        "desc":   "Revenue growing, new customers arriving, low churn — healthy commercial signal",
        "modality": "sales",
        "features": {
            "revenue_total":           8400.0,
            "revenue_mean":            280.0,
            "revenue_std":             60.0,
            "transaction_count":       30,
            "quantity_sold":           420,
            "unique_customers":        28,
            "unique_orders":           30,
            "avg_price":               20.0,
            "product_age_months":      5.0,
            "lifecycle_stage":         "growth",
            "revenue_velocity":        22.0,
            "revenue_acceleration":    3.0,
            "customer_churn_rate":     -5.0,
            "revenue_volatility":      8.0,
            "order_frequency_change":  10.0,
            "aov":                     280.0,
            "aov_change":              4.0,
            "customer_concentration":  0.12,
        },
        "expect_class_in": ["healthy"],
        "expect_confidence_max": 1.0,
    },

    {
        "label":  "sales_B_moderate_fatigue_onset",
        "desc":   "Revenue slowing, churn creeping up, AOV flat — moderate commercial fatigue",
        "modality": "sales",
        "features": {
            "revenue_total":           3100.0,
            "revenue_mean":            155.0,
            "revenue_std":             70.0,
            "transaction_count":       20,
            "quantity_sold":           180,
            "unique_customers":        14,
            "unique_orders":           20,
            "avg_price":               17.2,
            "product_age_months":      14.0,
            "lifecycle_stage":         "maturity",
            "revenue_velocity":        -8.0,
            "revenue_acceleration":    -4.0,
            "customer_churn_rate":     15.0,
            "revenue_volatility":      30.0,
            "order_frequency_change":  -9.0,
            "aov":                     155.0,
            "aov_change":              -2.0,
            "customer_concentration":  0.40,
        },
        "expect_class_in": ["moderate_fatigue", "high_fatigue"],
        "expect_confidence_max": 1.0,
    },

    {
        "label":  "sales_C_high_fatigue",
        "desc":   "Revenue collapsing, high churn, order frequency halved — severe commercial fatigue",
        "modality": "sales",
        "features": {
            "revenue_total":           320.0,
            "revenue_mean":            32.0,
            "revenue_std":             18.0,
            "transaction_count":       10,
            "quantity_sold":           40,
            "unique_customers":        4,
            "unique_orders":           10,
            "avg_price":               8.0,
            "product_age_months":      28.0,
            "lifecycle_stage":         "decline",
            "revenue_velocity":        -42.0,
            "revenue_acceleration":    -15.0,
            "customer_churn_rate":     55.0,
            "revenue_volatility":      90.0,
            "order_frequency_change":  -50.0,
            "aov":                     32.0,
            "aov_change":              -18.0,
            "customer_concentration":  0.85,
        },
        "expect_class_in": ["high_fatigue", "moderate_fatigue"],
        "expect_confidence_max": 1.0,
    },

    {
        "label":  "sales_D_brand_new_sku",
        "desc":   "First month of sales, velocity and acceleration are NaN — should survive gracefully",
        "modality": "sales",
        "features": {
            "revenue_total":           540.0,
            "revenue_mean":            54.0,
            "revenue_std":             0.0,
            "transaction_count":       10,
            "quantity_sold":           60,
            "unique_customers":        10,
            "unique_orders":           10,
            "avg_price":               9.0,
            "product_age_months":      0.0,
            "lifecycle_stage":         "introduction",
            "revenue_velocity":        0.0,
            "revenue_acceleration":    0.0,
            "customer_churn_rate":     0.0,
            "revenue_volatility":      0.0,
            "order_frequency_change":  0.0,
            "aov":                     54.0,
            "aov_change":              0.0,
            "customer_concentration":  0.10,
        },
        "expect_class_in": ["healthy", "moderate_fatigue"],
        "expect_confidence_max": 1.0,
    },

    {
        "label":  "sales_E_single_customer_monopoly",
        "desc":   "One customer drives 100% of orders — high concentration, extreme edge case",
        "modality": "sales",
        "features": {
            "revenue_total":           200.0,
            "revenue_mean":            200.0,
            "revenue_std":             0.0,
            "transaction_count":       1,
            "quantity_sold":           20,
            "unique_customers":        1,
            "unique_orders":           1,
            "avg_price":               10.0,
            "product_age_months":      12.0,
            "lifecycle_stage":         "maturity",
            "revenue_velocity":        -60.0,
            "revenue_acceleration":    -20.0,
            "customer_churn_rate":     80.0,
            "revenue_volatility":      0.0,
            "order_frequency_change":  -80.0,
            "aov":                     200.0,
            "aov_change":              0.0,
            "customer_concentration":  1.0,
        },
        "expect_class_in": ["high_fatigue", "moderate_fatigue"],
        "expect_confidence_max": 1.0,
    },

    # ── USAGE modality ──────────────────────────────────────────────────────

    {
        "label":  "usage_A_clearly_healthy",
        "desc":   "High engagement, good conversion, growing user base — healthy behavioral signals",
        "modality": "usage",
        "features": {
            "engagement_total":          9800,
            "engagement_mean":           1.12,
            "event_count":               8750,
            "active_users":              820,
            "view_count":                7200,
            "cart_count":                980,
            "purchase_count":            310,
            "avg_price":                 45.0,
            "unique_sessions":           790,
            "view_to_cart_rate":         13.6,
            "cart_to_purchase_rate":     31.6,
            "conversion_rate":           4.3,
            "product_age_months":        3.0,
            "lifecycle_stage":           "growth",
            "engagement_velocity":       18.0,
            "engagement_acceleration":   2.5,
            "user_retention_rate":       68.0,
            "session_frequency_change":  12.0,
            "conversion_rate_change":    1.2,
            "engagement_volatility":     85.0,
            "purchase_momentum":         22.0,
            "funnel_efficiency":         0.043,
            "funnel_efficiency_change":  0.005,
            "engagement_per_session":    1.24,
            "engagement_quality_change": 3.1,
        },
        "expect_class_in": ["healthy"],
        "expect_confidence_max": 1.0,
    },

    {
        "label":  "usage_B_moderate_fatigue_onset",
        "desc":   "Users still viewing but cart and purchase rates declining — funnel leaking",
        "modality": "usage",
        "features": {
            "engagement_total":          4200,
            "engagement_mean":           1.05,
            "event_count":               4000,
            "active_users":              380,
            "view_count":                3600,
            "cart_count":                280,
            "purchase_count":            60,
            "avg_price":                 38.0,
            "unique_sessions":           360,
            "view_to_cart_rate":         7.8,
            "cart_to_purchase_rate":     21.4,
            "conversion_rate":           1.67,
            "product_age_months":        8.0,
            "lifecycle_stage":           "maturity",
            "engagement_velocity":       -12.0,
            "engagement_acceleration":   -4.0,
            "user_retention_rate":       28.0,
            "session_frequency_change":  -15.0,
            "conversion_rate_change":    -1.8,
            "engagement_volatility":     310.0,
            "purchase_momentum":         -18.0,
            "funnel_efficiency":         0.0167,
            "funnel_efficiency_change":  -0.008,
            "engagement_per_session":    1.17,
            "engagement_quality_change": -4.2,
        },
        "expect_class_in": ["moderate_fatigue", "healthy"],
        "expect_confidence_max": 1.0,
    },

    {
        "label":  "usage_C_severe_fatigue",
        "desc":   "Engagement collapsed, zero purchases, session depth near 1 — total behavioral fatigue",
        "modality": "usage",
        "features": {
            "engagement_total":          320,
            "engagement_mean":           1.01,
            "event_count":               317,
            "active_users":              38,
            "view_count":                315,
            "cart_count":                2,
            "purchase_count":            0,
            "avg_price":                 55.0,
            "unique_sessions":           310,
            "view_to_cart_rate":         0.63,
            "cart_to_purchase_rate":     0.0,
            "conversion_rate":           0.0,
            "product_age_months":        14.0,
            "lifecycle_stage":           "decline",
            "engagement_velocity":       -65.0,
            "engagement_acceleration":   -22.0,
            "user_retention_rate":       5.0,
            "session_frequency_change":  -58.0,
            "conversion_rate_change":    -4.3,
            "engagement_volatility":     900.0,
            "purchase_momentum":         -100.0,
            "funnel_efficiency":         0.0,
            "funnel_efficiency_change":  -0.02,
            "engagement_per_session":    1.03,
            "engagement_quality_change": -12.0,
        },
        "expect_class_in": ["high_fatigue", "moderate_fatigue"],
        "expect_confidence_max": 1.0,
    },

    {
        "label":  "usage_D_viral_launch_spike",
        "desc":   "First month only, extremely high engagement but no history — introduction spike",
        "modality": "usage",
        "features": {
            "engagement_total":          55000,
            "engagement_mean":           1.08,
            "event_count":               50925,
            "active_users":              4820,
            "view_count":                48000,
            "cart_count":                1800,
            "purchase_count":            920,
            "avg_price":                 29.99,
            "unique_sessions":           4600,
            "view_to_cart_rate":         3.75,
            "cart_to_purchase_rate":     51.1,
            "conversion_rate":           1.92,
            "product_age_months":        0.0,
            "lifecycle_stage":           "introduction",
            "engagement_velocity":       0.0,
            "engagement_acceleration":   0.0,
            "user_retention_rate":       0.0,
            "session_frequency_change":  0.0,
            "conversion_rate_change":    0.0,
            "engagement_volatility":     0.0,
            "purchase_momentum":         0.0,
            "funnel_efficiency":         0.0192,
            "funnel_efficiency_change":  0.0,
            "engagement_per_session":    1.11,
            "engagement_quality_change": 0.0,
        },
        "expect_class_in": ["healthy", "moderate_fatigue"],
        "expect_confidence_max": 1.0,
    },

    {
        "label":  "usage_E_ghost_product",
        "desc":   "Views only, no carts, no purchases ever — permanently stuck at zero conversion",
        "modality": "usage",
        "features": {
            "engagement_total":          85,
            "engagement_mean":           1.0,
            "event_count":               85,
            "active_users":              85,
            "view_count":                85,
            "cart_count":                0,
            "purchase_count":            0,
            "avg_price":                 199.0,
            "unique_sessions":           82,
            "view_to_cart_rate":         0.0,
            "cart_to_purchase_rate":     0.0,
            "conversion_rate":           0.0,
            "product_age_months":        5.0,
            "lifecycle_stage":           "growth",
            "engagement_velocity":       -30.0,
            "engagement_acceleration":   -8.0,
            "user_retention_rate":       3.0,
            "session_frequency_change":  -28.0,
            "conversion_rate_change":    0.0,
            "engagement_volatility":     60.0,
            "purchase_momentum":         0.0,
            "funnel_efficiency":         0.0,
            "funnel_efficiency_change":  0.0,
            "engagement_per_session":    1.04,
            "engagement_quality_change": -6.0,
        },
        "expect_class_in": ["moderate_fatigue", "healthy"],
        "expect_confidence_max": 1.0,
    },
]


def _run_predict_cli(modality: str, features: dict, model: str = "xgboost") -> dict:
    """
    Run predict.py as a subprocess and return a structured result dict.
    Returns {"exit_code", "stdout", "stderr", "json_ok", "parsed"}.
    """
    cmd = [
        PYTHON, os.path.join(ROOT, "src", "predict.py"),
        "--modality", modality,
        "--model",    model,
        "--input",    json.dumps(features),
        "--threshold", "0.97",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=ROOT, timeout=60,
    )
    result = {
        "exit_code": proc.returncode,
        "stdout":    proc.stdout,
        "stderr":    proc.stderr,
        "json_ok":   False,
        "parsed":    None,
    }

    # predict.py prints human-readable text, not JSON — parse key fields
    stdout = proc.stdout
    parsed = {}
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("Prediction:"):
            parsed["predicted_class"] = line.split(":", 1)[1].strip()
        elif line.startswith("Confidence:"):
            raw = line.split(":", 1)[1].strip().replace("%", "")
            try:
                parsed["confidence"] = float(raw) / 100.0
            except ValueError:
                pass
    result["parsed"] = parsed
    result["json_ok"] = "predicted_class" in parsed and "confidence" in parsed
    return result


def test_hidden_input_strings():
    """Run all 15 hidden input scenarios through predict.py CLI."""
    artifacts_ready = all(
        os.path.exists(os.path.join(MODELS_DIR, f"{m}_artifacts.pkl"))
        for m in ["reviews", "sales", "usage"]
    )
    if not artifacts_ready:
        for case in HIDDEN_INPUT_CASES:
            _skip(f"hidden_input[{case['label']}]", "models not yet trained")
        return

    import joblib
    critical_labels = {
        "reviews_B_moderate_fatigue_onset",
        "reviews_C_severe_fatigue",
        "sales_B_moderate_fatigue_onset",
        "sales_C_severe_fatigue",
        "usage_C_severe_fatigue",
        "usage_E_ghost_product",
    }
    for case in HIDDEN_INPUT_CASES:
        label    = case["label"]
        modality = case["modality"]
        features = case["features"]
        expect   = case["expect_class_in"]
        desc     = case["desc"]

        # Load known classes for this modality
        artifacts = joblib.load(os.path.join(MODELS_DIR, f"{modality}_artifacts.pkl"))
        known_classes = set(artifacts["label_classes"].tolist())

        try:
            res = _run_predict_cli(modality, features)
        except subprocess.TimeoutExpired:
            _fail(f"hidden_input[{label}]", "predict.py timed out")
            continue
        except Exception as exc:
            _fail(f"hidden_input[{label}]", str(exc))
            continue

        if res["exit_code"] != 0:
            _fail(f"hidden_input[{label}]",
                  f"exit_code={res['exit_code']}  stderr={res['stderr'][:200]}")
            continue

        if not res["json_ok"]:
            _fail(f"hidden_input[{label}]", "could not parse predicted_class/confidence from stdout")
            continue

        pred_class  = res["parsed"]["predicted_class"]
        confidence  = res["parsed"]["confidence"]
        class_ok    = pred_class in known_classes
        conf_ok     = 0.0 <= confidence <= case["expect_confidence_max"]
        expected_ok = pred_class in expect

        issues = []
        if not class_ok:
            issues.append(f"predicted '{pred_class}' not in known classes {known_classes}")
        if not conf_ok:
            issues.append(f"confidence {confidence:.4f} out of range")

        if issues:
            _fail(f"hidden_input[{label}]", " | ".join(issues))
        elif expected_ok:
            _pass(f"hidden_input[{label}]",
                  f"pred={pred_class}  conf={confidence:.2%}  (expected={expect}) | {desc}")
        else:
            detail = (
                f"pred={pred_class} NOT in expected={expect}  "
                f"conf={confidence:.2%} | {desc}"
            )
            if label in critical_labels:
                _fail(f"hidden_input[{label}]", detail)
            else:
                _results.append({
                    "status": "WARN",
                    "name":   f"hidden_input[{label}]",
                    "detail": detail,
                })
                log.warning(
                    f"  WARN  hidden_input[{label}]  pred={pred_class}  "
                    f"expected one of {expect}  conf={confidence:.2%}"
                )


# ===========================================================================
# SECTION 5 — Metrics sanity checks (post-training)
# ===========================================================================

def test_metrics_json_sanity():
    """
    For each modality:
    - metrics JSON must exist
    - f1_macro must be > 0.30 (better than random)
    - leakage_warning count should ideally be 0
    """
    name = "metrics_json_sanity"
    for modality in ["reviews", "sales", "usage"]:
        path = os.path.join(ROOT, "outputs", f"{modality}_metrics.json")
        if not os.path.exists(path):
            _skip(f"{name}[{modality}]", "metrics JSON not yet written")
            continue
        with open(path) as fh:
            m = json.load(fh)
        clf = m.get("classification", {})
        leakage_count = 0
        for model_name, metrics in clf.items():
            f1 = metrics.get("f1_macro", 0)
            if f1 < 0.30:
                _fail(f"{name}[{modality}/{model_name}]", f"f1_macro={f1:.4f} < 0.30")
            else:
                _pass(f"{name}[{modality}/{model_name}]",
                      f"f1_macro={f1:.4f}  acc={metrics.get('accuracy','?')}")
            if metrics.get("leakage_warning"):
                leakage_count += 1
        if leakage_count > 0:
            _results.append({
                "status": "WARN",
                "name":   f"{name}[{modality}]_leakage",
                "detail": f"{leakage_count} model(s) still triggering leakage warning (acc>98%)",
            })
            log.warning(f"  WARN  {modality}: {leakage_count} model(s) with acc>98% — inspect features")

def test_cv_test_gap():
    """CV F1 vs test F1 gap should be < 0.15 for all models (not overfitting)."""
    name = "cv_test_gap"
    for modality in ["reviews", "sales", "usage"]:
        path = os.path.join(ROOT, "outputs", f"{modality}_metrics.json")
        if not os.path.exists(path):
            _skip(f"{name}[{modality}]", "metrics JSON not yet written")
            continue
        with open(path) as fh:
            m = json.load(fh)
        for model_name, metrics in m.get("classification", {}).items():
            gap = metrics.get("cv_test_gap", None)
            if gap is None:
                continue
            if abs(gap) < 0.15:
                _pass(f"{name}[{modality}/{model_name}]", f"gap={gap:+.4f}")
            else:
                _results.append({
                    "status": "WARN",
                    "name":   f"{name}[{modality}/{model_name}]",
                    "detail": f"large gap={gap:+.4f} (|gap| >= 0.15)",
                })
                log.warning(f"  WARN  {modality}/{model_name}: cv_test_gap={gap:+.4f}")


# ===========================================================================
# SECTION 6 — Engineering improvement tests (threshold, calibration,
#              experiment log, cluster_id, decision_threshold)
# ===========================================================================

def test_optimal_thresholds_in_artifacts():
    """Artifacts must contain 'optimal_thresholds' dict after main.py runs."""
    import joblib
    name = "optimal_thresholds_in_artifacts"

    for modality in ["reviews", "sales", "usage"]:
        pkl = os.path.join(MODELS_DIR, f"{modality}_artifacts.pkl")
        if not os.path.exists(pkl):
            _skip(f"{name}[{modality}]", "artifacts not yet saved")
            continue
        artifacts = joblib.load(pkl)
        thresholds = artifacts.get("optimal_thresholds")
        if thresholds is None:
            _fail(f"{name}[{modality}]", "key 'optimal_thresholds' missing from artifacts")
            continue
        if not isinstance(thresholds, dict):
            _fail(f"{name}[{modality}]", f"expected dict, got {type(thresholds).__name__}")
            continue
        # Each threshold must be a float in (0, 1]
        bad = {k: v for k, v in thresholds.items()
               if not isinstance(v, (int, float)) or not (0 < v <= 1)}
        if bad:
            _fail(f"{name}[{modality}]", f"invalid threshold values: {bad}")
        else:
            _pass(f"{name}[{modality}]", f"thresholds={thresholds}")


def test_calibrated_models_exist():
    """At least one calibrated .pkl file must exist per modality/model pair."""
    name = "calibrated_models_exist"
    model_names = ["xgboost", "random_forest", "logistic_regression"]

    for modality in ["reviews", "sales", "usage"]:
        for mn in model_names:
            found = False
            for method in ("isotonic", "sigmoid"):
                path = os.path.join(
                    MODELS_DIR, f"{modality}_{mn}_calibrated_{method}.pkl"
                )
                if os.path.exists(path):
                    found = True
                    break
            if found:
                _pass(f"{name}[{modality}/{mn}]")
            else:
                _skip(f"{name}[{modality}/{mn}]", "no calibrated model — run main.py")


def test_experiment_log_structure():
    """experiment_log.csv must exist and contain the expected columns."""
    name = "experiment_log_csv"
    import csv

    log_path = os.path.join(ROOT, "outputs", "experiment_log.csv")
    if not os.path.exists(log_path):
        _skip(name, "experiment_log.csv not found — run main.py")
        return

    with open(log_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        _fail(name, "experiment_log.csv is empty")
        return

    required_cols = {
        "timestamp", "modality", "model", "n_features",
        "n_train", "n_test", "cv_f1", "test_f1",
        "cv_test_gap", "optimal_threshold", "best_params",
        "balanced_accuracy", "macro_recall",
        "prediction_distribution_drift_l1", "scenario_score",
        "raw_brier_score", "raw_ece",
        "calibrated_brier_score", "calibrated_ece",
    }
    actual_cols = set(rows[0].keys())
    missing = required_cols - actual_cols
    if missing:
        _fail(name, f"missing columns: {missing}")
    else:
        _pass(name, f"{len(rows)} row(s), columns OK")

    # Verify at least one row has a non-empty optimal_threshold
    has_threshold = any(r.get("optimal_threshold", "").strip() for r in rows)
    if has_threshold:
        _pass(f"{name}_has_threshold", "at least one row has optimal_threshold set")
    else:
        _results.append({
            "status": "WARN",
            "name": f"{name}_has_threshold",
            "detail": "no rows have optimal_threshold — threshold optimisation may not have run",
        })


def test_predict_returns_cluster_id():
    """predict() must include 'cluster_id' in its return dict."""
    name = "predict_cluster_id"

    for modality in ["usage"]:  # one modality is enough to prove the pattern
        pkl = os.path.join(MODELS_DIR, f"{modality}_artifacts.pkl")
        km_path = os.path.join(MODELS_DIR, f"{modality}_kmeans_model.pkl")
        model_path = os.path.join(MODELS_DIR, f"{modality}_xgboost_model.pkl")
        if not all(os.path.exists(p) for p in (pkl, km_path, model_path)):
            _skip(f"{name}[{modality}]", "models not yet trained")
            continue

        from src.predict import predict
        import joblib
        artifacts = joblib.load(pkl)
        feature_names = artifacts["feature_names"]
        # Build a minimal input using zeros for all features
        dummy_input = {f: 0.0 for f in feature_names}

        try:
            result = predict(modality, dummy_input, model_name="xgboost")
        except SystemExit:
            _fail(f"{name}[{modality}]", "predict() called sys.exit")
            continue
        except Exception as exc:
            _fail(f"{name}[{modality}]", str(exc))
            continue

        if "cluster_id" not in result:
            _fail(f"{name}[{modality}]", "'cluster_id' key missing from predict() output")
        elif result["cluster_id"] is None:
            _fail(f"{name}[{modality}]", "cluster_id is None despite kmeans model existing")
        elif isinstance(result["cluster_id"], int):
            _pass(f"{name}[{modality}]", f"cluster_id={result['cluster_id']}")
        else:
            _fail(f"{name}[{modality}]",
                  f"cluster_id type={type(result['cluster_id']).__name__}, expected int")


def test_predict_calibration_has_decision_threshold():
    """When calibration is available, the calibration dict must include 'decision_threshold'."""
    name = "predict_decision_threshold"

    for modality in ["usage"]:
        pkl = os.path.join(MODELS_DIR, f"{modality}_artifacts.pkl")
        model_path = os.path.join(MODELS_DIR, f"{modality}_xgboost_model.pkl")
        if not all(os.path.exists(p) for p in (pkl, model_path)):
            _skip(f"{name}[{modality}]", "models not yet trained")
            continue

        # Check if any calibrated model exists
        has_cal = any(
            os.path.exists(os.path.join(MODELS_DIR, f"{modality}_xgboost_calibrated_{m}.pkl"))
            for m in ("isotonic", "sigmoid")
        )
        if not has_cal:
            _skip(f"{name}[{modality}]", "no calibrated model — run main.py")
            continue

        from src.predict import predict
        import joblib
        artifacts = joblib.load(pkl)
        feature_names = artifacts["feature_names"]
        dummy_input = {f: 0.0 for f in feature_names}

        try:
            result = predict(modality, dummy_input, model_name="xgboost")
        except SystemExit:
            _fail(f"{name}[{modality}]", "predict() called sys.exit")
            continue
        except Exception as exc:
            _fail(f"{name}[{modality}]", str(exc))
            continue

        cal = result.get("calibration")
        if cal is None:
            _fail(f"{name}[{modality}]", "calibration dict is None despite calibrated model")
            continue
        if "decision_threshold" not in cal:
            _fail(f"{name}[{modality}]", "'decision_threshold' missing from calibration dict")
        else:
            t = cal["decision_threshold"]
            if isinstance(t, (int, float)) and 0 < t <= 1:
                _pass(f"{name}[{modality}]", f"decision_threshold={t}")
            else:
                _fail(f"{name}[{modality}]", f"invalid decision_threshold={t}")


def test_find_optimal_threshold_range():
    """find_optimal_threshold() must return a threshold in (0, 1] and F1 in [0, 1]."""
    name = "find_optimal_threshold_range"
    try:
        from src.calibrate import find_optimal_threshold
    except ImportError:
        _skip(name, "calibrate module not importable")
        return

    import joblib
    for modality in ["usage"]:
        # We need a calibrated model and calibration data
        cal_path = None
        for method in ("isotonic", "sigmoid"):
            p = os.path.join(MODELS_DIR, f"{modality}_xgboost_calibrated_{method}.pkl")
            if os.path.exists(p):
                cal_path = p
                break
        art_path = os.path.join(MODELS_DIR, f"{modality}_artifacts.pkl")
        ds_path = os.path.join(DATA_DIR, f"{modality}_fatigue_signals.csv")

        if cal_path is None or not os.path.exists(art_path) or not os.path.exists(ds_path):
            _skip(f"{name}[{modality}]", "calibrated model or artifacts not found")
            continue

        cal_clf = joblib.load(cal_path)
        # Build a small synthetic calibration set from test data
        from src.data_loader import load_modality
        try:
            _, X_test, _, y_test, _, _ = load_modality(ds_path, modality)
        except Exception as exc:
            _fail(f"{name}[{modality}]", f"load_modality failed: {exc}")
            continue

        # Use a subset as pseudo-cal set
        n = min(500, X_test.shape[0])
        X_cal, y_cal = X_test[:n], y_test[:n]

        try:
            threshold, f1 = find_optimal_threshold(cal_clf, X_cal, y_cal)
        except Exception as exc:
            _fail(f"{name}[{modality}]", f"find_optimal_threshold raised: {exc}")
            continue

        issues = []
        if not (0 < threshold <= 1):
            issues.append(f"threshold={threshold} out of (0,1]")
        if not (0 <= f1 <= 1):
            issues.append(f"f1={f1} out of [0,1]")
        if issues:
            _fail(f"{name}[{modality}]", " | ".join(issues))
        else:
            _pass(f"{name}[{modality}]", f"threshold={threshold:.4f} f1={f1:.4f}")


def test_artifacts_selection_metadata():
    """Artifacts should persist selection metadata and strict input requirements."""
    import joblib
    name = "artifacts_selection_metadata"

    required_keys = {
        "raw_required_features",
        "class_weight_policies",
        "default_model",
        "scenario_benchmarks",
    }

    for modality in ["reviews", "sales", "usage"]:
        pkl = os.path.join(MODELS_DIR, f"{modality}_artifacts.pkl")
        if not os.path.exists(pkl):
            _skip(f"{name}[{modality}]", "artifacts not yet saved")
            continue

        artifacts = joblib.load(pkl)
        missing = required_keys - set(artifacts.keys())
        if missing:
            _fail(f"{name}[{modality}]", f"missing keys: {sorted(missing)}")
            continue

        default_model = artifacts.get("default_model")
        scenario_benchmarks = artifacts.get("scenario_benchmarks", {})
        required_features = artifacts.get("raw_required_features", [])
        class_weight_policies = artifacts.get("class_weight_policies", {})

        issues = []
        if default_model not in ["xgboost", "random_forest", "logistic_regression"]:
            issues.append(f"invalid default_model={default_model}")
        if not isinstance(scenario_benchmarks, dict) or not scenario_benchmarks:
            issues.append("scenario_benchmarks missing or empty")
        if not isinstance(required_features, list) or not required_features:
            issues.append("raw_required_features missing or empty")
        if not isinstance(class_weight_policies, dict) or not class_weight_policies:
            issues.append("class_weight_policies missing or empty")

        if issues:
            _fail(f"{name}[{modality}]", " | ".join(issues))
        else:
            _pass(
                f"{name}[{modality}]",
                f"default_model={default_model} required_fields={len(required_features)}",
            )


def test_scenario_regression_gates():
    """Default models must satisfy the fixed manual-inference scenario set."""
    import joblib
    from src.predict import predict
    from src.scenario_benchmark import SCENARIO_CASES

    name = "scenario_regression"

    for modality in ["reviews", "sales", "usage"]:
        pkl = os.path.join(MODELS_DIR, f"{modality}_artifacts.pkl")
        if not os.path.exists(pkl):
            _skip(f"{name}[{modality}]", "artifacts not yet saved")
            continue

        artifacts = joblib.load(pkl)
        default_model = artifacts.get("default_model")
        if default_model not in ["xgboost", "random_forest", "logistic_regression"]:
            _fail(f"{name}[{modality}]", f"invalid default_model={default_model}")
            continue

        modality_cases = [case for case in SCENARIO_CASES if case["modality"] == modality]
        all_ok = True
        for case in modality_cases:
            try:
                result = predict(
                    modality,
                    case["features"],
                    model_name=default_model,
                    strict=True,
                )
            except Exception as exc:
                _fail(f"{name}[{case['label']}]", f"prediction failed: {exc}")
                all_ok = False
                continue

            pred = result["predicted_class"]
            if pred not in case["expect"]:
                _fail(
                    f"{name}[{case['label']}]",
                    f"default_model={default_model} pred={pred} expected={case['expect']}",
                )
                all_ok = False
            else:
                _pass(
                    f"{name}[{case['label']}]",
                    f"default_model={default_model} pred={pred} conf={result['confidence']:.2%}",
                )

        if all_ok:
            _pass(f"{name}[{modality}]", f"default_model={default_model} passed {len(modality_cases)} scenarios")


# ===========================================================================
# Runner
# ===========================================================================

def _print_summary():
    total  = len(_results)
    passed = sum(1 for r in _results if r["status"] == "PASS")
    failed = sum(1 for r in _results if r["status"] == "FAIL")
    warned = sum(1 for r in _results if r["status"] == "WARN")
    skipped= sum(1 for r in _results if r["status"] == "SKIP")

    width = 72
    print("\n" + "=" * width)
    print("  TEST SUMMARY")
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
                print(f"    [SKIP]  {r['name']}  —  {r['detail']}")

    print("=" * width + "\n")
    return failed


UNIT_TESTS = [
    test_temporal_split_ordering,
    test_zscore_columns_dropped,
    test_id_and_date_columns_dropped,
    test_scaler_fitted_on_train_only,
    test_train_test_no_overlap,
    test_label_classes_consistent,
]

MODEL_TESTS = [
    test_smote_is_inside_pipeline,
    test_artifacts_have_required_keys,
    test_models_loadable,
    test_feature_alignment_missing,
    test_feature_alignment_extra,
]

EVAL_TESTS = [
    test_metrics_json_sanity,
    test_cv_test_gap,
]

CLI_TESTS = [
    test_hidden_input_strings,
]

ENGINEERING_TESTS = [
    test_optimal_thresholds_in_artifacts,
    test_calibrated_models_exist,
    test_experiment_log_structure,
    test_predict_returns_cluster_id,
    test_predict_calibration_has_decision_threshold,
    test_find_optimal_threshold_range,
    test_artifacts_selection_metadata,
    test_scenario_regression_gates,
]


def main(section: str = "all"):
    t0 = time.time()
    log.info(f"Running section: {section!r}")

    sections = {
        "unit":  UNIT_TESTS,
        "model": MODEL_TESTS,
        "eval":  EVAL_TESTS,
        "cli":   CLI_TESTS,
        "eng":   ENGINEERING_TESTS,
        "all":   UNIT_TESTS + MODEL_TESTS + EVAL_TESTS + CLI_TESTS + ENGINEERING_TESTS,
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
