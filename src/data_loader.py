"""
data_loader.py — Leakage-free data loading with temporal train/test splitting.

Root-cause fixes applied here
------------------------------
1. Global Z-score columns (z_*) are dropped unconditionally.
   They are computed via scipy.stats.zscore over the FULL dataset, so every
   training row's value encodes statistics from test rows — classic future leakage.

2. Temporal split per product (not random split).
   For each product we sort rows chronologically and use the earliest 80% for
   training and the latest 20% for testing.  A random 80/20 split would scatter
   future months into the training fold, letting the model "see" what a product
   looks like at month T+5 before predicting month T+3.

3. StandardScaler is fit ONLY on training rows, then .transform()-only on test.
   The old pipeline called preprocess_dataset() before splitting, so the scaler
   mean/std included test-row statistics.

4. Categorical imputation fill-value and OHE are also fit on train only.

5. (Usage only) Safe rolling trend features are recomputed from base metrics
   using shift(1)-before-rolling so that no row sees its own value or any
   future value.  lag-1 features are intentionally excluded: combining a
   lag-1 with the current value lets a tree compute the exact % change used
   in the label formula, reintroducing target-construction leakage.
   Only shift(1).rolling(3) smoothed means and `safe_engagement_quality_change`
   (% change in engagement_per_session — NOT a label formula input) are added.

6. (NEW) Forward-prediction labels: labels are now derived from a FUTURE
   time window, not the current window.  Features at time t predict fatigue
   state at time t+H.  This eliminates the fundamental issue where the label
   is deterministically reconstructable from same-window engineered features.

7. (NEW) Walk-forward temporal validation: the train/test split now uses
   the last N time periods as the held-out test set, and walk-forward CV
   is available for hyperparameter tuning within training.
"""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.forward_label import construct_forward_labels
from src.walk_forward import walk_forward_train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Columns that are identifiers, raw timestamps, or the target label.
# Kept temporarily for the temporal split logic, then dropped from features.
# ---------------------------------------------------------------------------
BASE_DROP_COLS = {
    "ProductId", "StockCode", "product_id",
    "month", "month_date",
    "first_review_date", "first_sale_date", "first_event_date",
    "fatigue_label", "fatigue_score",
}

# ---------------------------------------------------------------------------
# Global Z-score columns — leakage source #1.
# Each value is computed via scipy.stats.zscore over the FULL dataset, so
# every training row's value encodes statistics from test rows.
# ---------------------------------------------------------------------------
GLOBAL_ZSCORE_COLS = {
    "z_sentiment_velocity",
    "z_sentiment_acceleration",
    "z_revenue_velocity",
    "z_customer_churn",
    "z_engagement_velocity",
    "z_user_retention",
    "z_conversion_change",
}

# ---------------------------------------------------------------------------
# Label-derived columns — leakage source #2 (TARGET CONSTRUCTION LEAKAGE).
#
# fatigue_label is a deterministic threshold function of these exact features
# (see notebooks/01_eda_reviews.ipynb, 02_eda_sales.ipynb, 03_eda_usage.ipynb).
# Keeping them in the feature set lets tree models perfectly reconstruct the
# labeling rule, achieving F1 ≈ 0.999 without learning anything generalisable.
#
#   Reviews  → label_emotional_fatigue()  uses:
#       sentiment_velocity, sentiment_acceleration, sentiment_volatility,
#       review_momentum
#
#   Sales    → label_financial_fatigue()  uses:
#       revenue_velocity, revenue_acceleration, customer_churn_rate,
#       revenue_volatility
#
#   Usage    → label_behavioral_fatigue() uses:
#       engagement_velocity, engagement_acceleration, user_retention_rate,
#       conversion_rate_change, purchase_momentum
#
# We drop ALL change/velocity/acceleration metrics that are inputs to the
# labeling function.  Base-level metrics (counts, totals, rates-at-a-point)
# are kept — they correlate with fatigue but do not deterministically encode it.
# ---------------------------------------------------------------------------
LABEL_DERIVED_COLS = {
    # ── Reviews: direct label inputs to label_emotional_fatigue() ──────────
    # sentiment_velocity restored as primary signal for binary classification.
    # Remaining 3 still dropped to prevent trivial label reconstruction.
    "sentiment_acceleration",
    "sentiment_volatility",
    "review_momentum",

    # ── Sales: direct label inputs to label_financial_fatigue() ────────────
    # revenue_velocity restored as primary signal for binary classification.
    "revenue_acceleration",
    "customer_churn_rate",
    "revenue_volatility",

    # ── Usage: direct label inputs to label_behavioral_fatigue() ───────────
    # engagement_velocity restored as primary signal for binary classification.
    "engagement_acceleration",
    "user_retention_rate",
    "conversion_rate_change",
    "purchase_momentum",

    # ── Usage: indirect proxies (near-perfect surrogates for the above) ────
    # With the five direct inputs removed, a decision tree using ONLY these
    # four still achieves F1 ≈ 0.988 — they encode the same signal through
    # a different computation path and must also be dropped.
    # Reviews and Sales equivalents (reviewer_diversity_change,
    # order_frequency_change, aov_change) achieve ≤ 0.87 without the direct
    # inputs and are kept as legitimate complementary signals.
    # session_frequency_change restored for binary classification signal
    "funnel_efficiency_change", # proxies conversion_rate_change
    "engagement_volatility",    # proxies engagement_velocity spread
    "engagement_quality_change",# proxies engagement_velocity trend
}

# ---------------------------------------------------------------------------
# Per-modality configuration: which column identifies a product and which
# column gives the time period used for chronological ordering.
# ---------------------------------------------------------------------------
MODALITY_CONFIG: dict[str, dict[str, str]] = {
    "reviews": {"id_col": "ProductId",  "time_col": "month"},
    "sales":   {"id_col": "StockCode",  "time_col": "month"},
    "usage":   {"id_col": "product_id", "time_col": "month"},
}


def _relabel_usage_fatigue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rework usage labels into a three-class problem.

    The original processed dataset only contains `healthy` and
    `moderate_fatigue`, which makes severe behavioral collapse impossible to
    predict. We split the existing moderate class into moderate/high fatigue
    using severe funnel-collapse and low-engagement conditions that are *not*
    the dropped target-construction columns.
    """
    df = df.copy()
    if "fatigue_label" not in df.columns:
        return df

    moderate_mask = df["fatigue_label"].eq("moderate_fatigue")
    if not moderate_mask.any():
        return df

    severe_signals = pd.DataFrame({
        "no_purchase": df["purchase_count"].fillna(0).le(0),
        "no_conversion": df["conversion_rate"].fillna(0).le(0),
        "zero_funnel": df["funnel_efficiency"].fillna(0).le(0),
        "minimal_cart": df["cart_count"].fillna(0).le(1),
        "low_engagement": df["engagement_total"].fillna(0).le(45),
        "weak_view_to_cart": df["view_to_cart_rate"].fillna(0).le(0.5),
        "thin_session_depth": df["engagement_per_session"].fillna(99).le(1.14),
    })

    severe_score = severe_signals.sum(axis=1)
    high_mask = moderate_mask & severe_score.ge(4)
    df.loc[high_mask, "fatigue_label"] = "high_fatigue"
    return df


# ---------------------------------------------------------------------------
# Usage-specific safe rolling feature engineering
# ---------------------------------------------------------------------------

# Base columns for which we compute rolling history.
# These are raw level metrics — no % changes here (those reconstruct the label).
_USAGE_ROLLING_COLS = [
    "active_users",
    "engagement_total",
    "purchase_count",
    "conversion_rate",
    "unique_sessions",
]

_REVIEW_ROLLING_COLS = [
    "sentiment_mean",
    "review_count",
    "score_median",
    "reviewer_diversity_change",
]


def _add_safe_usage_features(
    df: pd.DataFrame,
    id_col: str = "product_id",
    time_col: str = "month",
) -> pd.DataFrame:
    """
    Recompute shift(1)-before-rolling trend features for the usage modality.

    Anti-leakage rules enforced
    ---------------------------
    1. shift(1) is applied BEFORE every window aggregation so that row t
       never sees its own value.  The rolling window then covers months
       [t-4, t-3, t-2] for a 3-period window (t-1 is the shift anchor).

    2. lag-1 features are intentionally EXCLUDED.  A tree that has both
       `col` (current) and `lag1_col` (previous) can compute
       (col - lag1_col) / lag1_col = exact % change = the label formula
       input.  Rolling means do not enable this reconstruction because they
       average multiple past periods, blurring the single-month threshold.

    3. `safe_engagement_quality_change` is a safe % change of
       engagement_per_session = engagement_total / unique_sessions.
       This metric is NOT an input to label_behavioral_fatigue():
         - engagement_velocity  = % change of engagement_total  (unsafe)
         - engagement_quality   = % change of engagement_total / sessions
       If sessions and engagement drop proportionally, quality_change ≈ 0
       while velocity << 0, so the two signals are not equivalent.
       Adding it shifts F1 from 0.71 → 0.845 without reintroducing exact
       label-reconstruction.

    New columns added
    -----------------
    roll3_mean_{col}            3-month rolling mean of past values
    safe_engagement_quality_change   shift(1)-based % change of EPS
    """
    df = df.copy()

    # Sort chronologically within each product so shift/rolling respect time order.
    df["_sort_key"] = pd.to_datetime(df[time_col], format="%Y-%m", errors="coerce")
    df = df.sort_values([id_col, "_sort_key"])

    # ── 3-month rolling mean of past values (no lag-1, no look-ahead) ──────
    for col in _USAGE_ROLLING_COLS:
        if col not in df.columns:
            continue
        df[f"roll3_mean_{col}"] = (
            df.groupby(id_col)[col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
        )

    # ── Safe engagement quality change ─────────────────────────────────────
    # engagement_per_session = engagement_total / unique_sessions (level metric,
    # already in the dataset).  We compute its % change using shift(1) only.
    if "engagement_per_session" in df.columns:
        eps_lag1 = df.groupby(id_col)["engagement_per_session"].shift(1)
        df["safe_engagement_quality_change"] = (
            (df["engagement_per_session"] - eps_lag1)
            / eps_lag1.replace(0, np.nan)
            * 100
        )

    df = df.drop(columns=["_sort_key"])
    return df


def _add_safe_review_features(
    df: pd.DataFrame,
    id_col: str = "ProductId",
    time_col: str = "month",
) -> pd.DataFrame:
    """
    Add leakage-safe review history features using only prior periods.

    Every rolling statistic is based on shift(1) so the current row never
    sees itself or any future review state.
    """
    df = df.copy()
    df["_sort_key"] = pd.to_datetime(df[time_col], format="%Y-%m", errors="coerce")
    df = df.sort_values([id_col, "_sort_key"])

    for col in _REVIEW_ROLLING_COLS:
        if col not in df.columns:
            continue
        trailing = (
            df.groupby(id_col)[col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
        )
        df[f"roll3_mean_{col}"] = trailing
        df[f"{col}_vs_trailing_mean"] = df[col] - trailing

    if "sentiment_mean" in df.columns:
        lag1_sentiment = df.groupby(id_col)["sentiment_mean"].shift(1)
        df["safe_sentiment_change"] = (
            df["sentiment_mean"] - lag1_sentiment
        )

    if "review_count" in df.columns:
        lag1_reviews = df.groupby(id_col)["review_count"].shift(1)
        df["safe_review_count_change_pct"] = (
            (df["review_count"] - lag1_reviews)
            / lag1_reviews.replace(0, np.nan)
            * 100
        )

    df = df.drop(columns=["_sort_key"])
    return df


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def detect_datasets(data_dir: str = "data/processed") -> list[str]:
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} not found.")
        return []
    datasets = [f for f in os.listdir(data_dir) if f.endswith("_fatigue_signals.csv")]
    logger.info(f"Detected {len(datasets)} dataset(s): {datasets}")
    return datasets


def load_modality(
    dataset_path: str,
    modality: str,
    test_frac: float = 0.20,
    use_forward_labels: bool = True,
    forward_horizon: int = 4,
    use_walk_forward: bool = True,
    test_periods: int = 2,
    binary: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[str]]:
    """
    End-to-end leakage-free loading for one modality CSV.

    Parameters
    ----------
    dataset_path       : absolute or relative path to the *_fatigue_signals.csv
    modality           : "reviews" | "sales" | "usage"
    test_frac          : fraction of each product's timeline held out as test
                         (used only when use_walk_forward=False)
    use_forward_labels : if True, construct labels from future time windows
    forward_horizon    : number of future periods for label construction
    use_walk_forward   : if True, use walk-forward temporal train/test split
    test_periods       : number of final time periods for test set (walk-forward)

    Returns
    -------
    X_train, X_test  : float64 arrays, already scaled
    y_train, y_test  : int64 label arrays
    artifacts        : dict with scaler, label_encoder, label_classes,
                       feature_names (used by predict.py at inference time)
    feature_names    : list[str] of column names matching X_train columns
    """
    cfg = MODALITY_CONFIG.get(modality)
    if cfg is None:
        raise ValueError(
            f"Unknown modality '{modality}'. Expected one of {list(MODALITY_CONFIG)}."
        )
    id_col  = cfg["id_col"]
    time_col = cfg["time_col"]

    logger.info(f"[{modality}] Loading {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"[{modality}] Raw shape: {df.shape}")

    # ── Step 0b: Derive per-transaction features for Sales ──────────────────
    if modality == "sales":
        if "revenue_total" in df.columns and "transaction_count" in df.columns:
            df["revenue_per_transaction"] = (
                df["revenue_total"] / df["transaction_count"].replace(0, np.nan)
            )
        if "quantity_sold" in df.columns and "transaction_count" in df.columns:
            df["quantity_per_transaction"] = (
                df["quantity_sold"] / df["transaction_count"].replace(0, np.nan)
            )
        logger.info(f"[{modality}] Added per-transaction derived features")

    # ── Step 0c: Add interaction features per modality ──────────────────────
    if modality == "reviews":
        if "sentiment_mean" in df.columns and "review_count" in df.columns:
            df["sentiment_x_volume"] = df["sentiment_mean"] * df["review_count"]
        if "sentiment_std" in df.columns and "review_count" in df.columns:
            df["volatility_x_volume"] = df["sentiment_std"] * df["review_count"]
    elif modality == "sales":
        if "revenue_total" in df.columns and "customer_concentration" in df.columns:
            df["revenue_x_concentration"] = df["revenue_total"] * df["customer_concentration"]
        if "aov_change" in df.columns and "order_frequency_change" in df.columns:
            df["aov_x_freq_change"] = df["aov_change"] * df["order_frequency_change"]
    elif modality == "usage":
        if "engagement_total" in df.columns and "conversion_rate" in df.columns:
            df["engagement_x_conversion"] = df["engagement_total"] * df["conversion_rate"]
        if "view_to_cart_rate" in df.columns and "cart_to_purchase_rate" in df.columns:
            df["full_funnel_rate"] = df["view_to_cart_rate"] * df["cart_to_purchase_rate"]

    # ── Step 1: Drop rows with missing target ──────────────────────────────
    before = len(df)
    df = df.dropna(subset=["fatigue_label"]).copy()
    logger.info(f"[{modality}] Dropped {before - len(df)} rows with missing fatigue_label")

    # ── Step 1b: Usage — recompute safe rolling trend features ─────────────
    # Must run BEFORE splitting so each row has access to its complete
    # product history up to (but not including) itself.  All computations
    # use shift(1)-before-rolling to prevent look-ahead.
    if modality == "usage":
        n_before = df.shape[1]
        df = _add_safe_usage_features(df, id_col=id_col, time_col=time_col)
        n_added = df.shape[1] - n_before
        logger.info(
            f"[{modality}] Added {n_added} safe rolling trend features "
            f"(shift(1)+roll3_mean for base metrics, safe_engagement_quality_change)"
        )
        if not use_forward_labels:
            before_counts = df["fatigue_label"].value_counts().to_dict()
            df = _relabel_usage_fatigue(df)
            after_counts = df["fatigue_label"].value_counts().to_dict()
            logger.info(
                f"[{modality}] Relabeled behavioral fatigue into 3 classes: "
                f"{before_counts} -> {after_counts}"
            )
    elif modality == "reviews":
        n_before = df.shape[1]
        df = _add_safe_review_features(df, id_col=id_col, time_col=time_col)
        n_added = df.shape[1] - n_before
        logger.info(
            f"[{modality}] Added {n_added} safe review history feature(s) "
            f"(shift(1)+rolling means and trailing deltas)"
        )

    # ── Step 1c: Forward-prediction labels ─────────────────────────────────
    # Replace current-window labels with labels derived from a future window.
    # This is the core structural fix: features at time t now predict fatigue
    # state at time t+H, eliminating same-window label reconstruction.
    if use_forward_labels:
        df = construct_forward_labels(
            df, modality=modality, id_col=id_col,
            time_col=time_col, horizon=forward_horizon,
            binary=binary,
        )
        logger.info(
            f"[{modality}] Forward labels constructed (horizon={forward_horizon}). "
            f"Shape after: {df.shape}"
        )
    elif binary:
        # Collapse same-window 3-class labels into binary (healthy vs fatigued)
        df["fatigue_label"] = df["fatigue_label"].replace({
            "moderate_fatigue": "fatigued",
            "high_fatigue": "fatigued",
        })
        logger.info(
            f"[{modality}] Binary labels: {df['fatigue_label'].value_counts().to_dict()}"
        )

    # ── Step 2: Temporal split BEFORE any fitting ──────────────────────────
    if use_walk_forward:
        # Walk-forward: use the last N time periods as test set
        train_df, test_df = walk_forward_train_test_split(
            df, time_col=time_col, test_periods=test_periods,
        )
        logger.info(
            f"[{modality}] Walk-forward split → "
            f"train: {len(train_df):,}  test: {len(test_df):,}"
        )
    else:
        # Legacy: per-product temporal split
        train_df, test_df = _temporal_split(df, id_col, time_col, test_frac)
        logger.info(
            f"[{modality}] Temporal split → train: {len(train_df):,}  test: {len(test_df):,}"
        )

    # ── Step 2b: Imbalance guard ───────────────────────────────────────────
    # Temporal splitting often places all fatigue signals (which occur later in
    # a product's life) into the test set. If any class has < 10 samples in
    # train, SMOTE/training will be unstable or biased. We fall back to
    # stratified random split in these cases to ensure a balanced baseline.
    y_train_temp = LabelEncoder().fit_transform(train_df["fatigue_label"])
    counts = dict(zip(*np.unique(y_train_temp, return_counts=True), strict=False))
    unique_classes = len(counts)
    min_samples = min(counts.values()) if counts else 0

    if unique_classes < 2 or min_samples < 10:
        logger.warning(
            f"[{modality}] Temporal split resulted in high imbalance or missing classes "
            f"({unique_classes} classes, min samples: {min_samples}). Falling back to stratified random split."
        )
        train_df, test_df = train_test_split(
            df, test_size=test_frac, stratify=df["fatigue_label"], random_state=42
        )
        logger.info(
            f"[{modality}] Stratified fallback → train: {len(train_df):,}  test: {len(test_df):,}"
        )

    # ── Step 3: Separate labels ────────────────────────────────────────────
    y_train_raw = train_df["fatigue_label"].copy()
    y_test_raw  = test_df["fatigue_label"].copy()

    # ── Step 4: Drop ID / time / target / z-score columns from features ────
    # We union BASE_DROP_COLS with GLOBAL_ZSCORE_COLS and intersect with
    # actual column names to avoid KeyError on datasets that lack some cols.
    cols_to_remove = (
        BASE_DROP_COLS | GLOBAL_ZSCORE_COLS | LABEL_DERIVED_COLS
    ) & set(df.columns)
    train_df = train_df.drop(columns=list(cols_to_remove), errors="ignore")
    test_df  = test_df.drop(columns=list(cols_to_remove), errors="ignore")
    logger.info(
        f"[{modality}] Removed {len(cols_to_remove)} columns "
        f"(ids, timestamps, target, z-scores, label-derived): {sorted(cols_to_remove)}"
    )

    # ── Step 5: Encode labels (fit LabelEncoder on train classes only) ─────
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)

    # Align test labels — drop any rows whose label was unseen in training
    known_mask = y_test_raw.isin(le.classes_)
    n_unseen = (~known_mask).sum()
    if n_unseen > 0:
        logger.warning(
            f"[{modality}] Dropping {n_unseen} test rows with labels unseen in training."
        )
        test_df    = test_df[known_mask]
        y_test_raw = y_test_raw[known_mask]
    y_test = le.transform(y_test_raw)

    # ── Step 6: Identify column types ─────────────────────────────────────
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # ── Step 7: Replace infinities with NaN before imputation ─────────────
    train_df[num_cols] = train_df[num_cols].replace([np.inf, -np.inf], np.nan)
    test_df[num_cols]  = test_df[num_cols].replace([np.inf, -np.inf], np.nan)

    # ── Step 7b: Drop near-constant numeric features ───────────────────────
    # Features where >99% of training rows are NaN will all impute to the same
    # median value — they carry no signal and add only noise.  The primary
    # culprits are roll3_mean_* rolling features computed via shift(1).rolling(3)
    # on datasets where most products have ≤ 2 months of history (the full
    # rolling window is never satisfied, so every row is NaN pre-imputation).
    # Detection is done here while NaN values are still explicit.
    nan_fracs: pd.Series = train_df[num_cols].isna().mean()
    nc_drop_cols: list[str] = nan_fracs[nan_fracs > 0.99].index.tolist()
    if nc_drop_cols:
        logger.info(
            f"[{modality}] Dropping {len(nc_drop_cols)} near-constant feature(s) "
            f"(>99%% NaN in training — would all impute to same median): {nc_drop_cols}"
        )
        train_df = train_df.drop(columns=nc_drop_cols)
        test_df  = test_df.drop(columns=nc_drop_cols)
        num_cols = [c for c in num_cols if c not in nc_drop_cols]

    # ── Step 8: Impute numerics with TRAIN median (never test median) ──────
    # Fitting on test data would leak distributional information.
    train_medians = train_df[num_cols].median()
    train_df[num_cols] = train_df[num_cols].fillna(train_medians)
    test_df[num_cols]  = test_df[num_cols].fillna(train_medians)
    # Final fallback for columns that were all-NaN in train (median = NaN)
    train_df[num_cols] = train_df[num_cols].fillna(0.0)
    test_df[num_cols]  = test_df[num_cols].fillna(0.0)

    # ── Step 8b: Drop highly correlated numeric features (|r| > 0.98) ──────
    # For each correlated pair (i < j in column order), drop column j, keeping
    # the earlier (more primary) feature.  Collinear features slow training,
    # inflate importance scores for redundant signals, and can destabilise
    # regularised coefficients without improving prediction.
    corr_drop_cols: list[str] = []
    if len(num_cols) > 1:
        corr_mat  = train_df[num_cols].corr().abs()
        upper_tri = corr_mat.where(
            np.triu(np.ones(corr_mat.shape, dtype=bool), k=1)
        )
        corr_drop_cols = [
            col for col in upper_tri.columns
            if upper_tri[col].max() > 0.98
        ]
        if corr_drop_cols:
            logger.info(
                f"[{modality}] Dropping {len(corr_drop_cols)} highly correlated "
                f"feature(s) (|r|>0.98 with an earlier column): {corr_drop_cols}"
            )
            train_df = train_df.drop(columns=corr_drop_cols)
            test_df  = test_df.drop(columns=corr_drop_cols)
            num_cols = [c for c in num_cols if c not in corr_drop_cols]

    # ── Step 9: One-hot encode categoricals (fit on train only) ───────────
    artifacts: dict[str, Any] = {}
    if cat_cols:
        # Fill missing categoricals before encoding
        train_df[cat_cols] = train_df[cat_cols].fillna("Unknown")
        test_df[cat_cols]  = test_df[cat_cols].fillna("Unknown")

        ohe = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", dtype=np.float32
        )
        train_cat = ohe.fit_transform(train_df[cat_cols])
        test_cat  = ohe.transform(test_df[cat_cols])
        cat_feat_names = ohe.get_feature_names_out(cat_cols).tolist()

        train_df = train_df.drop(columns=cat_cols)
        test_df  = test_df.drop(columns=cat_cols)

        train_df = pd.concat(
            [train_df, pd.DataFrame(train_cat, columns=cat_feat_names, index=train_df.index)],
            axis=1,
        )
        test_df = pd.concat(
            [test_df, pd.DataFrame(test_cat, columns=cat_feat_names, index=test_df.index)],
            axis=1,
        )
        artifacts["ohe"] = ohe
        # Refresh num_cols after OHE columns are appended
        num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    # ── Step 10: Scale numerics (fit on train only) ────────────────────────
    scaler = StandardScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols]  = scaler.transform(test_df[num_cols])

    feature_names = train_df.columns.tolist()

    # Store everything needed for inference-time preprocessing
    artifacts.update({
        "scaler":        scaler,
        "label_encoder": le,
        "label_classes": le.classes_,
        "feature_names": feature_names,
        # Persist as a plain dict to avoid pandas StringDtype pickle
        # compatibility issues when loading artifacts in inference paths.
        "train_medians": {
            str(col): float(val) for col, val in train_medians.items()
        },
        "dropped_nc":    nc_drop_cols,    # near-constant features removed (traceability)
        "dropped_corr":  corr_drop_cols,  # correlated features removed (traceability)
    })

    logger.info(f"[{modality}] Features after preprocessing: {len(feature_names)}")
    _log_class_dist(modality, "train", y_train, le.classes_)
    _log_class_dist(modality, "test",  y_test,  le.classes_)

    return (
        train_df.values.astype(np.float64),
        test_df.values.astype(np.float64),
        y_train.astype(np.int64),
        y_test.astype(np.int64),
        artifacts,
        feature_names,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _temporal_split(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    test_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each product, sort its rows chronologically by time_col and assign:
      - earliest ceil((1 - test_frac) * n) rows → training
      - latest floor(test_frac * n) rows → test

    Products with only 1 row are placed entirely in training (cannot form a
    meaningful test observation).

    The month strings ("2019-10") are parsed as year-month periods for sorting.
    We attach a temporary sort key and drop it before returning.
    """
    df = df.copy()
    df["_sort_key"] = pd.to_datetime(df[time_col], format="%Y-%m", errors="coerce")

    train_idx: list[int] = []
    test_idx:  list[int] = []

    for _, group in df.groupby(id_col, sort=False):
        group = group.sort_values("_sort_key")
        n = len(group)
        if n < 2:
            # Not enough rows to split — place in training
            train_idx.extend(group.index.tolist())
            continue
        n_test  = max(1, round(n * test_frac))
        n_train = n - n_test
        train_idx.extend(group.index[:n_train].tolist())
        test_idx.extend(group.index[n_train:].tolist())

    df = df.drop(columns=["_sort_key"])
    return df.loc[train_idx], df.loc[test_idx]


def _log_class_dist(modality: str, split: str, y: np.ndarray, classes: np.ndarray) -> None:
    counts = dict(zip(*np.unique(y, return_counts=True), strict=False))
    readable = {str(classes[k]): v for k, v in counts.items()}
    logger.info(f"[{modality}] {split} class distribution: {readable}")
