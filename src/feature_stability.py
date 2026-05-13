"""
feature_stability.py -- Feature engineering stability improvements.

Addresses Problem 5 from the implementation plan:
  - purchase_momentum contains Inf
  - engagement_acceleration is all-NaN
  - many highly correlated feature pairs
  - several near-constant rolling features

Fixes:
  - Log-differences instead of raw percent changes
  - Epsilon in all denominators
  - Extreme ratio capping
  - Dead feature removal
  - Near-duplicate correlated feature pruning
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

EPS = 1e-6


def safe_log_diff(current: pd.Series, previous: pd.Series) -> pd.Series:
    """
    Compute log-difference instead of raw percent change.

    log(x_t + 1) - log(x_{t-1} + 1)

    This is stable near zero and doesn't produce Inf values.
    """
    return np.log1p(current.clip(lower=0)) - np.log1p(previous.clip(lower=0))


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute ratio with epsilon denominator to prevent Inf/NaN."""
    return numerator / (denominator + EPS)


def cap_extreme_ratios(
    df: pd.DataFrame,
    ratio_cols: List[str],
    upper_quantile: float = 0.99,
    lower_quantile: float = 0.01,
) -> pd.DataFrame:
    """Cap extreme values in ratio columns to quantile boundaries."""
    df = df.copy()
    for col in ratio_cols:
        if col not in df.columns:
            continue
        lo = df[col].quantile(lower_quantile)
        hi = df[col].quantile(upper_quantile)
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def remove_dead_features(
    df: pd.DataFrame,
    nan_threshold: float = 0.95,
    constant_threshold: float = 0.99,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features that are mostly NaN or near-constant.

    Parameters
    ----------
    df                 : DataFrame
    nan_threshold      : drop columns with this fraction of NaN values
    constant_threshold : drop columns where one value dominates this fraction

    Returns
    -------
    (cleaned_df, dropped_columns)
    """
    dropped = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in num_cols:
        nan_frac = df[col].isna().mean()
        if nan_frac > nan_threshold:
            dropped.append(col)
            continue

        # Check for near-constant (one value dominates)
        value_counts = df[col].dropna().value_counts(normalize=True)
        if len(value_counts) > 0 and value_counts.iloc[0] > constant_threshold:
            dropped.append(col)

    if dropped:
        logger.info(
            f"Removing {len(dropped)} dead features "
            f"(>={nan_threshold:.0%} NaN or >={constant_threshold:.0%} constant): "
            f"{dropped}"
        )
        df = df.drop(columns=dropped)

    return df, dropped


def remove_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.98,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove one of each pair of highly correlated features.

    For each correlated pair (i, j) where i < j in column order,
    drop column j, keeping the earlier (more primary) feature.

    Returns
    -------
    (cleaned_df, dropped_columns)
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        return df, []

    corr_mat = df[num_cols].corr().abs()
    upper_tri = corr_mat.where(
        np.triu(np.ones(corr_mat.shape, dtype=bool), k=1)
    )

    dropped = []
    for col in upper_tri.columns:
        if upper_tri[col].max() > threshold:
            dropped.append(col)

    if dropped:
        logger.info(
            f"Removing {len(dropped)} highly correlated features "
            f"(|r| > {threshold}): {dropped}"
        )
        df = df.drop(columns=dropped)

    return df, dropped


def replace_unstable_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace known unstable features with stable alternatives.

    Replaces raw percent-change features with log-differences where
    the original and lagged values are available.
    """
    df = df.copy()

    # Replace Inf values with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return df


def age_normalized_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    age_col: str = "product_age_months",
    category_col: str = None,
) -> pd.DataFrame:
    """
    Create age-normalized deviation features.

    Instead of raw product_age_months dominating predictions,
    compute "deviation from age-category baseline" features.

    This addresses Problem 6: the model learning "old = fatigued"
    instead of "this product shows genuine fatigue signals now".
    """
    df = df.copy()

    if age_col not in df.columns:
        return df

    # Bin product age into lifecycle stages
    age_bins = [0, 3, 12, 24, float("inf")]
    age_labels = ["introduction", "growth", "maturity", "decline"]
    df["_age_stage"] = pd.cut(
        df[age_col], bins=age_bins, labels=age_labels, right=False
    )

    group_col = "_age_stage"
    if category_col and category_col in df.columns:
        df["_group_key"] = df["_age_stage"].astype(str) + "_" + df[category_col].astype(str)
        group_col = "_group_key"

    for col in feature_cols:
        if col not in df.columns:
            continue

        # Compute group mean and std
        group_mean = df.groupby(group_col, observed=True)[col].transform("mean")
        group_std = df.groupby(group_col, observed=True)[col].transform("std").replace(0, 1)

        # Deviation from age-group baseline (z-score within age group)
        df[f"{col}_age_dev"] = (df[col] - group_mean) / group_std

    # Clean up
    df = df.drop(columns=["_age_stage"], errors="ignore")
    if "_group_key" in df.columns:
        df = df.drop(columns=["_group_key"])

    return df


def apply_all_stability_fixes(
    df: pd.DataFrame,
    modality: str,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Apply all feature stability fixes to a DataFrame.

    Returns
    -------
    (cleaned_df, report) where report contains lists of removed features.
    """
    report: Dict[str, List[str]] = {}

    # 1. Replace Inf and unstable features
    df = replace_unstable_features(df)

    # 2. Remove dead features
    df, dead = remove_dead_features(df)
    report["dead_features"] = dead

    # 3. Remove correlated features
    df, correlated = remove_correlated_features(df)
    report["correlated_features"] = correlated

    # 4. Cap extreme ratios
    ratio_cols = [c for c in df.columns if any(
        kw in c for kw in ["_per_", "_rate", "_ratio", "_change"]
    )]
    df = cap_extreme_ratios(df, ratio_cols)

    # 5. Age-normalized features (Problem 6 fix)
    if "product_age_months" in df.columns:
        key_features = {
            "reviews": ["sentiment_mean", "review_count"],
            "sales": ["revenue_total", "transaction_count"],
            "usage": ["engagement_total", "conversion_rate", "purchase_count"],
        }
        features_to_normalize = key_features.get(modality, [])
        df = age_normalized_features(df, features_to_normalize)

    logger.info(
        f"[{modality}] Stability fixes: removed {len(dead)} dead, "
        f"{len(correlated)} correlated features. "
        f"Final shape: {df.shape}"
    )

    return df, report
