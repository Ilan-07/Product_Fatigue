"""
forward_label.py -- Forward-prediction label construction.

Core principle
--------------
Features at time t must come ONLY from a past window [t-W, t].
Labels at time t must come ONLY from a future window [t+1, t+H].

This eliminates the fundamental problem where the label is deterministically
reconstructable from same-window engineered features.  The model must now
learn to *forecast* fatigue rather than *classify* current-window signals.

Label construction
------------------
For each product-time row, we look H periods ahead and compute a composite
fatigue indicator from the future window's signals.  The fatigue indicator
is then bucketed into 3 classes: healthy / moderate / high fatigue.

The future fatigue indicator is computed differently per modality:
  - Reviews:  future sentiment decline + volatility increase
  - Sales:    future revenue decline + churn increase
  - Usage:    future engagement decline + conversion decline
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-modality future-window fatigue score computation
# ---------------------------------------------------------------------------

def _reviews_future_fatigue(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    horizon: int = 4,
) -> pd.Series:
    """
    Compute a forward-looking fatigue score for the Reviews modality.

    For each product-month at time t, the score captures how much sentiment
    deteriorates over the next `horizon` months:
      - decline in sentiment_mean
      - increase in sentiment std (volatility)
      - decline in review_count (engagement loss)

    Returns a Series aligned with df.index containing fatigue scores in [0, 1].
    """
    df = df.copy()
    df["_sort"] = pd.to_datetime(df[time_col], format="%Y-%m", errors="coerce")
    df = df.sort_values([id_col, "_sort"])

    scores = pd.Series(np.nan, index=df.index, dtype=float)

    for _, grp in df.groupby(id_col, sort=False):
        grp = grp.sort_values("_sort")
        idx = grp.index.tolist()
        n = len(idx)

        for i in range(n):
            future_start = i + 1
            future_end = min(i + 1 + horizon, n)
            if future_end <= future_start:
                continue  # not enough future data

            current = grp.iloc[i]
            future = grp.iloc[future_start:future_end]

            # Sentiment decline component
            sent_now = current.get("sentiment_mean", 0.0)
            sent_future = future["sentiment_mean"].mean() if "sentiment_mean" in future.columns else sent_now
            sent_decline = max(0, sent_now - sent_future) / max(abs(sent_now), 0.01)

            # Volatility increase component
            std_now = current.get("sentiment_std", 0.0) if pd.notna(current.get("sentiment_std")) else 0.0
            std_future = future["sentiment_std"].mean() if "sentiment_std" in future.columns else std_now
            vol_increase = max(0, std_future - std_now) / max(abs(std_now), 0.01)

            # Review count decline component
            rc_now = current.get("review_count", 1.0) if pd.notna(current.get("review_count")) else 1.0
            rc_future = future["review_count"].mean() if "review_count" in future.columns else rc_now
            rc_decline = max(0, rc_now - rc_future) / max(abs(rc_now), 0.01)

            # Composite score (weighted average, clipped to [0, 1])
            raw_score = 0.5 * sent_decline + 0.25 * vol_increase + 0.25 * rc_decline
            scores.loc[idx[i]] = np.clip(raw_score, 0.0, 1.0)

    return scores


def _sales_future_fatigue(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    horizon: int = 4,
) -> pd.Series:
    """
    Compute a forward-looking fatigue score for the Sales modality.

    Captures future revenue deterioration and customer erosion:
      - decline in revenue_total
      - decline in transaction_count
      - decline in avg_price (AOV erosion)
    """
    df = df.copy()
    df["_sort"] = pd.to_datetime(df[time_col], format="%Y-%m", errors="coerce")
    df = df.sort_values([id_col, "_sort"])

    scores = pd.Series(np.nan, index=df.index, dtype=float)

    for _, grp in df.groupby(id_col, sort=False):
        grp = grp.sort_values("_sort")
        idx = grp.index.tolist()
        n = len(idx)

        for i in range(n):
            future_start = i + 1
            future_end = min(i + 1 + horizon, n)
            if future_end <= future_start:
                continue

            current = grp.iloc[i]
            future = grp.iloc[future_start:future_end]

            # Revenue decline
            rev_now = current.get("revenue_total", 0.0) if pd.notna(current.get("revenue_total")) else 0.0
            rev_future = future["revenue_total"].mean() if "revenue_total" in future.columns else rev_now
            rev_decline = max(0, rev_now - rev_future) / max(abs(rev_now), 0.01)

            # Transaction count decline
            tc_now = current.get("transaction_count", 1.0) if pd.notna(current.get("transaction_count")) else 1.0
            tc_future = future["transaction_count"].mean() if "transaction_count" in future.columns else tc_now
            tc_decline = max(0, tc_now - tc_future) / max(abs(tc_now), 0.01)

            # AOV erosion
            aov_now = current.get("avg_price", 0.0) if pd.notna(current.get("avg_price")) else 0.0
            aov_future = future["avg_price"].mean() if "avg_price" in future.columns else aov_now
            aov_decline = max(0, aov_now - aov_future) / max(abs(aov_now), 0.01)

            raw_score = 0.5 * rev_decline + 0.3 * tc_decline + 0.2 * aov_decline
            scores.loc[idx[i]] = np.clip(raw_score, 0.0, 1.0)

    return scores


def _usage_future_fatigue(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    horizon: int = 4,
) -> pd.Series:
    """
    Compute a forward-looking fatigue score for the Usage modality.

    Captures future behavioral disengagement:
      - decline in engagement_total
      - decline in conversion_rate
      - decline in purchase_count
    """
    df = df.copy()
    df["_sort"] = pd.to_datetime(df[time_col], format="%Y-%m", errors="coerce")
    df = df.sort_values([id_col, "_sort"])

    scores = pd.Series(np.nan, index=df.index, dtype=float)

    for _, grp in df.groupby(id_col, sort=False):
        grp = grp.sort_values("_sort")
        idx = grp.index.tolist()
        n = len(idx)

        for i in range(n):
            future_start = i + 1
            future_end = min(i + 1 + horizon, n)
            if future_end <= future_start:
                continue

            current = grp.iloc[i]
            future = grp.iloc[future_start:future_end]

            # Engagement decline
            eng_now = current.get("engagement_total", 0.0) if pd.notna(current.get("engagement_total")) else 0.0
            eng_future = future["engagement_total"].mean() if "engagement_total" in future.columns else eng_now
            eng_decline = max(0, eng_now - eng_future) / max(abs(eng_now), 0.01)

            # Conversion decline
            conv_now = current.get("conversion_rate", 0.0) if pd.notna(current.get("conversion_rate")) else 0.0
            conv_future = future["conversion_rate"].mean() if "conversion_rate" in future.columns else conv_now
            conv_decline = max(0, conv_now - conv_future) / max(abs(conv_now), 0.01)

            # Purchase decline
            pc_now = current.get("purchase_count", 0.0) if pd.notna(current.get("purchase_count")) else 0.0
            pc_future = future["purchase_count"].mean() if "purchase_count" in future.columns else pc_now
            pc_decline = max(0, pc_now - pc_future) / max(abs(pc_now), 0.01)

            raw_score = 0.4 * eng_decline + 0.35 * conv_decline + 0.25 * pc_decline
            scores.loc[idx[i]] = np.clip(raw_score, 0.0, 1.0)

    return scores


# ---------------------------------------------------------------------------
# Score → class bucketing
# ---------------------------------------------------------------------------

def _score_to_class(
    scores: pd.Series,
    method: str = "quantile",
    thresholds: Optional[Tuple[float, float]] = None,
    binary: bool = False,
) -> pd.Series:
    """
    Convert continuous fatigue scores to class labels.

    Methods:
      "quantile" — use terciles (3-class) or median (binary) of the non-NaN score distribution
      "fixed"    — use fixed thresholds (default 0.33 / 0.66 for 3-class, 0.5 for binary)

    When binary=True, produces 2 classes: healthy / fatigued (using median split).
    NaN scores (rows without enough future data) remain NaN.
    """
    labels = pd.Series(np.nan, index=scores.index, dtype=object)
    valid = scores.dropna()

    if len(valid) == 0:
        return labels

    if binary:
        # Binary classification: median split
        if method == "fixed":
            threshold = (thresholds[0] if thresholds else 0.5)
        else:
            threshold = valid.median()

        labels.loc[valid.index] = np.where(
            valid <= threshold, "healthy", "fatigued"
        )
        return labels

    if method == "quantile":
        lo = valid.quantile(0.33)
        hi = valid.quantile(0.66)
    elif method == "fixed":
        lo, hi = thresholds or (0.33, 0.66)
    else:
        lo, hi = 0.33, 0.66

    # Ensure thresholds are distinct
    if lo >= hi:
        lo = valid.quantile(0.33)
        hi = valid.quantile(0.66)
    if lo >= hi:
        # Degenerate distribution — use fixed fallback
        lo, hi = 0.2, 0.5

    labels.loc[valid.index] = np.where(
        valid <= lo, "healthy",
        np.where(valid <= hi, "moderate_fatigue", "high_fatigue")
    )
    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_MODALITY_SCORERS = {
    "reviews": _reviews_future_fatigue,
    "sales": _sales_future_fatigue,
    "usage": _usage_future_fatigue,
}


def construct_forward_labels(
    df: pd.DataFrame,
    modality: str,
    id_col: str,
    time_col: str,
    horizon: int = 4,
    class_method: str = "quantile",
    binary: bool = False,
) -> pd.DataFrame:
    """
    Replace the existing fatigue_label with a forward-looking label.

    Steps:
      1. Compute a per-modality future fatigue score for each row
      2. Bucket scores into classes (binary or 3-class)
      3. Drop rows with NaN labels (insufficient future data)
      4. Store the raw score as 'fatigue_score' for interpretability

    Parameters
    ----------
    df        : DataFrame with product-time rows and feature columns
    modality  : "reviews" | "sales" | "usage"
    id_col    : product identifier column
    time_col  : time period column
    horizon   : number of future periods to look ahead (default: 4)
    class_method : "quantile" or "fixed" for class boundary selection
    binary    : if True, produce 2 classes (healthy / fatigued) instead of 3

    Returns
    -------
    DataFrame with updated 'fatigue_label' and new 'fatigue_score' column.
    Rows without sufficient future data are dropped.
    """
    scorer = _MODALITY_SCORERS.get(modality)
    if scorer is None:
        raise ValueError(f"Unknown modality '{modality}'. Expected one of {list(_MODALITY_SCORERS)}")

    logger.info(
        f"[{modality}] Constructing forward-prediction labels "
        f"(horizon={horizon}, method={class_method})"
    )

    n_before = len(df)
    scores = scorer(df, id_col, time_col, horizon)
    labels = _score_to_class(scores, method=class_method, binary=binary)

    df = df.copy()
    df["fatigue_score"] = scores
    df["fatigue_label"] = labels

    # Drop rows without future labels
    df = df.dropna(subset=["fatigue_label"])
    n_dropped = n_before - len(df)

    class_dist = df["fatigue_label"].value_counts().to_dict()
    logger.info(
        f"[{modality}] Forward labels: {n_dropped} rows dropped "
        f"(insufficient future data), {len(df)} rows remain. "
        f"Class distribution: {class_dist}"
    )

    return df
