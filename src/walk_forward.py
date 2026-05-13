"""
walk_forward.py -- Walk-forward temporal validation for fatigue prediction.

Why walk-forward?
-----------------
Standard stratified CV allows nearby time windows from the same product to
appear in both train and validation folds.  For a temporal problem like fatigue
prediction, this creates overly optimistic validation scores.

Walk-forward validation respects the time ordering:
  - Fold 1: train on periods 1-6,  validate on period 7
  - Fold 2: train on periods 1-7,  validate on period 8
  - Fold 3: train on periods 1-8,  validate on period 9
  ...

This ensures the model is always evaluated on strictly future data,
matching the real-world deployment scenario.

Product-aware splitting
-----------------------
Splits are done by (product, time) so that no product's future data leaks
into training.  Each fold's validation set contains all products that have
data in the validation period.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Generator

logger = logging.getLogger(__name__)


def get_time_periods(
    df: pd.DataFrame,
    time_col: str,
) -> List[str]:
    """
    Extract and sort unique time periods from the DataFrame.
    Handles year-month strings ("2019-10") and other sortable formats.
    """
    periods = df[time_col].dropna().unique().tolist()
    try:
        periods_dt = pd.to_datetime(periods, format="%Y-%m", errors="coerce")
        valid = [(p, dt) for p, dt in zip(periods, periods_dt) if pd.notna(dt)]
        valid.sort(key=lambda x: x[1])
        return [p for p, _ in valid]
    except Exception:
        return sorted(periods)


def walk_forward_splits(
    df: pd.DataFrame,
    time_col: str,
    min_train_periods: int = 6,
    val_periods: int = 1,
    expanding: bool = True,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate walk-forward train/validation index splits.

    Parameters
    ----------
    df                : DataFrame with a time_col column
    time_col          : column containing time period identifiers
    min_train_periods : minimum number of periods in the first training fold
    val_periods       : number of periods in each validation fold
    expanding         : if True, training window grows; if False, sliding window

    Yields
    ------
    (train_indices, val_indices) : numpy arrays of DataFrame indices
    """
    periods = get_time_periods(df, time_col)
    n_periods = len(periods)

    if n_periods < min_train_periods + val_periods:
        logger.warning(
            f"Only {n_periods} time periods available, need at least "
            f"{min_train_periods + val_periods} for walk-forward validation. "
            f"Falling back to single train/test split."
        )
        # Fallback: use first 80% as train, rest as validation
        cutoff = int(n_periods * 0.8)
        cutoff = max(1, cutoff)
        train_periods = set(periods[:cutoff])
        val_periods_set = set(periods[cutoff:])

        train_idx = df[df[time_col].isin(train_periods)].index.values
        val_idx = df[df[time_col].isin(val_periods_set)].index.values

        if len(train_idx) > 0 and len(val_idx) > 0:
            yield train_idx, val_idx
        return

    for i in range(min_train_periods, n_periods - val_periods + 1):
        if expanding:
            train_periods_set = set(periods[:i])
        else:
            # Sliding window: keep only the last min_train_periods
            start = max(0, i - min_train_periods)
            train_periods_set = set(periods[start:i])

        val_periods_set = set(periods[i:i + val_periods])

        train_idx = df[df[time_col].isin(train_periods_set)].index.values
        val_idx = df[df[time_col].isin(val_periods_set)].index.values

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        logger.debug(
            f"Walk-forward fold: train={sorted(train_periods_set)[-3:]}"
            f"..., val={sorted(val_periods_set)}, "
            f"train_n={len(train_idx)}, val_n={len(val_idx)}"
        )

        yield train_idx, val_idx


def walk_forward_train_test_split(
    df: pd.DataFrame,
    time_col: str,
    test_periods: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into train/test using the last `test_periods` as the held-out test set.

    This is the final evaluation split — walk_forward_splits() is used inside
    training for cross-validation.

    Parameters
    ----------
    df           : DataFrame with time_col
    time_col     : time period column
    test_periods : number of final periods reserved for testing

    Returns
    -------
    (train_df, test_df)
    """
    periods = get_time_periods(df, time_col)
    n_periods = len(periods)

    if n_periods <= test_periods:
        logger.warning(
            f"Only {n_periods} periods, can't reserve {test_periods} for test. "
            f"Using last 1 period as test."
        )
        test_periods = max(1, n_periods - 1)

    test_period_set = set(periods[-test_periods:])
    train_period_set = set(periods[:-test_periods])

    train_df = df[df[time_col].isin(train_period_set)].copy()
    test_df = df[df[time_col].isin(test_period_set)].copy()

    logger.info(
        f"Walk-forward train/test split: "
        f"train periods={len(train_period_set)}, test periods={len(test_period_set)}, "
        f"train rows={len(train_df)}, test rows={len(test_df)}"
    )

    return train_df, test_df


class WalkForwardCV:
    """
    sklearn-compatible cross-validator for walk-forward temporal validation.

    Can be passed directly to GridSearchCV(cv=WalkForwardCV(...)).
    """

    def __init__(
        self,
        time_col: str = "month",
        min_train_periods: int = 6,
        val_periods: int = 1,
        expanding: bool = True,
        df: Optional[pd.DataFrame] = None,
    ):
        self.time_col = time_col
        self.min_train_periods = min_train_periods
        self.val_periods = val_periods
        self.expanding = expanding
        self._df = df
        self._splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

    def set_df(self, df: pd.DataFrame) -> "WalkForwardCV":
        """Set the DataFrame used to compute time-based splits."""
        self._df = df
        self._splits = None
        return self

    def _compute_splits(self, X=None) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self._splits is not None:
            return self._splits

        if self._df is None:
            raise ValueError(
                "WalkForwardCV requires a DataFrame with time_col. "
                "Call .set_df(df) before using as a CV splitter."
            )

        splits = list(walk_forward_splits(
            self._df,
            time_col=self.time_col,
            min_train_periods=self.min_train_periods,
            val_periods=self.val_periods,
            expanding=self.expanding,
        ))

        if not splits:
            logger.warning("No valid walk-forward splits. Falling back to single split.")
            n = len(self._df)
            cutoff = int(n * 0.8)
            splits = [(np.arange(cutoff), np.arange(cutoff, n))]

        self._splits = splits
        return self._splits

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self._compute_splits(X))

    def split(self, X=None, y=None, groups=None):
        for train_idx, val_idx in self._compute_splits(X):
            yield train_idx, val_idx
