"""
train.py — Honest model training with SMOTE strictly inside CV folds.

Why imblearn Pipeline is non-negotiable
---------------------------------------
If you apply SMOTE to all of X_train before calling GridSearchCV, synthetic
samples from the minority class bleed into every validation fold — the model
is evaluated on data it effectively helped generate, inflating CV scores.

Using imblearn.pipeline.Pipeline(SMOTE, classifier) inside GridSearchCV
guarantees SMOTE is applied only to the k-1 training folds, never the
validation fold.  The scaler is already applied in data_loader.py (fit on
the full training split), so it does NOT appear in this pipeline — adding it
here would be redundant but harmless.

Hyperparameter grids are deliberately conservative to avoid overfitting on
the cross-validation objective.  For XGBoost, subsample and colsample_bytree
provide regularisation that the old pipeline lacked entirely.

Walk-forward CV support
-----------------------
When a WalkForwardCV splitter is provided, GridSearchCV uses temporal splits
instead of stratified k-fold.  This respects the time ordering of the data
and produces more realistic performance estimates.
"""

import os
import logging
import numpy as np
import joblib
from typing import Dict, Any, Tuple

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
SEED = 42


# ---------------------------------------------------------------------------
# Shared CV strategy — used by every GridSearchCV call
# ---------------------------------------------------------------------------

def _cv() -> StratifiedKFold:
    # shuffle=True ensures different fold boundaries per random seed,
    # reducing variance in the CV estimate.
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def _safe_smote(y: np.ndarray) -> SMOTE:
    """
    Return a SMOTE instance whose k_neighbors is safe for the smallest class.
    SMOTE requires at least k_neighbors + 1 samples in the minority class;
    if the minority class is tiny we reduce k_neighbors to avoid a ValueError.
    """
    min_class_size = int(np.min(np.bincount(y)))
    k = min(5, min_class_size - 1)
    k = max(1, k)   # never below 1
    return SMOTE(random_state=SEED, k_neighbors=k)


# ---------------------------------------------------------------------------
# Individual model trainers
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, modality: str = "",
    cv_splitter=None,
) -> Tuple[ImbPipeline, float]:
    """
    Returns (best_pipeline, best_cv_f1_macro).
    Pipeline: SMOTE → XGBClassifier.

    subsample and colsample_bytree are included because the old pipeline
    omitted them — stochastic subsampling is one of XGBoost's main
    regularisation mechanisms and their absence contributes to overfitting.
    """
    logger.info("Training XGBoost (SMOTE inside CV folds)...")

    pipe = ImbPipeline([
        ("smote", _safe_smote(y_train)),
        ("clf", XGBClassifier(
            random_state=SEED,
            eval_metric="mlogloss",
            tree_method="hist",     # faster on large datasets; equivalent to GPU hist
            n_jobs=-1,
            verbosity=0,
        )),
    ])

    # Keep one vetted configuration per family so retraining stays practical
    # in constrained environments while still regenerating consistent artifacts.
    param_grid = {
        "clf__n_estimators":     [200, 400],
        "clf__max_depth":        [4, 6],
        "clf__learning_rate":    [0.05, 0.1],
        "clf__subsample":        [0.8],
        "clf__colsample_bytree": [0.8],
        "clf__reg_alpha":        [0.0, 0.1],
        "clf__reg_lambda":       [1.0, 3.0],
        "clf__min_child_weight": [3],
    }

    gs = GridSearchCV(
        pipe, param_grid,
        cv=cv_splitter or _cv(), scoring="f1_macro",
        n_jobs=1, verbose=0, refit=True,
        error_score="raise",
    )
    gs.fit(X_train, y_train)
    logger.info(f"XGBoost — best params: {gs.best_params_}  CV F1: {gs.best_score_:.4f}")
    return gs.best_estimator_, float(gs.best_score_)


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, modality: str = "",
    cv_splitter=None,
) -> Tuple[ImbPipeline, float]:
    """
    Pipeline: SMOTE → RandomForestClassifier.
    class_weight='balanced' is kept as a belt-and-suspenders measure alongside
    SMOTE — they complement each other on heavily imbalanced data.
    """
    logger.info("Training Random Forest (SMOTE inside CV folds)...")

    pipe = ImbPipeline([
        ("smote", _safe_smote(y_train)),
        ("clf", RandomForestClassifier(
            random_state=SEED,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )),
    ])

    param_grid = {
        "clf__n_estimators":      [200, 400],
        "clf__max_depth":         [10, 15],
        "clf__min_samples_split": [5, 10],
        "clf__min_samples_leaf":  [2, 4],
    }

    gs = GridSearchCV(
        pipe, param_grid,
        cv=cv_splitter or _cv(), scoring="f1_macro",
        n_jobs=1, verbose=0, refit=True,
        error_score="raise",
    )
    gs.fit(X_train, y_train)
    logger.info(f"Random Forest — best params: {gs.best_params_}  CV F1: {gs.best_score_:.4f}")
    return gs.best_estimator_, float(gs.best_score_)


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray, modality: str = "",
    cv_splitter=None,
) -> Tuple[ImbPipeline, float]:
    """
    Pipeline: SMOTE → LogisticRegression.

    sklearn 1.8 deprecated 'penalty' in favour of 'l1_ratio':
      l1_ratio=0 ↔ L2 (ridge),  l1_ratio=1 ↔ L1 (lasso)

    Solver choice: lbfgs is the fastest solver for L2 on large dense datasets.
    We avoid saga here because saga can take many epochs to converge on 100k+
    row datasets — lbfgs uses second-order information and converges much faster.
    The trade-off is that lbfgs only supports L2, so l1_ratio is fixed at 0.
    """
    logger.info("Training Logistic Regression (SMOTE inside CV folds)...")

    pipe = ImbPipeline([
        ("smote", _safe_smote(y_train)),
        ("clf", LogisticRegression(
            random_state=SEED,
            class_weight="balanced",
            solver="lbfgs",     # fast second-order solver; supports L2 only
            # multi_class removed in sklearn 1.8 — lbfgs handles OvR automatically
            max_iter=1000,      # sufficient for lbfgs on scaled features
            # n_jobs removed in sklearn 1.8 for LogisticRegression
        )),
    ])

    # lbfgs only supports L2 → l1_ratio is not tuned; search only over C
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    }

    gs = GridSearchCV(
        pipe, param_grid,
        cv=cv_splitter or _cv(), scoring="f1_macro",
        n_jobs=1, verbose=0, refit=True,
        error_score="raise",
    )
    gs.fit(X_train, y_train)
    logger.info(f"LR — best params: {gs.best_params_}  CV F1: {gs.best_score_:.4f}")
    return gs.best_estimator_, float(gs.best_score_)


def train_kmeans(
    X_train: np.ndarray,
    k_range: range = range(2, 9),
) -> Tuple[KMeans, int, float]:
    """
    Select optimal k over k_range using silhouette score on (a subsample of)
    the training data.  K-Means is unsupervised — no SMOTE needed.

    Sub-samples to 20 000 rows for silhouette computation because silhouette
    is O(n²) and the training set may be very large.

    Returns (best_kmeans_model, best_k, best_silhouette_score).
    The returned model was trained on the full X_train (not the subsample).
    """
    logger.info("Training K-Means, selecting k by silhouette score...")

    # Sub-sample once for silhouette evaluation; train the winning model on full data
    if X_train.shape[0] > 20_000:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(X_train.shape[0], 20_000, replace=False)
        X_sil = X_train[idx]
    else:
        X_sil = X_train

    best_k, best_score, best_model = 2, -1.0, None

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init="auto")
        labels = km.fit_predict(X_sil)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(
            X_sil, labels,
            sample_size=min(10_000, len(X_sil)),
            random_state=SEED,
        )
        logger.info(f"  k={k}  silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
            # Re-train on full data with this k for deployment
            best_model = KMeans(n_clusters=k, random_state=SEED, n_init="auto").fit(X_train)

    logger.info(f"K-Means — best k={best_k}  silhouette={best_score:.4f}")
    return best_model, best_k, best_score


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def train_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    modality: str = "",
    cv_splitter=None,
) -> Dict[str, Any]:
    """
    Train all four model types.

    Parameters
    ----------
    X_train     : training features
    y_train     : training labels
    modality    : modality name for modality-specific config
    cv_splitter : optional walk-forward CV splitter (replaces StratifiedKFold)

    Returns
    -------
    {
      "xgboost":             {"pipeline": ImbPipeline, "cv_f1": float},
      "random_forest":       {"pipeline": ImbPipeline, "cv_f1": float},
      "logistic_regression": {"pipeline": ImbPipeline, "cv_f1": float},
      "kmeans":              {"model": KMeans, "best_k": int, "silhouette": float},
    }
    """
    results: Dict[str, Any] = {}

    xgb_pipe, xgb_cv = train_xgboost(X_train, y_train, modality=modality, cv_splitter=cv_splitter)
    results["xgboost"] = {"pipeline": xgb_pipe, "cv_f1": xgb_cv}

    rf_pipe, rf_cv = train_random_forest(X_train, y_train, modality=modality, cv_splitter=cv_splitter)
    results["random_forest"] = {"pipeline": rf_pipe, "cv_f1": rf_cv}

    lr_pipe, lr_cv = train_logistic_regression(X_train, y_train, modality=modality, cv_splitter=cv_splitter)
    results["logistic_regression"] = {"pipeline": lr_pipe, "cv_f1": lr_cv}

    km, best_k, km_sil = train_kmeans(X_train)
    results["kmeans"] = {"model": km, "best_k": best_k, "silhouette": km_sil}

    return results


def save_models(
    results: Dict[str, Any],
    output_dir: str = "models",
    prefix: str = "",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name, entry in results.items():
        # Classifiers are stored under "pipeline"; KMeans under "model"
        obj = entry.get("pipeline") or entry.get("model")
        path = os.path.join(output_dir, f"{prefix}{name}_model.pkl")
        joblib.dump(obj, path)
        logger.info(f"Saved {name} → {path}")
