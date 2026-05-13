"""
fusion.py -- Late fusion via meta-model for multimodal fatigue prediction.

Architecture (Problem 3 from implementation plan)
--------------------------------------------------
Stage 1: Train one branch model per modality (reviews, sales, usage).
Stage 2: Generate out-of-fold branch probabilities to avoid fusion leakage.
Stage 3: Build fusion training table: branch probabilities + shared metadata.
Stage 4: Train a meta-model (Logistic Regression, then XGBoost).
Stage 5: Evaluate on true holdout.

Canonical Fatigue Index (Problem 8)
-----------------------------------
FI = w1 * B + w2 * E + w3 * C
where B = behavioral, E = emotional, C = commercial degradation scores.
Weights can be expert-defined, optimized on validation, or learned by fusion.
"""

import logging
import os
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
SEED = 42


def generate_oof_probabilities(
    pipeline: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
) -> np.ndarray:
    """
    Generate out-of-fold (OOF) class probabilities from a trained pipeline.

    For each fold, the pipeline is retrained on k-1 folds and predicts
    probabilities on the held-out fold.  This prevents the fusion layer
    from seeing inflated in-sample probabilities.

    Parameters
    ----------
    pipeline : fitted imblearn Pipeline (will be cloned for each fold)
    X_train  : training features
    y_train  : training labels
    n_splits : number of CV folds for OOF generation

    Returns
    -------
    (n_samples, n_classes) array of OOF probabilities
    """
    from sklearn.base import clone

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    n_classes = len(np.unique(y_train))
    oof_proba = np.zeros((len(y_train), n_classes))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_fold_train, y_fold_train)
        oof_proba[val_idx] = fold_pipeline.predict_proba(X_fold_val)

        logger.debug(f"  OOF fold {fold+1}/{n_splits}: {len(val_idx)} predictions")

    return oof_proba


def build_fusion_table(
    branch_oof_probas: dict[str, np.ndarray],
    y_train: np.ndarray,
    shared_metadata: np.ndarray | None = None,
    shared_feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the fusion training table from branch OOF probabilities.

    Parameters
    ----------
    branch_oof_probas    : {"reviews": (n, 3), "sales": (n, 3), "usage": (n, 3)}
    y_train              : target labels
    shared_metadata      : optional shared features (e.g., product_age, category)
    shared_feature_names : names for shared metadata columns

    Returns
    -------
    (X_fusion, fusion_feature_names) : fusion feature matrix and column names
    """
    parts = []
    feature_names = []

    for modality, proba in branch_oof_probas.items():
        n_classes = proba.shape[1]
        class_labels = ["healthy", "moderate", "high"][:n_classes]
        for i, cls in enumerate(class_labels):
            col_name = f"{modality}_{cls}"
            parts.append(proba[:, i:i+1])
            feature_names.append(col_name)

    X_fusion = np.hstack(parts)

    if shared_metadata is not None:
        X_fusion = np.hstack([X_fusion, shared_metadata])
        if shared_feature_names:
            feature_names.extend(shared_feature_names)
        else:
            for i in range(shared_metadata.shape[1]):
                feature_names.append(f"shared_{i}")

    logger.info(
        f"Fusion table: {X_fusion.shape[0]} rows × "
        f"{X_fusion.shape[1]} features ({feature_names})"
    )

    return X_fusion, feature_names


def train_fusion_logistic(
    X_fusion: np.ndarray,
    y_fusion: np.ndarray,
) -> tuple[LogisticRegression, float]:
    """
    Train a Logistic Regression fusion model.

    This is the interpretable baseline fusion model.

    Returns
    -------
    (model, cv_f1_macro)
    """
    logger.info("Training Logistic Regression fusion model...")

    model = LogisticRegression(
        random_state=SEED,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        C=1.0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []

    for train_idx, val_idx in cv.split(X_fusion, y_fusion):
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_fusion[train_idx], y_fusion[train_idx])
        y_pred = fold_model.predict(X_fusion[val_idx])
        score = f1_score(y_fusion[val_idx], y_pred, average="macro", zero_division=0)
        cv_scores.append(score)

    cv_f1 = float(np.mean(cv_scores))

    # Fit on full data
    model.fit(X_fusion, y_fusion)

    logger.info(f"Fusion LR — CV F1 macro: {cv_f1:.4f}")
    return model, cv_f1


def train_fusion_xgboost(
    X_fusion: np.ndarray,
    y_fusion: np.ndarray,
) -> tuple[XGBClassifier, float]:
    """
    Train an XGBoost fusion model.

    This is the stronger fusion model, used after LR baseline is established.

    Returns
    -------
    (model, cv_f1_macro)
    """
    logger.info("Training XGBoost fusion model...")

    model = XGBClassifier(
        random_state=SEED,
        eval_metric="mlogloss",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []

    for train_idx, val_idx in cv.split(X_fusion, y_fusion):
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_fusion[train_idx], y_fusion[train_idx])
        y_pred = fold_model.predict(X_fusion[val_idx])
        score = f1_score(y_fusion[val_idx], y_pred, average="macro", zero_division=0)
        cv_scores.append(score)

    cv_f1 = float(np.mean(cv_scores))

    # Fit on full data
    model.fit(X_fusion, y_fusion)

    logger.info(f"Fusion XGBoost — CV F1 macro: {cv_f1:.4f}")
    return model, cv_f1


def compute_fatigue_index(
    branch_probas: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Compute a canonical fatigue index from branch probabilities.

    FI = w1 * B_high + w2 * E_high + w3 * C_high

    where *_high is the P(high_fatigue) from each branch.

    Parameters
    ----------
    branch_probas : {"reviews": (n, 3), "sales": (n, 3), "usage": (n, 3)}
    weights       : {"reviews": w1, "sales": w2, "usage": w3}

    Returns
    -------
    (n,) array of fatigue index values in [0, 1]
    """
    if weights is None:
        weights = {"reviews": 0.3, "sales": 0.3, "usage": 0.4}

    fi = np.zeros(len(next(iter(branch_probas.values()))))

    for modality, proba in branch_probas.items():
        w = weights.get(modality, 1.0 / len(branch_probas))
        # Use the "high fatigue" probability (last class)
        high_fatigue_prob = proba[:, -1] if proba.shape[1] > 1 else proba[:, 0]
        fi += w * high_fatigue_prob

    return np.clip(fi, 0.0, 1.0)


class FusionModel:
    """
    Unified fusion model that wraps branch models and meta-learner.

    Provides end-to-end prediction from branch features to final fatigue class.
    """

    def __init__(self):
        self.branch_pipelines: dict[str, Any] = {}
        self.meta_model: Any = None
        self.meta_model_type: str = "logistic_regression"
        self.fusion_feature_names: list[str] = []
        self.label_classes: np.ndarray | None = None
        self.branch_cv_scores: dict[str, float] = {}
        self.fusion_cv_f1: float = 0.0

    def fit(
        self,
        branch_data: dict[str, tuple[np.ndarray, np.ndarray]],
        branch_pipelines: dict[str, Any],
        y_train: np.ndarray,
        label_classes: np.ndarray,
        use_xgboost: bool = True,
    ) -> "FusionModel":
        """
        Fit the fusion model.

        Parameters
        ----------
        branch_data      : {"reviews": (X_train, y_train), "sales": ..., "usage": ...}
        branch_pipelines : {"reviews": pipeline, "sales": ..., "usage": ...}
        y_train          : unified target labels
        label_classes    : class name array
        use_xgboost      : if True, train both LR and XGBoost fusion, use best
        """
        self.branch_pipelines = branch_pipelines
        self.label_classes = label_classes

        # Generate OOF probabilities for each branch
        logger.info("Generating out-of-fold branch probabilities...")
        branch_oof = {}

        for modality, pipeline in branch_pipelines.items():
            if modality not in branch_data:
                continue
            X_mod, y_mod = branch_data[modality]
            oof = generate_oof_probabilities(pipeline, X_mod, y_mod)
            branch_oof[modality] = oof
            logger.info(f"  {modality}: OOF shape {oof.shape}")

        # Build fusion table
        X_fusion, self.fusion_feature_names = build_fusion_table(branch_oof, y_train)

        # Train fusion models
        lr_model, lr_f1 = train_fusion_logistic(X_fusion, y_train)

        if use_xgboost:
            xgb_model, xgb_f1 = train_fusion_xgboost(X_fusion, y_train)

            if xgb_f1 > lr_f1:
                self.meta_model = xgb_model
                self.meta_model_type = "xgboost"
                self.fusion_cv_f1 = xgb_f1
                logger.info(
                    f"Selected XGBoost fusion (F1={xgb_f1:.4f}) "
                    f"over LR fusion (F1={lr_f1:.4f})"
                )
            else:
                self.meta_model = lr_model
                self.meta_model_type = "logistic_regression"
                self.fusion_cv_f1 = lr_f1
                logger.info(
                    f"Selected LR fusion (F1={lr_f1:.4f}) "
                    f"over XGBoost fusion (F1={xgb_f1:.4f})"
                )
        else:
            self.meta_model = lr_model
            self.meta_model_type = "logistic_regression"
            self.fusion_cv_f1 = lr_f1

        return self

    def predict(
        self,
        branch_features: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict using the full fusion pipeline.

        Parameters
        ----------
        branch_features : {"reviews": X, "sales": X, "usage": X}

        Returns
        -------
        (predictions, probabilities)
        """
        # Get branch probabilities
        branch_probas = {}
        for modality, X in branch_features.items():
            if modality in self.branch_pipelines:
                pipeline = self.branch_pipelines[modality]
                branch_probas[modality] = pipeline.predict_proba(X)

        # Build fusion input
        parts = []
        for modality in sorted(branch_probas.keys()):
            parts.append(branch_probas[modality])

        X_fusion = np.hstack(parts)

        # Meta-model prediction
        predictions = self.meta_model.predict(X_fusion)
        probabilities = self.meta_model.predict_proba(X_fusion)

        return predictions, probabilities

    def predict_with_details(
        self,
        branch_features: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """
        Predict with full details including branch-level outputs.

        Returns
        -------
        {
          "final_class": int,
          "final_proba": ndarray,
          "branch_predictions": {modality: {"class": int, "proba": ndarray}},
          "fatigue_index": float,
          "fusion_model_type": str,
        }
        """
        branch_probas = {}
        branch_preds = {}

        for modality, X in branch_features.items():
            if modality in self.branch_pipelines:
                pipeline = self.branch_pipelines[modality]
                proba = pipeline.predict_proba(X)
                pred = proba.argmax(axis=1)
                branch_probas[modality] = proba
                branch_preds[modality] = {
                    "class": int(pred[0]) if len(pred) == 1 else pred,
                    "proba": proba[0] if len(proba) == 1 else proba,
                }

        # Build fusion input
        parts = []
        for modality in sorted(branch_probas.keys()):
            parts.append(branch_probas[modality])

        X_fusion = np.hstack(parts)

        # Meta prediction
        final_pred = self.meta_model.predict(X_fusion)
        final_proba = self.meta_model.predict_proba(X_fusion)

        # Fatigue index
        fi = compute_fatigue_index(branch_probas)

        return {
            "final_class": int(final_pred[0]) if len(final_pred) == 1 else final_pred,
            "final_proba": final_proba[0] if len(final_proba) == 1 else final_proba,
            "branch_predictions": branch_preds,
            "fatigue_index": float(fi[0]) if len(fi) == 1 else fi,
            "fusion_model_type": self.meta_model_type,
        }

    def save(self, output_dir: str = "models/fusion") -> None:
        """Save the fusion model and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.meta_model, os.path.join(output_dir, "champion.pkl"))
        joblib.dump({
            "meta_model_type": self.meta_model_type,
            "fusion_feature_names": self.fusion_feature_names,
            "fusion_cv_f1": self.fusion_cv_f1,
            "label_classes": self.label_classes,
        }, os.path.join(output_dir, "feature_manifest.json"))
        logger.info(f"Fusion model saved → {output_dir}")

    @classmethod
    def load(cls, model_dir: str = "models/fusion") -> "FusionModel":
        """Load a saved fusion model."""
        model = cls()
        model.meta_model = joblib.load(os.path.join(model_dir, "champion.pkl"))
        metadata = joblib.load(os.path.join(model_dir, "feature_manifest.json"))
        model.meta_model_type = metadata["meta_model_type"]
        model.fusion_feature_names = metadata["fusion_feature_names"]
        model.fusion_cv_f1 = metadata["fusion_cv_f1"]
        model.label_classes = metadata.get("label_classes")
        logger.info(
            f"Loaded fusion model ({model.meta_model_type}) "
            f"from {model_dir}, CV F1={model.fusion_cv_f1:.4f}"
        )
        return model
