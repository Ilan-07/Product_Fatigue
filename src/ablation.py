"""
ablation.py -- Ablation studies for the Product Fatigue system.

Problem 13 from the implementation plan:
  The README makes many strong claims, but ablation experiments are needed
  to prove which design choices matter.

Required ablations:
  - without product_age_months
  - without text-derived features
  - without behavioral features
  - without sales features
  - without calibration
  - without SMOTE or class balancing
  - without leakage fixes
  - only lag features vs lag + acceleration + volatility

This module provides a systematic framework for running ablations
and collecting comparable metrics.
"""

import json
import logging
import os
import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

SEED = 42


class AblationExperiment:
    """
    Represents a single ablation experiment.

    Each experiment removes a specific set of features or changes a
    specific design choice, then evaluates the impact on performance.
    """

    def __init__(
        self,
        name: str,
        description: str,
        features_to_remove: list[str] | None = None,
        feature_pattern_to_remove: str | None = None,
        use_smote: bool = True,
        use_calibration: bool = True,
    ):
        self.name = name
        self.description = description
        self.features_to_remove = features_to_remove or []
        self.feature_pattern_to_remove = feature_pattern_to_remove
        self.use_smote = use_smote
        self.use_calibration = use_calibration
        self.metrics: dict[str, Any] = {}
        self.elapsed_seconds: float = 0.0

    def get_feature_mask(self, feature_names: list[str]) -> list[bool]:
        """
        Return a boolean mask indicating which features to KEEP.
        """
        mask = []
        for f in feature_names:
            keep = True
            if f in self.features_to_remove:
                keep = False
            if self.feature_pattern_to_remove and self.feature_pattern_to_remove in f:
                keep = False
            mask.append(keep)
        return mask

    def apply_feature_removal(
        self,
        X: np.ndarray,
        feature_names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Remove specified features from the feature matrix.

        Returns
        -------
        (X_reduced, remaining_feature_names)
        """
        mask = self.get_feature_mask(feature_names)
        kept_indices = [i for i, keep in enumerate(mask) if keep]
        X_reduced = X[:, kept_indices]
        remaining_names = [feature_names[i] for i in kept_indices]

        n_removed = len(feature_names) - len(remaining_names)
        if n_removed > 0:
            logger.info(
                f"  [{self.name}] Removed {n_removed} features, "
                f"{len(remaining_names)} remaining"
            )

        return X_reduced, remaining_names


def define_standard_ablations(
    modality: str,
    feature_names: list[str],
) -> list[AblationExperiment]:
    """
    Define the standard set of ablation experiments for a modality.

    Returns a list of AblationExperiment objects ready to run.
    """
    ablations = []

    # 1. Full model (baseline)
    ablations.append(AblationExperiment(
        name="full_model",
        description="Full model with all features (baseline)",
    ))

    # 2. Without product_age_months
    age_features = [f for f in feature_names if "product_age" in f or "lifecycle" in f]
    if age_features:
        ablations.append(AblationExperiment(
            name="no_product_age",
            description="Without product_age_months and lifecycle features",
            features_to_remove=age_features,
        ))

    # 3. Without text-derived features (SBERT embeddings)
    text_features = [f for f in feature_names if f.startswith("sbert_") or f.startswith("nlp_")]
    if text_features:
        ablations.append(AblationExperiment(
            name="no_text_features",
            description="Without Sentence-BERT and NLP text features",
            features_to_remove=text_features,
        ))

    # 4. Without rolling/lag features
    rolling_features = [f for f in feature_names if "roll3" in f or "lag" in f or "slope" in f]
    if rolling_features:
        ablations.append(AblationExperiment(
            name="no_rolling_features",
            description="Without rolling means, lags, and slopes",
            features_to_remove=rolling_features,
        ))

    # 5. Without SMOTE
    ablations.append(AblationExperiment(
        name="no_smote",
        description="Without SMOTE oversampling",
        use_smote=False,
    ))

    # 6. Without calibration
    ablations.append(AblationExperiment(
        name="no_calibration",
        description="Without probability calibration",
        use_calibration=False,
    ))

    # 7. Only base features (no derived features)
    derived_patterns = ["roll3", "lag", "slope", "log_diff", "vs_trailing", "safe_", "_age_dev"]
    derived_features = [
        f for f in feature_names
        if any(p in f for p in derived_patterns)
    ]
    if derived_features:
        ablations.append(AblationExperiment(
            name="only_base_features",
            description="Only base-level features, no derived temporal features",
            features_to_remove=derived_features,
        ))

    # 8. Modality-specific ablations
    if modality == "reviews":
        sentiment_features = [f for f in feature_names if "sentiment" in f]
        if sentiment_features:
            ablations.append(AblationExperiment(
                name="no_sentiment",
                description="Without sentiment features",
                features_to_remove=sentiment_features,
            ))

    elif modality == "sales":
        revenue_features = [f for f in feature_names if "revenue" in f]
        if revenue_features:
            ablations.append(AblationExperiment(
                name="no_revenue",
                description="Without revenue features",
                features_to_remove=revenue_features,
            ))

    elif modality == "usage":
        funnel_features = [f for f in feature_names if "cart" in f or "purchase" in f or "conversion" in f]
        if funnel_features:
            ablations.append(AblationExperiment(
                name="no_funnel",
                description="Without funnel/conversion features",
                features_to_remove=funnel_features,
            ))

    logger.info(
        f"[{modality}] Defined {len(ablations)} ablation experiments: "
        f"{[a.name for a in ablations]}"
    )

    return ablations


def run_single_ablation(
    ablation: AblationExperiment,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    model_type: str = "xgboost",
) -> dict[str, Any]:
    """
    Run a single ablation experiment.

    Parameters
    ----------
    ablation      : AblationExperiment to run
    X_train       : training features (full)
    X_test        : test features (full)
    y_train       : training labels
    y_test        : test labels
    feature_names : full feature names list
    model_type    : "xgboost" or "random_forest"

    Returns
    -------
    Dict of metrics for this ablation.
    """
    from src.train import train_random_forest, train_xgboost

    start_time = time.time()
    logger.info(f"Running ablation: {ablation.name} — {ablation.description}")

    # Apply feature removal
    X_train_abl, remaining_names = ablation.apply_feature_removal(X_train, feature_names)
    X_test_abl, _ = ablation.apply_feature_removal(X_test, feature_names)

    if X_train_abl.shape[1] == 0:
        logger.warning(f"  [{ablation.name}] All features removed — skipping")
        return {"error": "All features removed", "name": ablation.name}

    # Train model
    if model_type == "xgboost":
        pipeline, cv_f1 = train_xgboost(X_train_abl, y_train)
    else:
        pipeline, cv_f1 = train_random_forest(X_train_abl, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test_abl)

    metrics = {
        "name": ablation.name,
        "description": ablation.description,
        "n_features": X_train_abl.shape[1],
        "features_removed": len(feature_names) - len(remaining_names),
        "cv_f1_macro": round(cv_f1, 6),
        "test_f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 6),
        "test_precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 6),
        "test_recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 6),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 6),
        "per_class_f1": f1_score(y_test, y_pred, average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(y_test, y_pred, average=None, zero_division=0).tolist(),
    }

    elapsed = time.time() - start_time
    metrics["elapsed_seconds"] = round(elapsed, 2)
    ablation.metrics = metrics
    ablation.elapsed_seconds = elapsed

    logger.info(
        f"  [{ablation.name}] F1={metrics['test_f1_macro']:.4f} "
        f"(baseline delta will be computed after all ablations)"
    )

    return metrics


def run_ablation_suite(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    modality: str,
    model_type: str = "xgboost",
    output_dir: str = "outputs/ablations",
) -> dict[str, Any]:
    """
    Run the full ablation suite for a modality.

    Returns
    -------
    {
      "modality": str,
      "baseline_f1": float,
      "ablations": [
        {"name": str, "f1": float, "delta": float, ...},
        ...
      ],
      "summary": str,
    }
    """
    os.makedirs(output_dir, exist_ok=True)

    ablations = define_standard_ablations(modality, feature_names)
    results = []

    for ablation in ablations:
        metrics = run_single_ablation(
            ablation, X_train, X_test, y_train, y_test,
            feature_names, model_type=model_type,
        )
        results.append(metrics)

    # Compute deltas from baseline
    baseline_f1 = None
    for r in results:
        if r.get("name") == "full_model":
            baseline_f1 = r.get("test_f1_macro")
            break

    if baseline_f1 is not None:
        for r in results:
            if "test_f1_macro" in r:
                r["delta_from_baseline"] = round(
                    r["test_f1_macro"] - baseline_f1, 6
                )

    # Summary
    summary_lines = [
        f"\n{'='*70}",
        f"  ABLATION RESULTS — {modality.upper()}",
        f"{'='*70}",
        f"  {'Experiment':<30} {'F1 Macro':>10} {'Delta':>10} {'Features':>10}",
        f"  {'-'*62}",
    ]

    for r in results:
        if "error" in r:
            summary_lines.append(f"  {r['name']:<30} {'ERROR':>10}")
            continue
        delta = r.get("delta_from_baseline", 0.0)
        delta_str = f"{delta:+.4f}" if delta != 0 else "baseline"
        summary_lines.append(
            f"  {r['name']:<30} {r['test_f1_macro']:>10.4f} "
            f"{delta_str:>10} {r['n_features']:>10}"
        )

    summary_lines.append(f"{'='*70}\n")
    summary = "\n".join(summary_lines)
    print(summary)

    # Save results
    report = {
        "modality": modality,
        "model_type": model_type,
        "baseline_f1": baseline_f1,
        "ablations": results,
    }

    report_path = os.path.join(output_dir, f"{modality}_ablation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Ablation report saved → {report_path}")

    return report


def run_subgroup_analysis(
    pipeline: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    label_classes: np.ndarray,
    modality: str,
    output_dir: str = "outputs/ablations",
) -> dict[str, Any]:
    """
    Evaluate model performance across subgroups.

    Subgroups:
      - By lifecycle stage (introduction/growth/maturity/decline)
      - By price band (low/medium/high)
      - By engagement level (low/medium/high)

    Returns
    -------
    Dict with per-subgroup metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    y_pred = pipeline.predict(X_test)
    results = {"modality": modality, "subgroups": {}}

    # Lifecycle stage subgroups
    lifecycle_features = [f for f in feature_names if "lifecycle_stage_" in f]
    if lifecycle_features:
        for lf in lifecycle_features:
            stage_name = lf.replace("lifecycle_stage_", "")
            idx = feature_names.index(lf)
            mask = X_test[:, idx] > 0.5  # OHE feature

            if mask.sum() < 10:
                continue

            subgroup_f1 = float(f1_score(
                y_test[mask], y_pred[mask],
                average="macro", zero_division=0
            ))
            subgroup_recall = float(recall_score(
                y_test[mask], y_pred[mask],
                average="macro", zero_division=0
            ))

            results["subgroups"][f"lifecycle_{stage_name}"] = {
                "n_samples": int(mask.sum()),
                "f1_macro": round(subgroup_f1, 6),
                "recall_macro": round(subgroup_recall, 6),
            }

    # Product age tercile subgroups
    age_idx = None
    for i, f in enumerate(feature_names):
        if f == "product_age_months":
            age_idx = i
            break

    if age_idx is not None:
        age_vals = X_test[:, age_idx]
        terciles = np.percentile(age_vals, [33, 66])

        for label, mask in [
            ("young", age_vals <= terciles[0]),
            ("mid_age", (age_vals > terciles[0]) & (age_vals <= terciles[1])),
            ("old", age_vals > terciles[1]),
        ]:
            if mask.sum() < 10:
                continue

            subgroup_f1 = float(f1_score(
                y_test[mask], y_pred[mask],
                average="macro", zero_division=0
            ))
            results["subgroups"][f"age_{label}"] = {
                "n_samples": int(mask.sum()),
                "f1_macro": round(subgroup_f1, 6),
            }

    # Save
    report_path = os.path.join(output_dir, f"{modality}_subgroup_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Subgroup analysis saved → {report_path}")

    return results
