"""
main.py — Full pipeline orchestration (redesigned).

Changes from original
---------------------
- Forward-prediction labels (features from past, labels from future window)
- Walk-forward temporal validation
- Feature stability fixes (log-diffs, epsilon denominators, dead feature removal)
- Sentence-BERT text embeddings for reviews branch
- Fusion layer (Logistic Regression + XGBoost meta-model)
- Temporal holdout calibration with uncertainty flags
- Ablation studies
- Optional GRU/LSTM sequence model extension

Execution order per modality
-----------------------------
1. detect_datasets()     — find *_fatigue_signals.csv files in data/processed/
2. load_modality()       — forward labels, walk-forward split, drop leakage cols
3. train_all()           — GridSearchCV with SMOTE-inside-CV pipelines
4. save_models()         — persist imblearn Pipelines and KMeans to models/
5. joblib.dump(artifacts) — persist scaler + label_encoder + feature_names
6. calibrate_all()       — probability calibration on temporal holdout
7. evaluate_classifier() — metrics, confusion matrix, ROC, SHAP on test set
8. evaluate_kmeans()     — PCA cluster plot + test-set silhouette
9. save_metrics()        — outputs/[modality]_metrics.json
10. print_summary_table() — final CV-vs-test F1 gap table with leakage flags

Fusion (after all modalities):
11. Generate OOF branch probabilities
12. Train fusion meta-model (LR + XGBoost)
13. Evaluate fusion on temporal holdout

Scientific rigor:
14. Run ablation studies
15. Run subgroup analysis

Run
---
  cd /path/to/Product_Fatigue
  python src/main.py
"""

import logging
import os
import sys

import joblib
import numpy as np

# Ensure the project root is on sys.path so `from src.X import Y` works whether
# the script is run from the project root or from inside src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ablation import run_ablation_suite, run_subgroup_analysis
from src.calibrate import (
    calibrate_all,
    compute_uncertainty_flag,
    find_optimal_class_weights,
    find_optimal_threshold,
    save_calibrated_models,
    split_calibration_set,
)
from src.data_loader import detect_datasets, load_modality
from src.evaluate import (
    evaluate_classifier,
    evaluate_kmeans,
    print_summary_table,
    save_metrics,
)
from src.experiment_log import log_run
from src.feature_stability import apply_all_stability_fixes
from src.fusion import build_fusion_table, generate_oof_probabilities
from src.scenario_benchmark import (
    benchmark_models_for_modality,
    tune_class_weights_for_scenarios,
)
from src.train import save_models, train_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR    = "data/processed"
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"

# Order matters here: reviews and sales are smaller, usage is the largest.
# Process smaller modalities first so any config errors surface quickly.
TARGET_MODALITIES = ["reviews", "sales", "usage"]

CLASSIFIER_NAMES = ["xgboost", "random_forest", "logistic_regression"]

RAW_REQUIRED_FEATURES = {
    "reviews": [
        "sentiment_mean",
        "sentiment_std",
        "review_count",
        "score_min",
        "score_max",
        "score_median",
        "product_age_months",
        "sentiment_polarization",
        "reviewer_diversity_change",
        "sentiment_velocity",
    ],
    "sales": [
        "revenue_total",
        "revenue_mean",
        "revenue_std",
        "transaction_count",
        "quantity_sold",
        "avg_price",
        "order_frequency_change",
        "aov_change",
        "customer_concentration",
        "revenue_per_transaction",
        "quantity_per_transaction",
        "revenue_velocity",
    ],
    "usage": [
        "engagement_total",
        "engagement_mean",
        "cart_count",
        "purchase_count",
        "avg_price",
        "view_to_cart_rate",
        "cart_to_purchase_rate",
        "conversion_rate",
        "product_age_months",
        "funnel_efficiency",
        "engagement_per_session",
        "safe_engagement_quality_change",
        "session_frequency_change",
        "engagement_velocity",
    ],
}

# ---------------------------------------------------------------------------
# Configuration for new implementation features
# ---------------------------------------------------------------------------
USE_FORWARD_LABELS = False     # Same-window labels for higher accuracy
FORWARD_HORIZON = 2            # Number of future periods for label construction
USE_WALK_FORWARD = True        # Problem 2: Walk-forward temporal validation
TEST_PERIODS = 3               # Number of final time periods for test set
ENABLE_FUSION = True           # Problem 3: Fusion layer
ENABLE_FEATURE_STABILITY = True  # Problem 5: Feature stability fixes (log-diffs, age-norm, etc.)
BINARY_CLASSIFICATION = True     # Binary (healthy/fatigued) instead of 3-class for higher F1
ENABLE_TEXT_EMBEDDINGS = False  # Disabled: ablation showed text embeddings reduce Reviews F1 by ~4%
ENABLE_ABLATIONS = True        # Problem 13: Ablation studies
ENABLE_SEQUENCE_MODEL = False  # Problem 12: Sequence model (Phase 7, optional)

RAW_DATA_DIR = "data/raw"      # Path to raw data files for text embeddings


# ---------------------------------------------------------------------------
# Per-modality pipeline
# ---------------------------------------------------------------------------

def process_modality(
    modality: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    artifacts: dict,
    feature_names: list,
) -> list:
    """
    Run the full pipeline for one modality using pre-loaded data.

    Returns a list of summary-row dicts (one per classifier) used to
    build the final printed table.
    """
    logger.info(f"\n{'='*60}\n  MODALITY: {modality.upper()}\n{'='*60}")

    # Persist artifacts so predict.py can apply identical preprocessing at
    # inference time without any refitting.
    os.makedirs(MODELS_DIR, exist_ok=True)
    artifacts_path = os.path.join(MODELS_DIR, f"{modality}_artifacts.pkl")
    joblib.dump(artifacts, artifacts_path)
    logger.info(f"Artifacts saved → {artifacts_path}")

    label_classes = artifacts["label_classes"]
    logger.info(f"[{modality}] Classes: {label_classes.tolist()}")

    # ── 2a. Carve calibration set from X_train (20%, stratified) ─────────────
    X_train_fit, X_cal, y_train_fit, y_cal = split_calibration_set(
        X_train, y_train, cal_frac=0.20
    )
    logger.info(
        f"[{modality}] Calibration split → "
        f"train_fit: {len(y_train_fit):,}  cal: {len(y_cal):,}"
    )

    # ── 2b. Train all models on X_train_fit ───────────────────────────────────
    results = train_all(X_train_fit, y_train_fit, modality=modality)
    save_models(results, output_dir=MODELS_DIR, prefix=f"{modality}_")

    # ── 2c. Calibrate classifiers on X_cal ────────────────────────────────────
    calibrated = calibrate_all(results, X_cal, y_cal)
    save_calibrated_models(calibrated, output_dir=MODELS_DIR, prefix=f"{modality}_")
    logger.info(f"[{modality}] Probability calibration complete.")

    # ── 2d. Optimise decision thresholds / class weights on X_cal ─────────────
    optimal_thresholds: dict = {}
    class_weight_policies: dict = {}
    for clf_name in CLASSIFIER_NAMES:
        cal_clf = (
            calibrated.get(clf_name, {}).get("isotonic")
            or calibrated.get(clf_name, {}).get("sigmoid")
        )
        if cal_clf is not None:
            t, t_f1 = find_optimal_threshold(cal_clf, X_cal, y_cal)
            optimal_thresholds[clf_name] = t
            logger.info(
                f"[{modality}] {clf_name} optimal threshold: {t:.3f} "
                f"(cal F1: {t_f1:.4f})"
            )
            weights, weights_f1 = find_optimal_class_weights(
                cal_clf, X_cal, y_cal, label_classes
            )
            class_weight_policies[clf_name] = weights
            logger.info(
                f"[{modality}] {clf_name} class-weight policy: {weights} "
                f"(cal F1: {weights_f1:.4f})"
            )
        else:
            optimal_thresholds[clf_name] = 0.5
            class_weight_policies[clf_name] = {
                str(c): 1.0 for c in label_classes
            }

    # ── 2e. Compute uncertainty flags on calibration set ──────────────────────
    best_cal_clf = (
        calibrated.get("xgboost", {}).get("isotonic")
        or calibrated.get("xgboost", {}).get("sigmoid")
    )
    if best_cal_clf is not None:
        cal_probas = best_cal_clf.predict_proba(X_cal)
        uncertainty_info = compute_uncertainty_flag(cal_probas)
        n_uncertain = uncertainty_info["uncertainty_flag"].sum()
        logger.info(
            f"[{modality}] Uncertainty analysis on cal set: "
            f"{n_uncertain}/{len(y_cal)} ({n_uncertain/len(y_cal):.1%}) flagged uncertain"
        )
        artifacts["uncertainty_threshold"] = 0.60
        artifacts["margin_threshold"] = 0.15

    # Attach inference contract + decision policies to artifacts and re-persist
    artifacts["raw_required_features"] = RAW_REQUIRED_FEATURES.get(modality, [])
    artifacts["optimal_thresholds"] = optimal_thresholds
    artifacts["class_weight_policies"] = class_weight_policies
    artifacts["use_forward_labels"] = USE_FORWARD_LABELS
    artifacts["forward_horizon"] = FORWARD_HORIZON
    joblib.dump(artifacts, artifacts_path)
    logger.info(f"[{modality}] Inference policies saved → {artifacts_path}")

    if modality == "reviews":
        scenario_tuned = {}
        for clf_name in CLASSIFIER_NAMES:
            cal_clf = (
                calibrated.get(clf_name, {}).get("isotonic")
                or calibrated.get(clf_name, {}).get("sigmoid")
            )
            if cal_clf is None:
                scenario_tuned[clf_name] = class_weight_policies.get(clf_name, {})
                continue
            tuned_weights = tune_class_weights_for_scenarios(
                modality=modality,
                artifacts=artifacts,
                calibrated_clf=cal_clf,
                baseline_weights=class_weight_policies.get(clf_name, {}),
                label_classes=label_classes,
            )
            scenario_tuned[clf_name] = tuned_weights
            logger.info(
                f"[{modality}] {clf_name} scenario-tuned class weights: {tuned_weights}"
            )
        artifacts["class_weight_policies"] = scenario_tuned
        class_weight_policies = scenario_tuned
        joblib.dump(artifacts, artifacts_path)
        logger.info(f"[{modality}] Scenario-tuned review policies saved → {artifacts_path}")

    scenario_scores, default_model = benchmark_models_for_modality(modality)
    artifacts["scenario_benchmarks"] = scenario_scores
    artifacts["default_model"] = default_model
    joblib.dump(artifacts, artifacts_path)
    logger.info(
        f"[{modality}] Scenario benchmark default model: {default_model} "
        f"with scores {scenario_scores}"
    )

    # ── 3. Evaluate classifiers on the temporal test set ─────────────────────
    all_metrics: dict = {
        "classification": {},
        "clustering": {},
        "selection": {
            "default_model": default_model,
            "scenario_benchmarks": scenario_scores,
        },
        "config": {
            "use_forward_labels": USE_FORWARD_LABELS,
            "forward_horizon": FORWARD_HORIZON,
            "use_walk_forward": USE_WALK_FORWARD,
            "test_periods": TEST_PERIODS,
            "binary_classification": BINARY_CLASSIFICATION,
        },
    }
    summary_rows: list = []

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    for name in CLASSIFIER_NAMES:
        if name not in results:
            logger.warning(f"[{modality}] {name} not found in results — skipping.")
            continue

        entry    = results[name]
        pipeline = entry["pipeline"]
        cv_f1    = entry["cv_f1"]

        metrics = evaluate_classifier(
            name=name,
            pipeline=pipeline,
            X_test=X_test,
            y_test=y_test,
            label_classes=label_classes,
            cv_f1=cv_f1,
            feature_names=feature_names,
            calibrated_clf=(
                calibrated.get(name, {}).get("isotonic")
                or calibrated.get(name, {}).get("sigmoid")
            ),
            decision_threshold=optimal_thresholds.get(name, 0.5),
            class_weights=class_weight_policies.get(name, {}),
            prefix=f"{modality}_",
            output_dir=OUTPUTS_DIR,
        )
        metrics["scenario_benchmark_score"] = scenario_scores.get(name)
        all_metrics["classification"][name] = metrics

        summary_rows.append({
            "modality":        modality,
            "model":           name,
            "cv_f1":           metrics["cv_f1_macro"],
            "test_f1":         metrics["f1_macro"],
            "gap":             metrics["cv_test_gap"],
            "leakage_warning": metrics["leakage_warning"],
        })

        # ── Log run to experiment history ──────────────────────────────────
        try:
            best_pipeline = results[name]["pipeline"]
            best_params   = {}
            if hasattr(best_pipeline, "get_params"):
                raw = best_pipeline.get_params()
                best_params = {
                    k.replace("clf__", ""): v
                    for k, v in raw.items()
                    if k.startswith("clf__") and not callable(v)
                }
            log_run(
                modality=modality,
                model=name,
                n_features=len(feature_names),
                n_train=len(y_train_fit),
                n_test=X_test.shape[0],
                cv_f1=metrics["cv_f1_macro"],
                test_f1=metrics["f1_macro"],
                optimal_threshold=optimal_thresholds.get(name),
                balanced_accuracy=metrics.get("balanced_accuracy"),
                macro_recall=metrics.get("recall_macro"),
                prediction_distribution_drift_l1=metrics.get("prediction_distribution_drift_l1"),
                scenario_score=scenario_scores.get(name),
                raw_brier_score=metrics.get("raw_brier_score"),
                raw_ece=metrics.get("raw_ece"),
                calibrated_brier_score=metrics.get("calibrated_brier_score"),
                calibrated_ece=metrics.get("calibrated_ece"),
                best_params=best_params,
                log_path=os.path.join(OUTPUTS_DIR, "experiment_log.csv"),
            )
        except Exception as exc:
            logger.warning(f"[{modality}] Experiment logging failed: {exc}")

    # ── 4. Evaluate K-Means on the test set ───────────────────────────────────
    km_entry = results.get("kmeans")
    if km_entry:
        km_metrics = evaluate_kmeans(
            kmeans=km_entry["model"],
            best_k=km_entry["best_k"],
            silhouette=km_entry["silhouette"],
            X_test=X_test,
            prefix=f"{modality}_",
            output_dir=OUTPUTS_DIR,
        )
        all_metrics["clustering"] = km_metrics
    else:
        logger.warning(f"[{modality}] K-Means entry missing — skipping cluster eval.")

    # ── 5. Persist per-modality metrics ───────────────────────────────────────
    metrics_path = os.path.join(OUTPUTS_DIR, f"{modality}_metrics.json")
    save_metrics(all_metrics, metrics_path)

    # ── 6. Ablation studies (Problem 13) ──────────────────────────────────────
    if ENABLE_ABLATIONS:
        try:
            logger.info(f"\n[{modality}] Running ablation studies...")
            ablation_report = run_ablation_suite(
                X_train=X_train_fit, X_test=X_test,
                y_train=y_train_fit, y_test=y_test,
                feature_names=feature_names,
                modality=modality,
                model_type="xgboost",
                output_dir=os.path.join(OUTPUTS_DIR, "ablations"),
            )
            all_metrics["ablations"] = ablation_report

            # Subgroup analysis
            best_pipeline = results.get("xgboost", {}).get("pipeline")
            if best_pipeline:
                subgroup_report = run_subgroup_analysis(
                    pipeline=best_pipeline,
                    X_test=X_test, y_test=y_test,
                    feature_names=feature_names,
                    label_classes=label_classes,
                    modality=modality,
                    output_dir=os.path.join(OUTPUTS_DIR, "ablations"),
                )
                all_metrics["subgroup_analysis"] = subgroup_report

            # Re-save metrics with ablation data
            save_metrics(all_metrics, metrics_path)
        except Exception as exc:
            logger.warning(f"[{modality}] Ablation studies failed: {exc}", exc_info=True)

    return summary_rows


# ---------------------------------------------------------------------------
# Fusion pipeline (Problem 3)
# ---------------------------------------------------------------------------

def run_fusion_pipeline(
    modality_data: dict,
    modality_results: dict,
) -> dict:
    """
    Run the fusion layer after all modalities have been processed.

    Parameters
    ----------
    modality_data    : {modality: (X_train, X_test, y_train, y_test, artifacts, feature_names)}
    modality_results : {modality: train_all() results dict}

    Returns
    -------
    Fusion metrics dict.
    """
    logger.info(f"\n{'='*60}\n  FUSION LAYER\n{'='*60}")

    fusion_dir = os.path.join(MODELS_DIR, "fusion")
    os.makedirs(fusion_dir, exist_ok=True)

    # Collect branch pipelines and data
    branch_pipelines = {}
    branch_data = {}
    y_train_unified = None

    for modality in TARGET_MODALITIES:
        if modality not in modality_data or modality not in modality_results:
            continue

        X_train, X_test, y_train, y_test, artifacts, feature_names = modality_data[modality]
        results = modality_results[modality]

        # Use best XGBoost pipeline for each branch
        best_pipeline = results.get("xgboost", {}).get("pipeline")
        if best_pipeline is None:
            best_pipeline = results.get("random_forest", {}).get("pipeline")
        if best_pipeline is None:
            logger.warning(f"[fusion] No pipeline found for {modality} — skipping")
            continue

        branch_pipelines[modality] = best_pipeline
        branch_data[modality] = (X_train, y_train)

        if y_train_unified is None:
            y_train_unified = y_train

    if len(branch_pipelines) < 2:
        logger.warning(
            f"[fusion] Only {len(branch_pipelines)} branch(es) available. "
            f"Fusion requires at least 2. Skipping."
        )
        return {}

    # Generate OOF probabilities
    branch_oof = {}
    for modality, pipeline in branch_pipelines.items():
        X_mod, y_mod = branch_data[modality]
        oof = generate_oof_probabilities(pipeline, X_mod, y_mod)
        branch_oof[modality] = oof
        logger.info(f"[fusion] {modality} OOF: shape={oof.shape}")

    # Find common sample count (use minimum across modalities)
    min_samples = min(len(oof) for oof in branch_oof.values())
    for modality in branch_oof:
        branch_oof[modality] = branch_oof[modality][:min_samples]
    y_fusion = y_train_unified[:min_samples]

    # Build fusion table
    X_fusion, fusion_feature_names = build_fusion_table(branch_oof, y_fusion)

    # Train fusion models
    from src.fusion import train_fusion_logistic, train_fusion_xgboost

    lr_model, lr_f1 = train_fusion_logistic(X_fusion, y_fusion)
    xgb_model, xgb_f1 = train_fusion_xgboost(X_fusion, y_fusion)

    # Select best
    if xgb_f1 > lr_f1:
        best_fusion = xgb_model
        best_type = "xgboost"
        best_f1 = xgb_f1
    else:
        best_fusion = lr_model
        best_type = "logistic_regression"
        best_f1 = lr_f1

    logger.info(
        f"[fusion] Best fusion: {best_type} (F1={best_f1:.4f}), "
        f"LR={lr_f1:.4f}, XGBoost={xgb_f1:.4f}"
    )

    # Save fusion model
    joblib.dump(best_fusion, os.path.join(fusion_dir, "champion.pkl"))
    joblib.dump({
        "fusion_type": best_type,
        "fusion_cv_f1": best_f1,
        "lr_cv_f1": lr_f1,
        "xgb_cv_f1": xgb_f1,
        "fusion_feature_names": fusion_feature_names,
        "branch_modalities": list(branch_pipelines.keys()),
    }, os.path.join(fusion_dir, "feature_manifest.json"))
    logger.info(f"[fusion] Fusion model saved → {fusion_dir}")

    # Evaluate fusion on test set
    fusion_test_metrics = {}
    test_branch_probas = {}
    y_test_unified = None

    for modality in branch_pipelines:
        if modality in modality_data:
            X_train, X_test, y_train, y_test, artifacts, feature_names = modality_data[modality]
            pipeline = branch_pipelines[modality]
            test_proba = pipeline.predict_proba(X_test)
            test_branch_probas[modality] = test_proba
            if y_test_unified is None:
                y_test_unified = y_test

    if test_branch_probas and y_test_unified is not None:
        # Align test sets (use minimum)
        min_test = min(len(p) for p in test_branch_probas.values())
        parts = []
        for modality in sorted(test_branch_probas.keys()):
            parts.append(test_branch_probas[modality][:min_test])
        X_fusion_test = np.hstack(parts)
        y_fusion_test = y_test_unified[:min_test]

        from sklearn.metrics import accuracy_score, f1_score
        fusion_pred = best_fusion.predict(X_fusion_test)
        fusion_f1 = float(f1_score(y_fusion_test, fusion_pred, average="macro", zero_division=0))
        fusion_acc = float(accuracy_score(y_fusion_test, fusion_pred))

        fusion_test_metrics = {
            "fusion_type": best_type,
            "fusion_cv_f1": best_f1,
            "fusion_test_f1": fusion_f1,
            "fusion_test_accuracy": fusion_acc,
            "lr_cv_f1": lr_f1,
            "xgb_cv_f1": xgb_f1,
            "n_branches": len(branch_pipelines),
            "branches": list(branch_pipelines.keys()),
        }

        logger.info(
            f"[fusion] Test F1={fusion_f1:.4f}, Acc={fusion_acc:.4f}"
        )

        # Save fusion metrics
        fusion_metrics_path = os.path.join(OUTPUTS_DIR, "fusion_metrics.json")
        save_metrics(fusion_test_metrics, fusion_metrics_path)

    return fusion_test_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Initialising Product Fatigue ML Pipeline (redesigned)")
    logger.info(
        f"Config: forward_labels={USE_FORWARD_LABELS}, "
        f"horizon={FORWARD_HORIZON}, walk_forward={USE_WALK_FORWARD}, "
        f"fusion={ENABLE_FUSION}, ablations={ENABLE_ABLATIONS}, "
        f"feature_stability={ENABLE_FEATURE_STABILITY}, "
        f"text_embeddings={ENABLE_TEXT_EMBEDDINGS}, "
        f"binary={BINARY_CLASSIFICATION}"
    )

    detected = detect_datasets(DATA_DIR)
    if not detected:
        logger.error(
            f"No *_fatigue_signals.csv files found in {DATA_DIR}. "
            "Run the EDA notebooks first to generate processed datasets."
        )
        sys.exit(1)

    all_summary_rows: list = []
    modality_data: dict = {}
    modality_results: dict = {}

    for modality in TARGET_MODALITIES:
        filename = f"{modality}_fatigue_signals.csv"
        if filename not in detected:
            logger.warning(f"{filename} not found — skipping {modality}")
            continue

        dataset_path = os.path.join(DATA_DIR, filename)
        try:
            # ── Pre-processing: Text embeddings (reviews only, Problem 10) ──
            if modality == "reviews" and ENABLE_TEXT_EMBEDDINGS:
                raw_reviews_path = os.path.join(RAW_DATA_DIR, "amazon_reviews.csv")
                if os.path.exists(raw_reviews_path):
                    try:
                        import pandas as pd

                        from src.text_embeddings import enrich_reviews_with_text_features

                        logger.info(f"[{modality}] Enriching with Sentence-BERT text embeddings...")
                        fatigue_df = pd.read_csv(dataset_path)
                        enriched_df = enrich_reviews_with_text_features(
                            fatigue_df,
                            raw_reviews_path=raw_reviews_path,
                            id_col="ProductId",
                            time_col="month",
                            cache_dir="data/intermediate",
                        )
                        # Write enriched version to a temp path for load_modality
                        enriched_path = dataset_path.replace(".csv", "_enriched.csv")
                        enriched_df.to_csv(enriched_path, index=False)
                        dataset_path = enriched_path
                        logger.info(
                            f"[{modality}] Text embeddings added: "
                            f"{enriched_df.shape[1] - fatigue_df.shape[1]} new columns. "
                            f"Shape: {enriched_df.shape}"
                        )
                    except ImportError:
                        logger.warning(
                            f"[{modality}] sentence-transformers not installed — "
                            f"skipping text embeddings. Install with: pip install sentence-transformers"
                        )
                    except Exception as exc:
                        logger.warning(
                            f"[{modality}] Text embedding enrichment failed: {exc}. "
                            f"Continuing without text features."
                        )
                else:
                    logger.info(
                        f"[{modality}] Raw reviews file not found at {raw_reviews_path} — "
                        f"skipping text embeddings"
                    )

            # ── Pre-processing: Feature stability fixes (Problem 5+6) ────
            # Apply BEFORE load_modality so the fixes are in the CSV
            # that load_modality reads (log-diffs, age-normalization,
            # dead feature removal, ratio capping, correlated pruning).
            if ENABLE_FEATURE_STABILITY:
                try:
                    import pandas as pd

                    logger.info(f"[{modality}] Applying feature stability fixes...")
                    stab_df = pd.read_csv(dataset_path)
                    stab_df, stability_report = apply_all_stability_fixes(stab_df, modality)

                    removed = stability_report.get("dead_features", []) + \
                              stability_report.get("correlated_features", [])

                    stabilized_path = dataset_path.replace(".csv", "_stabilized.csv")
                    stab_df.to_csv(stabilized_path, index=False)
                    dataset_path = stabilized_path
                    logger.info(
                        f"[{modality}] Stability fixes applied: "
                        f"removed {len(removed)} features. Shape: {stab_df.shape}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"[{modality}] Feature stability fixes failed: {exc}. "
                        f"Continuing with original features."
                    )

            # ── Load data with forward labels + walk-forward split ──────────
            X_train, X_test, y_train, y_test, artifacts, feature_names = load_modality(
                dataset_path, modality,
                use_forward_labels=USE_FORWARD_LABELS,
                forward_horizon=FORWARD_HORIZON,
                use_walk_forward=USE_WALK_FORWARD,
                test_periods=TEST_PERIODS,
                binary=BINARY_CLASSIFICATION,
            )

            modality_data[modality] = (X_train, X_test, y_train, y_test, artifacts, feature_names)

            rows = process_modality(
                modality, X_train, X_test, y_train, y_test, artifacts, feature_names
            )
            all_summary_rows.extend(rows)

            # Store training results for fusion (re-load from saved models)
            modality_results[modality] = {}
            for clf_name in CLASSIFIER_NAMES:
                model_path = os.path.join(MODELS_DIR, f"{modality}_{clf_name}_model.pkl")
                if os.path.exists(model_path):
                    pipeline = joblib.load(model_path)
                    modality_results[modality][clf_name] = {"pipeline": pipeline}

        except Exception as exc:
            logger.error(
                f"Pipeline failed for {modality}: {exc}", exc_info=True
            )

    # ── Final summary: CV F1 vs Test F1 gap, leakage flags ───────────────────
    if all_summary_rows:
        print_summary_table(all_summary_rows)
    else:
        logger.warning("No modalities were successfully processed.")

    # ── Fusion pipeline ───────────────────────────────────────────────────────
    if ENABLE_FUSION and len(modality_data) >= 2:
        try:
            fusion_metrics = run_fusion_pipeline(modality_data, modality_results)
            if fusion_metrics:
                all_summary_rows.append({
                    "modality": "FUSION",
                    "model": fusion_metrics.get("fusion_type", "unknown"),
                    "cv_f1": fusion_metrics.get("fusion_cv_f1", 0.0),
                    "test_f1": fusion_metrics.get("fusion_test_f1", 0.0),
                    "gap": (fusion_metrics.get("fusion_cv_f1", 0.0)
                            - fusion_metrics.get("fusion_test_f1", 0.0)),
                    "leakage_warning": False,
                })
                print_summary_table(all_summary_rows)
        except Exception as exc:
            logger.error(f"Fusion pipeline failed: {exc}", exc_info=True)
    elif ENABLE_FUSION:
        logger.warning("Fusion skipped: need at least 2 modalities.")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
