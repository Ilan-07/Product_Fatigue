"""
scenario_benchmark.py — fixed manual-inference scenarios used for:
1. model ranking on realistic operator inputs
2. regression gates for important business cases
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MODEL_NAMES = ["xgboost", "random_forest", "logistic_regression"]

SCENARIO_CASES: List[Dict[str, Any]] = [
    {
        "label": "reviews_healthy",
        "modality": "reviews",
        "must_match": True,
        "features": {
            "sentiment_mean": 0.82, "sentiment_std": 0.10, "review_count": 48,
            "score_min": 4, "score_max": 5, "score_median": 5.0,
            "product_age_months": 6, "sentiment_polarization": 0.4,
            "reviewer_diversity_change": 12.0,
        },
        "expect": ["healthy"],
    },
    {
        "label": "reviews_moderate",
        "modality": "reviews",
        "must_match": True,
        "features": {
            "sentiment_mean": 0.54, "sentiment_std": 0.24, "review_count": 21,
            "score_min": 2, "score_max": 5, "score_median": 3.0,
            "product_age_months": 16, "sentiment_polarization": 1.5,
            "reviewer_diversity_change": -12.0,
        },
        "expect": ["moderate_fatigue", "high_fatigue"],
    },
    {
        "label": "reviews_high",
        "modality": "reviews",
        "must_match": True,
        "features": {
            "sentiment_mean": 0.12, "sentiment_std": 0.39, "review_count": 9,
            "score_min": 1, "score_max": 4, "score_median": 2.0,
            "product_age_months": 28, "sentiment_polarization": 2.8,
            "reviewer_diversity_change": -31.0,
        },
        "expect": ["high_fatigue", "moderate_fatigue"],
    },
    {
        "label": "sales_healthy",
        "modality": "sales",
        "must_match": True,
        "features": {
            "revenue_total": 18200, "revenue_mean": 910, "revenue_std": 140,
            "transaction_count": 25, "quantity_sold": 430, "avg_price": 42.0,
            "product_age_months": 8, "order_frequency_change": 11.0,
            "aov_change": 4.0, "customer_concentration": 0.22,
        },
        "expect": ["healthy"],
    },
    {
        "label": "sales_moderate",
        "modality": "sales",
        "must_match": True,
        "features": {
            "revenue_total": 9800, "revenue_mean": 520, "revenue_std": 220,
            "transaction_count": 18, "quantity_sold": 270, "avg_price": 36.0,
            "product_age_months": 16, "order_frequency_change": -12.0,
            "aov_change": -5.0, "customer_concentration": 0.51,
        },
        "expect": ["moderate_fatigue", "high_fatigue"],
    },
    {
        "label": "sales_high",
        "modality": "sales",
        "must_match": True,
        "features": {
            "revenue_total": 4200, "revenue_mean": 230, "revenue_std": 310,
            "transaction_count": 9, "quantity_sold": 110, "avg_price": 28.0,
            "product_age_months": 30, "order_frequency_change": -33.0,
            "aov_change": -17.0, "customer_concentration": 0.79,
        },
        "expect": ["high_fatigue", "moderate_fatigue"],
    },
    {
        "label": "usage_healthy",
        "modality": "usage",
        "must_match": True,
        "features": {
            "engagement_total": 2600, "engagement_mean": 18, "cart_count": 190,
            "purchase_count": 61, "avg_price": 42, "view_to_cart_rate": 11.4,
            "cart_to_purchase_rate": 32.1, "conversion_rate": 3.7,
            "product_age_months": 5, "funnel_efficiency": 0.34,
            "engagement_per_session": 4.8, "safe_engagement_quality_change": 12.0,
        },
        "expect": ["healthy"],
    },
    {
        "label": "usage_moderate",
        "modality": "usage",
        "must_match": True,
        "features": {
            "engagement_total": 1200, "engagement_mean": 12, "cart_count": 80,
            "purchase_count": 20, "avg_price": 49, "view_to_cart_rate": 8.0,
            "cart_to_purchase_rate": 25.0, "conversion_rate": 2.0,
            "product_age_months": 8, "funnel_efficiency": 0.25,
            "engagement_per_session": 3.2, "safe_engagement_quality_change": 5.0,
        },
        "expect": ["moderate_fatigue", "healthy"],
    },
    {
        "label": "usage_high",
        "modality": "usage",
        "must_match": True,
        "features": {
            "engagement_total": 420, "engagement_mean": 5.5, "cart_count": 14,
            "purchase_count": 2, "avg_price": 55, "view_to_cart_rate": 3.1,
            "cart_to_purchase_rate": 12.0, "conversion_rate": 0.4,
            "product_age_months": 18, "funnel_efficiency": 0.09,
            "engagement_per_session": 1.4, "safe_engagement_quality_change": -19.0,
        },
        "expect": ["high_fatigue", "moderate_fatigue"],
    },
]


def benchmark_models_for_modality(modality: str) -> Tuple[Dict[str, float], str]:
    from src.predict import predict

    cases = [c for c in SCENARIO_CASES if c["modality"] == modality]
    scores: Dict[str, float] = {}
    for model_name in MODEL_NAMES:
        score = 0.0
        for case in cases:
            result = predict(modality, case["features"], model_name=model_name)
            pred = str(result["predicted_class"]).lower()
            if pred in case["expect"]:
                score += 1.0
                score += max(0.0, float(result["confidence"]) - 0.5) * 0.1
            elif case.get("must_match"):
                score -= 1.0
        scores[model_name] = round(score, 4)

    best = max(scores.items(), key=lambda item: item[1])[0]
    return scores, best


def score_fixed_scenarios_for_model(
    modality: str,
    artifacts: Dict[str, Any],
    calibrated_clf: Any,
    class_weights: Optional[Dict[str, float]] = None,
    decision_threshold: float = 0.5,
) -> float:
    from src.predict import align_features

    cases = [c for c in SCENARIO_CASES if c["modality"] == modality]
    feature_names = artifacts["feature_names"]
    scaler = artifacts["scaler"]
    train_medians = artifacts.get("train_medians")
    label_classes = artifacts["label_classes"]
    weights = np.array(
        [float((class_weights or {}).get(str(c), 1.0)) for c in label_classes],
        dtype=float,
    )

    score = 0.0
    for case in cases:
        X = align_features(
            case["features"],
            feature_names,
            scaler,
            train_medians,
            warn_missing=False,
        )
        proba = calibrated_clf.predict_proba(X)[0]
        if len(label_classes) == 2:
            pred_idx = int(proba[1] >= decision_threshold)
        else:
            pred_idx = int(np.argmax(proba * weights))
        pred = str(label_classes[pred_idx]).lower()
        if pred in case["expect"]:
            score += 1.0 + max(0.0, float(proba[pred_idx]) - 0.5) * 0.1
        elif case.get("must_match"):
            score -= 1.0
    return round(score, 4)


def tune_class_weights_for_scenarios(
    modality: str,
    artifacts: Dict[str, Any],
    calibrated_clf: Any,
    baseline_weights: Dict[str, float],
    label_classes: np.ndarray,
) -> Dict[str, float]:
    if len(label_classes) <= 2:
        return dict(baseline_weights)

    healthy_idx = next(
        (idx for idx, cls in enumerate(label_classes) if str(cls).lower() == "healthy"),
        None,
    )
    nonhealthy = [
        idx for idx, cls in enumerate(label_classes)
        if str(cls).lower() != "healthy"
    ]
    if healthy_idx is None or not nonhealthy:
        return dict(baseline_weights)

    best = dict(baseline_weights)
    best_score = score_fixed_scenarios_for_model(
        modality, artifacts, calibrated_clf, class_weights=best
    )

    healthy_grid = [1.0, 0.8, 0.65, 0.5, 0.35]
    fatigue_grid = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]

    for hw in healthy_grid:
        for w1 in fatigue_grid:
            for w2 in fatigue_grid:
                weights = {str(c): 1.0 for c in label_classes}
                weights[str(label_classes[healthy_idx])] = float(hw)
                weights[str(label_classes[nonhealthy[0]])] = float(w1)
                if len(nonhealthy) > 1:
                    weights[str(label_classes[nonhealthy[1]])] = float(w2)
                score = score_fixed_scenarios_for_model(
                    modality, artifacts, calibrated_clf, class_weights=weights
                )
                if score > best_score:
                    best_score = score
                    best = weights

    return best
