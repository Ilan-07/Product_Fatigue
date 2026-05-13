import json
import logging
import math
import numpy as np
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_fastapi_instrumentator import Instrumentator

from .schemas import (
    DashboardPredictRequest,
    PredictionResponse,
    ReviewFeatures,
    SalesFeatures,
    UsageFeatures,
    FusionPredictionRequest,
    FusionPredictionResponse,
    BranchPrediction,
)

logger = logging.getLogger("fatigue_inference_api")

app = FastAPI(
    title="Product Fatigue Inference API",
    description="Multi-dataset ML inference endpoints for fatigue prediction",
    version="2.0.0",
)

# Register versioned routers
from .v1 import router as v1_router
from .v2 import router as v2_router
app.include_router(v1_router)
app.include_router(v2_router)

# CORS configuration for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Prometheus instrumentation globally (cannot be done inside an event loop)
Instrumentator().instrument(app).expose(app)

ROOT_DIR = Path(__file__).resolve().parents[2]
API_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOCAL_MLFLOW_DB = ROOT_DIR / "docker" / "mlruns" / "mlflow.db"
LOCAL_MLFLOW_ARTIFACTS_DIR = ROOT_DIR / "docker" / "mlruns" / "artifacts"
MODEL_PRIORITY = ("xgboost", "random_forest", "logistic_regression")
TARGET_MODALITIES = ("reviews", "sales", "usage")
TEMPLATES = Jinja2Templates(directory=str(API_DIR / "templates"))

app.mount("/dashboard/static", StaticFiles(directory=str(API_DIR / "static")), name="dashboard-static")

# In-memory caches
loaded_models: Dict[str, Any] = {}
loaded_preprocessors: Dict[str, Dict[str, Any]] = {}
loaded_model_versions: Dict[str, str] = {}

MODALITY_LABELS = {
    "reviews": "Reviews",
    "sales": "Sales",
    "usage": "Usage",
}

CLUSTER_CONTEXT = {
    "reviews": [
        "Stable review cadence with low sentiment drift.",
        "Polarized catalog cluster with patchy reviewer loyalty.",
        "Momentum-sensitive products where sentiment swings drive health.",
        "Mature products with elevated fatigue recovery risk.",
    ],
    "sales": [
        "Steady revenue products with broad customer spread.",
        "Promotion-sensitive items with volatile order frequency.",
        "High-concentration revenue cluster with churn exposure.",
        "Late-lifecycle products with margin compression signals.",
    ],
    "usage": [
        "Habit-driven products with durable repeat behavior.",
        "Funnel-friction cluster where intent leaks before purchase.",
        "Session-light products with fragile engagement depth.",
        "Recovery-sensitive cluster dependent on conversion lift.",
    ],
}

DASHBOARD_CONFIG: Dict[str, Dict[str, Any]] = {
    "reviews": {
        "scenario_feature": "review_count",
        "fields": [
            {"name": "sentiment_mean", "label": "Sentiment Mean", "type": "number", "step": 0.01, "min": -1, "max": 1, "default": 0.62},
            {"name": "sentiment_std", "label": "Sentiment Std", "type": "number", "step": 0.01, "min": 0, "max": 1.5, "default": 0.21},
            {"name": "review_count", "label": "Review Count", "type": "number", "step": 1, "min": 0, "max": 200, "default": 24},
            {"name": "score_min", "label": "Min Score", "type": "number", "step": 0.5, "min": 0, "max": 5, "default": 2},
            {"name": "score_max", "label": "Max Score", "type": "number", "step": 0.5, "min": 0, "max": 5, "default": 5},
            {"name": "score_median", "label": "Median Score", "type": "number", "step": 0.5, "min": 0, "max": 5, "default": 3.5},
            {"name": "product_age_months", "label": "Product Age", "type": "number", "step": 1, "min": 0, "max": 60, "default": 14},
            {"name": "sentiment_polarization", "label": "Polarization", "type": "number", "step": 0.1, "min": 0, "max": 5, "default": 1.2},
            {"name": "reviewer_diversity_change", "label": "Reviewer Diversity Change %", "type": "number", "step": 0.1, "min": -100, "max": 100, "default": -8},
        ],
        "samples": {
            "healthy": {
                "label": "Healthy Baseline",
                "features": {
                    "sentiment_mean": 0.82, "sentiment_std": 0.10, "review_count": 48,
                    "score_min": 4, "score_max": 5, "score_median": 5.0,
                    "product_age_months": 6, "sentiment_polarization": 0.4,
                    "reviewer_diversity_change": 12.0,
                },
            },
            "moderate": {
                "label": "Moderate Fatigue",
                "features": {
                    "sentiment_mean": 0.54, "sentiment_std": 0.24, "review_count": 21,
                    "score_min": 2, "score_max": 5, "score_median": 3.0,
                    "product_age_months": 16, "sentiment_polarization": 1.5,
                    "reviewer_diversity_change": -12.0,
                },
            },
            "high": {
                "label": "High Fatigue",
                "features": {
                    "sentiment_mean": 0.12, "sentiment_std": 0.39, "review_count": 9,
                    "score_min": 1, "score_max": 4, "score_median": 2.0,
                    "product_age_months": 28, "sentiment_polarization": 2.8,
                    "reviewer_diversity_change": -31.0,
                },
            },
            "polarized": {
                "label": "Polarized",
                "features": {
                    "sentiment_mean": 0.45, "sentiment_std": 0.85, "review_count": 65,
                    "score_min": 1, "score_max": 5, "score_median": 3.0,
                    "product_age_months": 12, "sentiment_polarization": 4.2,
                    "reviewer_diversity_change": 5.0,
                },
            },
            "recovering": {
                "label": "Recovering",
                "features": {
                    "sentiment_mean": 0.68, "sentiment_std": 0.15, "review_count": 32,
                    "score_min": 3, "score_max": 5, "score_median": 4.0,
                    "product_age_months": 24, "sentiment_polarization": 0.6,
                    "reviewer_diversity_change": 25.0,
                },
            },
        },
    },
    "sales": {
        "scenario_feature": "revenue_total",
        "fields": [
            {"name": "revenue_total", "label": "Revenue Total", "type": "number", "step": 10, "min": 0, "max": 100000, "default": 12400},
            {"name": "revenue_mean", "label": "Revenue Mean", "type": "number", "step": 1, "min": 0, "max": 10000, "default": 620},
            {"name": "revenue_std", "label": "Revenue Std", "type": "number", "step": 1, "min": 0, "max": 10000, "default": 180},
            {"name": "transaction_count", "label": "Transactions", "type": "number", "step": 1, "min": 0, "max": 500, "default": 20},
            {"name": "quantity_sold", "label": "Quantity Sold", "type": "number", "step": 1, "min": 0, "max": 10000, "default": 340},
            {"name": "avg_price", "label": "Average Price", "type": "number", "step": 0.5, "min": 0, "max": 1000, "default": 36.5},
            {"name": "product_age_months", "label": "Product Age", "type": "number", "step": 1, "min": 0, "max": 60, "default": 18},
            {"name": "order_frequency_change", "label": "Order Frequency Change %", "type": "number", "step": 0.1, "min": -100, "max": 100, "default": -9.5},
            {"name": "aov_change", "label": "AOV Change %", "type": "number", "step": 0.1, "min": -100, "max": 100, "default": -4.0},
            {"name": "customer_concentration", "label": "Customer Concentration", "type": "number", "step": 0.01, "min": 0, "max": 1, "default": 0.46},
        ],
        "samples": {
            "healthy": {
                "label": "Healthy Baseline",
                "features": {
                    "revenue_total": 18200, "revenue_mean": 910, "revenue_std": 140,
                    "transaction_count": 25, "quantity_sold": 430, "avg_price": 42.0,
                    "product_age_months": 8, "order_frequency_change": 11.0,
                    "aov_change": 4.0, "customer_concentration": 0.22,
                },
            },
            "moderate": {
                "label": "Moderate Fatigue",
                "features": {
                    "revenue_total": 9800, "revenue_mean": 520, "revenue_std": 220,
                    "transaction_count": 18, "quantity_sold": 270, "avg_price": 36.0,
                    "product_age_months": 16, "order_frequency_change": -12.0,
                    "aov_change": -5.0, "customer_concentration": 0.51,
                },
            },
            "high": {
                "label": "High Fatigue",
                "features": {
                    "revenue_total": 4200, "revenue_mean": 230, "revenue_std": 310,
                    "transaction_count": 9, "quantity_sold": 110, "avg_price": 28.0,
                    "product_age_months": 30, "order_frequency_change": -33.0,
                    "aov_change": -17.0, "customer_concentration": 0.79,
                },
            },
            "stagnant": {
                "label": "Stagnant",
                "features": {
                    "revenue_total": 8500, "revenue_mean": 450, "revenue_std": 45,
                    "transaction_count": 12, "quantity_sold": 200, "avg_price": 42.0,
                    "product_age_months": 36, "order_frequency_change": 0.5,
                    "aov_change": -1.2, "customer_concentration": 0.35,
                },
            },
            "recovering": {
                "label": "Recovering",
                "features": {
                    "revenue_total": 12000, "revenue_mean": 650, "revenue_std": 200,
                    "transaction_count": 22, "quantity_sold": 380, "avg_price": 31.0,
                    "product_age_months": 14, "order_frequency_change": 18.0,
                    "aov_change": 8.5, "customer_concentration": 0.28,
                },
            },
        },
    },
    "usage": {
        "scenario_feature": "engagement_total",
        "fields": [
            {"name": "engagement_total", "label": "Engagement Total", "type": "number", "step": 10, "min": 0, "max": 100000, "default": 1200},
            {"name": "engagement_mean", "label": "Engagement Mean", "type": "number", "step": 0.1, "min": 0, "max": 1000, "default": 12},
            {"name": "cart_count", "label": "Cart Count", "type": "number", "step": 1, "min": 0, "max": 10000, "default": 80},
            {"name": "purchase_count", "label": "Purchase Count", "type": "number", "step": 1, "min": 0, "max": 10000, "default": 20},
            {"name": "avg_price", "label": "Average Price", "type": "number", "step": 0.5, "min": 0, "max": 1000, "default": 49},
            {"name": "view_to_cart_rate", "label": "View to Cart %", "type": "number", "step": 0.1, "min": 0, "max": 100, "default": 8.0},
            {"name": "cart_to_purchase_rate", "label": "Cart to Purchase %", "type": "number", "step": 0.1, "min": 0, "max": 100, "default": 25.0},
            {"name": "conversion_rate", "label": "Conversion Rate %", "type": "number", "step": 0.1, "min": 0, "max": 100, "default": 2.0},
            {"name": "product_age_months", "label": "Product Age", "type": "number", "step": 1, "min": 0, "max": 60, "default": 8},
            {"name": "funnel_efficiency", "label": "Funnel Efficiency", "type": "number", "step": 0.01, "min": 0, "max": 1, "default": 0.25},
            {"name": "engagement_per_session", "label": "Engagement per Session", "type": "number", "step": 0.1, "min": 0, "max": 100, "default": 3.2},
            {"name": "safe_engagement_quality_change", "label": "Engagement Quality Change %", "type": "number", "step": 0.1, "min": -100, "max": 100, "default": 5.0},
        ],
        "samples": {
            "healthy": {
                "label": "Healthy Baseline",
                "features": {
                    "engagement_total": 2600, "engagement_mean": 18, "cart_count": 190,
                    "purchase_count": 61, "avg_price": 42, "view_to_cart_rate": 11.4,
                    "cart_to_purchase_rate": 32.1, "conversion_rate": 3.7,
                    "product_age_months": 5, "funnel_efficiency": 0.34,
                    "engagement_per_session": 4.8, "safe_engagement_quality_change": 12.0,
                },
            },
            "moderate": {
                "label": "Moderate Fatigue",
                "features": {
                    "engagement_total": 1200, "engagement_mean": 12, "cart_count": 80,
                    "purchase_count": 20, "avg_price": 49, "view_to_cart_rate": 8.0,
                    "cart_to_purchase_rate": 25.0, "conversion_rate": 2.0,
                    "product_age_months": 8, "funnel_efficiency": 0.25,
                    "engagement_per_session": 3.2, "safe_engagement_quality_change": 5.0,
                },
            },
            "high": {
                "label": "High Fatigue",
                "features": {
                    "engagement_total": 420, "engagement_mean": 5.5, "cart_count": 14,
                    "purchase_count": 2, "avg_price": 55, "view_to_cart_rate": 3.1,
                    "cart_to_purchase_rate": 12.0, "conversion_rate": 0.4,
                    "product_age_months": 18, "funnel_efficiency": 0.09,
                    "engagement_per_session": 1.4, "safe_engagement_quality_change": -19.0,
                },
            },
            "viral": {
                "label": "Viral Boost",
                "features": {
                    "engagement_total": 85000, "engagement_mean": 450, "cart_count": 4200,
                    "purchase_count": 850, "avg_price": 35, "view_to_cart_rate": 18.5,
                    "cart_to_purchase_rate": 42.0, "conversion_rate": 7.5,
                    "product_age_months": 2, "funnel_efficiency": 0.65,
                    "engagement_per_session": 12.4, "safe_engagement_quality_change": 85.0,
                },
            },
            "ghost": {
                "label": "Ghost Product",
                "features": {
                    "engagement_total": 50, "engagement_mean": 1.2, "cart_count": 2,
                    "purchase_count": 0, "avg_price": 120, "view_to_cart_rate": 0.5,
                    "cart_to_purchase_rate": 0.0, "conversion_rate": 0.0,
                    "product_age_months": 48, "funnel_efficiency": 0.01,
                    "engagement_per_session": 0.5, "safe_engagement_quality_change": -45.0,
                },
            },
        },
    },
}


def _dataset_path(modality: str) -> Path:
    return DATA_DIR / f"{modality}_fatigue_signals.csv"


def _lifecycle_stage_from_age(product_age_months: float) -> str:
    if product_age_months < 3:
        return "introduction"
    if product_age_months < 12:
        return "growth"
    if product_age_months < 24:
        return "maturity"
    return "decline"


def _augment_request_features(modality: str, raw_features: Dict[str, Any]) -> Dict[str, Any]:
    features = dict(raw_features)

    age = features.get("product_age_months")
    if age is not None:
        try:
            features["lifecycle_stage"] = _lifecycle_stage_from_age(float(age))
        except (TypeError, ValueError):
            pass

    if modality == "usage":
        engagement_total = features.get("engagement_total")
        unique_sessions = features.get("unique_sessions")
        if (
            engagement_total is not None
            and unique_sessions not in (None, 0)
            and "engagement_per_session" not in features
        ):
            try:
                features["engagement_per_session"] = float(engagement_total) / float(unique_sessions)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

    return features


def _build_inference_row(
    raw_features: Dict[str, Any],
    feature_names: list[str],
    scaler: Any,
    train_medians: Optional[pd.Series] = None,
) -> pd.DataFrame:
    features = dict(raw_features)
    lifecycle_stage = features.pop("lifecycle_stage", None)
    median_lookup = (
        train_medians
        if isinstance(train_medians, dict)
        else train_medians.to_dict() if train_medians is not None else {}
    )

    if lifecycle_stage:
        stage = str(lifecycle_stage).strip().lower()
        for col in feature_names:
            if col.startswith("lifecycle_stage_"):
                features[col] = 1.0 if col == f"lifecycle_stage_{stage}" else 0.0

    row_pre_scale = []
    for idx, name in enumerate(feature_names):
        if name in features:
            value = features[name]
        elif name in median_lookup:
            value = median_lookup[name]
        elif hasattr(scaler, "mean_") and idx < len(scaler.mean_):
            value = scaler.mean_[idx]
        else:
            value = 0.0
        row_pre_scale.append(float(value))

    row_df = pd.DataFrame([row_pre_scale], columns=feature_names)
    row_scaled = scaler.transform(row_df)
    return pd.DataFrame(row_scaled, columns=feature_names)


def _latest_local_mlflow_model_info(modality: str) -> Optional[Tuple[Path, str]]:
    if not LOCAL_MLFLOW_DB.exists():
        return None

    model_name = f"fatigue-{modality}-model"
    conn = sqlite3.connect(LOCAL_MLFLOW_DB)
    try:
        row = conn.execute(
            """
            SELECT version, storage_location
            FROM model_versions
            WHERE name = ?
            ORDER BY CAST(version AS INTEGER) DESC
            LIMIT 1
            """,
            (model_name,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return None

    version, storage_location = row
    prefix = "mlflow-artifacts:/"
    if not storage_location or not storage_location.startswith(prefix):
        return None

    relative_dir = storage_location[len(prefix):].strip("/")
    model_path = LOCAL_MLFLOW_ARTIFACTS_DIR / relative_dir / "model.pkl"
    if not model_path.exists():
        return None

    return model_path, f"local-mlflow-v{version}"


def _local_model_candidates(modality: str) -> list[Tuple[Path, str]]:
    candidates: list[Tuple[Path, str]] = []

    mlflow_info = _latest_local_mlflow_model_info(modality)
    if mlflow_info is not None:
        candidates.append(mlflow_info)

    for model_name in MODEL_PRIORITY:
        path = MODELS_DIR / f"{modality}_{model_name}_model.pkl"
        if path.exists():
            candidates.append((path, f"local-{model_name}"))

    return candidates


def _load_preprocessor(modality: str) -> Dict[str, Any]:
    artifacts_path = MODELS_DIR / f"{modality}_artifacts.pkl"
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts not found for {modality}: {artifacts_path}")

    artifacts = joblib.load(artifacts_path)
    return {
        "feature_names": artifacts["feature_names"],
        "scaler": artifacts["scaler"],
        "train_medians": artifacts.get("train_medians"),
        "label_classes": artifacts["label_classes"],
        "raw_required_features": artifacts.get("raw_required_features", []),
        "default_model": artifacts.get("default_model"),
        "scenario_benchmarks": artifacts.get("scenario_benchmarks", {}),
    }


def _load_matching_model(modality: str, expected_features: int) -> Tuple[Any, str]:
    for path, version in _local_model_candidates(modality):
        try:
            model = joblib.load(path)
        except Exception as exc:
            logger.warning(f"Could not load candidate model for {modality} from {path}: {exc}")
            continue

        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None and n_features != expected_features:
            logger.info(
                f"Skipping {path} for {modality}: model expects {n_features} features, "
                f"preprocessor produces {expected_features}."
            )
            continue

        return model, version

    raise RuntimeError(
        f"No compatible model found for {modality} with {expected_features} features."
    )


def _load_metrics(modality: str) -> Dict[str, Any]:
    path = OUTPUTS_DIR / f"{modality}_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _champion_model(modality: str) -> str:
    artifacts_path = MODELS_DIR / f"{modality}_artifacts.pkl"
    if artifacts_path.exists():
        try:
            artifacts = joblib.load(artifacts_path)
            default_model = artifacts.get("default_model")
            if default_model in MODEL_PRIORITY:
                return default_model
        except Exception as exc:
            logger.warning(f"Could not load default model from artifacts for {modality}: {exc}")
    metrics = _load_metrics(modality)
    classification = metrics.get("classification", {})
    if not classification:
        return "xgboost"
    return max(
        classification.items(),
        key=lambda item: item[1].get("f1_macro", 0.0),
    )[0]


def _safe_last_retrained() -> Optional[str]:
    paths = list(MODELS_DIR.glob("*_artifacts.pkl")) + list(MODELS_DIR.glob("*_model.pkl"))
    if not paths:
        return None
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    return datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc).astimezone().isoformat()


def _fatigue_score(predicted_class: str, confidence: float) -> float:
    label = predicted_class.lower()
    if "high" in label:
        score = 72 + confidence * 28
    elif "moderate" in label:
        score = 45 + confidence * 24
    else:
        score = 10 + (1 - confidence) * 18
    return round(float(max(0.0, min(100.0, score))), 2)


def _risk_band(score: float) -> str:
    if score >= 70:
        return "High Fatigue"
    if score >= 40:
        return "Moderate Fatigue"
    return "Healthy"


def _cluster_blurb(modality: str, cluster_id: Optional[int]) -> str:
    if cluster_id is None:
        return "Cluster context unavailable for this prediction."
    options = CLUSTER_CONTEXT.get(modality, ["Behavior pattern cluster."])
    return options[cluster_id % len(options)]


def _load_class_rates() -> List[Dict[str, Any]]:
    path = DATA_DIR / "fatigue_rates_comparison.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "dataset": str(row["Dataset"]).lower(),
            "healthy": float(row["Healthy"]),
            "moderate": float(row.get("Moderate Fatigue", 0.0)),
            "high": float(row.get("High Fatigue", 0.0)),
        })
    return rows


def _load_key_metrics_summary() -> List[Dict[str, Any]]:
    path = DATA_DIR / "key_metrics_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        {
            "dataset": str(row["Dataset"]).lower(),
            "dimension": row["Dimension"],
            "signal": row["Primary Signal"],
            "avg_signal_value": row["Avg Signal Value"],
            "high_fatigue_products": int(row["High Fatigue Products"]),
        }
        for _, row in df.iterrows()
    ]


def _event_markers(modality: str, _features: Dict[str, Any], months: int) -> List[Dict[str, Any]]:
    pivot_a = max(1, months // 3)
    pivot_b = max(2, (months * 2) // 3)
    if modality == "reviews":
        return [
            {"month": pivot_a, "label": "Review dip", "detail": "Review cadence softened."},
            {"month": pivot_b, "label": "Sentiment wobble", "detail": "Polarity spread widened."},
        ]
    if modality == "sales":
        return [
            {"month": pivot_a, "label": "Churn pulse", "detail": "Order frequency slipped."},
            {"month": pivot_b, "label": "Revenue stall", "detail": "AOV recovery flattened."},
        ]
    return [
        {"month": pivot_a, "label": "Funnel leak", "detail": "View-to-cart conversion cooled."},
        {"month": pivot_b, "label": "Usage softness", "detail": "Session depth lost momentum."},
    ]


def _build_trajectory(
    modality: str,
    risk_score: float,
    confidence: float,
    months: int,
    features: Dict[str, Any],
) -> Dict[str, Any]:
    start = max(8.0, min(92.0, risk_score - 18.0))
    drift = (risk_score - start) / max(1, months - 1)
    fatigue, conf = [], []
    for idx in range(months):
        wobble = math.sin(idx / 1.8) * 2.2
        fatigue.append(round(max(0.0, min(100.0, start + drift * idx + wobble)), 2))
        conf.append(round(max(0.25, min(0.99, confidence - 0.08 + (idx / max(1, months - 1)) * 0.08)), 3))
    trend_pct = round(fatigue[-1] - fatigue[-2], 2) if len(fatigue) > 1 else 0.0
    labels = [f"M-{months - i - 1}" if i < months - 1 else "Now" for i in range(months)]
    return {
        "labels": labels,
        "fatigue": fatigue,
        "confidence": conf,
        "thresholds": {"healthy": 35, "high": 70},
        "trend_vs_last_period": trend_pct,
        "events": _event_markers(modality, features, months),
    }


def _inference_completeness(features: Dict[str, Any]) -> float:
    usable = [v for v in features.values() if isinstance(v, (int, float))]
    if not usable:
        return 0.0
    non_zero = sum(1 for v in usable if abs(float(v)) > 1e-9)
    return round(non_zero / len(usable), 3)


def _natural_summary(modality: str, predicted_class: str, risk_score: float, top_features: List[str]) -> str:
    causes = ", ".join(top_features[:2]) if top_features else "the current signal mix"
    dimension = {
        "reviews": "sentiment",
        "sales": "commercial",
        "usage": "behavioral",
    }[modality]
    direction = "rising" if risk_score >= 40 else "contained"
    return (
        f"Fatigue risk is {direction}, primarily driven by {causes} across the "
        f"{dimension} layer. The current classification is {predicted_class.replace('_', ' ')}."
    )


def _build_alerts_and_actions(
    modality: str,
    result: Dict[str, Any],
    completeness: float,
) -> Tuple[List[str], List[str]]:
    alerts = list(result.get("warnings", []))
    top_features = list(result.get("shap_top5_features", {}).keys())
    if completeness < 0.7:
        alerts.append("Low data completeness detected. Confidence may be optimistic for sparse inputs.")
    if modality == "reviews" and "reviewer_diversity_change" in top_features:
        alerts.append("Reviewer diversity is a primary fatigue driver.")
    if modality == "sales" and "customer_concentration" in top_features:
        alerts.append("Revenue concentration risk is elevated.")
    if modality == "usage" and "funnel_efficiency" in top_features:
        alerts.append("Usage funnel degradation is materially impacting risk.")

    actions: List[str] = []
    if modality == "reviews":
        actions = [
            "Refresh review acquisition and reactivate high-quality reviewers.",
            "Investigate negative sentiment themes before sentiment volatility compounds.",
            "Audit lifecycle messaging for mature products with softening cadence.",
        ]
    elif modality == "sales":
        actions = [
            "Reduce customer concentration through broader acquisition campaigns.",
            "Test pricing or bundle adjustments to stabilize order frequency and AOV.",
            "Prioritize churn intervention on products showing repeat-order decay.",
        ]
    else:
        actions = [
            "Repair funnel leakage before increasing top-of-funnel traffic.",
            "Increase depth of engagement with onboarding or in-product prompts.",
            "Run conversion experiments on the highest-friction session path.",
        ]
    return alerts[:4], actions[:3]


def _scenario_features(
    features: Dict[str, Any],
    scenario_feature: Optional[str],
    scenario_delta_pct: float,
) -> Dict[str, Any]:
    if not scenario_feature or scenario_feature not in features:
        return dict(features)
    updated = dict(features)
    value = updated.get(scenario_feature)
    if isinstance(value, (int, float)):
        updated[scenario_feature] = float(value) * (1.0 + scenario_delta_pct / 100.0)
    return updated


def _compose_dashboard_prediction(
    modality: str,
    features: Dict[str, Any],
    time_range_months: int,
    compare_features: Optional[Dict[str, Any]] = None,
    scenario_feature: Optional[str] = None,
    scenario_delta_pct: float = 0.0,
    product_name: Optional[str] = None,
) -> Dict[str, Any]:
    from src.predict import predict as rich_predict

    model_name = _champion_model(modality)
    enriched_features = _augment_request_features(modality, features)
    started = time.perf_counter()
    result = rich_predict(modality, enriched_features, model_name=model_name, strict=True)
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

    confidence = float(result["confidence"])
    predicted_class = result["predicted_class"]
    risk_score = _fatigue_score(predicted_class, confidence)
    band = _risk_band(risk_score)
    metrics = _load_metrics(modality).get("classification", {}).get(model_name, {})
    completeness = float(result.get("completeness", _inference_completeness(enriched_features)))
    trajectory = _build_trajectory(modality, risk_score, confidence, time_range_months, enriched_features)
    top_features = list(result.get("shap_top5_features", {}).keys())
    alerts, actions = _build_alerts_and_actions(modality, result, completeness)

    response: Dict[str, Any] = {
        "product_name": product_name or "Manual product",
        "modality": modality,
        "modality_label": MODALITY_LABELS[modality],
        "model_name": model_name,
        "model_version": loaded_model_versions.get(modality, f"local-{model_name}"),
        "last_retrained": _safe_last_retrained(),
        "prediction": result,
        "risk_score": risk_score,
        "risk_band": band,
        "trajectory": trajectory,
        "trend_vs_last_period": trajectory["trend_vs_last_period"],
        "cluster_context": _cluster_blurb(modality, result.get("cluster_id")),
        "natural_summary": _natural_summary(modality, predicted_class, risk_score, top_features),
        "alerts": alerts,
        "recommended_actions": actions,
        "completeness": completeness,
        "model_health": {
            "f1_macro": metrics.get("f1_macro"),
            "roc_auc_ovr_macro": metrics.get("roc_auc_ovr_macro"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "macro_recall": metrics.get("recall_macro"),
            "cv_test_gap": metrics.get("cv_test_gap"),
            "prediction_distribution_drift_l1": metrics.get("prediction_distribution_drift_l1"),
            "scenario_benchmark_score": metrics.get("scenario_benchmark_score"),
            "raw_brier_score": metrics.get("raw_brier_score"),
            "raw_ece": metrics.get("raw_ece"),
            "calibrated_brier_score": metrics.get("calibrated_brier_score"),
            "calibrated_ece": metrics.get("calibrated_ece"),
            "leakage_warning": metrics.get("leakage_warning"),
            "api_latency_ms": latency_ms,
        },
    }

    if compare_features:
        compare_result = rich_predict(
            modality,
            _augment_request_features(modality, compare_features),
            model_name=model_name,
            strict=True,
        )
        compare_score = _fatigue_score(compare_result["predicted_class"], float(compare_result["confidence"]))
        response["compare"] = {
            "prediction": compare_result,
            "risk_score": compare_score,
            "risk_band": _risk_band(compare_score),
            "delta_vs_primary": round(compare_score - risk_score, 2),
        }

    if scenario_feature:
        scenario_input = _scenario_features(enriched_features, scenario_feature, scenario_delta_pct)
        scenario_result = rich_predict(modality, scenario_input, model_name=model_name, strict=True)
        scenario_score = _fatigue_score(scenario_result["predicted_class"], float(scenario_result["confidence"]))
        response["scenario"] = {
            "feature": scenario_feature,
            "delta_pct": scenario_delta_pct,
            "prediction": scenario_result,
            "risk_score": scenario_score,
            "delta_vs_primary": round(scenario_score - risk_score, 2),
        }

    return response


def _dashboard_context() -> Dict[str, Any]:
    modality_cards = []
    for modality in TARGET_MODALITIES:
        metrics = _load_metrics(modality)
        champion = _champion_model(modality)
        champ_metrics = metrics.get("classification", {}).get(champion, {})
        modality_cards.append({
            "modality": modality,
            "label": MODALITY_LABELS[modality],
            "champion_model": champion,
            "model_version": loaded_model_versions.get(modality, f"local-{champion}"),
            "f1_macro": champ_metrics.get("f1_macro"),
            "roc_auc": champ_metrics.get("roc_auc_ovr_macro"),
            "accuracy": champ_metrics.get("accuracy"),
            "balanced_accuracy": champ_metrics.get("balanced_accuracy"),
            "scenario_benchmark_score": champ_metrics.get("scenario_benchmark_score"),
            "calibrated_ece": champ_metrics.get("calibrated_ece"),
            "cluster_metrics": metrics.get("clustering", {}),
        })
    return {
        "modalities": DASHBOARD_CONFIG,
        "modality_cards": modality_cards,
        "fatigue_rates": _load_class_rates(),
        "key_metrics": _load_key_metrics_summary(),
        "last_retrained": _safe_last_retrained(),
        "model_versions": loaded_model_versions,
        "api_status": {
            "loaded_models": list(loaded_models.keys()),
            "model_versions": loaded_model_versions,
        },
    }


@app.on_event("startup")
def load_models() -> None:
    """Load preprocessors and compatible models for all modalities."""
    loaded_models.clear()
    loaded_preprocessors.clear()
    loaded_model_versions.clear()

    for modality in TARGET_MODALITIES:
        try:
            preprocessor = _load_preprocessor(modality)
            model, version = _load_matching_model(
                modality, expected_features=len(preprocessor["feature_names"])
            )
            loaded_preprocessors[modality] = preprocessor
            loaded_models[modality] = model
            loaded_model_versions[modality] = version
            logger.info(f"Loaded {modality} model ({version}).")
        except Exception as exc:
            logger.warning(f"Could not initialize {modality}: {exc}")


def _predict(modality: str, features_dict: dict) -> PredictionResponse:
    if modality not in loaded_models or modality not in loaded_preprocessors:
        raise HTTPException(
            status_code=503,
            detail=f"Model for '{modality}' is not loaded or unavailable.",
        )

    try:
        from src.predict import predict as core_predict
        result = core_predict(
            modality,
            features_dict,
            model_name=_champion_model(modality),
            strict=True,
        )
        
        return PredictionResponse(
            modality=modality,
            fatigue_status=result["predicted_class"],
            probability=result["confidence"],
            model_version=result.get("model_version", loaded_model_versions.get(modality, "1.1.0"))
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Inference error for {modality}: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/predict/reviews", response_model=PredictionResponse)
def predict_reviews(req: ReviewFeatures) -> PredictionResponse:
    return _predict("reviews", req.model_dump())


@app.post("/predict/sales", response_model=PredictionResponse)
def predict_sales(req: SalesFeatures) -> PredictionResponse:
    return _predict("sales", req.model_dump())


@app.post("/predict/usage", response_model=PredictionResponse)
def predict_usage(req: UsageFeatures) -> PredictionResponse:
    return _predict("usage", req.model_dump())


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(
        "dashboard.html",
        {"request": request, "title": "Product Fatigue Intelligence"},
    )


@app.get("/dashboard/api/context")
def dashboard_context() -> Dict[str, Any]:
    return _dashboard_context()


@app.post("/dashboard/api/predict/{modality}")
def dashboard_predict(modality: str, req: DashboardPredictRequest) -> Dict[str, Any]:
    if modality not in TARGET_MODALITIES:
        raise HTTPException(status_code=404, detail=f"Unknown modality '{modality}'.")
    try:
        return _compose_dashboard_prediction(
            modality=modality,
            features=req.features,
            compare_features=req.compare_features,
            time_range_months=req.time_range_months,
            scenario_feature=req.scenario_feature,
            scenario_delta_pct=req.scenario_delta_pct,
            product_name=req.product_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Dashboard inference error for {modality}: {exc}")
        raise HTTPException(status_code=500, detail=f"Dashboard prediction failed: {exc}")


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "up",
        "loaded_models": list(loaded_models.keys()),
        "model_versions": loaded_model_versions,
    }


@app.post("/predict")
def predict_generic(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic predict endpoint.  Expects JSON with 'modality' and 'features' keys.
    Delegates to the modality-specific prediction logic.
    """
    modality = req.get("modality")
    features = req.get("features")
    if modality not in TARGET_MODALITIES:
        raise HTTPException(status_code=422, detail=f"Invalid modality '{modality}'. Must be one of {TARGET_MODALITIES}.")
    if not features or not isinstance(features, dict):
        raise HTTPException(status_code=422, detail="'features' must be a non-empty dict.")
    return _predict(modality, features).model_dump()


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    """Return model metadata for every loaded modality and fusion model."""
    info: Dict[str, Any] = {}
    for modality in TARGET_MODALITIES:
        champion = _champion_model(modality)
        metrics = _load_metrics(modality)
        champ_metrics = metrics.get("classification", {}).get(champion, {})
        clustering = metrics.get("clustering", {})

        # Enrich with forward-label config from artifacts
        art_path = MODELS_DIR / f"{modality}_artifacts.pkl"
        forward_config: Dict[str, Any] = {}
        if art_path.exists():
            try:
                art = joblib.load(art_path)
                forward_config = {
                    "use_forward_labels": art.get("use_forward_labels", False),
                    "forward_horizon": art.get("forward_horizon"),
                    "uncertainty_threshold": art.get("uncertainty_threshold"),
                    "default_model": art.get("default_model", champion),
                }
            except Exception:
                pass

        info[modality] = {
            "champion_model": champion,
            "model_version": loaded_model_versions.get(modality, f"local-{champion}"),
            "last_retrained": _safe_last_retrained(),
            "n_features": len(loaded_preprocessors.get(modality, {}).get("feature_names", [])),
            "label_classes": list(loaded_preprocessors.get(modality, {}).get("label_classes", [])),
            "f1_macro": champ_metrics.get("f1_macro"),
            "roc_auc_ovr_macro": champ_metrics.get("roc_auc_ovr_macro"),
            "accuracy": champ_metrics.get("accuracy"),
            "balanced_accuracy": champ_metrics.get("balanced_accuracy"),
            "cv_test_gap": champ_metrics.get("cv_test_gap"),
            "leakage_warning": champ_metrics.get("leakage_warning"),
            "calibrated_ece": champ_metrics.get("calibrated_ece"),
            "clustering": clustering,
            **forward_config,
        }

    # Fusion model info
    fusion_manifest = MODELS_DIR / "fusion" / "feature_manifest.json"
    if fusion_manifest.exists():
        try:
            import json as _json
            manifest = _json.loads(fusion_manifest.read_text())
            info["fusion"] = {
                "fusion_type": manifest.get("fusion_type"),
                "fusion_cv_f1": manifest.get("fusion_cv_f1"),
                "branch_modalities": manifest.get("branch_modalities", []),
                "n_fusion_features": len(manifest.get("fusion_feature_names", [])),
            }
        except Exception:
            pass

    return {
        "models": info,
        "loaded_modalities": list(loaded_models.keys()),
        "model_versions": loaded_model_versions,
    }


@app.get("/metrics/models")
def get_metrics() -> Dict[str, Any]:
    """Return evaluation metrics for all modalities."""
    result: Dict[str, Any] = {}
    for modality in TARGET_MODALITIES:
        result[modality] = _load_metrics(modality)
    return result


@app.get("/pipeline/status")
def pipeline_status() -> Dict[str, Any]:
    """Return the operational status of the ML pipeline and all models."""
    modality_status: Dict[str, Any] = {}
    forward_labels = False
    walk_forward = False

    for modality in TARGET_MODALITIES:
        artifacts_path = MODELS_DIR / f"{modality}_artifacts.pkl"
        has_artifacts = artifacts_path.exists()
        has_model = modality in loaded_models
        metrics = _load_metrics(modality)
        classification = metrics.get("classification", {})
        champion = _champion_model(modality)
        champ_metrics = classification.get(champion, {})

        # Check forward-label config
        if has_artifacts and not forward_labels:
            try:
                art = joblib.load(artifacts_path)
                forward_labels = art.get("use_forward_labels", False)
                walk_forward = art.get("use_walk_forward", False)
            except Exception:
                pass

        modality_status[modality] = {
            "status": "ready" if has_model and has_artifacts else "unavailable",
            "artifacts_present": has_artifacts,
            "model_loaded": has_model,
            "champion_model": champion,
            "model_version": loaded_model_versions.get(modality),
            "f1_macro": champ_metrics.get("f1_macro"),
            "leakage_warning": champ_metrics.get("leakage_warning"),
            "available_classifiers": list(classification.keys()),
        }

    fusion_ready = (MODELS_DIR / "fusion" / "champion.pkl").exists()
    all_ready = all(v["status"] == "ready" for v in modality_status.values())
    degraded_warnings = []
    for m, s in modality_status.items():
        if s["status"] != "ready":
            degraded_warnings.append(f"{m} branch model not available")
    if not fusion_ready:
        degraded_warnings.append("Fusion model not available — run main.py with ENABLE_FUSION=True")

    return {
        "pipeline_status": "operational" if all_ready else "degraded",
        "modalities": modality_status,
        "fusion_ready": fusion_ready,
        "forward_labels": forward_labels,
        "walk_forward_validation": walk_forward,
        "degraded_warnings": degraded_warnings,
        "last_retrained": _safe_last_retrained(),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
    }


@app.post("/predict/fusion", response_model=FusionPredictionResponse)
async def predict_fusion(request: FusionPredictionRequest):
    """
    Unified multimodal fusion prediction endpoint.

    Accepts features for any combination of branches, runs branch inference,
    applies the fusion model, and returns the final fatigue prediction with
    uncertainty flags and branch-level details.
    """
    fusion_model_path = MODELS_DIR / "fusion" / "champion.pkl"
    if not fusion_model_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Fusion model not available. Run main.py with ENABLE_FUSION=True."
        )

    try:
        fusion_model = joblib.load(fusion_model_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load fusion model: {exc}")

    # Run branch predictions
    branch_predictions = {}
    branch_probas = {}
    branch_features_map = {
        "reviews": request.reviews_features,
        "sales": request.sales_features,
        "usage": request.usage_features,
    }

    for modality, features in branch_features_map.items():
        if features is None:
            continue

        try:
            # Import predict module
            import sys
            sys.path.insert(0, str(ROOT_DIR))
            from src.predict import predict as run_prediction

            result = run_prediction(modality, features)
            pred_class = result["predicted_class"]
            all_probs = result["all_probabilities"]

            branch_predictions[modality] = BranchPrediction(
                fatigue_class=pred_class,
                healthy=all_probs.get("healthy", 0.0),
                moderate=all_probs.get("moderate_fatigue", 0.0),
                high=all_probs.get("high_fatigue", 0.0),
            )

            # Collect probabilities for fusion
            branch_probas[modality] = np.array([
                all_probs.get("healthy", 0.0),
                all_probs.get("moderate_fatigue", 0.0),
                all_probs.get("high_fatigue", 0.0),
            ])

        except Exception as exc:
            logger.warning(f"Branch prediction failed for {modality}: {exc}")

    if not branch_probas:
        raise HTTPException(
            status_code=400,
            detail="No valid branch predictions. Provide features for at least one modality."
        )

    # Build fusion input
    parts = []
    for modality in sorted(branch_probas.keys()):
        parts.append(branch_probas[modality])
    X_fusion = np.array([np.concatenate(parts)])

    # Run fusion prediction
    try:
        fusion_pred = fusion_model.predict(X_fusion)[0]
        fusion_proba = fusion_model.predict_proba(X_fusion)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fusion prediction failed: {exc}")

    # Map prediction to class name
    class_names = ["healthy", "moderate_fatigue", "high_fatigue"]
    fused_class = class_names[fusion_pred] if fusion_pred < len(class_names) else "unknown"
    fused_prob = float(fusion_proba.max())

    # Confidence band and uncertainty
    if fused_prob >= 0.80:
        confidence_band = "high"
    elif fused_prob >= 0.60:
        confidence_band = "medium"
    else:
        confidence_band = "low"

    # Margin between top-2 classes
    sorted_probs = np.sort(fusion_proba)[::-1]
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    uncertainty_flag = fused_prob < 0.60 or margin < 0.15

    # Recommended action
    if fused_class == "high_fatigue" and not uncertainty_flag:
        recommended_action = "urgent intervention"
    elif fused_class == "moderate_fatigue":
        recommended_action = "proactive monitoring"
    elif uncertainty_flag:
        recommended_action = "manual review needed"
    else:
        recommended_action = "monitor"

    # Driver summary
    drivers = []
    if "usage" in branch_predictions:
        bp = branch_predictions["usage"]
        if bp.high > 0.5:
            drivers.append("strong behavioral fatigue signal")
    if "reviews" in branch_predictions:
        bp = branch_predictions["reviews"]
        if bp.high > 0.3 or bp.moderate > 0.5:
            drivers.append("emotional deterioration detected")
    if "sales" in branch_predictions:
        bp = branch_predictions["sales"]
        if bp.high > 0.3 or bp.moderate > 0.5:
            drivers.append("commercial performance declining")
    driver_summary = "; ".join(drivers) if drivers else "No dominant fatigue driver"

    # Fatigue index
    fi = 0.0
    weights = {"reviews": 0.3, "sales": 0.3, "usage": 0.4}
    for modality, proba in branch_probas.items():
        fi += weights.get(modality, 0.33) * proba[2]  # P(high)

    return FusionPredictionResponse(
        product_id=request.product_id,
        history_window=request.history_window,
        prediction_horizon=request.prediction_horizon,
        fatigue_class=fused_class,
        fused_probability=round(fused_prob, 4),
        confidence_band=confidence_band,
        uncertainty_flag=uncertainty_flag,
        branch_predictions=branch_predictions,
        top_contributors=[],
        driver_summary=driver_summary,
        recommended_action=recommended_action,
        model_versions={m: "v2.0" for m in branch_probas},
        fatigue_index=round(float(fi), 4),
    )
