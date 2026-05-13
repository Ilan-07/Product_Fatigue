"""
src/api/v1.py
=============
V1 API routes — mirrors the original single-modality prediction endpoints.
These are backward-compatible and use the legacy per-branch prediction flow.
"""
from fastapi import APIRouter, HTTPException
from typing import Any, Dict

from .schemas import PredictionResponse, ReviewFeatures, SalesFeatures, UsageFeatures

router = APIRouter(prefix="/v1", tags=["v1"])


def _get_predict_fn():
    """Lazy import to avoid circular dependency at module load."""
    from .main import _predict
    return _predict


@router.post("/predict/reviews", response_model=PredictionResponse)
def v1_predict_reviews(req: ReviewFeatures) -> PredictionResponse:
    return _get_predict_fn()("reviews", req.model_dump())


@router.post("/predict/sales", response_model=PredictionResponse)
def v1_predict_sales(req: SalesFeatures) -> PredictionResponse:
    return _get_predict_fn()("sales", req.model_dump())


@router.post("/predict/usage", response_model=PredictionResponse)
def v1_predict_usage(req: UsageFeatures) -> PredictionResponse:
    return _get_predict_fn()("usage", req.model_dump())


@router.get("/health")
def v1_health() -> Dict[str, Any]:
    from .main import loaded_models, loaded_model_versions
    return {
        "status": "up",
        "api_version": "v1",
        "loaded_models": list(loaded_models.keys()),
        "model_versions": loaded_model_versions,
    }
