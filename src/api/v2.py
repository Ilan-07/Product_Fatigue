"""
src/api/v2.py
=============
V2 API routes — uses the new fusion + forward-label architecture.
Includes fusion prediction, pipeline status, and enhanced model info.
"""
from fastapi import APIRouter, HTTPException
from typing import Any, Dict

from .schemas import (
    PredictionResponse,
    ReviewFeatures,
    SalesFeatures,
    UsageFeatures,
    FusionPredictionRequest,
    FusionPredictionResponse,
)

router = APIRouter(prefix="/v2", tags=["v2"])


def _get_predict_fn():
    from .main import _predict
    return _predict


@router.post("/predict/reviews", response_model=PredictionResponse)
def v2_predict_reviews(req: ReviewFeatures) -> PredictionResponse:
    return _get_predict_fn()("reviews", req.model_dump())


@router.post("/predict/sales", response_model=PredictionResponse)
def v2_predict_sales(req: SalesFeatures) -> PredictionResponse:
    return _get_predict_fn()("sales", req.model_dump())


@router.post("/predict/usage", response_model=PredictionResponse)
def v2_predict_usage(req: UsageFeatures) -> PredictionResponse:
    return _get_predict_fn()("usage", req.model_dump())


@router.post("/predict/fusion", response_model=FusionPredictionResponse)
def v2_predict_fusion(request: FusionPredictionRequest) -> FusionPredictionResponse:
    """V2 fusion prediction — delegates to the main fusion endpoint logic."""
    from .main import predict_fusion as _fusion
    import asyncio
    # The main endpoint is async; call it directly since FastAPI handles both
    return asyncio.get_event_loop().run_until_complete(_fusion(request))


@router.get("/pipeline/status")
def v2_pipeline_status() -> Dict[str, Any]:
    from .main import pipeline_status as _status
    return _status()


@router.get("/model/info")
def v2_model_info() -> Dict[str, Any]:
    from .main import model_info as _info
    return _info()


@router.get("/health")
def v2_health() -> Dict[str, Any]:
    from .main import loaded_models, loaded_model_versions, MODELS_DIR
    from pathlib import Path
    return {
        "status": "up",
        "api_version": "v2",
        "loaded_models": list(loaded_models.keys()),
        "model_versions": loaded_model_versions,
        "fusion_available": (MODELS_DIR / "fusion" / "champion.pkl").exists(),
    }
