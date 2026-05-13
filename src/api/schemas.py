from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class ReviewFeatures(BaseModel):
    sentiment_mean: float = Field(..., description="Mean sentiment score (-1.0 to 1.0)")
    sentiment_std: float = Field(..., ge=0.0, description="Std dev of sentiment scores")
    review_count: float = Field(..., ge=0.0, description="Review count for the period")
    score_min: float = Field(..., ge=0.0, description="Minimum review score")
    score_max: float = Field(..., ge=0.0, description="Maximum review score")
    score_median: float = Field(..., ge=0.0, description="Median review score")
    product_age_months: int = Field(..., ge=0, description="Age of product in months since first review")
    sentiment_polarization: float = Field(..., ge=0.0, description="Sentiment polarization/spread feature")
    reviewer_diversity_change: float = Field(..., description="Change in reviewer diversity")

class SalesFeatures(BaseModel):
    revenue_total: float = Field(..., ge=0.0, description="Total revenue for the period")
    revenue_mean: float = Field(..., ge=0.0, description="Average revenue per transaction window")
    revenue_std: float = Field(..., ge=0.0, description="Std dev of revenue")
    transaction_count: float = Field(..., ge=0.0, description="Transaction count for the period")
    quantity_sold: float = Field(..., ge=0.0, description="Units sold")
    avg_price: float = Field(..., ge=0.0, description="Average selling price")
    product_age_months: int = Field(..., ge=0)
    order_frequency_change: float = Field(..., description="% change in orders per customer")
    aov_change: float = Field(..., description="% change in average order value")
    customer_concentration: float = Field(..., ge=0.0, description="Customer concentration metric")

class UsageFeatures(BaseModel):
    engagement_total: float = Field(..., ge=0.0)
    engagement_mean: float = Field(..., ge=0.0)
    cart_count: float = Field(..., ge=0.0)
    purchase_count: float = Field(..., ge=0.0)
    avg_price: float = Field(..., ge=0.0)
    view_to_cart_rate: float = Field(..., ge=0.0, le=100.0)
    cart_to_purchase_rate: float = Field(..., ge=0.0, le=100.0)
    conversion_rate: float = Field(..., ge=0.0, le=100.0)
    product_age_months: int = Field(..., ge=0)
    funnel_efficiency: float = Field(..., ge=0.0)
    engagement_per_session: float = Field(..., ge=0.0)
    safe_engagement_quality_change: float = Field(..., description="Shift(1) based % change in Engagement per session")

class PredictionResponse(BaseModel):
    modality: str = Field(..., description="The dataset dimension used")
    fatigue_status: str = Field(..., description="Predicted fatigue class label")
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence/probability of the prediction")
    model_version: Optional[str] = Field(None, description="MLflow registry version used")


class BranchPrediction(BaseModel):
    fatigue_class: str = Field(..., description="Predicted fatigue class for this branch")
    healthy: float = Field(0.0, description="P(healthy)")
    moderate: float = Field(0.0, description="P(moderate_fatigue)")
    high: float = Field(0.0, description="P(high_fatigue)")


class FusionPredictionRequest(BaseModel):
    reviews_features: Optional[Dict[str, Any]] = Field(None, description="Reviews branch features")
    sales_features: Optional[Dict[str, Any]] = Field(None, description="Sales branch features")
    usage_features: Optional[Dict[str, Any]] = Field(None, description="Usage branch features")
    product_id: Optional[str] = Field(None, description="Product identifier")
    history_window: Optional[str] = Field(None, description="Historical window description")
    prediction_horizon: Optional[str] = Field(None, description="Prediction horizon description")


class FusionPredictionResponse(BaseModel):
    product_id: Optional[str] = None
    history_window: Optional[str] = None
    prediction_horizon: Optional[str] = None
    fatigue_class: str = Field(..., description="Final fused fatigue class")
    fused_probability: float = Field(..., description="Fused calibrated probability")
    confidence_band: str = Field(..., description="high / medium / low")
    uncertainty_flag: bool = Field(False, description="True if prediction is uncertain")
    branch_predictions: Dict[str, BranchPrediction] = Field(
        default_factory=dict, description="Per-branch predictions"
    )
    top_contributors: list = Field(default_factory=list, description="Top contributing features")
    driver_summary: str = Field("", description="Short description of what drives the prediction")
    recommended_action: str = Field("monitor", description="Suggested action level")
    model_versions: Dict[str, str] = Field(default_factory=dict, description="Model version metadata")
    fatigue_index: Optional[float] = Field(None, description="Canonical fatigue index (0-1)")


class PipelineStatusResponse(BaseModel):
    reviews_ready: bool = False
    sales_ready: bool = False
    usage_ready: bool = False
    fusion_ready: bool = False
    forward_labels: bool = False
    walk_forward_validation: bool = False
    degraded_warnings: list = Field(default_factory=list)


class DashboardPredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Raw feature payload for the selected modality")
    compare_features: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional second payload for compare mode",
    )
    time_range_months: int = Field(
        default=12,
        ge=6,
        le=24,
        description="How many months to render in the synthetic trajectory",
    )
    scenario_feature: Optional[str] = Field(
        default=None,
        description="Feature key to perturb for scenario simulation",
    )
    scenario_delta_pct: float = Field(
        default=0.0,
        ge=-50.0,
        le=50.0,
        description="Percentage delta to apply to scenario_feature",
    )
    product_name: Optional[str] = Field(
        default=None,
        description="Friendly product label for dashboard copy",
    )
