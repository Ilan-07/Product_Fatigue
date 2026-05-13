import os
import logging
import pandas as pd
import mlflow
from prefect import flow, task
from src.models.train_model import train_xgb_model

logger = logging.getLogger("training_pipeline")

@task
def load_features(modality: str) -> pd.DataFrame:
    """Load the standardized feature set from the DVC-tracked feature layer."""
    path = f"data/features/{modality}_features.parquet"
    if not os.path.exists(path):
        # Fallback to the existing processed CSVs for this mock/migration
        path = f"data/processed/{modality}_fatigue_signals.csv"
        return pd.read_csv(path, low_memory=False)
    return pd.read_parquet(path)


@task
def train_and_register_modality(modality: str, df: pd.DataFrame):
    """
    Subflow logic:
    1. Preprocess (split, drop leakage, scale, OHE).
    2. Train XGBoost.
    3. Log to MLflow.
    """
    from src.data_loader import load_modality
    
    logger.info(f"Running ML pipeline for {modality}...")
    
    # For a real pipeline we'd decouple data_loader.load_modality from the CSV
    # but we'll simulate the load mechanism here.
    with mlflow.start_run(run_name=f"train_{modality}_xgboost"):
        # Log model metadata
        mlflow.log_param("modality", modality)
        mlflow.log_param("architecture", "XGBoost")
        
        # 1. Feature Engineering (Simulated)
        # 2. Train Model
        clf, f1_macro = train_xgb_model(df, modality)
        
        # 3. Log results and Register Model
        mlflow.log_metric("f1_macro", f1_macro)
        
        mlflow.sklearn.log_model(
            sk_model=clf,
            name=f"model_{modality}",
            registered_model_name=f"fatigue-{modality}-model"
        )
        logger.info(f"[{modality}] F1 Macro: {f1_macro:.4f}. Model registered.")

@flow(name="Multi-Dataset Master Training Flow")
def master_training_pipeline():
    """
    Orchestrates the unified MLflow training process natively in Prefect.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("Product_Fatigue_Production_V2")
    
    for modality in ["reviews", "sales", "usage"]:
        try:
            df = load_features(modality)
            train_and_register_modality(modality, df)
        except Exception as exc:
            logger.error(f"Failed processing {modality}: {exc}")

if __name__ == "__main__":
    master_training_pipeline()
