import logging
import pandas as pd
from prefect import flow, task
from typing import Optional

logger = logging.getLogger("prefect_pipeline")

@task(retries=2, retry_delay_seconds=30)
def extract_usage_data(raw_data_path: str) -> pd.DataFrame:
    """Extract raw behavioral data, applying initial filtering logic."""
    logger.info(f"Extracting raw data from {raw_data_path}")
    # Simulate loading the 10% sample
    df = pd.read_csv(raw_data_path)
    return df

@task
def engineer_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Implement the behavioral feature engineering logic for fatigue."""
    logger.info("Computing session depth, conversion funnel, and quality metrics...")
    
    # In a full migration, the notebook's logic is extracted into src/features/
    # For this template, we simulate the output generation process.
    if "event_type" in df.columns:
        # Example processing:
        # df['is_purchase'] = (df['event_type'] == 'purchase').astype(int)
        pass 
        
    logger.info("Feature engineering complete.")
    return df

@task
def persist_feature_layer(df: pd.DataFrame, output_path: str):
    """Save the engineered dataset to standard unified feature format (Parquet)."""
    logger.info(f"Saving feature layer to {output_path}")
    df.to_parquet(output_path, index=False)

@flow(name="Usage Modality Pipeline", description="ETL for Behavioral Fatigue dataset")
def dataset_usage_pipeline(
    raw_path: str = "data/raw/behavior_combined_sampled_10pct.csv",
    output_path: str = "data/features/usage_features.parquet"
):
    """
    DAG execution flow for taking raw event logs and producing 
    the behavioral fatigue feature store dataset.
    """
    try:
        raw_df = extract_usage_data(raw_path)
        feature_df = engineer_behavioral_features(raw_df)
        persist_feature_layer(feature_df, output_path)
        
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}")
        raise

if __name__ == "__main__":
    dataset_usage_pipeline()
