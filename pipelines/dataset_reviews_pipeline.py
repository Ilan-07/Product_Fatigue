import logging
import pandas as pd
from prefect import flow, task

logger = logging.getLogger("prefect_pipeline")

@task(retries=2, retry_delay_seconds=30)
def extract_reviews_data(raw_data_path: str) -> pd.DataFrame:
    """Extract raw Amazon Reviews dataset."""
    logger.info(f"Extracting raw data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    return df

@task
def engineer_emotional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling sentiment, velocity, and review momentum."""
    logger.info("Computing sentiment velocity and volatility metrics...")
    
    # Notebook logic migrates here
    # df['sentiment_velocity'] = ...
    
    return df

@task
def persist_feature_layer(df: pd.DataFrame, output_path: str):
    logger.info(f"Saving feature layer to {output_path}")
    df.to_parquet(output_path, index=False)

@flow(name="Reviews Modality Pipeline", description="ETL for Emotional Fatigue dataset")
def dataset_reviews_pipeline(
    raw_path: str = "data/raw/amazon_reviews.csv",
    output_path: str = "data/features/reviews_features.parquet"
):
    try:
        raw_df = extract_reviews_data(raw_path)
        feature_df = engineer_emotional_features(raw_df)
        persist_feature_layer(feature_df, output_path)
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}")
        raise

if __name__ == "__main__":
    dataset_reviews_pipeline()
