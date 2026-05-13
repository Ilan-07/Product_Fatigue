import logging

import pandas as pd
from prefect import flow, task

logger = logging.getLogger("prefect_pipeline")

@task(retries=2, retry_delay_seconds=30)
def extract_sales_data(raw_data_path: str) -> pd.DataFrame:
    logger.info(f"Extracting raw data from {raw_data_path} (note: supports excel/csv)")
    df = pd.read_csv(raw_data_path) # Assumes CSV is created from excel in ingest
    return df

@task
def engineer_commercial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute revenue velocity, customer churn rate, etc."""
    logger.info("Computing revenue trajectory metrics...")
    
    # Notebook logic migrates here
    # df['customer_churn_rate'] = ...
    
    return df

@task
def persist_feature_layer(df: pd.DataFrame, output_path: str):
    logger.info(f"Saving feature layer to {output_path}")
    df.to_parquet(output_path, index=False)

@flow(name="Sales Modality Pipeline", description="ETL for Commercial Fatigue dataset")
def dataset_sales_pipeline(
    raw_path: str = "data/raw/online_retail.csv",
    output_path: str = "data/features/sales_features.parquet"
):
    try:
        raw_df = extract_sales_data(raw_path)
        feature_df = engineer_commercial_features(raw_df)
        persist_feature_layer(feature_df, output_path)
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}")
        raise

if __name__ == "__main__":
    dataset_sales_pipeline()
