"""
Kubeflow Pipeline Components for Credit Risk Model
Component 1: Data Loader - Load and split the dataset
"""

from kfp import dsl
from kfp.dsl import Dataset, Output, Input, Metrics


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.2"]
)
def data_loader(
    dataset_url: str,
    test_size: float,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    metrics: Output[Metrics]
):
    """Load credit card dataset and split into train/test sets."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load dataset
    print(f"Loading dataset from {dataset_url}...")
    df = pd.read_csv(dataset_url)
    
    # Rename columns
    column_names = {
        'X1': 'LIMIT_BAL', 'X2': 'SEX', 'X3': 'EDUCATION', 'X4': 'MARRIAGE', 'X5': 'AGE',
        'X6': 'PAY_0', 'X7': 'PAY_2', 'X8': 'PAY_3', 'X9': 'PAY_4', 'X10': 'PAY_5', 'X11': 'PAY_6',
        'X12': 'BILL_AMT1', 'X13': 'BILL_AMT2', 'X14': 'BILL_AMT3', 'X15': 'BILL_AMT4', 
        'X16': 'BILL_AMT5', 'X17': 'BILL_AMT6',
        'X18': 'PAY_AMT1', 'X19': 'PAY_AMT2', 'X20': 'PAY_AMT3', 'X21': 'PAY_AMT4', 
        'X22': 'PAY_AMT5', 'X23': 'PAY_AMT6',
        'Y': 'DEFAULT'
    }
    df = df.rename(columns=column_names)
    
    # Split data
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['DEFAULT']
    )
    
    # Save outputs
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    
    # Log metrics
    metrics.log_metric("total_samples", len(df))
    metrics.log_metric("train_samples", len(train_df))
    metrics.log_metric("test_samples", len(test_df))
    metrics.log_metric("default_rate", float(df['DEFAULT'].mean()))
    
    print(f"✅ Data loaded: {len(train_df)} train, {len(test_df)} test samples")
