"""
Kubeflow Pipeline Components for Credit Risk Model
Component 2: Feature Engineer - Apply feature engineering transformations
"""

from kfp import dsl
from kfp.dsl import Dataset, Output, Input, Model, Metrics


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.2", "numpy==1.24.3", "joblib==1.3.2"]
)
def feature_engineer(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    train_features: Output[Dataset],
    test_features: Output[Dataset],
    scaler_artifact: Output[Model],
    metrics: Output[Metrics]
):
    """Apply feature engineering transformations to the dataset."""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    # Load data
    train_df = pd.read_csv(train_data.path)
    test_df = pd.read_csv(test_data.path)
    
    def engineer_features(df):
        """Apply feature engineering transformations."""
        pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
        amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        
        # Payment behavior features
        df['LATE_PAYMENTS'] = (df[pay_cols] > 0).sum(axis=1)
        df['MAX_DELAY'] = df[pay_cols].max(axis=1)
        df['AVG_DELAY'] = df[pay_cols].mean(axis=1)
        df['SEVERE_DELAY'] = (df[pay_cols] >= 2).sum(axis=1)
        df['EVER_2MONTH_LATE'] = (df[pay_cols] >= 2).any(axis=1).astype(int)
        df['RECENT_DELAY_WEIGHTED'] = df['PAY_0'] * 3 + df['PAY_2'] * 2 + df['PAY_3']
        
        # Aggregates
        df['AVG_BILL_AMT'] = df[bill_cols].mean(axis=1)
        df['AVG_PAY_AMT'] = df[amt_cols].mean(axis=1)
        df['TOTAL_BILL'] = df[bill_cols].sum(axis=1)
        df['TOTAL_PAY'] = df[amt_cols].sum(axis=1)
        
        # Ratios
        df['UTILIZATION'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
        df['AVG_UTILIZATION'] = df['AVG_BILL_AMT'] / (df['LIMIT_BAL'] + 1)
        df['PAY_RATIO'] = df['TOTAL_PAY'] / (df['TOTAL_BILL'] + 1)
        df['RECENT_PAY_RATIO'] = df['PAY_AMT1'] / (df['BILL_AMT1'] + 1)
        
        # Trends
        df['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
        df['PAY_TREND'] = df['PAY_AMT1'] - df['PAY_AMT6']
        df['INCREASING_DEBT'] = (df['BILL_TREND'] > 0).astype(int)
        
        # Interactions
        df['LIMIT_AGE'] = df['LIMIT_BAL'] / df['AGE']
        df['DELAY_UTIL'] = df['AVG_DELAY'] * df['AVG_UTILIZATION']
        
        # Categorical
        df['HIGH_EDUCATION'] = (df['EDUCATION'] <= 2).astype(int)
        df['SINGLE'] = (df['MARRIAGE'] == 2).astype(int)
        
        # Handle infinity
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df
    
    # Apply feature engineering
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # Separate features and target
    feature_cols = [col for col in train_df.columns if col != 'DEFAULT']
    
    X_train = train_df[feature_cols]
    y_train = train_df['DEFAULT']
    X_test = test_df[feature_cols]
    y_test = test_df['DEFAULT']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create output DataFrames
    train_out = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_out['DEFAULT'] = y_train.values
    
    test_out = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_out['DEFAULT'] = y_test.values
    
    # Save outputs
    train_out.to_csv(train_features.path, index=False)
    test_out.to_csv(test_features.path, index=False)
    joblib.dump(scaler, scaler_artifact.path)
    
    # Log metrics
    metrics.log_metric("num_features", len(feature_cols))
    metrics.log_metric("engineered_features", 22)
    
    print(f"✅ Feature engineering complete: {len(feature_cols)} features")
