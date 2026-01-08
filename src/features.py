
import pandas as pd
import numpy as np

def engineer_features(df):
    """Apply feature engineering transformations."""
    # Ensure columns exist (for robustness)
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # Validation: Check if columns exist
    missing_cols = [c for c in pay_cols + bill_cols + amt_cols + ['LIMIT_BAL', 'AGE', 'EDUCATION', 'MARRIAGE'] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

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
