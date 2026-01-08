
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from features import engineer_features

def test_feature_engineering_structure():
    """Test if feature engineering returns expected columns"""
    # Create dummy data
    data = pd.DataFrame({
        'LIMIT_BAL': [50000], 'SEX': [1], 'EDUCATION': [2], 'MARRIAGE': [1], 'AGE': [35],
        'PAY_0': [0], 'PAY_2': [0], 'PAY_3': [0], 'PAY_4': [0], 'PAY_5': [0], 'PAY_6': [0],
        'BILL_AMT1': [2000], 'BILL_AMT2': [1900], 'BILL_AMT3': [1800],
        'BILL_AMT4': [1700], 'BILL_AMT5': [1600], 'BILL_AMT6': [1500],
        'PAY_AMT1': [200], 'PAY_AMT2': [180], 'PAY_AMT3': [160],
        'PAY_AMT4': [140], 'PAY_AMT5': [120], 'PAY_AMT6': [100],
        'default.payment.next.month': [0]
    })
    
    # Run engineering
    X = engineer_features(data)
    
    # Checks
    assert X is not None
    feature_names = X.columns.tolist()
    assert len(feature_names) > 0
    assert 'UTILIZATION' in feature_names
    assert 'PAY_RATIO' in feature_names

def test_feature_engineering_outliers():
    """Test handling of division by zero (infinity)"""
    data = pd.DataFrame({
        'LIMIT_BAL': [0], 'SEX': [1], 'EDUCATION': [2], 'MARRIAGE': [1], 'AGE': [35],
        'PAY_0': [0], 'PAY_2': [0], 'PAY_3': [0], 'PAY_4': [0], 'PAY_5': [0], 'PAY_6': [0],
        'BILL_AMT1': [2000], 'BILL_AMT2': [1900], 'BILL_AMT3': [1800],
        'BILL_AMT4': [1700], 'BILL_AMT5': [1600], 'BILL_AMT6': [1500],
        'PAY_AMT1': [0], 'PAY_AMT2': [0], 'PAY_AMT3': [0],
        'PAY_AMT4': [0], 'PAY_AMT5': [0], 'PAY_AMT6': [0],
        'default.payment.next.month': [0]
    })
    
    X = engineer_features(data)
    
    # Check for NaNs or Infs in processed array
    assert not X.isnull().values.any()
    assert not np.isinf(X.values).any()

if __name__ == "__main__":
    pytest.main([__file__])
