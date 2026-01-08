
import requests
import json
import numpy as np
import pandas as pd
import joblib

# Configuration
INGRESS_HOST = "localhost"  # Use localhost with port forward or ingress IP
INGRESS_PORT = 8081         # Port forward port
SERVICE_HOSTNAME = "credit-risk.local"
MODEL_NAME = "credit-risk-model"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.txt"

def load_artifacts():
    print("Loading scaler...")
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    return scaler, feature_names

def engineer_features(data, feature_names):
    df = pd.DataFrame([data])
    
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # Feature Engineering (Same as training)
    df['LATE_PAYMENTS'] = (df[pay_cols] > 0).sum(axis=1)
    df['MAX_DELAY'] = df[pay_cols].max(axis=1)
    df['AVG_DELAY'] = df[pay_cols].mean(axis=1)
    df['SEVERE_DELAY'] = (df[pay_cols] >= 2).sum(axis=1)
    df['EVER_2MONTH_LATE'] = (df[pay_cols] >= 2).any(axis=1).astype(int)
    df['RECENT_DELAY_WEIGHTED'] = df['PAY_0'] * 3 + df['PAY_2'] * 2 + df['PAY_3']
    
    df['AVG_BILL_AMT'] = df[bill_cols].mean(axis=1)
    df['AVG_PAY_AMT'] = df[amt_cols].mean(axis=1)
    df['TOTAL_BILL'] = df[bill_cols].sum(axis=1)
    df['TOTAL_PAY'] = df[amt_cols].sum(axis=1)
    
    df['UTILIZATION'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
    df['AVG_UTILIZATION'] = df['AVG_BILL_AMT'] / (df['LIMIT_BAL'] + 1)
    df['PAY_RATIO'] = df['TOTAL_PAY'] / (df['TOTAL_BILL'] + 1)
    df['RECENT_PAY_RATIO'] = df['PAY_AMT1'] / (df['BILL_AMT1'] + 1)
    
    df['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
    df['PAY_TREND'] = df['PAY_AMT1'] - df['PAY_AMT6']
    df['INCREASING_DEBT'] = (df['BILL_TREND'] > 0).astype(int)
    
    df['LIMIT_AGE'] = df['LIMIT_BAL'] / df['AGE']
    df['DELAY_UTIL'] = df['AVG_DELAY'] * df['AVG_UTILIZATION']
    
    df['HIGH_EDUCATION'] = (df['EDUCATION'] <= 2).astype(int)
    df['SINGLE'] = (df['MARRIAGE'] == 2).astype(int)
    
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df[feature_names]

def test_prediction():
    scaler, feature_names = load_artifacts()
    
    # Sample input
    sample = {
        'LIMIT_BAL': 50000, 'SEX': 1, 'EDUCATION': 2, 'MARRIAGE': 1, 'AGE': 35,
        'PAY_0': 0, 'PAY_2': 0, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
        'BILL_AMT1': 20000, 'BILL_AMT2': 19000, 'BILL_AMT3': 18000,
        'BILL_AMT4': 17000, 'BILL_AMT5': 16000, 'BILL_AMT6': 15000,
        'PAY_AMT1': 2000, 'PAY_AMT2': 1800, 'PAY_AMT3': 1600,
        'PAY_AMT4': 1400, 'PAY_AMT5': 1200, 'PAY_AMT6': 1000
    }
    
    # Preprocess
    print("Preprocessing input...")
    features_df = engineer_features(sample, feature_names)
    features_scaled = scaler.transform(features_df)
    
    # Prepare request (V1 format)
    payload = {
        "instances": features_scaled.tolist()
    }
    
    url = f"http://{INGRESS_HOST}:{INGRESS_PORT}/v1/models/{MODEL_NAME}:predict"
    headers = {"Host": SERVICE_HOSTNAME}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction success!")
            
            # Log to Monitoring Service
            try:
                monitor_url = "http://localhost:8002/log"
                print(f"Logging to {monitor_url}...")
                requests.post(monitor_url, json=payload)
                print("✅ Logged to monitoring service")
            except Exception as e:
                print(f"⚠️ Failed to log to monitor: {e}")
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")

            
    except Exception as e:
        print(f"❌ Connection failed: {e}")

    print("Running 60 verification requests to trigger drift detection...")
    for i in range(60):
        try:
            requests.post(url, json=payload, headers=headers)
            # Log to Monitor
            requests.post("http://localhost:8002/log", json=payload)
            if i % 10 == 0:
                print(f"Request {i}/60 sent")
        except:
            pass
    print("Traffic generation complete.")

if __name__ == "__main__":
    test_prediction()
