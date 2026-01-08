#!/usr/bin/env python3
"""
Credit Risk Model - End-to-End Simulation Test
Demonstrates the full MLOps pipeline with real predictions and monitoring
"""

import requests
import pandas as pd
import numpy as np
import joblib
import time
import random

# Configuration
KSERVE_URL = "http://localhost:8081/v1/models/credit-risk-model:predict"
SERVICE_HOSTNAME = "credit-risk.local"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.txt"

def load_artifacts():
    """Load model artifacts"""
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    return scaler, feature_names

def engineer_features(data, feature_names):
    """Apply feature engineering transformations"""
    df = pd.DataFrame([data])
    
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # Feature Engineering
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

def generate_normal_customer():
    """Generate a typical customer profile"""
    return {
        'LIMIT_BAL': random.choice([20000, 50000, 100000, 200000, 500000]),
        'SEX': random.choice([1, 2]),
        'EDUCATION': random.choice([1, 2, 3, 4]),
        'MARRIAGE': random.choice([1, 2, 3]),
        'AGE': random.randint(21, 65),
        'PAY_0': random.choice([0, 0, 0, 0, -1, -1, 1, 2]),
        'PAY_2': random.choice([0, 0, 0, -1, -1, 1]),
        'PAY_3': random.choice([0, 0, 0, -1, -1]),
        'PAY_4': random.choice([0, 0, -1, -1]),
        'PAY_5': random.choice([0, 0, -1]),
        'PAY_6': random.choice([0, 0, -1]),
        'BILL_AMT1': random.randint(0, 50000),
        'BILL_AMT2': random.randint(0, 50000),
        'BILL_AMT3': random.randint(0, 50000),
        'BILL_AMT4': random.randint(0, 50000),
        'BILL_AMT5': random.randint(0, 50000),
        'BILL_AMT6': random.randint(0, 50000),
        'PAY_AMT1': random.randint(0, 10000),
        'PAY_AMT2': random.randint(0, 10000),
        'PAY_AMT3': random.randint(0, 10000),
        'PAY_AMT4': random.randint(0, 10000),
        'PAY_AMT5': random.randint(0, 10000),
        'PAY_AMT6': random.randint(0, 10000),
    }

def generate_risky_customer():
    """Generate a high-risk customer profile (likely to default)"""
    return {
        'LIMIT_BAL': random.choice([10000, 20000, 30000]),  # Low limit
        'SEX': random.choice([1, 2]),
        'EDUCATION': random.choice([3, 4]),  # Lower education
        'MARRIAGE': random.choice([1, 2, 3]),
        'AGE': random.randint(21, 35),  # Younger
        'PAY_0': random.choice([2, 3, 4, 5]),  # Severely late
        'PAY_2': random.choice([2, 3, 4]),
        'PAY_3': random.choice([1, 2, 3]),
        'PAY_4': random.choice([1, 2]),
        'PAY_5': random.choice([0, 1, 2]),
        'PAY_6': random.choice([0, 1]),
        'BILL_AMT1': random.randint(5000, 30000),  # High bills
        'BILL_AMT2': random.randint(5000, 30000),
        'BILL_AMT3': random.randint(4000, 25000),
        'BILL_AMT4': random.randint(3000, 20000),
        'BILL_AMT5': random.randint(2000, 15000),
        'BILL_AMT6': random.randint(1000, 10000),
        'PAY_AMT1': random.randint(0, 500),  # Very low payments
        'PAY_AMT2': random.randint(0, 500),
        'PAY_AMT3': random.randint(0, 500),
        'PAY_AMT4': random.randint(0, 500),
        'PAY_AMT5': random.randint(0, 500),
        'PAY_AMT6': random.randint(0, 500),
    }

def send_prediction(customer, scaler, feature_names):
    """Send prediction request to KServe"""
    features_df = engineer_features(customer, feature_names)
    features_scaled = scaler.transform(features_df)
    
    payload = {"instances": features_scaled.tolist()}
    headers = {"Host": SERVICE_HOSTNAME}
    
    try:
        response = requests.post(KSERVE_URL, json=payload, headers=headers, timeout=5)
        if response.status_code == 200:
            result = response.json()
            return result["predictions"][0]
    except Exception as e:
        pass
    return None

def run_simulation(duration_seconds=60, requests_per_second=2):
    """Run the simulation test"""
    print("=" * 60)
    print("🚀 Credit Risk Model - End-to-End Simulation Test")
    print("=" * 60)
    
    scaler, feature_names = load_artifacts()
    print(f"✅ Loaded model artifacts")
    print(f"📊 Running simulation for {duration_seconds} seconds...")
    print(f"⚡ Target: {requests_per_second} requests/second")
    print("-" * 60)
    
    start_time = time.time()
    total_requests = 0
    successful = 0
    high_risk_predictions = 0
    low_risk_predictions = 0
    
    interval = 1.0 / requests_per_second
    
    while (time.time() - start_time) < duration_seconds:
        # Randomly choose customer type (80% normal, 20% risky)
        if random.random() < 0.8:
            customer = generate_normal_customer()
            customer_type = "Normal"
        else:
            customer = generate_risky_customer()
            customer_type = "Risky"
        
        prediction = send_prediction(customer, scaler, feature_names)
        total_requests += 1
        
        if prediction is not None:
            successful += 1
            risk_level = "HIGH" if prediction > 0.5 else "LOW"
            
            if prediction > 0.5:
                high_risk_predictions += 1
            else:
                low_risk_predictions += 1
            
            # Progress indicator every 10 requests
            if total_requests % 10 == 0:
                elapsed = time.time() - start_time
                rate = successful / elapsed
                print(f"[{elapsed:5.1f}s] Requests: {total_requests} | "
                      f"Success: {successful} | "
                      f"Rate: {rate:.1f}/s | "
                      f"High Risk: {high_risk_predictions} | "
                      f"Low Risk: {low_risk_predictions}")
        
        time.sleep(interval)
    
    # Final Summary
    elapsed = time.time() - start_time
    print("-" * 60)
    print("📈 SIMULATION RESULTS")
    print("-" * 60)
    print(f"Duration:        {elapsed:.1f} seconds")
    print(f"Total Requests:  {total_requests}")
    print(f"Successful:      {successful} ({100*successful/total_requests:.1f}%)")
    print(f"Avg Rate:        {successful/elapsed:.2f} req/s")
    print(f"High Risk:       {high_risk_predictions} ({100*high_risk_predictions/successful:.1f}%)")
    print(f"Low Risk:        {low_risk_predictions} ({100*low_risk_predictions/successful:.1f}%)")
    print("=" * 60)
    print("✅ Simulation complete! Check Grafana dashboard for metrics.")

if __name__ == "__main__":
    import sys
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rate = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run_simulation(duration_seconds=duration, requests_per_second=rate)
