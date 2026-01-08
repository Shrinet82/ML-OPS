
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Gauge, CollectorRegistry
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from typing import List, Dict, Any
import uvicorn
import os

# Configuration
REFERENCE_DATA_PATH = "data/credit_card_default.csv"
WINDOW_SIZE = 50  # Calculate drift every 50 requests

app = FastAPI()

# Prometheus Registry
registry = CollectorRegistry()
metrics_app = make_asgi_app(registry=registry)
app.mount("/metrics", metrics_app)

# Metrics
DRIFT_SCORE = Gauge('credit_risk_drift_score', 'Data Drift Score', registry=registry)
DRIFT_DETECTED = Gauge('credit_risk_drift_detected', 'Data Drift Detected (1=Yes, 0=No)', registry=registry)

# Global Store
reference_data = None
current_window = []

# Feature columns (must match training data)
FEATURE_COLS = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

class PredictionLog(BaseModel):
    instances: List[List[float]]

@app.on_event("startup")
def load_reference_data():
    global reference_data
    print("Loading reference data...")
    try:
        if os.path.exists(REFERENCE_DATA_PATH):
            df = pd.read_csv(REFERENCE_DATA_PATH)
            # Ensure we only use feature columns
            reference_data = df[FEATURE_COLS].sample(n=1000, random_state=42) # Sample for performance
            print(f"Reference data loaded: {len(reference_data)} rows")
        else:
            print("Running in container without local data? Generate dummy reference for demo.")
            reference_data = pd.DataFrame(np.random.rand(1000, 23), columns=FEATURE_COLS)
    except Exception as e:
        print(f"Error loading reference data: {e}")
        # Fallback
        reference_data = pd.DataFrame(np.random.rand(1000, 23), columns=FEATURE_COLS)

def calculate_drift(current_data: pd.DataFrame):
    print("Calculating drift...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Extract metrics
    # Evidently JSON storage is complex, simple extraction:
    results = report.as_dict()
    drift_share = results['metrics'][0]['result']['drift_share']
    drift_detected = results['metrics'][0]['result']['dataset_drift']
    
    print(f"Drift Share: {drift_share}, Detected: {drift_detected}")
    
    DRIFT_SCORE.set(drift_share)
    DRIFT_DETECTED.set(1 if drift_detected else 0)

@app.post("/log")
def log_prediction(log: PredictionLog, background_tasks: BackgroundTasks):
    global current_window
    
    # Append new data
    # Assuming instances are in order of FEATURE_COLS
    for instance in log.instances:
        current_window.append(instance)
    
    # Check window size
    if len(current_window) >= WINDOW_SIZE:
        print(f"Window full ({len(current_window)}), triggering drift calculation")
        # Create DataFrame from window
        current_df = pd.DataFrame(current_window, columns=FEATURE_COLS)
        
        # Calculate drift in background
        background_tasks.add_task(calculate_drift, current_df)
        
        # Reset window (sliding or tumbling? Tumbling for simplicity)
        current_window = []
        
    return {"status": "logged", "window_size": len(current_window)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
