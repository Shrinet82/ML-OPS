#!/usr/bin/env python3
"""
FastAPI Credit Risk Prediction Service
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
import os

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predict credit card default risk using XGBoost ensemble model",
    version="1.0.0"
)

# Load model and scaler
MODEL_PATH = "models/xgboost_baseline.json"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.txt"

model = None
scaler = None
feature_names = None

@app.on_event("startup")
async def load_model():
    global model, scaler, feature_names
    
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    scaler = joblib.load(SCALER_PATH)
    
    with open(FEATURES_PATH, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"✅ Model loaded with {len(feature_names)} features")


class CreditInput(BaseModel):
    """Input schema for credit risk prediction"""
    LIMIT_BAL: float = Field(..., description="Credit limit")
    SEX: int = Field(..., description="Gender (1=male, 2=female)")
    EDUCATION: int = Field(..., description="Education (1=graduate, 2=university, 3=high school, 4=others)")
    MARRIAGE: int = Field(..., description="Marital status (1=married, 2=single, 3=others)")
    AGE: int = Field(..., description="Age in years")
    PAY_0: int = Field(..., description="Repayment status in Sep (-1=paid, 1=delay 1mo, 2=delay 2mo...)")
    PAY_2: int = Field(..., description="Repayment status in Aug")
    PAY_3: int = Field(..., description="Repayment status in Jul")
    PAY_4: int = Field(..., description="Repayment status in Jun")
    PAY_5: int = Field(..., description="Repayment status in May")
    PAY_6: int = Field(..., description="Repayment status in Apr")
    BILL_AMT1: float = Field(..., description="Bill amount in Sep")
    BILL_AMT2: float = Field(..., description="Bill amount in Aug")
    BILL_AMT3: float = Field(..., description="Bill amount in Jul")
    BILL_AMT4: float = Field(..., description="Bill amount in Jun")
    BILL_AMT5: float = Field(..., description="Bill amount in May")
    BILL_AMT6: float = Field(..., description="Bill amount in Apr")
    PAY_AMT1: float = Field(..., description="Payment amount in Sep")
    PAY_AMT2: float = Field(..., description="Payment amount in Aug")
    PAY_AMT3: float = Field(..., description="Payment amount in Jul")
    PAY_AMT4: float = Field(..., description="Payment amount in Jun")
    PAY_AMT5: float = Field(..., description="Payment amount in May")
    PAY_AMT6: float = Field(..., description="Payment amount in Apr")


class PredictionResponse(BaseModel):
    """Output schema for prediction"""
    default_probability: float
    prediction: str
    risk_level: str
    confidence: float


def engineer_features(data: dict) -> pd.DataFrame:
    """Apply same feature engineering as training"""
    df = pd.DataFrame([data])
    
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # Payment behavior
    df['LATE_PAYMENTS'] = (df[pay_cols] > 0).sum(axis=1)
    df['MAX_DELAY'] = df[pay_cols].max(axis=1)
    df['AVG_DELAY'] = df[pay_cols].mean(axis=1)
    df['SEVERE_DELAY'] = (df[pay_cols] >= 2).sum(axis=1)
    df['EVER_2MONTH_LATE'] = (df[pay_cols] >= 2).any(axis=1).astype(int)
    df['RECENT_DELAY_WEIGHTED'] = df['PAY_0'] * 3 + df['PAY_2'] * 2 + df['PAY_3']
    
    # Bill & Payment aggregates
    df['AVG_BILL_AMT'] = df[bill_cols].mean(axis=1)
    df['AVG_PAY_AMT'] = df[amt_cols].mean(axis=1)
    df['TOTAL_BILL'] = df[bill_cols].sum(axis=1)
    df['TOTAL_PAY'] = df[amt_cols].sum(axis=1)
    
    # Utilization ratios
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
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df[feature_names]


@app.get("/")
async def root():
    return {"message": "Credit Risk Prediction API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: CreditInput):
    """Predict credit default probability"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to dict and engineer features
        data = input_data.model_dump()
        features_df = engineer_features(data)
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Predict
        probability = float(model.predict_proba(features_scaled)[0, 1])
        prediction = "DEFAULT" if probability >= 0.5 else "NO DEFAULT"
        
        # Risk level
        if probability < 0.2:
            risk_level = "LOW"
        elif probability < 0.5:
            risk_level = "MEDIUM"
        elif probability < 0.7:
            risk_level = "HIGH"
        else:
            risk_level = "VERY HIGH"
        
        # Confidence
        confidence = abs(probability - 0.5) * 2
        
        return PredictionResponse(
            default_probability=round(probability, 4),
            prediction=prediction,
            risk_level=risk_level,
            confidence=round(confidence, 4)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get model information"""
    metrics_path = "models/metrics_summary.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}
    
    return {
        "model_type": "XGBoost + LightGBM Ensemble",
        "features": len(feature_names) if feature_names else 0,
        "metrics": metrics
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
