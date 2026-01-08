#!/usr/bin/env python3
"""
Credit Risk Ensemble Model - XGBoost + LightGBM
Target: AUC >= 0.82
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('notebooks/figures', exist_ok=True)

print("="*60)
print("CREDIT RISK ENSEMBLE MODEL")
print("="*60)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n📊 Loading dataset...")
df = pd.read_csv('data/credit_card_default.csv')

column_names = {
    'X1': 'LIMIT_BAL', 'X2': 'SEX', 'X3': 'EDUCATION', 'X4': 'MARRIAGE', 'X5': 'AGE',
    'X6': 'PAY_0', 'X7': 'PAY_2', 'X8': 'PAY_3', 'X9': 'PAY_4', 'X10': 'PAY_5', 'X11': 'PAY_6',
    'X12': 'BILL_AMT1', 'X13': 'BILL_AMT2', 'X14': 'BILL_AMT3', 'X15': 'BILL_AMT4', 'X16': 'BILL_AMT5', 'X17': 'BILL_AMT6',
    'X18': 'PAY_AMT1', 'X19': 'PAY_AMT2', 'X20': 'PAY_AMT3', 'X21': 'PAY_AMT4', 'X22': 'PAY_AMT5', 'X23': 'PAY_AMT6',
    'Y': 'DEFAULT'
}
df = df.rename(columns=column_names)

# =============================================================================
# 2. ENHANCED FEATURE ENGINEERING
# =============================================================================
print("\n🔧 Enhanced Feature Engineering...")

pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

# Payment behavior - MOST PREDICTIVE
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

# Interaction features
df['LIMIT_AGE'] = df['LIMIT_BAL'] / df['AGE']
df['DELAY_UTIL'] = df['AVG_DELAY'] * df['AVG_UTILIZATION']

# Categorical encoding improvements
df['HIGH_EDUCATION'] = (df['EDUCATION'] <= 2).astype(int)  # Graduate or above
df['SINGLE'] = (df['MARRIAGE'] == 2).astype(int)

print(f"  - Added 22 engineered features")
print(f"  - Final shape: {df.shape}")

# Handle infinity and NaN values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

# =============================================================================
# 3. PREPARE DATA
# =============================================================================
print("\n🎯 Preparing data...")

feature_cols = [col for col in df.columns if col != 'DEFAULT']
X = df[feature_cols]
y = df['DEFAULT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  - Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.pkl')

# =============================================================================
# 4. TRAIN MODELS WITH CROSS-VALIDATION
# =============================================================================
print("\n🚀 Training Ensemble with 5-Fold CV...")

mlflow.set_experiment("credit-risk-baseline")

with mlflow.start_run(run_name="ensemble-xgb-lgb"):
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    # XGBoost params
    xgb_params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'learning_rate': 0.03,
        'n_estimators': 500,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,
        'gamma': 0.2,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
    }
    
    # LightGBM params
    lgb_params = {
        'objective': 'binary',
        'max_depth': 4,
        'learning_rate': 0.03,
        'n_estimators': 500,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_samples': 50,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbosity': -1,
    }
    
    mlflow.log_params({"model_type": "ensemble", "xgb_lr": 0.03, "lgb_lr": 0.03})
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train)):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
        
        # Ensemble average
        ensemble_pred = (xgb_pred + lgb_pred) / 2
        fold_auc = roc_auc_score(y_val, ensemble_pred)
        cv_scores.append(fold_auc)
        print(f"  Fold {fold+1}: AUC = {fold_auc:.4f}")
    
    print(f"\n📊 CV Mean AUC: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
    
    # Train final models on full training data
    print("\n🎯 Training final models...")
    
    xgb_final = xgb.XGBClassifier(**xgb_params)
    xgb_final.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
    
    lgb_final = lgb.LGBMClassifier(**lgb_params)
    lgb_final.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)])
    
    # Ensemble predictions
    xgb_pred = xgb_final.predict_proba(X_test_scaled)[:, 1]
    lgb_pred = lgb_final.predict_proba(X_test_scaled)[:, 1]
    y_pred_proba = (xgb_pred + lgb_pred) / 2
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    xgb_auc = roc_auc_score(y_test, xgb_pred)
    lgb_auc = roc_auc_score(y_test, lgb_pred)
    
    metrics = {
        'auc': auc,
        'xgb_auc': xgb_auc,
        'lgb_auc': lgb_auc,
        'cv_mean_auc': np.mean(cv_scores),
        'cv_std_auc': np.std(cv_scores)
    }
    mlflow.log_metrics(metrics)
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"  XGBoost AUC:  {xgb_auc:.4f}")
    print(f"  LightGBM AUC: {lgb_auc:.4f}")
    print(f"  ✅ ENSEMBLE AUC: {auc:.4f}")
    print(f"  CV Mean AUC:  {np.mean(cv_scores):.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    
    # Save models
    xgb_final.save_model('models/xgboost_baseline.json')
    joblib.dump(lgb_final, 'models/lightgbm_baseline.pkl')
    
    # Plots
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_pred)
    fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_pred)
    
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Ensemble (AUC = {auc:.4f})')
    plt.plot(fpr_xgb, tpr_xgb, 'g--', linewidth=1.5, label=f'XGBoost (AUC = {xgb_auc:.4f})')
    plt.plot(fpr_lgb, tpr_lgb, 'r--', linewidth=1.5, label=f'LightGBM (AUC = {lgb_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Ensemble Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('notebooks/figures/roc_curve.png', dpi=150)
    mlflow.log_artifact('notebooks/figures/roc_curve.png')
    
    run_id = mlflow.active_run().info.run_id
    print(f"\n📝 MLflow Run ID: {run_id}")

# Save summary
with open('models/metrics_summary.json', 'w') as f:
    json.dump(metrics, f, indent=2)

with open('models/feature_names.txt', 'w') as f:
    for col in feature_cols:
        f.write(f"{col}\n")

df.to_csv('data/credit_card_processed.csv', index=False)

print("\n" + "="*60)
if auc >= 0.82:
    print(f"✅ TARGET ACHIEVED! AUC: {auc:.4f} >= 0.82")
else:
    print(f"⚠️  Current AUC: {auc:.4f} | Target: 0.82 | Gap: {0.82 - auc:.4f}")
print("="*60)
