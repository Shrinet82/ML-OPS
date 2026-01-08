"""
Kubeflow Pipeline Components for Credit Risk Model
Component 3: Train Model - Train XGBoost model
"""

from kfp import dsl
from kfp.dsl import Dataset, Output, Input, Model, Metrics


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.2", "xgboost==2.0.3", "joblib==1.3.2"]
)
def train_model(
    train_features: Input[Dataset],
    test_features: Input[Dataset],
    max_depth: int,
    learning_rate: float,
    n_estimators: int,
    model_artifact: Output[Model],
    metrics: Output[Metrics]
):
    """Train XGBoost model on the engineered features."""
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    import joblib
    
    # Load data
    train_df = pd.read_csv(train_features.path)
    test_df = pd.read_csv(test_features.path)
    
    X_train = train_df.drop('DEFAULT', axis=1)
    y_train = train_df['DEFAULT']
    X_test = test_df.drop('DEFAULT', axis=1)
    y_test = test_df['DEFAULT']
    
    # Calculate class weight
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    # Train model
    print(f"Training XGBoost: depth={max_depth}, lr={learning_rate}, n_est={n_estimators}")
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log metrics
    metrics.log_metric("auc", float(auc))
    metrics.log_metric("accuracy", float(accuracy))
    metrics.log_metric("precision", float(precision))
    metrics.log_metric("recall", float(recall))
    metrics.log_metric("f1", float(f1))
    
    # Save model
    model.save_model(model_artifact.path)
    
    print(f"✅ Model trained: AUC={auc:.4f}, Accuracy={accuracy:.4f}")
