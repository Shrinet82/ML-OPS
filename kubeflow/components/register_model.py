"""
Kubeflow Pipeline Components for Credit Risk Model
Component 5: Register Model - Register model in MLflow registry
"""

from kfp import dsl
from kfp.dsl import Dataset, Output, Input, Model, Metrics, Artifact


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["mlflow==2.9.2", "xgboost==2.0.3", "pandas==2.0.3", "scikit-learn==1.3.2"]
)
def register_model(
    model_artifact: Input[Model],
    scaler_artifact: Input[Model],
    validation_report: Input[Artifact],
    model_name: str,
    stage: str,
    registered_model: Output[Artifact],
    metrics: Output[Metrics]
):
    """Register validated model in MLflow registry."""
    import mlflow
    import mlflow.xgboost
    import xgboost as xgb
    import joblib
    import json
    import os
    
    # Set MLflow tracking URI (use environment variable or default)
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Load validation report
    with open(validation_report.path, 'r') as f:
        validation_result = json.load(f)
    
    if not validation_result.get("passed", False):
        print("❌ Model validation failed - skipping registration")
        metrics.log_metric("registered", 0)
        return
    
    # Load model
    model = xgb.XGBClassifier()
    model.load_model(model_artifact.path)
    
    # Set experiment
    mlflow.set_experiment("credit-risk-pipeline")
    
    with mlflow.start_run(run_name=f"pipeline-{model_name}"):
        # Log model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Log validation metrics
        mlflow.log_metric("auc", validation_result["auc"])
        mlflow.log_metric("validation_passed", int(validation_result["passed"]))
        
        # Log scaler
        mlflow.log_artifact(scaler_artifact.path, artifact_path="preprocessing")
        
        # Log validation report
        mlflow.log_artifact(validation_report.path, artifact_path="validation")
        
        run_id = mlflow.active_run().info.run_id
    
    # Create registration info
    registration_info = {
        "model_name": model_name,
        "run_id": run_id,
        "stage": stage,
        "auc": validation_result["auc"],
        "mlflow_uri": mlflow_uri
    }
    
    with open(registered_model.path, 'w') as f:
        json.dump(registration_info, f, indent=2)
    
    # Log metrics
    metrics.log_metric("registered", 1)
    metrics.log_metric("auc", float(validation_result["auc"]))
    
    print(f"✅ Model registered: {model_name} (run_id: {run_id})")
    print(f"   Stage: {stage}")
    print(f"   AUC: {validation_result['auc']:.4f}")
