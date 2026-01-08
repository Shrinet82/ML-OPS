"""
Credit Risk MLOps Pipeline - Kubeflow Pipeline Definition
5-Stage DAG: Load → Engineer → Train → Validate → Register
"""

from kfp import dsl
from kfp import compiler

# Import components
from components.data_loader import data_loader
from components.feature_engineer import feature_engineer
from components.train_model import train_model
from components.validate_model import validate_model
from components.register_model import register_model


@dsl.pipeline(
    name="credit-risk-pipeline",
    description="End-to-end MLOps pipeline for credit risk scoring model"
)
def credit_risk_pipeline(
    dataset_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
    test_size: float = 0.2,
    max_depth: int = 4,
    learning_rate: float = 0.03,
    n_estimators: int = 500,
    auc_threshold: float = 0.75,
    model_name: str = "credit-risk-model",
    stage: str = "staging"
):
    """
    Credit Risk Scoring Pipeline
    
    Args:
        dataset_url: URL or path to the credit card dataset
        test_size: Fraction of data for testing
        max_depth: XGBoost max tree depth
        learning_rate: XGBoost learning rate
        n_estimators: Number of boosting rounds
        auc_threshold: Minimum AUC required for model validation
        model_name: Name for model registration
        stage: MLflow model stage (staging/production)
    """
    
    # Step 1: Load and split data
    load_task = data_loader(
        dataset_url=dataset_url,
        test_size=test_size
    )
    load_task.set_display_name("1. Load Data")
    
    # Step 2: Feature engineering
    feature_task = feature_engineer(
        train_data=load_task.outputs["train_data"],
        test_data=load_task.outputs["test_data"]
    )
    feature_task.set_display_name("2. Feature Engineering")
    
    # Step 3: Train model
    train_task = train_model(
        train_features=feature_task.outputs["train_features"],
        test_features=feature_task.outputs["test_features"],
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators
    )
    train_task.set_display_name("3. Train XGBoost")
    
    # Step 4: Validate model
    validate_task = validate_model(
        model_artifact=train_task.outputs["model_artifact"],
        test_features=feature_task.outputs["test_features"],
        auc_threshold=auc_threshold
    )
    validate_task.set_display_name("4. Validate Model")
    
    # Step 5: Register model (only if validation passed)
    register_task = register_model(
        model_artifact=train_task.outputs["model_artifact"],
        scaler_artifact=feature_task.outputs["scaler_artifact"],
        validation_report=validate_task.outputs["validation_report"],
        model_name=model_name,
        stage=stage
    )
    register_task.set_display_name("5. Register Model")


if __name__ == "__main__":
    # Compile pipeline to YAML
    compiler.Compiler().compile(
        pipeline_func=credit_risk_pipeline,
        package_path="pipeline.yaml"
    )
    print("✅ Pipeline compiled to pipeline.yaml")
