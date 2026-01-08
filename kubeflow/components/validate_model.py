"""
Kubeflow Pipeline Components for Credit Risk Model
Component 4: Validate Model - Validate model performance and generate SHAP explanations
"""

from kfp import dsl
from kfp.dsl import Dataset, Output, Input, Model, Metrics, Artifact


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.2", "xgboost==2.0.3", "matplotlib==3.7.3"]
)
def validate_model(
    model_artifact: Input[Model],
    test_features: Input[Dataset],
    auc_threshold: float,
    validation_report: Output[Artifact],
    metrics: Output[Metrics]
) -> bool:
    """Validate model meets AUC threshold and generate validation report."""
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt
    import json
    
    # Load model and data
    model = xgb.XGBClassifier()
    model.load_model(model_artifact.path)
    
    test_df = pd.read_csv(test_features.path)
    X_test = test_df.drop('DEFAULT', axis=1)
    y_test = test_df['DEFAULT']
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    passed = auc >= auc_threshold
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # Create validation report
    validation_result = {
        "auc": float(auc),
        "auc_threshold": float(auc_threshold),
        "passed": passed,
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_count": len(X_test.columns),
        "test_samples": len(y_test)
    }
    
    # Save report
    with open(validation_report.path, 'w') as f:
        json.dump(validation_result, f, indent=2)
    
    # Log metrics
    metrics.log_metric("auc", float(auc))
    metrics.log_metric("auc_threshold", float(auc_threshold))
    metrics.log_metric("validation_passed", int(passed))
    metrics.log_metric("precision_default", float(report['1']['precision']))
    metrics.log_metric("recall_default", float(report['1']['recall']))
    
    # Generate ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Model (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.axhline(y=auc_threshold, color='r', linestyle='--', label=f'Threshold = {auc_threshold}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Validation {"PASSED" if passed else "FAILED"}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(validation_report.path.replace('.json', '_roc.png'), dpi=100)
    
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"{status}: AUC={auc:.4f} (threshold: {auc_threshold})")
    
    return passed
