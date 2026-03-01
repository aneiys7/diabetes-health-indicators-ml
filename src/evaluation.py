# src/evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay, 
    ConfusionMatrixDisplay, 
    classification_report, 
    mean_absolute_error, 
    r2_score
)

def plot_binary_results(model, X_test, y_test, model_name="Binary Model"):
    """
    Plots the ROC Curve and prints the Classification Report for Binary Tasks.
    """
    print(f"--- {model_name} Performance ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Plot ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC Curve: {model_name}")
    plt.show()

def plot_multiclass_results(model, X_test, y_test, labels=None):
    """
    Plots a Confusion Matrix for Multiclass Staging.
    """
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=labels, cmap='Blues', ax=ax
    )
    plt.title("Multiclass Confusion Matrix (Diabetes Stages)")
    plt.show()

def plot_regression_results(y_test, y_pred):
    """
    Shows a Scatter Plot of Actual vs Predicted Risk Scores.
    """
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
    plt.xlabel('Actual Risk Score')
    plt.ylabel('Predicted Risk Score')
    plt.title(f'Regression: Actual vs Predicted (R2: {r2:.2f}, MAE: {mae:.2f})')
    plt.show()