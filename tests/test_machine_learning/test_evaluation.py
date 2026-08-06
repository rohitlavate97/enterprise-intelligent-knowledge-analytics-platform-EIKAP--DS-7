import pytest
import numpy as np
from machine_learning.evaluation import ClassificationMetrics, RegressionMetrics, MLEvaluator

def test_classification_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.6, 0.4]])
    
    metrics = ClassificationMetrics.calculate(y_true, y_pred, y_prob)
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["f1"] > 0
    assert metrics["roc_auc"] > 0

def test_ece_calculation(ml_evaluator):
    y_true = np.array([0, 1, 0, 1])
    # perfectly calibrated
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])
    
    ece = ml_evaluator.expected_calibration_error(y_true, y_prob, n_bins=2)
    assert ece == 0.0
    
def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    
    metrics = RegressionMetrics.calculate(y_true, y_pred)
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["mse"] > 0
    assert metrics["r2"] > 0
