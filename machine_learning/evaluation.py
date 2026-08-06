import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

class ClassificationMetrics:
    @staticmethod
    def calculate(y_true: Any, y_pred: Any, y_prob: Any = None) -> Dict[str, float]:
        metrics = {}
        metrics["f1"] = float(f1_score(y_true, y_pred, average="macro"))
        if y_prob is not None:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        return metrics

class RegressionMetrics:
    @staticmethod
    def calculate(y_true: Any, y_pred: Any) -> Dict[str, float]:
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))
        }

class MLEvaluator:
    def __init__(self):
        pass

    def expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        if y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
            
        bins = np.linspace(0., 1., n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        
        bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))
        
        nonzero = bin_total != 0
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]
        
        ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
        return float(ece)

    def evaluate_classification(self, model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        metrics = ClassificationMetrics.calculate(y_test, y_pred, y_prob)
        if y_prob is not None and len(np.unique(y_test)) == 2:
            metrics["ece"] = self.expected_calibration_error(np.array(y_test), np.array(y_prob))
            
        return metrics
