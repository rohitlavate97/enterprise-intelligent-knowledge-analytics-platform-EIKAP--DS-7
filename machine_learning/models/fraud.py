from typing import Any, Dict, List
import numpy as np
from xgboost import XGBClassifier
from .base import BaseMLModel

class FraudDetectionModel(BaseMLModel):
    def __init__(self, scale_pos_weight: float = 1.0, random_state: int = 42):
        self.model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=random_state, use_label_encoder=False, eval_metric="logloss")
        self.feature_names = []

    def fit(self, X: Any, y: Any, **kwargs) -> "FraudDetectionModel":
        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)
        elif isinstance(X, np.ndarray):
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            self.feature_names = [f"feature_{i}" for i in range(len(X[0]))]
        self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self.model.predict_proba(X)

    def explain(self, X: Any = None) -> Dict[str, Any]:
        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        top_factors = []
        for i in sorted_indices[:5]:
            top_factors.append({
                "feature": self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                "importance": float(importances[i])
            })
        return {"top_factors": top_factors, "model_type": "XGBoost"}

    def get_human_review_recommendations(self, probabilities: np.ndarray, threshold: float = 0.8) -> List[Dict[str, Any]]:
        # For restricted models, do NOT include "action" or "decision"
        recommendations = []
        for prob in probabilities:
            if prob > threshold:
                recommendations.append({
                    "recommendation": "Flag for human analyst review",
                    "reason": f"High probability of fraud ({prob:.2f})"
                })
            else:
                recommendations.append({
                    "recommendation": "Standard processing",
                    "reason": f"Low probability of fraud ({prob:.2f})"
                })
        return recommendations
