from typing import Any, Dict, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from .base import BaseMLModel

class CreditRiskModel(BaseMLModel):
    def __init__(self, random_state: int = 42):
        base_model = LogisticRegression(random_state=random_state, class_weight="balanced", max_iter=1000)
        # Using Isotonic regression to ensure good probability calibration
        self.model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
        self.feature_names = []
        self.base_model_coef_ = None

    def fit(self, X: Any, y: Any, **kwargs) -> "CreditRiskModel":
        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)
        elif isinstance(X, np.ndarray):
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            self.feature_names = [f"feature_{i}" for i in range(len(X[0]))]
            
        self.model.fit(X, y)
        
        # Train a standalone base model to extract coefficients for explainability
        base_for_explain = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
        base_for_explain.fit(X, y)
        self.base_model_coef_ = base_for_explain.coef_[0]
        return self

    def predict(self, X: Any) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self.model.predict_proba(X)

    def explain(self, X: Any = None) -> Dict[str, Any]:
        if self.base_model_coef_ is None:
            return {"error": "Model not fitted"}
            
        importances = np.abs(self.base_model_coef_)
        sorted_indices = np.argsort(importances)[::-1]
        top_factors = []
        for i in sorted_indices[:5]:
            top_factors.append({
                "feature": self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                "importance": float(importances[i]),
                "direction": "positive" if self.base_model_coef_[i] > 0 else "negative"
            })
        return {"top_factors": top_factors, "model_type": "CalibratedLogisticRegression"}

    def get_human_review_recommendations(self, probabilities: np.ndarray, threshold: float = 0.5) -> List[Dict[str, Any]]:
        # Restricted framing: no 'decision' or 'action'
        recommendations = []
        for prob in probabilities:
            if prob > threshold:
                recommendations.append({
                    "recommendation": "Refer to loan officer",
                    "reason": f"High risk profile ({prob:.2f})"
                })
            else:
                recommendations.append({
                    "recommendation": "Standard review process",
                    "reason": f"Lower risk profile ({prob:.2f})"
                })
        return recommendations
