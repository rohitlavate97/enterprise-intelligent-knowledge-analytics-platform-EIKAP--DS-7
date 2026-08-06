from typing import Any, Dict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base import BaseMLModel

class CustomerChurnModel(BaseMLModel):
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.feature_names = []

    def fit(self, X: Any, y: Any, **kwargs) -> "CustomerChurnModel":
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
        return {"top_factors": top_factors, "model_type": "RandomForest"}
