from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class BaseMLModel(ABC):
    @abstractmethod
    def fit(self, X: Any, y: Any, **kwargs) -> "BaseMLModel":
        raise NotImplementedError("fit must be implemented")

    @abstractmethod
    def predict(self, X: Any) -> Any:
        raise NotImplementedError("predict must be implemented")

    @abstractmethod
    def predict_proba(self, X: Any) -> Any:
        raise NotImplementedError("predict_proba must be implemented")

    @abstractmethod
    def explain(self, X: Any) -> Dict[str, Any]:
        raise NotImplementedError("explain must be implemented")
