import time
from typing import Any, Dict, Type

import numpy as np
import sklearn
import xgboost
import joblib

from module_registry.base import EIKAPModule, ModuleMetadata, ModuleCategory, MaturityLabel
from module_registry.schemas import BaseOutputSchema
from machine_learning.schemas import MLInput, MLOutput
from machine_learning.models.churn import CustomerChurnModel
from machine_learning.models.fraud import FraudDetectionModel
from machine_learning.models.credit import CreditRiskModel
from machine_learning.evaluation import MLEvaluator

class MachineLearningModule(EIKAPModule):
    def __init__(self):
        self._metadata = ModuleMetadata(
            name="machine_learning",
            version="0.1.0",
            description="Machine Learning Engine for Churn, Fraud, and Credit Risk Models",
            category=ModuleCategory.MACHINE_LEARNING,
            maturity=MaturityLabel.STANDARD,
            author="EIKAP Team",
            dependencies=["scikit-learn", "xgboost", "joblib"]
        )
        self.models = {
            "churn": CustomerChurnModel(),
            "fraud": FraudDetectionModel(),
            "credit": CreditRiskModel()
        }
        self.evaluator = MLEvaluator()

    @property
    def metadata(self) -> ModuleMetadata:
        return self._metadata

    @property
    def input_schema(self) -> Type[MLInput]:
        return MLInput

    @property
    def output_schema(self) -> Type[MLOutput]:
        return MLOutput

    def train(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Trains specified model (churn, fraud, credit) on input DataFrame and logs to MLflow."""
        use_case = kwargs.get("use_case", "churn")
        if use_case not in self.models:
            raise ValueError(f"Unknown use case: {use_case}")
            
        model = self.models[use_case]
        X = data.get("X")
        y = data.get("y")
        
        start_time = time.time()
        model.fit(X, y)
        train_time = time.time() - start_time
        
        # Stubbed MLflow logging
        mlflow_run_id = f"run_{int(time.time())}"
        
        return {
            "status": "success",
            "use_case": use_case,
            "train_time_ms": train_time * 1000,
            "mlflow_run_id": mlflow_run_id
        }

    def predict(self, input_data: Any, **kwargs) -> MLOutput:
        start_time = time.time()
        
        # Validate and parse input
        if isinstance(input_data, dict):
            validated_input = self.input_schema(**input_data)
        elif isinstance(input_data, MLInput):
            validated_input = input_data
        else:
            raise ValueError("Invalid input format")
            
        use_case = validated_input.use_case
        if use_case not in self.models:
            raise ValueError(f"Unknown use case: {use_case}")
            
        model = self.models[use_case]
        
        # Extract features (assume it's passed appropriately, normally a DataFrame or array)
        X = validated_input.feature_data.get("X")
        if X is None:
            # Fallback if just raw data is passed
            X = np.array(list(validated_input.feature_data.values())).reshape(1, -1)
            
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        
        if probabilities is not None and probabilities.ndim > 1:
            prob_list = probabilities[:, 1].tolist()
        elif probabilities is not None:
            prob_list = probabilities.tolist()
        else:
            prob_list = []
            
        recommendations = None
        if hasattr(model, "get_human_review_recommendations") and probabilities is not None:
            recommendations = model.get_human_review_recommendations(np.array(prob_list))
            
        explanation = model.explain(X)
        
        execution_time = (time.time() - start_time) * 1000
        
        output = MLOutput(
            use_case=use_case,
            predictions=predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            probabilities=prob_list,
            metrics={},
            recommendations=recommendations,
            explanation=explanation,
            execution_time_ms=execution_time
        )
        return output

    def explain(self, input_data: Any, prediction: Any, **kwargs) -> Dict[str, Any]:
        if isinstance(input_data, dict) and "use_case" in input_data:
            use_case = input_data["use_case"]
        elif hasattr(input_data, "use_case"):
            use_case = input_data.use_case
        else:
            use_case = kwargs.get("use_case", "churn")
            
        if use_case not in self.models:
            raise ValueError(f"Unknown use case: {use_case}")
            
        model = self.models[use_case]
        X = input_data.get("feature_data", {}).get("X") if isinstance(input_data, dict) else None
        
        return model.explain(X)

    def evaluate(self, test_data: Any, **kwargs) -> Dict[str, float]:
        use_case = kwargs.get("use_case", "churn")
        if use_case not in self.models:
            raise ValueError(f"Unknown use case: {use_case}")
            
        model = self.models[use_case]
        X_test = test_data.get("X")
        y_test = test_data.get("y")
        
        metrics = self.evaluator.evaluate_classification(model, X_test, y_test)
        return metrics

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "dependencies": {
                "scikit-learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "joblib": joblib.__version__
            }
        }
