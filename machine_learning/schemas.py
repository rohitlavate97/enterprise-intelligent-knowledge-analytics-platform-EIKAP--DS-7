from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class MLInput(BaseModel):
    use_case: str = Field(default="churn", description="Use case: churn, fraud, or credit")
    feature_data: Dict[str, Any] = Field(default_factory=dict, description="Input features for the model")

class MLOutput(BaseModel):
    use_case: str = Field(description="Use case that was executed")
    predictions: List[Any] = Field(description="Predicted class labels or continuous values")
    probabilities: List[float] = Field(description="Predicted probabilities (if applicable)")
    metrics: Dict[str, float] = Field(description="Evaluation metrics if evaluated")
    recommendations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Human review framing recommendations")
    explanation: Optional[Dict[str, Any]] = Field(default=None, description="Explanation for the predictions")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
