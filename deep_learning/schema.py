from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class DLInput(BaseModel):
    model_architecture: str = Field(default="mlp")
    input_dim: int = Field(default=10)
    epochs: int = Field(default=20)
    batch_size: int = Field(default=32)
    feature_data: Dict[str, Any] = Field(default_factory=dict)

class DLOutput(BaseModel):
    model_architecture: str
    loss_history: List[float]
    predictions: List[float]
    metrics: Dict[str, float]
    execution_time_ms: float
