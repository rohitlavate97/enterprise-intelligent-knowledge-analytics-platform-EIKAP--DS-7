from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List
import time
import uuid

class BaseInputSchema(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier")
    timestamp: float = Field(default_factory=time.time, description="Timestamp of the request")

class BaseOutputSchema(BaseModel):
    prediction: Any = Field(description="The primary output/prediction")
    confidence: Optional[float] = Field(default=None, description="Confidence score if applicable")
    explanation: Optional[Dict[str, Any]] = Field(default=None, description="Explanation for the prediction")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    maturity_label: str = Field(description="Maturity label (standard/restricted)")
    module_name: str = Field(description="Name of the module that generated the output")
    timestamp: float = Field(default_factory=time.time, description="Timestamp of the response")
    request_id: str = Field(description="Unique request identifier matching the input")

class PredictionResponse(BaseModel):
    success: bool
    data: Optional[BaseOutputSchema]
    error: Optional[str]
    latency_ms: float

class HealthCheckResponse(BaseModel):
    status: str
    module_name: str
    version: str
    checks: Dict[str, Any]

class ComplianceReportResponse(BaseModel):
    module_name: str
    timestamp: str
    all_passed: bool
    summary: str
    checks: List[Dict[str, Any]]

class ModuleInfoResponse(BaseModel):
    name: str
    version: str
    description: str
    category: str
    maturity: str
    author: str
    dependencies: List[str]
    tags: List[str]
    requires_gpu: bool
