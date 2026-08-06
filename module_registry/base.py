"""EIKAPModule plugin interface.

Every business-use-case module in EIKAP implements this shared interface,
enabling the Universal Module Contract to be enforced uniformly across
all 15 modules regardless of discipline (stats/ML/DL/CV/NLP/RAG).
"""
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
import time

from module_registry.schemas import BaseOutputSchema

class MaturityLabel(str, Enum):
    STANDARD = "standard"
    RESTRICTED = "restricted"

class ModuleCategory(str, Enum):
    DATA_PIPELINE = "data_pipeline"
    ANALYTICS = "analytics"
    STATISTICS = "statistics"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    COMPUTER_VISION = "computer_vision"
    NLP = "nlp"
    RAG = "rag"

@dataclass
class ModuleMetadata:
    name: str
    version: str
    description: str
    category: ModuleCategory
    maturity: MaturityLabel
    author: str
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    latency_target_ms: Optional[float] = None
    requires_gpu: bool = False

class EIKAPModule(ABC):
    
    @property
    @abstractmethod
    def metadata(self) -> ModuleMetadata:
        pass
        
    @property
    @abstractmethod
    def input_schema(self) -> Type[BaseModel]:
        pass
        
    @property
    @abstractmethod
    def output_schema(self) -> Type[BaseModel]:
        pass
        
    @abstractmethod
    def train(self, data: Any, **kwargs) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def predict(self, input_data: Any, **kwargs) -> Any:
        pass
        
    @abstractmethod
    def explain(self, input_data: Any, prediction: Any, **kwargs) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def evaluate(self, test_data: Any, **kwargs) -> Dict[str, float]:
        pass
        
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        pass
        
    def run_with_latency(self, input_data: Any, **kwargs) -> tuple[Any, float]:
        start = time.time()
        result = self.predict(input_data, **kwargs)
        end = time.time()
        latency_ms = (end - start) * 1000
        return result, latency_ms
        
    def validate_input(self, data: Any) -> BaseModel:
        if isinstance(data, dict):
            return self.input_schema(**data)
        elif isinstance(data, BaseModel):
            return self.input_schema.model_validate(data.model_dump())
        return self.input_schema.model_validate(data)
        
    def validate_output(self, data: Any) -> BaseModel:
        if isinstance(data, dict):
            return self.output_schema(**data)
        elif isinstance(data, BaseModel):
            return self.output_schema.model_validate(data.model_dump())
        return self.output_schema.model_validate(data)
        
    def to_api_response(self, prediction: Any, explanation: Dict[str, Any], latency_ms: float, request_id: str = "unknown") -> Dict[str, Any]:
        schema = BaseOutputSchema(
            prediction=prediction,
            explanation=explanation,
            maturity_label=self.metadata.maturity.value,
            module_name=self.metadata.name,
            request_id=request_id
        )
        return {
            "success": True,
            "data": schema.model_dump(),
            "latency_ms": latency_ms
        }
        
    def is_restricted(self) -> bool:
        return self.metadata.maturity == MaturityLabel.RESTRICTED
        
    def get_module_info(self) -> Dict[str, Any]:
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "description": self.metadata.description,
            "category": self.metadata.category.value,
            "maturity": self.metadata.maturity.value,
            "author": self.metadata.author,
            "dependencies": self.metadata.dependencies,
            "tags": self.metadata.tags,
            "requires_gpu": self.metadata.requires_gpu,
            "latency_target_ms": self.metadata.latency_target_ms
        }
