import pytest
from typing import Any, Dict, Type
from pydantic import BaseModel
from module_registry.base import EIKAPModule, ModuleMetadata, MaturityLabel, ModuleCategory
from module_registry.registry import module_registry as global_registry

class DummyInput(BaseModel):
    value: str

class DummyOutput(BaseModel):
    result: str

class MockStandardModule(EIKAPModule):
    def __init__(self):
        self._meta = ModuleMetadata(
            name="mock_standard",
            version="1.0.0",
            description="A standard mock module",
            category=ModuleCategory.MACHINE_LEARNING,
            maturity=MaturityLabel.STANDARD,
            author="Test Author"
        )
        
    @property
    def metadata(self) -> ModuleMetadata:
        return self._meta
        
    @property
    def input_schema(self) -> Type[BaseModel]:
        return DummyInput
        
    @property
    def output_schema(self) -> Type[BaseModel]:
        return DummyOutput
        
    def train(self, data: Any, **kwargs) -> Dict[str, Any]:
        return {"loss": 0.1}
        
    def predict(self, input_data: Any, **kwargs) -> Any:
        return {"result": "ok"}
        
    def explain(self, input_data: Any, prediction: Any, **kwargs) -> Dict[str, Any]:
        return {"feature_importance": [0.5, 0.5]}
        
    def evaluate(self, test_data: Any, **kwargs) -> Dict[str, float]:
        return {"accuracy": 0.99}
        
    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok"}

class MockRestrictedModule(EIKAPModule):
    def __init__(self, fails_framing=False):
        self._meta = ModuleMetadata(
            name="mock_restricted",
            version="1.0.0",
            description="A restricted mock module",
            category=ModuleCategory.MACHINE_LEARNING,
            maturity=MaturityLabel.RESTRICTED,
            author="Test Author"
        )
        self.fails_framing = fails_framing
        
    @property
    def metadata(self) -> ModuleMetadata:
        return self._meta
        
    @property
    def input_schema(self) -> Type[BaseModel]:
        return DummyInput
        
    @property
    def output_schema(self) -> Type[BaseModel]:
        return DummyOutput
        
    def train(self, data: Any, **kwargs) -> Dict[str, Any]:
        return {"loss": 0.1}
        
    def predict(self, input_data: Any, **kwargs) -> Any:
        if self.fails_framing:
            return {"action": "deny"}
        return {"recommendation": "review", "explanation": "flagged"}
        
    def explain(self, input_data: Any, prediction: Any, **kwargs) -> Dict[str, Any]:
        return {"reason": "risk score"}
        
    def evaluate(self, test_data: Any, **kwargs) -> Dict[str, float]:
        return {"f1": 0.9}
        
    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok"}


@pytest.fixture
def mock_standard_module():
    return MockStandardModule()

@pytest.fixture
def mock_restricted_module():
    return MockRestrictedModule()

@pytest.fixture
def sample_input():
    return {"value": "test"}

@pytest.fixture
def sample_output():
    return {"result": "ok"}

@pytest.fixture
def clean_registry():
    # Clear registry manually
    modules_to_remove = list(global_registry._modules.keys())
    for name in modules_to_remove:
        global_registry.unregister(name)
    yield global_registry
    modules_to_remove = list(global_registry._modules.keys())
    for name in modules_to_remove:
        global_registry.unregister(name)
