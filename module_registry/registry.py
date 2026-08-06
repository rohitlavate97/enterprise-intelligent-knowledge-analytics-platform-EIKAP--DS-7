import threading
from typing import Dict, List, Optional
from module_registry.base import EIKAPModule, ModuleMetadata, ModuleCategory, MaturityLabel

class ModuleRegistry:
    def __init__(self):
        self._modules: Dict[str, EIKAPModule] = {}
        self._lock = threading.Lock()
        self._compliance_status: Dict[str, Dict] = {}

    def register(self, module_instance: EIKAPModule) -> None:
        name = module_instance.metadata.name
        with self._lock:
            self._validate_unique_name(name)
            self._modules[name] = module_instance

    def unregister(self, module_name: str) -> None:
        with self._lock:
            if module_name in self._modules:
                del self._modules[module_name]

    def get(self, module_name: str) -> EIKAPModule:
        with self._lock:
            if module_name not in self._modules:
                raise KeyError(f"Module '{module_name}' not found in registry.")
            return self._modules[module_name]

    def list_modules(self) -> List[ModuleMetadata]:
        with self._lock:
            return [m.metadata for m in self._modules.values()]

    def list_by_category(self, category: ModuleCategory) -> List[ModuleMetadata]:
        with self._lock:
            return [m.metadata for m in self._modules.values() if m.metadata.category == category]

    def list_restricted(self) -> List[ModuleMetadata]:
        with self._lock:
            return [m.metadata for m in self._modules.values() if m.metadata.maturity == MaturityLabel.RESTRICTED]

    def get_compliance_status(self, module_name: str) -> Dict:
        with self._lock:
            return self._compliance_status.get(module_name, {})
            
    def update_compliance_status(self, module_name: str, status: Dict) -> None:
        with self._lock:
            self._compliance_status[module_name] = status

    def _validate_unique_name(self, name: str) -> None:
        if name in self._modules:
            raise ValueError(f"Module with name '{name}' is already registered.")

module_registry = ModuleRegistry()
