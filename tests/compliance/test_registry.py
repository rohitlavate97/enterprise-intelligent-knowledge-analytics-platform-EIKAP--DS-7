import pytest
import threading
from module_registry.base import ModuleCategory
from module_registry.registry import ModuleRegistry

def test_register_unregister(clean_registry, mock_standard_module):
    clean_registry.register(mock_standard_module)
    assert clean_registry.get("mock_standard") == mock_standard_module
    
    clean_registry.unregister("mock_standard")
    with pytest.raises(KeyError):
        clean_registry.get("mock_standard")

def test_duplicate_name_rejection(clean_registry, mock_standard_module):
    clean_registry.register(mock_standard_module)
    with pytest.raises(ValueError):
        clean_registry.register(mock_standard_module)

def test_list_by_category(clean_registry, mock_standard_module, mock_restricted_module):
    mock_restricted_module._meta.category = ModuleCategory.NLP
    clean_registry.register(mock_standard_module)
    clean_registry.register(mock_restricted_module)
    
    ml_modules = clean_registry.list_by_category(ModuleCategory.MACHINE_LEARNING)
    assert len(ml_modules) == 1
    assert ml_modules[0].name == "mock_standard"
    
    nlp_modules = clean_registry.list_by_category(ModuleCategory.NLP)
    assert len(nlp_modules) == 1
    assert nlp_modules[0].name == "mock_restricted"

def test_list_restricted(clean_registry, mock_standard_module, mock_restricted_module):
    clean_registry.register(mock_standard_module)
    clean_registry.register(mock_restricted_module)
    
    restricted = clean_registry.list_restricted()
    assert len(restricted) == 1
    assert restricted[0].name == "mock_restricted"

def test_thread_safety(clean_registry, mock_standard_module):
    registry = ModuleRegistry()
    
    def register_func(idx):
        import copy
        mod = copy.copy(mock_standard_module)
        mod._meta.name = f"mod_{idx}"
        registry.register(mod)

    threads = []
    for i in range(10):
        t = threading.Thread(target=register_func, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    assert len(registry.list_modules()) == 10
