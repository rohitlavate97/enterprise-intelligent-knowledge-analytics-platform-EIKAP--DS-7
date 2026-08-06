from data_pipeline.module import DataPipelineModule
from module_registry.base import MaturityLabel, ModuleCategory

def test_module_metadata():
    m = DataPipelineModule()
    assert m.metadata.name == "data_pipeline"
    assert m.metadata.category == ModuleCategory.DATA_PIPELINE

def test_module_is_standard():
    m = DataPipelineModule()
    assert m.metadata.maturity == MaturityLabel.STANDARD
    assert not m.is_restricted()

def test_module_health_check():
    m = DataPipelineModule()
    assert m.health_check()["status"] == "ok"

def test_module_compliance_contract():
    pass
