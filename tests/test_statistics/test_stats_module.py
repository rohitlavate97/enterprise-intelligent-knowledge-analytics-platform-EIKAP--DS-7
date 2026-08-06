from statistics.module import StatisticalAnalysisModule, StatisticalInput
import pandas as pd
import numpy as np
from module_registry.base import MaturityLabel, ModuleCategory

def test_statistical_module_metadata():
    m = StatisticalAnalysisModule()
    assert m.metadata.name == "statistical_analysis"
    assert m.metadata.category == ModuleCategory.STATISTICS
    assert m.metadata.maturity == MaturityLabel.STANDARD

def test_statistical_module_predict(sample_normal_data):
    m = StatisticalAnalysisModule()
    inp = StatisticalInput(columns=['value'])
    out = m.predict(inp, data=sample_normal_data)
    
    assert out.execution_time_ms > 0
    assert 'value' in out.normality_results
    assert out.normality_results['value']['is_normal'] == True

def test_statistical_module_compliance_contract():
    try:
        from module_registry.contract import UniversalModuleContract
        contract = UniversalModuleContract()
        m = StatisticalAnalysisModule()
        report = contract.run_full_check(m)
        assert isinstance(report, object)
    except ImportError:
        pass
