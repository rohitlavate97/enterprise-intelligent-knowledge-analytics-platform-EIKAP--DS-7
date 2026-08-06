import pytest
from module_registry.contract import UniversalModuleContract
from tests.compliance.conftest import MockRestrictedModule, MockStandardModule

def test_check_metadata(mock_standard_module):
    contract = UniversalModuleContract()
    result = contract.check_metadata(mock_standard_module)
    assert result.passed is True
    
    mock_standard_module._meta.author = ""
    result2 = contract.check_metadata(mock_standard_module)
    assert result2.passed is False
    assert "Missing required metadata fields" in result2.message

def test_check_input_output_schemas(mock_standard_module):
    contract = UniversalModuleContract()
    result = contract.check_input_output_schemas(mock_standard_module)
    assert result.passed is True

def test_check_human_review_framing_standard(mock_standard_module):
    contract = UniversalModuleContract()
    result = contract.check_human_review_framing(mock_standard_module, {"action": "auto_approve"})
    assert result.passed is True
    assert "N/A" in result.message

def test_check_human_review_framing_restricted_pass():
    contract = UniversalModuleContract()
    module = MockRestrictedModule(fails_framing=False)
    result = contract.check_human_review_framing(module, module.predict({}))
    assert result.passed is True

def test_check_human_review_framing_restricted_fail():
    contract = UniversalModuleContract()
    module = MockRestrictedModule(fails_framing=True)
    result = contract.check_human_review_framing(module, module.predict({}))
    assert result.passed is False
    assert "action/decision fields" in result.message

def test_check_latency_benchmark(mock_standard_module):
    contract = UniversalModuleContract()
    result = contract.check_latency_benchmark(mock_standard_module, {"value": "test"}, 1000.0)
    assert result.passed is True
    
    result2 = contract.check_latency_benchmark(mock_standard_module, {"value": "test"}, 0.0)
    assert result2.passed is False

def test_run_full_check(mock_standard_module, sample_input, sample_output):
    contract = UniversalModuleContract()
    report = contract.run_full_check(
        module=mock_standard_module,
        sample_input=sample_input,
        sample_prediction=sample_output,
        sample_output=sample_output,
        train_data=[],
        test_data=[],
        target_ms=1000.0
    )
    assert report.all_passed is True
    assert len(report.checks) > 0
    assert report.module_name == "mock_standard"
