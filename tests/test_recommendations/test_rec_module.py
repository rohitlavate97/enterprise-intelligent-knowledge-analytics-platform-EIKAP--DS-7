import pytest
import pandas as pd
from module_registry.contract import UniversalModuleContract
from recommendations.module import RecommendationModule, RecommendationInput

def test_rec_module_metadata():
    """Test RecommendationModule metadata."""
    module = RecommendationModule()
    meta = module.metadata
    
    assert meta.name == "recommendation_engine"
    assert meta.version == "0.1.0"
    assert "MACHINE_LEARNING" in str(meta.category)

def test_rec_module_predict(sample_interactions, sample_items):
    """Test full predict lifecycle of RecommendationModule."""
    module = RecommendationModule()
    module.train(data=sample_interactions, items=sample_items)
    
    input_data = RecommendationInput(
        user_id=1,
        top_k=2,
        cf_weight=0.5
    )
    
    output = module.predict(input_data)
    assert output.user_id == 1
    assert len(output.recommendations) <= 2
    assert "num_recommendations" in output.metrics
    assert output.execution_time_ms >= 0.0

def test_rec_module_compliance_contract():
    """Test RecommendationModule against UniversalModuleContract."""
    module = RecommendationModule()
    contract = UniversalModuleContract()
    
    result = contract.run_full_check(module)
    assert result.all_passed is True
    # no result.errors in ContractReport, we just check all_passed
