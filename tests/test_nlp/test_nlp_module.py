import pytest
from nlp import NLPModule
from module_registry.base import MaturityLabel

def test_nlp_module_metadata():
    module = NLPModule()
    meta = module.metadata
    assert meta.name == "nlp_analytics"
    assert meta.version == "0.1.0"
    assert meta.maturity == MaturityLabel.STANDARD
    assert "nltk" in meta.dependencies
    assert "scikit-learn" in meta.dependencies

def test_nlp_module_predict_sentiment():
    module = NLPModule()
    input_data = {"task": "sentiment", "text": "This is a good day."}
    output = module.predict(input_data)
    assert output.task == "sentiment"
    assert "sentiment" in output.results
    assert output.results["sentiment"] == "positive"

def test_nlp_module_predict_resume(sample_resume, sample_job_description):
    module = NLPModule()
    input_data = {
        "task": "resume", 
        "text": sample_resume, 
        "job_description": sample_job_description
    }
    output = module.predict(input_data)
    assert output.task == "resume"
    assert "match_score" in output.results
    assert "fairness" in output.results
    assert "framing" in output.results
    
    assert "action" not in output.results["framing"]
    assert "rejection" not in output.results["framing"]

def test_nlp_module_compliance_contract():
    module = NLPModule()
    assert hasattr(module, "train")
    assert hasattr(module, "predict")
    assert hasattr(module, "explain")
    assert hasattr(module, "evaluate")
    assert hasattr(module, "health_check")
    
    health = module.health_check()
    assert "status" in health
    
    eval_res = module.evaluate(None)
    assert "accuracy" in eval_res
    assert "f1_score" in eval_res
