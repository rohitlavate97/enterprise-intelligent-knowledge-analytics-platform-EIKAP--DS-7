import pytest
import numpy as np
from machine_learning.models.fraud import FraudDetectionModel
from module_registry.contract import UniversalModuleContract
from machine_learning.module import MachineLearningModule

def test_fraud_fit_predict(fraud_data, fraud_model):
    _, X_test, _, _ = fraud_data
    preds = fraud_model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})

def test_fraud_human_review_framing(fraud_model):
    probs = np.array([0.1, 0.9])
    recs = fraud_model.get_human_review_recommendations(probs, threshold=0.5)
    
    assert len(recs) == 2
    for rec in recs:
        assert "recommendation" in rec
        assert "reason" in rec
        assert "action" not in rec
        assert "decision" not in rec

def test_fraud_contract_compliance(fraud_data):
    X_train, X_test, y_train, y_test = fraud_data
    
    module = MachineLearningModule()
    
    # Train module with fraud data
    train_res = module.train({"X": X_train, "y": y_train}, use_case="fraud")
    assert train_res["status"] == "success"
    
    # Get a sample output
    sample_input = module.input_schema(use_case="fraud", feature_data={"X": X_test})
    sample_output = module.predict(sample_input)
    
    # Check compliance using UniversalModuleContract
    contract = UniversalModuleContract()
    
    # Mock is_restricted for testing purpose to True to enforce restricted checks
    module.is_restricted = lambda: True
    
    check_res = contract.check_human_review_framing(module, sample_output)
    
    # Contract asserts
    assert check_res.passed
