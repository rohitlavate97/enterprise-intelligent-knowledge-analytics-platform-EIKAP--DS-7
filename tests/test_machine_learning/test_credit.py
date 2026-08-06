import pytest
import numpy as np
from machine_learning.models.credit import CreditRiskModel
from machine_learning.evaluation import MLEvaluator

def test_credit_calibration(credit_data, credit_model, ml_evaluator):
    _, X_test, _, y_test = credit_data
    probs = credit_model.predict_proba(X_test)
    
    # Verify probability calibration shape and values
    assert probs.shape == (len(X_test), 2)
    assert np.all((probs >= 0) & (probs <= 1))
    
    # Check ECE
    ece = ml_evaluator.expected_calibration_error(y_test, probs[:, 1])
    assert ece < 0.15

def test_credit_human_review_framing(credit_model):
    probs = np.array([0.2, 0.8])
    recs = credit_model.get_human_review_recommendations(probs, threshold=0.5)
    
    assert len(recs) == 2
    for rec in recs:
        assert "recommendation" in rec
        assert "reason" in rec
        assert "action" not in rec
        assert "decision" not in rec
