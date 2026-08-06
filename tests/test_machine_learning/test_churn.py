import pytest
import numpy as np
from machine_learning.models.churn import CustomerChurnModel

def test_churn_fit_predict(churn_data, churn_model):
    _, X_test, _, _ = churn_data
    preds = churn_model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})

def test_churn_predict_proba(churn_data, churn_model):
    _, X_test, _, _ = churn_data
    probs = churn_model.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert np.all((probs >= 0) & (probs <= 1))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)

def test_churn_top_factors(churn_model):
    explanation = churn_model.explain()
    assert "top_factors" in explanation
    assert "model_type" in explanation
    assert len(explanation["top_factors"]) <= 5
    assert explanation["model_type"] == "RandomForest"
