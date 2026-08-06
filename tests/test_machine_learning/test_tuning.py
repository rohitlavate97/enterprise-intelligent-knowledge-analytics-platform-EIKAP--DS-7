import pytest
from sklearn.ensemble import RandomForestClassifier
from machine_learning.tuning import HyperparameterTuner

def test_grid_search_tuning(churn_data):
    X_train, _, y_train, _ = churn_data
    
    estimator = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [5, 10],
        "max_depth": [2, None]
    }
    
    tuner = HyperparameterTuner(estimator, param_grid, cv=2)
    best_model = tuner.tune(X_train, y_train)
    
    assert best_model is not None
    assert tuner.best_params_ is not None
    assert "n_estimators" in tuner.best_params_
    assert "max_depth" in tuner.best_params_
