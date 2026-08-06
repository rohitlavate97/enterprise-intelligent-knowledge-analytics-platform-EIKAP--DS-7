import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from machine_learning.models.churn import CustomerChurnModel
from machine_learning.models.fraud import FraudDetectionModel
from machine_learning.models.credit import CreditRiskModel
from machine_learning.evaluation import MLEvaluator
from machine_learning.module import MachineLearningModule

@pytest.fixture
def churn_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return X[:150], X[150:], y[:150], y[150:]

@pytest.fixture
def fraud_data():
    X, y = make_classification(n_samples=200, n_features=10, weights=[0.9, 0.1], random_state=42)
    return X[:150], X[150:], y[:150], y[150:]

@pytest.fixture
def credit_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return X[:150], X[150:], y[:150], y[150:]

@pytest.fixture
def churn_model(churn_data):
    model = CustomerChurnModel(n_estimators=10, random_state=42)
    X_train, _, y_train, _ = churn_data
    model.fit(X_train, y_train)
    return model

@pytest.fixture
def fraud_model(fraud_data):
    model = FraudDetectionModel(scale_pos_weight=9.0, random_state=42)
    X_train, _, y_train, _ = fraud_data
    model.fit(X_train, y_train)
    return model

@pytest.fixture
def credit_model(credit_data):
    model = CreditRiskModel(random_state=42)
    X_train, _, y_train, _ = credit_data
    model.fit(X_train, y_train)
    return model

@pytest.fixture
def ml_evaluator():
    return MLEvaluator()

@pytest.fixture
def ml_module():
    return MachineLearningModule()
