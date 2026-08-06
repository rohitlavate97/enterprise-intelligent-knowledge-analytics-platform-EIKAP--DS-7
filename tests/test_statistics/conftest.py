
import pytest
import pandas as pd
import numpy as np
from statistics import (
    HypothesisTester,
    NormalityTester,
    CorrelationAnalyzer,
    DistributionFitter,
    StatisticalFeatureSelector
)

@pytest.fixture
def sample_normal_data():
    np.random.seed(42)
    return pd.DataFrame({'value': np.random.normal(loc=0, scale=1, size=100)})

@pytest.fixture
def sample_non_normal_data():
    np.random.seed(42)
    return pd.DataFrame({'value': np.random.exponential(scale=1, size=100)})

@pytest.fixture
def sample_contingency_table():
    return np.array([[10, 20], [20, 40]])

@pytest.fixture
def hypothesis_tester():
    return HypothesisTester()

@pytest.fixture
def normality_tester():
    return NormalityTester()

@pytest.fixture
def correlation_analyzer():
    return CorrelationAnalyzer()

@pytest.fixture
def distribution_fitter():
    return DistributionFitter()

@pytest.fixture
def feature_selector():
    return StatisticalFeatureSelector()
