import numpy as np
import pandas as pd

def test_pearson_correlation(correlation_analyzer):
    x = pd.Series(np.linspace(0, 10, 50))
    y = x * 2 + np.random.normal(0, 0.1, 50)
    res = correlation_analyzer.pearson_correlation(x, y)
    assert res['r'] > 0.9

def test_spearman_correlation(correlation_analyzer):
    x = pd.Series(np.linspace(1, 10, 50))
    y = x ** 3
    res = correlation_analyzer.spearman_correlation(x, y)
    assert np.isclose(res['r'], 1.0)

def test_partial_correlation(correlation_analyzer):
    np.random.seed(42)
    z = np.random.normal(0, 1, 100)
    x = z + np.random.normal(0, 0.1, 100)
    y = z + np.random.normal(0, 0.1, 100)
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    res = correlation_analyzer.partial_correlation(df, 'x', 'y', ['z'])
    assert abs(res['r']) < 0.3
