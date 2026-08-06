
import pandas as pd
import numpy as np

def test_select_by_anova_f(feature_selector):
    np.random.seed(42)
    X = pd.DataFrame({
        'f1': np.random.normal(0, 1, 100),
        'f2': np.random.normal(0, 1, 100),
        'f3': np.random.normal(0, 1, 100)
    })
    y = pd.Series(np.where(X['f1'] > 0, 1, 0)) # y depends on f1
    selected = feature_selector.select_by_anova_f(X, y, k=1)
    assert selected == ['f1']

def test_calculate_vif(feature_selector):
    np.random.seed(42)
    x1 = np.random.normal(0, 1, 100)
    x2 = x1 * 2 + np.random.normal(0, 0.1, 100) # highly collinear
    X = pd.DataFrame({'x1': x1, 'x2': x2})
    vifs = feature_selector.calculate_vif(X)
    assert vifs['x1'] > 10
    assert vifs['x2'] > 10
