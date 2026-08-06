
import numpy as np
import pandas as pd

def test_one_sample_ttest(hypothesis_tester, sample_normal_data):
    res = hypothesis_tester.one_sample_ttest(sample_normal_data['value'], popmean=0)
    assert not res.significant
    
def test_two_sample_ttest(hypothesis_tester):
    np.random.seed(42)
    data1 = pd.Series(np.random.normal(0, 1, 100))
    data2 = pd.Series(np.random.normal(0.5, 1, 100))
    res = hypothesis_tester.two_sample_ttest(data1, data2)
    assert res.significant
    
def test_one_way_anova(hypothesis_tester):
    np.random.seed(42)
    g1 = pd.Series(np.random.normal(0, 1, 50))
    g2 = pd.Series(np.random.normal(1, 1, 50))
    g3 = pd.Series(np.random.normal(2, 1, 50))
    res = hypothesis_tester.one_way_anova(g1, g2, g3)
    assert res.significant
    
def test_chi_square_test(hypothesis_tester, sample_contingency_table):
    res = hypothesis_tester.chi_square_test(sample_contingency_table)
    assert not res.significant
    
def test_mann_whitney_u_test(hypothesis_tester):
    np.random.seed(42)
    d1 = pd.Series(np.random.exponential(1, 50))
    d2 = pd.Series(np.random.exponential(5, 50))
    res = hypothesis_tester.mann_whitney_u_test(d1, d2)
    assert res.significant
