
def test_shapiro_wilk_normal(normality_tester, sample_normal_data):
    res = normality_tester.shapiro_wilk(sample_normal_data['value'])
    assert res.is_normal
    assert res.p_value > 0.05
    
def test_shapiro_wilk_non_normal(normality_tester, sample_non_normal_data):
    res = normality_tester.shapiro_wilk(sample_non_normal_data['value'])
    assert not res.is_normal
    assert res.p_value <= 0.05
    
def test_auto_select_test_type(normality_tester, sample_normal_data):
    res = normality_tester.auto_select_test(sample_normal_data['value'])
    assert res.test_name == "Shapiro-Wilk"
    assert res.is_normal
