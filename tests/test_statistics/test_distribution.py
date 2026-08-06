
def test_fit_norm_distribution(distribution_fitter, sample_normal_data):
    res = distribution_fitter.fit_norm_distribution(sample_normal_data['value'])
    assert res.distribution == "norm"
    
def test_find_best_distribution(distribution_fitter, sample_normal_data):
    res = distribution_fitter.find_best_distribution(sample_normal_data['value'])
    assert res.distribution == "norm"

def test_confidence_interval(distribution_fitter, sample_normal_data):
    ci = distribution_fitter.confidence_interval(sample_normal_data['value'])
    assert ci[0] < ci[1]
