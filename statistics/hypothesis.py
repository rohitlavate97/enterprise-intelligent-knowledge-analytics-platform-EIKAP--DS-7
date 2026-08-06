
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import scipy.stats as stats
import pandas as pd
import numpy as np

class HypothesisTestResult(BaseModel):
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    recommendation: str = ""

class HypothesisTester:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def one_sample_ttest(self, data: pd.Series, popmean: float) -> HypothesisTestResult:
        stat, p = stats.ttest_1samp(data.dropna(), popmean)
        sig = p < self.alpha
        return HypothesisTestResult(test_name="1-Sample T-Test", statistic=stat, p_value=p, significant=sig, recommendation="Reject H0" if sig else "Fail to reject H0")

    def two_sample_ttest(self, data1: pd.Series, data2: pd.Series) -> HypothesisTestResult:
        stat, p = stats.ttest_ind(data1.dropna(), data2.dropna())
        sig = p < self.alpha
        return HypothesisTestResult(test_name="2-Sample T-Test", statistic=stat, p_value=p, significant=sig)

    def one_way_anova(self, *groups: pd.Series) -> HypothesisTestResult:
        cleaned_groups = [g.dropna() for g in groups]
        stat, p = stats.f_oneway(*cleaned_groups)
        sig = p < self.alpha
        return HypothesisTestResult(test_name="One-Way ANOVA", statistic=stat, p_value=p, significant=sig)

    def chi_square_test(self, contingency_table: np.ndarray) -> HypothesisTestResult:
        stat, p, dof, expected = stats.chi2_contingency(contingency_table)
        sig = p < self.alpha
        return HypothesisTestResult(test_name="Chi-Square Test", statistic=stat, p_value=p, significant=sig)

    def mann_whitney_u_test(self, data1: pd.Series, data2: pd.Series) -> HypothesisTestResult:
        stat, p = stats.mannwhitneyu(data1.dropna(), data2.dropna())
        sig = p < self.alpha
        return HypothesisTestResult(test_name="Mann-Whitney U Test", statistic=stat, p_value=p, significant=sig)
