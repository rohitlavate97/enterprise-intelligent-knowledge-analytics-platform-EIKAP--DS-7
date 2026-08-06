
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import scipy.stats as stats
import pandas as pd

class NormalityTestResult(BaseModel):
    test_name: str
    statistic: float
    p_value: float
    is_normal: bool

class NormalityTester:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def shapiro_wilk(self, data: pd.Series) -> NormalityTestResult:
        stat, p = stats.shapiro(data.dropna())
        is_normal = p > self.alpha
        return NormalityTestResult(test_name="Shapiro-Wilk", statistic=stat, p_value=p, is_normal=is_normal)
        
    def auto_select_test(self, data: pd.Series) -> NormalityTestResult:
        # For small sample size use Shapiro, else D'Agostino's K-squared test
        clean_data = data.dropna()
        if len(clean_data) < 5000:
            return self.shapiro_wilk(clean_data)
        else:
            stat, p = stats.normaltest(clean_data)
            is_normal = p > self.alpha
            return NormalityTestResult(test_name="D'Agostino's K-squared test", statistic=stat, p_value=p, is_normal=is_normal)
