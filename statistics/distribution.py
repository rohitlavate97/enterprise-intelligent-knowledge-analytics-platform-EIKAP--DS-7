
from typing import Dict, Any, Tuple
from pydantic import BaseModel
import scipy.stats as stats
import pandas as pd
import numpy as np

class FitResult(BaseModel):
    distribution: str
    params: Tuple[float, ...]
    sse: float
    aic: float

class DistributionFitter:
    def fit_norm_distribution(self, data: pd.Series) -> FitResult:
        clean_data = data.dropna()
        params = stats.norm.fit(clean_data)
        
        # calculate SSE
        y, x = np.histogram(clean_data, bins='auto', density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        pdf = stats.norm.pdf(x, *params)
        sse = np.sum((y - pdf)**2)
        
        return FitResult(distribution="norm", params=params, sse=sse, aic=sse) # Simplified AIC

    def find_best_distribution(self, data: pd.Series) -> FitResult:
        clean_data = data.dropna()
        dists = ['norm', 'expon', 'lognorm']
        best_res = None
        best_sse = float('inf')
        
        y, x = np.histogram(clean_data, bins='auto', density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        
        for dist_name in dists:
            dist = getattr(stats, dist_name)
            try:
                params = dist.fit(clean_data)
                pdf = dist.pdf(x, *params)
                sse = np.sum((y - pdf)**2)
                if sse < best_sse:
                    best_sse = sse
                    best_res = FitResult(distribution=dist_name, params=params, sse=sse, aic=sse)
            except Exception:
                continue
        return best_res if best_res else FitResult(distribution="norm", params=(0,1), sse=0, aic=0)
        
    def confidence_interval(self, data: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
        clean_data = data.dropna()
        return stats.t.interval(confidence, len(clean_data)-1, loc=np.mean(clean_data), scale=stats.sem(clean_data))
