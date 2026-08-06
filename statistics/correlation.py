from typing import Dict, Any
import pandas as pd
import scipy.stats as stats
import numpy as np

class CorrelationAnalyzer:
    def pearson_correlation(self, data1: pd.Series, data2: pd.Series) -> Dict[str, float]:
        df = pd.DataFrame({'x': data1, 'y': data2}).dropna()
        if len(df) < 2: return {'r': 0.0, 'p': 1.0}
        r, p = stats.pearsonr(df['x'], df['y'])
        return {'r': r, 'p': p}

    def spearman_correlation(self, data1: pd.Series, data2: pd.Series) -> Dict[str, float]:
        df = pd.DataFrame({'x': data1, 'y': data2}).dropna()
        if len(df) < 2: return {'r': 0.0, 'p': 1.0}
        r, p = stats.spearmanr(df['x'], df['y'])
        return {'r': r, 'p': p}

    def partial_correlation(self, df: pd.DataFrame, x: str, y: str, covar: list) -> Dict[str, float]:
        # Simple partial correlation using OLS residuals
        df_clean = df[[x, y] + covar].dropna()
        if len(df_clean) < 3: return {'r': 0.0, 'p': 1.0}
        
        # We need to regress x on covar and y on covar, then correlate residuals
        from sklearn.linear_model import LinearRegression
        
        X_cov = df_clean[covar].values
        
        model_x = LinearRegression().fit(X_cov, df_clean[x].values)
        res_x = df_clean[x].values - model_x.predict(X_cov)
        
        model_y = LinearRegression().fit(X_cov, df_clean[y].values)
        res_y = df_clean[y].values - model_y.predict(X_cov)
        
        r, p = stats.pearsonr(res_x, res_y)
        return {'r': r, 'p': p}
