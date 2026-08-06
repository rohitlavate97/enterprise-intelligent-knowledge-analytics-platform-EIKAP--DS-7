from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LinearRegression

class StatisticalFeatureSelector:
    def select_by_anova_f(self, df_X: pd.DataFrame, y: pd.Series, k: int = 5) -> List[str]:
        df_clean = df_X.copy()
        df_clean['target'] = y
        df_clean = df_clean.dropna()
        X = df_clean.drop(columns=['target'])
        Y = df_clean['target']
        if X.empty: return []
        
        f_vals, p_vals = f_classif(X, Y)
        scores = pd.Series(f_vals, index=X.columns)
        scores = scores.fillna(0)
        return scores.nlargest(k).index.tolist()

    def calculate_vif(self, df_X: pd.DataFrame) -> Dict[str, float]:
        df_clean = df_X.dropna()
        if df_clean.empty or df_clean.shape[1] < 2: return {}
        
        vif_data = {}
        for col in df_clean.columns:
            y = df_clean[col].values
            X = df_clean.drop(columns=[col]).values
            
            if X.shape[1] == 0:
                vif_data[col] = 1.0
                continue
                
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            
            # VIF = 1 / (1 - R^2)
            if r2 == 1.0:
                vif = float('inf')
            else:
                vif = 1.0 / (1.0 - r2)
            vif_data[col] = vif
            
        return vif_data
