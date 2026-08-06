import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from shared.logging import get_logger

class OutlierDetector:
    def __init__(self):
        self.logger = get_logger(__name__)

    def _get_target_columns(self, df: pd.DataFrame, columns: Optional[List[str]]) -> List[str]:
        if columns is not None:
            return [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def detect_iqr(self, df: pd.DataFrame, columns: Optional[List[str]] = None, factor: float = 1.5) -> Dict[str, Dict[str, Any]]:
        results = {}
        if df.empty:
            return results
            
        target_cols = self._get_target_columns(df, columns)
        
        for col in target_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (factor * iqr)
            upper_bound = q3 + (factor * iqr)
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            results[col] = {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_count": len(outliers),
                "outlier_indices": outliers.index.tolist(),
                "outlier_percentage": (len(outliers) / len(df)) * 100
            }
            
        return results

    def detect_zscore(self, df: pd.DataFrame, columns: Optional[List[str]] = None, threshold: float = 3.0) -> Dict[str, Dict[str, Any]]:
        results = {}
        if df.empty:
            return results
            
        target_cols = self._get_target_columns(df, columns)
        
        for col in target_cols:
            mean = df[col].mean()
            std = df[col].std()
            
            if pd.isna(std) or std == 0:
                continue
                
            z_scores = np.abs((df[col] - mean) / std)
            outliers = df[z_scores > threshold]
            
            results[col] = {
                "lower_bound": float(mean - threshold * std),
                "upper_bound": float(mean + threshold * std),
                "outlier_count": len(outliers),
                "outlier_indices": outliers.index.tolist(),
                "outlier_percentage": (len(outliers) / len(df)) * 100
            }
            
        return results

    def detect_modified_zscore(self, df: pd.DataFrame, columns: Optional[List[str]] = None, threshold: float = 3.5) -> Dict[str, Dict[str, Any]]:
        results = {}
        if df.empty:
            return results
            
        target_cols = self._get_target_columns(df, columns)
        
        for col in target_cols:
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            
            if pd.isna(mad) or mad == 0:
                continue
                
            mod_z_scores = 0.6745 * np.abs(df[col] - median) / mad
            outliers = df[mod_z_scores > threshold]
            
            bound_diff = (threshold * mad) / 0.6745
            
            results[col] = {
                "lower_bound": float(median - bound_diff),
                "upper_bound": float(median + bound_diff),
                "outlier_count": len(outliers),
                "outlier_indices": outliers.index.tolist(),
                "outlier_percentage": (len(outliers) / len(df)) * 100
            }
            
        return results

    def cap_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        if df.empty:
            return df.copy()
            
        df_capped = df.copy()
        
        if method == 'iqr':
            detection_res = self.detect_iqr(df_capped, columns, factor)
        elif method == 'zscore':
            detection_res = self.detect_zscore(df_capped, columns, factor)
        elif method == 'modified_zscore':
            detection_res = self.detect_modified_zscore(df_capped, columns, factor)
        else:
            self.logger.warning(f"Unknown detection method: {method}")
            return df_capped
            
        for col, stats in detection_res.items():
            df_capped[col] = df_capped[col].clip(lower=stats['lower_bound'], upper=stats['upper_bound'])
            
        return df_capped

    def remove_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        if df.empty:
            return df.copy()
            
        if method == 'iqr':
            detection_res = self.detect_iqr(df, columns, factor)
        elif method == 'zscore':
            detection_res = self.detect_zscore(df, columns, factor)
        elif method == 'modified_zscore':
            detection_res = self.detect_modified_zscore(df, columns, factor)
        else:
            self.logger.warning(f"Unknown detection method: {method}")
            return df.copy()
            
        outlier_indices = set()
        for col, stats in detection_res.items():
            outlier_indices.update(stats['outlier_indices'])
            
        return df.drop(index=list(outlier_indices))

    def get_outlier_report(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        if df.empty:
            return {}
            
        report = {}
        iqr_res = self.detect_iqr(df, columns, factor=1.5)
        zscore_res = self.detect_zscore(df, columns, threshold=3.0)
        mod_zscore_res = self.detect_modified_zscore(df, columns, threshold=3.5)
        
        target_cols = self._get_target_columns(df, columns)
        
        for col in target_cols:
            report[col] = {
                "iqr_outliers": iqr_res.get(col, {}).get("outlier_count", 0),
                "zscore_outliers": zscore_res.get(col, {}).get("outlier_count", 0),
                "modified_zscore_outliers": mod_zscore_res.get(col, {}).get("outlier_count", 0),
                "iqr_percentage": iqr_res.get(col, {}).get("outlier_percentage", 0.0)
            }
            
        return report
