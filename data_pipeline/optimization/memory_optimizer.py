import pandas as pd
import numpy as np
from typing import Dict, Any
from shared.logging import get_logger

class MemoryOptimizer:
    def __init__(self):
        self.logger = get_logger(__name__)

    def optimize(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        if df.empty:
            return df.copy()
            
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        
        optimized = self.downcast_integers(df)
        optimized = self.downcast_floats(optimized)
        optimized = self.categorize_strings(optimized)
        
        end_mem = optimized.memory_usage(deep=True).sum() / 1024**2
        
        if verbose:
            savings = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
            self.logger.info(f"Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({savings:.1f}% reduction)")
            
        return optimized

    def downcast_integers(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
            
        res = df.copy()
        int_cols = res.select_dtypes(include=['int']).columns
        
        for col in int_cols:
            c_min = res[col].min()
            c_max = res[col].max()
            
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                res[col] = res[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                res[col] = res[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                res[col] = res[col].astype(np.int32)
                
        return res

    def downcast_floats(self, df: pd.DataFrame, precision: str = 'float32') -> pd.DataFrame:
        if df.empty:
            return df.copy()
            
        res = df.copy()
        float_cols = res.select_dtypes(include=['float']).columns
        
        for col in float_cols:
            res[col] = res[col].astype(precision)
            
        return res

    def categorize_strings(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        if df.empty:
            return df.copy()
            
        res = df.copy()
        obj_cols = res.select_dtypes(include=['object']).columns
        
        num_rows = len(res)
        if num_rows == 0:
            return res
            
        for col in obj_cols:
            num_unique = res[col].nunique()
            if num_unique / num_rows < threshold:
                res[col] = res[col].astype('category')
                
        return res

    def get_memory_usage(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"total_mb": 0.0, "per_column": {}, "optimization_potential_mb": 0.0}
            
        mem_series = df.memory_usage(deep=True)
        total_mem = mem_series.sum() / 1024**2
        
        per_col = {}
        for col in df.columns:
            mem = mem_series[col] / 1024**2
            per_col[col] = {
                "dtype": str(df[col].dtype),
                "memory_mb": float(mem),
                "percentage": float(mem / total_mem * 100) if total_mem > 0 else 0.0
            }
            
        # Estimate savings
        savings_est = self.estimate_savings(df)
        opt_potential = total_mem * (savings_est.get("estimated_savings_percentage", 0.0) / 100)
        
        return {
            "total_mb": float(total_mem),
            "per_column": per_col,
            "optimization_potential_mb": float(opt_potential)
        }

    def estimate_savings(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {"estimated_savings_percentage": 0.0}
            
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        
        # very rough estimate without full conversion
        estimated_end_mem = start_mem
        
        # strings
        obj_cols = df.select_dtypes(include=['object']).columns
        num_rows = len(df)
        if num_rows > 0:
            for col in obj_cols:
                num_unique = df[col].nunique()
                if num_unique / num_rows < 0.5:
                    col_mem = df[col].memory_usage(deep=True) / 1024**2
                    cat_mem = (num_rows * 2 + num_unique * 50) / 1024**2  # rough approx
                    estimated_end_mem -= max(0, col_mem - cat_mem)
                    
        # ints
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            col_mem = df[col].memory_usage(deep=True) / 1024**2
            estimated_end_mem -= col_mem * 0.75 # assume int16 avg
            
        # floats
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            col_mem = df[col].memory_usage(deep=True) / 1024**2
            estimated_end_mem -= col_mem * 0.5
            
        savings_pct = 100 * (start_mem - estimated_end_mem) / start_mem if start_mem > 0 else 0.0
        
        return {
            "estimated_savings_percentage": float(savings_pct)
        }
