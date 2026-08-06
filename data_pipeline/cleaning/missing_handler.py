import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from shared.logging import get_logger

class MissingValueHandler:
    def __init__(self):
        self.logger = get_logger(__name__)

    def analyze_missing(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {
                "per_column": {},
                "total_missing_cells": 0,
                "total_cells": 0,
                "overall_missing_percentage": 0.0
            }
            
        total_cells = df.size
        total_missing = int(df.isna().sum().sum())
        
        per_column = {}
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            if missing_count > 0:
                per_column[col] = {
                    "count": missing_count,
                    "percentage": (missing_count / len(df)) * 100,
                    "dtype": str(df[col].dtype),
                    "pattern": "missing" if missing_count > 0 else "complete"
                }
                
        return {
            "per_column": per_column,
            "total_missing_cells": total_missing,
            "total_cells": total_cells,
            "overall_missing_percentage": (total_missing / total_cells * 100) if total_cells > 0 else 0.0
        }

    def drop_missing(self, df: pd.DataFrame, axis: int = 0, threshold: float = 0.5) -> pd.DataFrame:
        if df.empty:
            return df.copy()
            
        if axis == 0:
            thresh_count = int(len(df.columns) * (1 - threshold))
            return df.dropna(axis=0, thresh=thresh_count)
        else:
            thresh_count = int(len(df) * (1 - threshold))
            return df.dropna(axis=1, thresh=thresh_count)

    def fill_numeric(self, df: pd.DataFrame, strategy: str = 'median', columns: Optional[List[str]] = None) -> pd.DataFrame:
        if df.empty:
            return df.copy()
            
        df_filled = df.copy()
        target_cols = columns if columns is not None else df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in target_cols:
            if col not in df.columns or df[col].isna().sum() == 0:
                continue
                
            if strategy == 'median':
                val = df[col].median()
                df_filled[col] = df_filled[col].fillna(val)
            elif strategy == 'mean':
                val = df[col].mean()
                df_filled[col] = df_filled[col].fillna(val)
            elif strategy == 'zero':
                df_filled[col] = df_filled[col].fillna(0)
            elif strategy == 'interpolate':
                df_filled[col] = df_filled[col].interpolate(method='linear')
            else:
                self.logger.warning(f"Unknown numeric fill strategy: {strategy}. Skipping column {col}.")
                
        return df_filled

    def fill_categorical(self, df: pd.DataFrame, strategy: str = 'mode', columns: Optional[List[str]] = None) -> pd.DataFrame:
        if df.empty:
            return df.copy()
            
        df_filled = df.copy()
        target_cols = columns if columns is not None else df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in target_cols:
            if col not in df.columns or df[col].isna().sum() == 0:
                continue
                
            if strategy == 'mode':
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df_filled[col] = df_filled[col].fillna(mode_val.iloc[0])
            elif strategy == 'constant' or strategy == 'UNKNOWN':
                df_filled[col] = df_filled[col].fillna('UNKNOWN')
            else:
                self.logger.warning(f"Unknown categorical fill strategy: {strategy}. Skipping column {col}.")
                
        return df_filled

    def fill_temporal(self, df: pd.DataFrame, strategy: str = 'ffill', columns: Optional[List[str]] = None) -> pd.DataFrame:
        if df.empty:
            return df.copy()
            
        df_filled = df.copy()
        target_cols = columns if columns is not None else df.select_dtypes(include=['datetime64', 'timedelta64']).columns.tolist()
        
        for col in target_cols:
            if col not in df.columns or df[col].isna().sum() == 0:
                continue
                
            if strategy == 'ffill':
                df_filled[col] = df_filled[col].ffill()
            elif strategy == 'bfill':
                df_filled[col] = df_filled[col].bfill()
            elif strategy == 'interpolate':
                df_filled[col] = df_filled[col].interpolate(method='linear')
            else:
                self.logger.warning(f"Unknown temporal fill strategy: {strategy}. Skipping column {col}.")
                
        return df_filled

    def smart_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
            
        df_filled = df.copy()
        
        num_cols = df_filled.select_dtypes(include=[np.number]).columns.tolist()
        df_filled = self.fill_numeric(df_filled, strategy='median', columns=num_cols)
        
        cat_cols = df_filled.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        df_filled = self.fill_categorical(df_filled, strategy='mode', columns=cat_cols)
        
        dt_cols = df_filled.select_dtypes(include=['datetime64', 'timedelta64']).columns.tolist()
        df_filled = self.fill_temporal(df_filled, strategy='ffill', columns=dt_cols)
        
        return df_filled

    def create_missing_indicators(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        if df.empty:
            return df.copy()
            
        df_ind = df.copy()
        target_cols = columns if columns is not None else df.columns.tolist()
        
        for col in target_cols:
            if col in df.columns:
                if df[col].isna().any():
                    df_ind[f"{col}_is_missing"] = df[col].isna().astype(int)
                    
        return df_ind
