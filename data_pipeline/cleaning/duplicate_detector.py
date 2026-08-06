import pandas as pd
import numpy as np
import difflib
from typing import Optional, List, Dict, Any, Tuple
from shared.logging import get_logger

class DuplicateDetector:
    def __init__(self):
        self.logger = get_logger(__name__)

    def find_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
        if df.empty:
            return df.copy()
        return df[df.duplicated(subset=subset, keep=keep)]

    def find_near_duplicates(self, df: pd.DataFrame, columns: List[str], threshold: float = 0.85) -> List[Tuple[int, int, float]]:
        if df.empty or len(df) < 2:
            return []
            
        near_duplicates = []
        indices = df.index.tolist()
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1 = indices[i]
                idx2 = indices[j]
                
                total_sim = 0.0
                valid_cols = 0
                
                for col in columns:
                    val1 = str(df.loc[idx1, col])
                    val2 = str(df.loc[idx2, col])
                    
                    if pd.isna(df.loc[idx1, col]) or pd.isna(df.loc[idx2, col]):
                        continue
                        
                    sim = difflib.SequenceMatcher(None, val1, val2).ratio()
                    total_sim += sim
                    valid_cols += 1
                
                if valid_cols > 0:
                    avg_sim = total_sim / valid_cols
                    if avg_sim >= threshold:
                        near_duplicates.append((idx1, idx2, avg_sim))
                        
        return near_duplicates

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
        if df.empty:
            return df.copy()
            
        initial_len = len(df)
        cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
        removed_count = initial_len - len(cleaned_df)
        self.logger.info(f"Removed {removed_count} duplicate rows.")
        return cleaned_df

    def get_duplicate_stats(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        if df.empty:
            return {
                "total_rows": 0,
                "duplicate_count": 0,
                "duplicate_percentage": 0.0,
                "first_duplicate_indices": []
            }
            
        total_rows = len(df)
        duplicates = df[df.duplicated(subset=subset, keep='first')]
        duplicate_count = len(duplicates)
        
        return {
            "total_rows": total_rows,
            "duplicate_count": duplicate_count,
            "duplicate_percentage": (duplicate_count / total_rows * 100) if total_rows > 0 else 0.0,
            "first_duplicate_indices": duplicates.index.tolist()
        }

    def deduplicate_by_priority(self, df: pd.DataFrame, subset: List[str], priority_column: str, ascending: bool = False) -> pd.DataFrame:
        if df.empty:
            return df.copy()
            
        sorted_df = df.sort_values(by=priority_column, ascending=ascending)
        deduplicated = sorted_df.drop_duplicates(subset=subset, keep='first')
        return deduplicated.sort_index()
