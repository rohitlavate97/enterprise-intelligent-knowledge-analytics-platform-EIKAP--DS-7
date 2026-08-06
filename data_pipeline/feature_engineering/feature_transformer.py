import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from shared.logging import get_logger

class FeatureTransformer:
    def __init__(self):
        self.logger = get_logger(__name__)

    def create_date_features(self, df: pd.DataFrame, column: str, drop_original: bool = False) -> pd.DataFrame:
        if df.empty or column not in df.columns:
            return df.copy()
            
        res = df.copy()
        try:
            series = pd.to_datetime(res[column])
            res[f"{column}_year"] = series.dt.year
            res[f"{column}_month"] = series.dt.month
            res[f"{column}_day"] = series.dt.day
            res[f"{column}_day_of_week"] = series.dt.dayofweek
            res[f"{column}_hour"] = series.dt.hour
            res[f"{column}_quarter"] = series.dt.quarter
            res[f"{column}_is_weekend"] = series.dt.dayofweek.isin([5, 6]).astype(int)
            res[f"{column}_is_month_start"] = series.dt.is_month_start.astype(int)
            res[f"{column}_is_month_end"] = series.dt.is_month_end.astype(int)
            res[f"{column}_days_since_epoch"] = (series - pd.Timestamp("1970-01-01")).dt.days
            
            if drop_original:
                res = res.drop(columns=[column])
        except Exception as e:
            self.logger.error(f"Error creating date features for {column}: {str(e)}")
            
        return res

    def create_interaction_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        res = df.copy()
        if len(columns) < 2:
            return res
            
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                if col1 in res.columns and col2 in res.columns:
                    res[f"{col1}_mul_{col2}"] = res[col1] * res[col2]
        return res

    def create_polynomial_features(self, df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
        res = df.copy()
        for col in columns:
            if col in res.columns:
                for d in range(2, degree + 1):
                    res[f"{col}_deg_{d}"] = res[col] ** d
        return res

    def create_binned_features(self, df: pd.DataFrame, column: str, bins: int = 5, strategy: str = 'quantile') -> pd.DataFrame:
        res = df.copy()
        if column not in res.columns:
            return res
            
        new_col = f"{column}_binned"
        if strategy == 'quantile':
            res[new_col] = pd.qcut(res[column], q=bins, labels=False, duplicates='drop')
        else:
            res[new_col] = pd.cut(res[column], bins=bins, labels=False)
            
        return res

    def create_lag_features(self, df: pd.DataFrame, column: str, lags: List[int], sort_column: Optional[str] = None) -> pd.DataFrame:
        res = df.copy()
        if column not in res.columns:
            return res
            
        if sort_column and sort_column in res.columns:
            res = res.sort_values(by=sort_column)
            
        for lag in lags:
            res[f"{column}_lag_{lag}"] = res[column].shift(lag)
            
        if sort_column:
            res = res.sort_index()
            
        return res

    def create_rolling_features(self, df: pd.DataFrame, column: str, windows: List[int], agg_funcs: List[str] = ['mean', 'std'], sort_column: Optional[str] = None) -> pd.DataFrame:
        res = df.copy()
        if column not in res.columns:
            return res
            
        if sort_column and sort_column in res.columns:
            res = res.sort_values(by=sort_column)
            
        for window in windows:
            for func in agg_funcs:
                res[f"{column}_rolling_{window}_{func}"] = res[column].rolling(window=window, min_periods=1).agg(func)
                
        if sort_column:
            res = res.sort_index()
            
        return res

    def encode_categorical(self, df: pd.DataFrame, columns: List[str], method: str = 'label') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        res = df.copy()
        encoding_map = {}
        
        for col in columns:
            if col not in res.columns:
                continue
                
            if method == 'label':
                cats = res[col].astype('category').cat.categories
                mapping = {cat: i for i, cat in enumerate(cats)}
                res[f"{col}_encoded"] = res[col].map(mapping)
                encoding_map[col] = {"method": method, "mapping": mapping}
            elif method == 'frequency':
                freqs = res[col].value_counts(normalize=True).to_dict()
                res[f"{col}_encoded"] = res[col].map(freqs)
                encoding_map[col] = {"method": method, "mapping": freqs}
                
        return res, encoding_map

    def create_text_features(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        res = df.copy()
        if column not in res.columns:
            return res
            
        s = res[column].astype(str).fillna("")
        res[f"{column}_length"] = s.str.len()
        res[f"{column}_word_count"] = s.str.split().str.len()
        res[f"{column}_avg_word_length"] = res[f"{column}_length"] / res[f"{column}_word_count"].replace(0, 1)
        res[f"{column}_has_digits"] = s.str.contains(r'\d').astype(int)
        res[f"{column}_has_special_chars"] = s.str.contains(r'[^A-Za-z0-9\s]').astype(int)
        
        uppercase_count = s.apply(lambda x: sum(1 for c in x if c.isupper()))
        res[f"{column}_uppercase_ratio"] = uppercase_count / res[f"{column}_length"].replace(0, 1)
        
        return res

    def create_ratio_features(self, df: pd.DataFrame, numerator_cols: List[str], denominator_cols: List[str]) -> pd.DataFrame:
        res = df.copy()
        for num_col in numerator_cols:
            for den_col in denominator_cols:
                if num_col in res.columns and den_col in res.columns:
                    new_col = f"{num_col}_ratio_{den_col}"
                    res[new_col] = np.where(res[den_col] == 0, 0, res[num_col] / res[den_col])
        return res
