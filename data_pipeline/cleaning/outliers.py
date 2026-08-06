import pandas as pd
import numpy as np

def cap_outliers(df: pd.DataFrame, method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
    df_out = df.copy()
    for col in df_out.columns:
        if pd.api.types.is_numeric_dtype(df_out[col]):
            if method == 'iqr':
                q1 = df_out[col].quantile(0.25)
                q3 = df_out[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - factor * iqr
                upper = q3 + factor * iqr
                df_out[col] = np.clip(df_out[col], lower, upper)
            elif method == 'zscore':
                mean = df_out[col].mean()
                std = df_out[col].std()
                lower = mean - factor * std
                upper = mean + factor * std
                df_out[col] = np.clip(df_out[col], lower, upper)
    return df_out
