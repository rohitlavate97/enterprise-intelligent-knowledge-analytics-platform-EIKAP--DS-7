import pandas as pd

def fill_missing(df: pd.DataFrame, strategy_numeric: str = 'mean', strategy_categorical: str = 'mode') -> pd.DataFrame:
    df_out = df.copy()
    for col in df_out.columns:
        if pd.api.types.is_numeric_dtype(df_out[col]):
            if strategy_numeric == 'mean':
                val = df_out[col].mean()
            elif strategy_numeric == 'median':
                val = df_out[col].median()
            else:
                val = 0
            df_out[col] = df_out[col].fillna(val)
        else:
            if strategy_categorical == 'mode':
                val = df_out[col].mode()[0] if not df_out[col].mode().empty else "Missing"
            else:
                val = "Missing"
            df_out[col] = df_out[col].fillna(val)
    return df_out
