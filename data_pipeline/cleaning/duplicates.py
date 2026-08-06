import pandas as pd

def drop_duplicates(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset)
