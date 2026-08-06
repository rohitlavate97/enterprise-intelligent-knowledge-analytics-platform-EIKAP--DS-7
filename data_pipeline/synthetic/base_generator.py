from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from typing import List

class BaseSyntheticGenerator(ABC):
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    @abstractmethod
    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass

    def _add_noise(self, series: pd.Series, noise_level: float = 0.05) -> pd.Series:
        if not pd.api.types.is_numeric_dtype(series):
            return series
        std = series.std() if series.std() != 0 else 1.0
        noise = self.rng.normal(0, noise_level * std, len(series))
        return series + noise

    def _introduce_missing(self, df: pd.DataFrame, columns: List[str], missing_rate: float = 0.05) -> pd.DataFrame:
        df_out = df.copy()
        for col in columns:
            if col in df_out.columns:
                mask = self.rng.random(len(df_out)) < missing_rate
                df_out.loc[mask, col] = np.nan
        return df_out

    def _add_duplicates(self, df: pd.DataFrame, duplicate_rate: float = 0.02) -> pd.DataFrame:
        n_duplicates = int(len(df) * duplicate_rate)
        if n_duplicates == 0:
            return df
        indices = self.rng.choice(df.index, n_duplicates, replace=True)
        duplicates = df.loc[indices].copy()
        return pd.concat([df, duplicates], ignore_index=True)
