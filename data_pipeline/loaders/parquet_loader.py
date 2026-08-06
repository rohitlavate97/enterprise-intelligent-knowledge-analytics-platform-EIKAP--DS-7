"""Parquet data loader for EIKAP data pipeline."""
from pathlib import Path
from typing import Any, Dict, List, Union
import pandas as pd

from data_pipeline.loaders.base import BaseLoader
from shared.exceptions import DataLoadError


class ParquetLoader(BaseLoader):
    """Loader for Parquet columnar files."""

    def _get_supported_extensions(self) -> List[str]:
        return [".parquet"]

    def load(self, source: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
        path = self.validate_source(source)
        try:
            self.logger.info(f"Loading Parquet file {path.name}")
            df = pd.read_parquet(path, **kwargs)
            self.logger.info(f"Successfully loaded {path.name}: {len(df)} rows, {len(df.columns)} columns")
            return self._post_load_hook(df, path)
        except Exception as e:
            self.logger.error(f"Error loading Parquet file {path.name}: {str(e)}")
            raise DataLoadError(f"Failed to load Parquet file {path}: {str(e)}") from e

    def get_schema(self, source: Union[str, Path]) -> Dict[str, str]:
        """Return column names and string representation of dtypes."""
        df = self.load(source)
        return {col: str(dtype) for col, dtype in df.dtypes.items()}
