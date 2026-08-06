"""CSV and TSV data loader for EIKAP data pipeline."""
from pathlib import Path
from typing import Any, Generator, List, Union
import pandas as pd

from data_pipeline.loaders.base import BaseLoader
from shared.exceptions import DataLoadError


class CSVLoader(BaseLoader):
    """Loader for CSV and TSV files with encoding auto-detection and chunking."""

    def _get_supported_extensions(self) -> List[str]:
        return [".csv", ".tsv"]

    def load(self, source: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
        """
        Load CSV data into a pandas DataFrame.
        Attempts utf-8, latin-1, and cp1252 encodings automatically if not specified.
        """
        path = self.validate_source(source)
        
        # Auto-detect delimiter for TSV
        if path.suffix.lower() == ".tsv" and "sep" not in kwargs and "delimiter" not in kwargs:
            kwargs["sep"] = "\t"
            
        encodings = [kwargs.pop("encoding", None)] if "encoding" in kwargs else ["utf-8", "latin-1", "cp1252"]
        encodings = [e for e in encodings if e is not None]
        
        last_exception = None
        for enc in encodings:
            try:
                self.logger.info(f"Attempting to load {path.name} with encoding={enc}")
                df = pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
                self.logger.info(f"Successfully loaded {path.name}: {len(df)} rows, {len(df.columns)} columns")
                return self._post_load_hook(df, path)
            except (UnicodeDecodeError, Exception) as e:
                last_exception = e
                continue
                
        self.logger.error(f"Failed to load {path.name} with tested encodings: {last_exception}")
        raise DataLoadError(f"Could not load CSV file {path}: {last_exception}") from last_exception

    def load_chunked(self, source: Union[str, Path], chunksize: int = 10000, **kwargs: Any) -> Generator[pd.DataFrame, None, None]:
        """Yield chunks of DataFrame for large CSV files."""
        path = self.validate_source(source)
        for chunk in pd.read_csv(path, chunksize=chunksize, **kwargs):
            yield chunk
