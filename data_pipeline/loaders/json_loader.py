"""JSON and JSONL data loader for EIKAP data pipeline."""
from pathlib import Path
from typing import Any, List, Union
import pandas as pd

from data_pipeline.loaders.base import BaseLoader
from shared.exceptions import DataLoadError


class JSONLoader(BaseLoader):
    """Loader for JSON and JSON Lines (.jsonl) files."""

    def _get_supported_extensions(self) -> List[str]:
        return [".json", ".jsonl"]

    def load(self, source: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
        path = self.validate_source(source)
        
        # Auto-detect lines parameter for .jsonl
        if path.suffix.lower() == ".jsonl" and "lines" not in kwargs:
            kwargs["lines"] = True

        try:
            self.logger.info(f"Loading JSON file {path.name}")
            df = pd.read_json(path, **kwargs)
            self.logger.info(f"Successfully loaded {path.name}: {len(df)} rows, {len(df.columns)} columns")
            return self._post_load_hook(df, path)
        except Exception as e:
            self.logger.error(f"Error loading JSON file {path.name}: {str(e)}")
            raise DataLoadError(f"Failed to load JSON file {path}: {str(e)}") from e
