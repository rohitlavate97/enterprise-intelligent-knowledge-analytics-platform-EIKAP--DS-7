"""Base data loader interface for EIKAP data pipeline."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
import datetime
import pandas as pd
from shared.logging import get_logger
from shared.exceptions import DataLoadError


class BaseLoader(ABC):
    """Abstract Base Class for all file loaders in EIKAP."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def load(self, source: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
        """Load data from the source path into a pandas DataFrame."""
        pass

    @abstractmethod
    def _get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions (e.g. ['.csv', '.tsv'])."""
        pass

    def load_batch(self, sources: List[Union[str, Path]], **kwargs: Any) -> pd.DataFrame:
        """Load multiple files and concatenate them into a single DataFrame."""
        dfs = []
        for src in sources:
            df = self.load(src, **kwargs)
            dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def validate_source(self, source: Union[str, Path]) -> Path:
        """Validate that file exists and extension matches supported types."""
        path = Path(source)
        if not path.exists():
            raise DataLoadError(f"File not found: {path}")
        if not path.is_file():
            raise DataLoadError(f"Source is not a file: {path}")
        ext = path.suffix.lower()
        supported = self._get_supported_extensions()
        if ext not in supported:
            raise DataLoadError(
                f"Unsupported extension '{ext}' for {self.__class__.__name__}. Supported: {supported}"
            )
        return path

    def _post_load_hook(self, df: pd.DataFrame, source: Path) -> pd.DataFrame:
        """Optional hook for post-load processing."""
        return df

    def get_file_info(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Return metadata about the source file."""
        path = self.validate_source(source)
        stat = path.stat()
        return {
            "path": str(path.absolute()),
            "name": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "modified_time": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }


class LoaderFactory:
    """Factory class to manage and instantiate data loaders by file extension."""

    _registry: Dict[str, Type[BaseLoader]] = {}

    @classmethod
    def register(cls, extension: str, loader_class: Type[BaseLoader]) -> None:
        """Register a loader class for a specific extension (e.g. '.csv')."""
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        cls._registry[ext] = loader_class

    @classmethod
    def create(cls, source: Union[str, Path]) -> BaseLoader:
        """Auto-detect file type and return appropriate loader instance."""
        path = Path(source)
        ext = path.suffix.lower()
        if ext not in cls._registry:
            raise DataLoadError(
                f"No loader registered for extension '{ext}'. Registered: {list(cls._registry.keys())}"
            )
        loader_cls = cls._registry[ext]
        return loader_cls()

    @classmethod
    def get_loader(cls, file_type: str) -> BaseLoader:
        """Alias for create / get loader by file extension or name."""
        file_type = file_type.lower()
        if not file_type.startswith("."):
            file_type = f".{file_type}"
        return cls.create(f"dummy{file_type}")
