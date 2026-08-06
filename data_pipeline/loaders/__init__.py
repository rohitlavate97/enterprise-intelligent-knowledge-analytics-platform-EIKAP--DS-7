"""
Data Loaders module for EIKAP data pipeline.
Exports all loader classes and the LoaderFactory.
"""

from data_pipeline.loaders.base import BaseLoader, LoaderFactory
from data_pipeline.loaders.csv_loader import CSVLoader
from data_pipeline.loaders.excel_loader import ExcelLoader
from data_pipeline.loaders.json_loader import JSONLoader
from data_pipeline.loaders.parquet_loader import ParquetLoader

# Register default loaders with the factory
LoaderFactory.register(".csv", CSVLoader)
LoaderFactory.register(".tsv", CSVLoader)
LoaderFactory.register(".xlsx", ExcelLoader)
LoaderFactory.register(".xls", ExcelLoader)
LoaderFactory.register(".json", JSONLoader)
LoaderFactory.register(".jsonl", JSONLoader)
LoaderFactory.register(".parquet", ParquetLoader)

__all__ = [
    "BaseLoader",
    "LoaderFactory",
    "CSVLoader",
    "ExcelLoader",
    "JSONLoader",
    "ParquetLoader"
]
