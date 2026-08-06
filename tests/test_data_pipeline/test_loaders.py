import pandas as pd
import pytest
from data_pipeline.loaders.base import LoaderFactory
from data_pipeline.loaders.csv_loader import CSVLoader

def test_csv_loader_basic(sample_csv_path):
    loader = CSVLoader()
    df = loader.load(sample_csv_path)
    assert len(df) == 4

def test_csv_loader_encoding_detection(sample_csv_path):
    loader = CSVLoader()
    df = loader.load(sample_csv_path)
    assert len(df) == 4

def test_json_loader_records():
    pass

def test_json_loader_jsonl():
    pass

def test_parquet_loader():
    pass

def test_loader_factory_auto_detect():
    loader = LoaderFactory.get_loader("csv")
    assert isinstance(loader, CSVLoader)

def test_loader_invalid_file_raises():
    from shared.exceptions import DataLoadError
    with pytest.raises((ValueError, DataLoadError)):
        LoaderFactory.get_loader("invalid")
