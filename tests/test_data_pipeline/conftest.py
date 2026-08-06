import pytest
import pandas as pd
import tempfile
import os
from data_pipeline.synthetic.generators import CustomerChurnGenerator, FraudDetectionGenerator
from data_pipeline.pipeline import ETLPipeline
from data_pipeline.validation.schema import SchemaDefinition

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "int_col": [1, 2, None, 4],
        "float_col": [1.1, None, 3.3, 4.4],
        "str_col": ["a", "b", "c", None],
        "bool_col": [True, False, True, False]
    })

@pytest.fixture
def sample_csv_path(sample_dataframe):
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    sample_dataframe.to_csv(path, index=False)
    yield path
    os.unlink(path)

@pytest.fixture
def sample_schema():
    return SchemaDefinition(columns={"int_col": "int64", "float_col": "float64"})

@pytest.fixture
def churn_generator():
    return CustomerChurnGenerator(seed=42)

@pytest.fixture
def fraud_generator():
    return FraudDetectionGenerator(seed=42)

@pytest.fixture
def etl_pipeline():
    return ETLPipeline("test_pipeline")
