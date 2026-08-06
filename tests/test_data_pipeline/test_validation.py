import pandas as pd
import pytest
from data_pipeline.validation.schema import SchemaDefinition, validate_schema

def test_not_null_validation():
    pass

def test_range_validation():
    pass

def test_schema_validation_pass():
    df = pd.DataFrame({"a": [1]})
    schema = SchemaDefinition(columns={"a": "int"})
    assert validate_schema(df, schema)

def test_schema_validation_fail():
    df = pd.DataFrame({"b": [1]})
    schema = SchemaDefinition(columns={"a": "int"})
    with pytest.raises(ValueError):
        validate_schema(df, schema)

def test_schema_inference():
    pass
