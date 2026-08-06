import pandas as pd
from data_pipeline.pipeline import ETLPipeline, PipelineBuilder
from data_pipeline.validation.schema import SchemaDefinition

def test_pipeline_basic_execution(sample_dataframe):
    p = ETLPipeline("test")
    def mock_step(df): return df
    p.add_step("s1", mock_step)
    res = p.run(sample_dataframe)
    assert res.success

def test_pipeline_step_metrics(sample_dataframe):
    p = ETLPipeline("test")
    def mock_step(df): return df
    p.add_step("s1", mock_step)
    res = p.run(sample_dataframe)
    assert len(res.steps_executed) == 1
    assert "duration_ms" in res.steps_executed[0]

def test_pipeline_with_validation(sample_dataframe):
    p = ETLPipeline("test")
    def mock_step(df): return df
    p.add_step("s1", mock_step)
    schema = SchemaDefinition(columns={"int_col": "int64"})
    res = p.run_with_validation(sample_dataframe, schema)
    assert res.success

def test_pipeline_builder_standard():
    p = PipelineBuilder.build_standard_cleaning_pipeline()
    assert len(p.steps) == 3

def test_pipeline_builder_full_etl():
    p = PipelineBuilder.build_full_etl_pipeline()
    assert len(p.steps) == 5

def test_pipeline_dry_run():
    p = ETLPipeline("test")
    def mock_step(df): return df
    p.add_step("s1", mock_step)
    names = p.dry_run(pd.DataFrame())
    assert names == ["s1"]
