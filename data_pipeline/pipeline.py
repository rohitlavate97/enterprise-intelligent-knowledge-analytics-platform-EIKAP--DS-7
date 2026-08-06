import time
import logging
import pandas as pd
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from data_pipeline.validation.schema import SchemaDefinition, validate_schema

@dataclass
class PipelineStep:
    name: str
    function: Callable
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class PipelineResult:
    success: bool
    data: pd.DataFrame
    steps_executed: List[Dict[str, Any]]
    total_duration_ms: float
    summary: str

class ETLPipeline:
    def __init__(self, name: str):
        self.name = name
        self.steps: List[PipelineStep] = []
        self.logger = logging.getLogger(f"ETLPipeline.{name}")
        self.logger.setLevel(logging.INFO)

    def add_step(self, name: str, function: Callable, params: Dict = None, enabled: bool = True) -> 'ETLPipeline':
        if params is None:
            params = {}
        self.steps.append(PipelineStep(name=name, function=function, params=params, enabled=enabled))
        return self

    def remove_step(self, name: str) -> 'ETLPipeline':
        self.steps = [step for step in self.steps if step.name != name]
        return self

    def run(self, df: pd.DataFrame) -> PipelineResult:
        current_df = df.copy()
        steps_executed = []
        start_time_total = time.time()
        
        for step in self.steps:
            if not step.enabled:
                continue
                
            step_start = time.time()
            rows_before, cols_before = current_df.shape
            
            try:
                current_df = step.function(current_df, **step.params)
            except Exception as e:
                self.logger.error(f"Step {step.name} failed: {str(e)}")
                return PipelineResult(
                    success=False,
                    data=current_df,
                    steps_executed=steps_executed,
                    total_duration_ms=(time.time() - start_time_total) * 1000,
                    summary=f"Pipeline failed at step {step.name}: {str(e)}"
                )
                
            step_end = time.time()
            rows_after, cols_after = current_df.shape
            
            steps_executed.append({
                "name": step.name,
                "duration_ms": (step_end - step_start) * 1000,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "columns_before": cols_before,
                "columns_after": cols_after
            })
            
        total_duration = (time.time() - start_time_total) * 1000
        return PipelineResult(
            success=True,
            data=current_df,
            steps_executed=steps_executed,
            total_duration_ms=total_duration,
            summary=f"Pipeline '{self.name}' completed successfully in {total_duration:.2f}ms"
        )

    def run_with_validation(self, df: pd.DataFrame, schema: Optional[SchemaDefinition] = None) -> PipelineResult:
        if schema:
            validate_schema(df, schema)
        result = self.run(df)
        if schema and result.success:
            validate_schema(result.data, schema)
        return result

    def dry_run(self, df: pd.DataFrame) -> List[str]:
        return [step.name for step in self.steps if step.enabled]

    def get_step_names(self) -> List[str]:
        return [step.name for step in self.steps]

    def reset(self) -> 'ETLPipeline':
        self.steps = []
        return self

class PipelineBuilder:
    @classmethod
    def build_standard_cleaning_pipeline(cls) -> ETLPipeline:
        from data_pipeline.cleaning.duplicates import drop_duplicates
        from data_pipeline.cleaning.missing import fill_missing
        from data_pipeline.cleaning.outliers import cap_outliers
        
        pipeline = ETLPipeline("Standard Cleaning")
        pipeline.add_step("dedup", drop_duplicates)
        pipeline.add_step("missing_fill", fill_missing, {"strategy_numeric": "median"})
        pipeline.add_step("outlier_cap", cap_outliers, {"method": "iqr", "factor": 1.5})
        return pipeline

    @classmethod
    def build_full_etl_pipeline(cls) -> ETLPipeline:
        def mock_validate(df): return df
        def mock_clean(df): return df
        def mock_feature_engineer(df): return df
        def mock_profile(df): return df
        def mock_optimize(df): return df
        
        pipeline = ETLPipeline("Full ETL")
        pipeline.add_step("validate", mock_validate)
        pipeline.add_step("clean", mock_clean)
        pipeline.add_step("feature_engineer", mock_feature_engineer)
        pipeline.add_step("profile", mock_profile)
        pipeline.add_step("optimize", mock_optimize)
        return pipeline
