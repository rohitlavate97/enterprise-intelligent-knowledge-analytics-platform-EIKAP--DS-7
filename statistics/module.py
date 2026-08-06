import time
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field
import pandas as pd
from module_registry.base import EIKAPModule, ModuleMetadata, ModuleCategory, MaturityLabel

from statistics.hypothesis import HypothesisTester
from statistics.normality import NormalityTester
from statistics.correlation import CorrelationAnalyzer
from statistics.distribution import DistributionFitter
from statistics.feature_selection import StatisticalFeatureSelector
from statistics.reports import StatisticalReportGenerator

class StatisticalInput(BaseModel):
    target_col: str = ""
    columns: List[str] = Field(default_factory=list)
    test_type: str = "auto"
    alpha: float = 0.05

class StatisticalOutput(BaseModel):
    normality_results: Dict[str, Any]
    correlation_matrix: Dict[str, Any]
    hypothesis_tests: Dict[str, Any]
    best_distributions: Dict[str, Any]
    vif_scores: Dict[str, Any]
    summary_report: str
    execution_time_ms: float

class StatisticalAnalysisModule(EIKAPModule):
    @property
    def metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            name="statistical_analysis",
            version="0.1.0",
            description="Statistical Analysis, Hypothesis Testing & Distribution Fitting Engine",
            category=ModuleCategory.STATISTICS,
            maturity=MaturityLabel.STANDARD,
            author="EIKAP Team"
        )

    @property
    def input_schema(self) -> Type[BaseModel]:
        return StatisticalInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return StatisticalOutput

    def train(self, data: Any, **kwargs) -> Dict[str, Any]:
        return {"status": "trained"}

    def predict(self, input_data: Any, data: pd.DataFrame = None, **kwargs) -> Any:
        start = time.time()
        
        normality_tester = NormalityTester(alpha=input_data.alpha)
        correlation_analyzer = CorrelationAnalyzer()
        dist_fitter = DistributionFitter()
        feat_selector = StatisticalFeatureSelector()
        reporter = StatisticalReportGenerator()

        cols = input_data.columns if input_data.columns else (data.select_dtypes(include=['number']).columns.tolist() if data is not None else [])
        
        normality_results = {}
        best_dists = {}
        if data is not None:
            for c in cols:
                if c in data.columns:
                    normality_results[c] = normality_tester.auto_select_test(data[c]).model_dump()
                    best_dists[c] = dist_fitter.find_best_distribution(data[c]).model_dump()
                    
        corr_matrix = {}
        if data is not None and len(cols) >= 2:
            corr_matrix = correlation_analyzer.pearson_correlation(data[cols[0]], data[cols[1]])
            
        vif_scores = feat_selector.calculate_vif(data[cols]) if data is not None else {}
        
        results = {
            "normality_results": normality_results,
            "hypothesis_tests": {}
        }
        
        report = reporter.generate_statistical_report(results)
        
        exec_time = (time.time() - start) * 1000
        
        return StatisticalOutput(
            normality_results=normality_results,
            correlation_matrix=corr_matrix,
            hypothesis_tests={},
            best_distributions=best_dists,
            vif_scores=vif_scores,
            summary_report=report,
            execution_time_ms=exec_time
        )

    def explain(self, input_data: Any, prediction: Any, **kwargs) -> Dict[str, Any]:
        return {"explanation": "p-values, effect sizes, and recommendations"}

    def evaluate(self, test_data: Any, **kwargs) -> Dict[str, float]:
        return {"power": 0.8}

    def health_check(self) -> Dict[str, Any]:
        try:
            import scipy
            return {"status": "ok", "scipy_version": scipy.__version__}
        except ImportError:
            return {"status": "error", "message": "Missing dependencies"}
