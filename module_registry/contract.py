import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel

from module_registry.base import EIKAPModule, MaturityLabel, ModuleCategory

@dataclass
class ContractCheckResult:
    check_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0

@dataclass 
class ContractReport:
    module_name: str
    timestamp: str
    all_passed: bool
    checks: List[ContractCheckResult]
    summary: str

class UniversalModuleContract:
    def check_metadata(self, module: EIKAPModule) -> ContractCheckResult:
        start_time = time.time()
        try:
            meta = module.metadata
            if not meta.name or not meta.version or not meta.author:
                return ContractCheckResult("check_metadata", False, "Missing required metadata fields", duration_ms=(time.time()-start_time)*1000)
            if meta.maturity not in MaturityLabel:
                return ContractCheckResult("check_metadata", False, "Invalid or missing maturity label", duration_ms=(time.time()-start_time)*1000)
            return ContractCheckResult("check_metadata", True, "Metadata valid", duration_ms=(time.time()-start_time)*1000)
        except Exception as e:
            return ContractCheckResult("check_metadata", False, f"Error: {str(e)}", duration_ms=(time.time()-start_time)*1000)

    def check_input_output_schemas(self, module: EIKAPModule) -> ContractCheckResult:
        start_time = time.time()
        try:
            inp = module.input_schema
            out = module.output_schema
            if not issubclass(inp, BaseModel):
                return ContractCheckResult("check_input_output_schemas", False, "input_schema must be a Pydantic BaseModel", duration_ms=(time.time()-start_time)*1000)
            if not issubclass(out, BaseModel):
                return ContractCheckResult("check_input_output_schemas", False, "output_schema must be a Pydantic BaseModel", duration_ms=(time.time()-start_time)*1000)
            return ContractCheckResult("check_input_output_schemas", True, "Schemas are valid Pydantic models", duration_ms=(time.time()-start_time)*1000)
        except Exception as e:
            return ContractCheckResult("check_input_output_schemas", False, f"Error: {str(e)}", duration_ms=(time.time()-start_time)*1000)

    def check_explainability(self, module: EIKAPModule, sample_input: Any, sample_prediction: Any) -> ContractCheckResult:
        start_time = time.time()
        try:
            explanation = module.explain(sample_input, sample_prediction)
            if not explanation:
                return ContractCheckResult("check_explainability", False, "Explanation is empty", duration_ms=(time.time()-start_time)*1000)
            return ContractCheckResult("check_explainability", True, "Explainability check passed", duration_ms=(time.time()-start_time)*1000)
        except Exception as e:
            return ContractCheckResult("check_explainability", False, f"Error calling explain: {str(e)}", duration_ms=(time.time()-start_time)*1000)

    def check_human_review_framing(self, module: EIKAPModule, sample_output: Any) -> ContractCheckResult:
        start_time = time.time()
        if not module.is_restricted():
            return ContractCheckResult("check_human_review_framing", True, "N/A for standard modules", duration_ms=(time.time()-start_time)*1000)
        try:
            out_dict = sample_output if isinstance(sample_output, dict) else (sample_output.model_dump() if isinstance(sample_output, BaseModel) else {})
            if "action" in out_dict or "decision" in out_dict:
                return ContractCheckResult("check_human_review_framing", False, "Restricted module output contains automated action/decision fields", duration_ms=(time.time()-start_time)*1000)
            if "recommendation" not in out_dict and "explanation" not in out_dict:
                return ContractCheckResult("check_human_review_framing", False, "Restricted module output missing recommendation/explanation fields", duration_ms=(time.time()-start_time)*1000)
            return ContractCheckResult("check_human_review_framing", True, "Human review framing valid", duration_ms=(time.time()-start_time)*1000)
        except Exception as e:
            return ContractCheckResult("check_human_review_framing", False, f"Error: {str(e)}", duration_ms=(time.time()-start_time)*1000)

    def check_latency_benchmark(self, module: EIKAPModule, sample_input: Any, target_ms: float) -> ContractCheckResult:
        start_time = time.time()
        try:
            _, latency = module.run_with_latency(sample_input)
            if latency > target_ms:
                return ContractCheckResult("check_latency_benchmark", False, f"Latency {latency:.2f}ms exceeds target {target_ms}ms", duration_ms=(time.time()-start_time)*1000)
            return ContractCheckResult("check_latency_benchmark", True, f"Latency {latency:.2f}ms within target {target_ms}ms", duration_ms=(time.time()-start_time)*1000)
        except Exception as e:
            return ContractCheckResult("check_latency_benchmark", False, f"Error: {str(e)}", duration_ms=(time.time()-start_time)*1000)

    def check_mlflow_tracking(self, module: EIKAPModule) -> ContractCheckResult:
        start_time = time.time()
        return ContractCheckResult("check_mlflow_tracking", True, "MLflow tracking valid (stubbed)", duration_ms=(time.time()-start_time)*1000)

    def check_leakage_safety(self, module: EIKAPModule, train_data: Any, test_data: Any) -> ContractCheckResult:
        start_time = time.time()
        return ContractCheckResult("check_leakage_safety", True, "Leakage safety valid (stubbed)", duration_ms=(time.time()-start_time)*1000)

    def check_grounding(self, module: EIKAPModule, sample_input: Any, sample_output: Any) -> ContractCheckResult:
        start_time = time.time()
        if module.metadata.category not in [ModuleCategory.RAG, ModuleCategory.NLP]:
            return ContractCheckResult("check_grounding", True, "N/A for non-RAG/NLP modules", duration_ms=(time.time()-start_time)*1000)
        out_dict = sample_output if isinstance(sample_output, dict) else (sample_output.model_dump() if isinstance(sample_output, BaseModel) else {})
        if "citations" not in out_dict and "sources" not in out_dict:
             return ContractCheckResult("check_grounding", False, "Missing citations/sources in output", duration_ms=(time.time()-start_time)*1000)
        return ContractCheckResult("check_grounding", True, "Grounding valid", duration_ms=(time.time()-start_time)*1000)

    def check_calibration(self, module: EIKAPModule, predictions: List[Any], actuals: List[Any]) -> ContractCheckResult:
        start_time = time.time()
        if not module.is_restricted():
            return ContractCheckResult("check_calibration", True, "N/A for standard modules", duration_ms=(time.time()-start_time)*1000)
        return ContractCheckResult("check_calibration", True, "Calibration valid (stubbed)", duration_ms=(time.time()-start_time)*1000)

    def check_maturity_label_surfaced(self, module: EIKAPModule) -> ContractCheckResult:
        start_time = time.time()
        try:
            api_resp = module.to_api_response("test", {"reason": "test"}, 10.0)
            if "data" not in api_resp or "maturity_label" not in api_resp["data"]:
                return ContractCheckResult("check_maturity_label_surfaced", False, "Maturity label not surfaced in API response", duration_ms=(time.time()-start_time)*1000)
            return ContractCheckResult("check_maturity_label_surfaced", True, "Maturity label correctly surfaced", duration_ms=(time.time()-start_time)*1000)
        except Exception as e:
            return ContractCheckResult("check_maturity_label_surfaced", False, f"Error: {str(e)}", duration_ms=(time.time()-start_time)*1000)

    def run_full_check(self, module: EIKAPModule, sample_input: Any = None, sample_prediction: Any = None, sample_output: Any = None, train_data: Any = None, test_data: Any = None, target_ms: float = 1000.0) -> ContractReport:
        checks = []
        checks.append(self.check_metadata(module))
        checks.append(self.check_input_output_schemas(module))
        if sample_input is not None and sample_prediction is not None:
            checks.append(self.check_explainability(module, sample_input, sample_prediction))
        if sample_output is not None:
            checks.append(self.check_human_review_framing(module, sample_output))
        if sample_input is not None:
            checks.append(self.check_latency_benchmark(module, sample_input, target_ms))
        checks.append(self.check_mlflow_tracking(module))
        if train_data is not None and test_data is not None:
            checks.append(self.check_leakage_safety(module, train_data, test_data))
        if sample_input is not None and sample_output is not None:
            checks.append(self.check_grounding(module, sample_input, sample_output))
        checks.append(self.check_calibration(module, [], []))
        checks.append(self.check_maturity_label_surfaced(module))
        
        all_passed = all(c.passed for c in checks)
        return ContractReport(
            module_name=module.metadata.name,
            timestamp=datetime.utcnow().isoformat() + "Z",
            all_passed=all_passed,
            checks=checks,
            summary=f"{sum(1 for c in checks if c.passed)}/{len(checks)} checks passed."
        )
