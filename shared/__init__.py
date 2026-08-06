"""Shared module components."""

from .config import get_settings, Settings
from .logging import get_logger
from .exceptions import (
    EIKAPBaseError, ConfigurationError, ValidationError, PipelineError,
    ModelTrainingError, ModelInferenceError, DataLoadError, ModuleComplianceError,
    AuthenticationError, AuthorizationError, RateLimitError, PromptInjectionError,
    LeakageDetectedError, GroundingViolationError, CalibrationError
)
from .types import (
    DataFrameType, ArrayType, PathLike, JSONDict, MetricsDict, MaturityLabel,
    PredictionResult, ModuleInfo
)
from .utils import (
    generate_request_id, ensure_directory, load_yaml, merge_dicts, timer,
    retry, validate_file_extension, hash_file, sizeof_fmt
)
from .dependency_injection import DIContainer, Lifetime, get_container, inject

__all__ = [
    "get_settings", "Settings",
    "get_logger",
    "EIKAPBaseError", "ConfigurationError", "ValidationError", "PipelineError",
    "ModelTrainingError", "ModelInferenceError", "DataLoadError", "ModuleComplianceError",
    "AuthenticationError", "AuthorizationError", "RateLimitError", "PromptInjectionError",
    "LeakageDetectedError", "GroundingViolationError", "CalibrationError",
    "DataFrameType", "ArrayType", "PathLike", "JSONDict", "MetricsDict", "MaturityLabel",
    "PredictionResult", "ModuleInfo",
    "generate_request_id", "ensure_directory", "load_yaml", "merge_dicts", "timer",
    "retry", "validate_file_extension", "hash_file", "sizeof_fmt",
    "DIContainer", "inject"
]
