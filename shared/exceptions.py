"""Centralized exception hierarchy for EIKAP."""
from typing import Optional, Dict, Any

class EIKAPBaseError(Exception):
    """Base exception for all EIKAP errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: str = "INTERNAL_ERROR", http_status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        self.http_status_code = http_status_code

class ConfigurationError(EIKAPBaseError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "CONFIG_ERROR", 500)

class ValidationError(EIKAPBaseError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "VALIDATION_ERROR", 400)

class PipelineError(EIKAPBaseError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "PIPELINE_ERROR", 500)

class ModelTrainingError(EIKAPBaseError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "TRAINING_ERROR", 500)

class ModelInferenceError(EIKAPBaseError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "INFERENCE_ERROR", 500)

class DataLoadError(EIKAPBaseError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "DATA_LOAD_ERROR", 400)

class ModuleComplianceError(EIKAPBaseError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "COMPLIANCE_ERROR", 403)

class AuthenticationError(EIKAPBaseError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "AUTH_ERROR", 401)

class AuthorizationError(EIKAPBaseError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "FORBIDDEN_ERROR", 403)

class RateLimitError(EIKAPBaseError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "RATE_LIMIT_ERROR", 429)

class PromptInjectionError(EIKAPBaseError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "PROMPT_INJECTION_ERROR", 400)

class LeakageDetectedError(EIKAPBaseError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "LEAKAGE_DETECTED", 400)

class GroundingViolationError(EIKAPBaseError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "GROUNDING_VIOLATION", 400)

class CalibrationError(EIKAPBaseError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, "CALIBRATION_ERROR", 500)
