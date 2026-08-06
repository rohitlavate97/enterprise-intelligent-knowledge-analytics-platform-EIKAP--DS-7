"""Shared configuration loading and management."""
import os
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class AppSettings(BaseModel):
    name: str = "EIKAP"
    version: str = "0.1.0"
    description: str = ""
    env: str = "development"
    debug: bool = False

class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: str = "json"
    handlers: List[str] = Field(default_factory=lambda: ["console"])
    file_path: Optional[str] = None

class DatabaseSettings(BaseModel):
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

class RedisSettings(BaseModel):
    url: str
    max_connections: int = 20

class MLflowSettings(BaseModel):
    tracking_uri: str
    artifact_root: str
    experiment_prefix: str = "eikap"

class ApiSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: List[str] = Field(default_factory=list)
    rate_limit: Dict[str, str] = Field(default_factory=dict)

class SecuritySettings(BaseModel):
    secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    allowed_upload_extensions: List[str] = Field(default_factory=list)
    max_upload_size_mb: int = 100

class ModuleSettings(BaseModel):
    compliance: Dict[str, bool] = Field(default_factory=dict)
    latency_targets_ms: Dict[str, int] = Field(default_factory=dict)

class Settings(BaseSettings):
    app: AppSettings
    logging: LoggingSettings
    database: DatabaseSettings
    redis: RedisSettings
    mlflow: MLflowSettings
    api: ApiSettings
    security: SecuritySettings
    modules: ModuleSettings

_settings_instance: Optional[Settings] = None
_settings_lock = Lock()

def _resolve_env_vars(config_dict: Any) -> Any:
    """Recursively resolve ${ENV_VAR:default} interpolation in YAML values."""
    if isinstance(config_dict, dict):
        return {k: _resolve_env_vars(v) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        return [_resolve_env_vars(v) for v in config_dict]
    elif isinstance(config_dict, str):
        pattern = re.compile(r'\$\{([^}^{]+)\}')
        
        def replace(match: re.Match) -> str:
            env_var = match.group(1)
            parts = env_var.split(':', 1)
            name = parts[0]
            default = parts[1] if len(parts) > 1 else ""
            return os.environ.get(name, default)
            
        return pattern.sub(replace, config_dict)
    return config_dict

def _merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

def get_settings() -> Settings:
    """Get the application settings singleton, initializing if necessary."""
    global _settings_instance
    if _settings_instance is not None:
        return _settings_instance

    with _settings_lock:
        if _settings_instance is not None:
            return _settings_instance

        env = os.environ.get("EIKAP_ENV", "development")
        config_dir = Path(__file__).parent.parent / "configs"
        
        default_path = config_dir / "default.yaml"
        env_path = config_dir / f"{env}.yaml"
        
        try:
            with open(default_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
                
            if env_path.exists():
                with open(env_path, 'r') as f:
                    env_data = yaml.safe_load(f) or {}
                config_data = _merge_dicts(config_data, env_data)
                
            resolved_data = _resolve_env_vars(config_data)
            _settings_instance = Settings(**resolved_data)
            return _settings_instance
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
