"""Utility functions."""
import uuid
import time
from pathlib import Path
import yaml
import hashlib
from typing import Dict, Any, Callable, TypeVar, cast
from functools import wraps
from tenacity import retry as tenacity_retry, stop_after_attempt, wait_exponential

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())

def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with open(path, 'r') as f:
        return cast(Dict[str, Any], yaml.safe_load(f) or {})

def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

class timer:
    """Context manager for latency measurement."""
    def __init__(self, name: str, logger: Any = None):
        self.name = name
        self.logger = logger
        self.start_time = 0.0

    def __enter__(self) -> "timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed = (time.perf_counter() - self.start_time) * 1000
        msg = f"{self.name} took {elapsed:.2f} ms"
        if self.logger:
            self.logger.info(msg, latency_ms=elapsed)
        else:
            print(msg)

def retry(attempts: int = 3, min_wait: float = 1.0, max_wait: float = 10.0) -> Callable[[F], F]:
    """Decorator to retry a function with exponential backoff."""
    return tenacity_retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=min_wait, max=max_wait)
    )

def validate_file_extension(filename: str, allowed: list[str]) -> bool:
    """Check if file extension is allowed."""
    ext = Path(filename).suffix.lower()
    return ext in allowed

def hash_file(path: str | Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def sizeof_fmt(num_bytes: int) -> str:
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0 # type: ignore
    return f"{num_bytes:.1f} PB"
