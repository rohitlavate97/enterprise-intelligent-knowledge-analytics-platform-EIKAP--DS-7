"""Structured logging setup using structlog."""
import logging
import sys
from typing import Any, Dict
import structlog
from pathlib import Path
from pydantic import BaseModel
from .config import get_settings

def setup_logging() -> None:
    """Initialize structured logging configuration."""
    settings = get_settings()
    log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.logging.format == "json" else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handlers = []
    
    if "console" in settings.logging.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(console_handler)

    if "file" in settings.logging.handlers and settings.logging.file_path:
        log_path = Path(settings.logging.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )

def get_logger(module_name: str) -> structlog.BoundLogger:
    """Get a structured logger bound with the module name."""
    if not structlog.is_configured():
        setup_logging()
    return structlog.get_logger(module_name)
