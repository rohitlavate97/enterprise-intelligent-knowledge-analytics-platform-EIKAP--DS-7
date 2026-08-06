import time
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import os
from shared.config import get_settings
from shared.logging import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up EIKAP API...")
    yield
    # Shutdown
    logger.info("Shutting down EIKAP API...")

app = FastAPI(
    title="EIKAP API",
    description="Enterprise Intelligent Knowledge & Analytics Platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "version": "0.1.0",
        "environment": os.getenv("EIKAP_ENV", "development"),
        "timestamp": time.time()
    }

@app.get("/ready", response_model=Dict[str, bool])
async def readiness_check() -> Dict[str, bool]:
    # Placeholder logic for checking internal services (e.g., DB, Redis)
    return {"ready": True}

@app.get("/", response_model=Dict[str, Any])
async def root() -> Dict[str, Any]:
    return {
        "platform": "EIKAP",
        "message": "Welcome to the Enterprise Intelligent Knowledge & Analytics Platform"
    }
