"""Shared type definitions."""
from typing import TypeVar, Union, Dict, Any, Literal, Optional, TypedDict
from pathlib import Path
import pandas as pd
import numpy as np

DataFrameType = pd.DataFrame
ArrayType = np.ndarray
PathLike = Union[str, Path]
JSONDict = Dict[str, Any]
MetricsDict = Dict[str, float]
MaturityLabel = Literal['standard', 'restricted']

class PredictionResult(TypedDict):
    prediction: Any
    confidence: float
    explanation: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]

class ModuleInfo(TypedDict):
    name: str
    version: str
    description: str
    dependencies: list[str]
