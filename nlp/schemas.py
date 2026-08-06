from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class NLPInput(BaseModel):
    task: str = Field(default="sentiment")
    text: str = Field(default="")
    text_list: List[str] = Field(default_factory=list)
    job_description: str = Field(default="")

class NLPOutput(BaseModel):
    task: str
    results: Dict[str, Any]
    execution_time_ms: float
