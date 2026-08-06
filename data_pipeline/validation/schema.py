import pandas as pd
from typing import Dict, Any, Optional
from pydantic import BaseModel

class SchemaDefinition(BaseModel):
    columns: Dict[str, str]

def validate_schema(df: pd.DataFrame, schema: SchemaDefinition) -> bool:
    for col, dtype in schema.columns.items():
        if col not in df.columns:
            raise ValueError(f"Column missing: {col}")
    return True
