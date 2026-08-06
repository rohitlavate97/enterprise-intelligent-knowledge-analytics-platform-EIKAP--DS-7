"""
Schema validation module for EIKAP data pipeline.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
import yaml
import copy

from data_pipeline.validation.data_validator import ValidationResult, DataValidator
from shared.logging import get_logger


@dataclass
class ColumnSchema:
    """Definition of a single column's schema."""
    name: str
    dtype: str
    nullable: bool = True
    unique: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    regex_pattern: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d = {
            "name": self.name,
            "dtype": self.dtype,
            "nullable": self.nullable,
            "unique": self.unique,
        }
        if self.min_value is not None: d["min_value"] = self.min_value
        if self.max_value is not None: d["max_value"] = self.max_value
        if self.allowed_values is not None: d["allowed_values"] = self.allowed_values
        if self.regex_pattern is not None: d["regex_pattern"] = self.regex_pattern
        if self.description: d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnSchema":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            dtype=data["dtype"],
            nullable=data.get("nullable", True),
            unique=data.get("unique", False),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            allowed_values=data.get("allowed_values"),
            regex_pattern=data.get("regex_pattern"),
            description=data.get("description", "")
        )


@dataclass
class SchemaDefinition:
    """Definition of a tabular dataset schema."""
    name: str
    version: str
    columns: List[ColumnSchema]
    min_rows: int = 0
    max_rows: Optional[int] = None
    strict: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
            "strict": self.strict,
            "columns": [col.to_dict() for col in self.columns]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaDefinition":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            min_rows=data.get("min_rows", 0),
            max_rows=data.get("max_rows"),
            strict=data.get("strict", False),
            columns=[ColumnSchema.from_dict(c) for c in data.get("columns", [])]
        )

    def to_yaml(self, path: str) -> None:
        """Serialize schema to YAML file."""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> "SchemaDefinition":
        """Load schema from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class SchemaValidator:
    """Validates DataFrames against SchemaDefinitions."""
    
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.data_validator = DataValidator()

    def validate(self, df: pd.DataFrame, schema: SchemaDefinition) -> ValidationResult:
        """
        Validate a DataFrame against a schema definition.
        
        Args:
            df: DataFrame to validate.
            schema: SchemaDefinition to validate against.
            
        Returns:
            Comprehensive ValidationResult.
        """
        errors = []
        warnings = []
        
        # 1. Row count validation
        row_count = len(df)
        if row_count < schema.min_rows:
            errors.append({
                "rule": "min_rows",
                "message": f"Dataset has {row_count} rows, minimum required is {schema.min_rows}.",
                "severity": "error"
            })
        if schema.max_rows is not None and row_count > schema.max_rows:
            errors.append({
                "rule": "max_rows",
                "message": f"Dataset has {row_count} rows, maximum allowed is {schema.max_rows}.",
                "severity": "error"
            })

        # 2. Column presence validation
        df_columns = set(df.columns)
        schema_columns = {col.name: col for col in schema.columns}
        schema_col_names = set(schema_columns.keys())
        
        missing_cols = schema_col_names - df_columns
        for col in missing_cols:
            errors.append({
                "column": col,
                "rule": "missing_column",
                "message": f"Required column '{col}' is missing.",
                "severity": "error"
            })
            
        extra_cols = df_columns - schema_col_names
        if schema.strict and extra_cols:
            for col in extra_cols:
                errors.append({
                    "column": col,
                    "rule": "extra_column",
                    "message": f"Column '{col}' is not defined in strict schema.",
                    "severity": "error"
                })

        # 3. Construct rules for DataValidator
        validation_rules = []
        for col_name, col_schema in schema_columns.items():
            if col_name not in df_columns:
                continue
                
            # Type constraint (handled by coercion attempt if possible, but strict checking here)
            validation_rules.append({
                "type": "dtype",
                "column": col_name,
                "expected_dtype": col_schema.dtype
            })
            
            # Nullable constraint
            if not col_schema.nullable:
                validation_rules.append({
                    "type": "not_null",
                    "columns": [col_name],
                    "threshold": 0.0
                })
                
            # Unique constraint
            if col_schema.unique:
                validation_rules.append({
                    "type": "unique",
                    "columns": [col_name]
                })
                
            # Range constraint
            if col_schema.min_value is not None or col_schema.max_value is not None:
                min_val = col_schema.min_value if col_schema.min_value is not None else float('-inf')
                max_val = col_schema.max_value if col_schema.max_value is not None else float('inf')
                validation_rules.append({
                    "type": "range",
                    "column": col_name,
                    "min_val": min_val,
                    "max_val": max_val
                })
                
            # Allowed values constraint
            if col_schema.allowed_values is not None:
                validation_rules.append({
                    "type": "category",
                    "column": col_name,
                    "allowed_values": col_schema.allowed_values
                })
                
            # Regex constraint
            if col_schema.regex_pattern is not None:
                validation_rules.append({
                    "type": "regex",
                    "column": col_name,
                    "pattern": col_schema.regex_pattern
                })

        # Apply rules
        if validation_rules:
            rule_results = self.data_validator.validate_all(df, validation_rules)
            errors.extend(rule_results.errors)
            warnings.extend(rule_results.warnings)

        is_valid = len(errors) == 0
        summary = {
            "total_rows": row_count,
            "schema_name": schema.name,
            "schema_version": schema.version,
            "error_count": len(errors),
            "warning_count": len(warnings)
        }
        
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, summary=summary)

    def infer_schema(self, df: pd.DataFrame, name: str = "inferred", version: str = "1.0") -> SchemaDefinition:
        """
        Auto-generate a schema from a DataFrame.
        
        Args:
            df: DataFrame to infer schema from.
            name: Name of the schema.
            version: Version string.
            
        Returns:
            Inferred SchemaDefinition.
        """
        columns = []
        for col_name in df.columns:
            series = df[col_name]
            dtype_str = str(series.dtype)
            
            # Infer nullable
            nullable = series.isnull().any()
            
            # Infer unique
            unique = series.is_unique and not nullable # If there are nulls, typically not unique in a strict sense
            
            # Infer ranges for numeric
            min_val = None
            max_val = None
            if pd.api.types.is_numeric_dtype(series):
                min_val = float(series.min()) if not pd.isna(series.min()) else None
                max_val = float(series.max()) if not pd.isna(series.max()) else None
                
            columns.append(ColumnSchema(
                name=str(col_name),
                dtype=dtype_str,
                nullable=bool(nullable),
                unique=bool(unique),
                min_value=min_val,
                max_value=max_val
            ))
            
        return SchemaDefinition(
            name=name,
            version=version,
            columns=columns,
            min_rows=0,
            strict=False
        )

    def compare_schemas(self, schema1: SchemaDefinition, schema2: SchemaDefinition) -> Dict[str, Any]:
        """
        Compare two schemas and return the differences.
        
        Args:
            schema1: First schema.
            schema2: Second schema.
            
        Returns:
            Dictionary detailing differences.
        """
        diff = {
            "missing_in_2": [],
            "missing_in_1": [],
            "type_mismatches": [],
            "constraint_changes": []
        }
        
        s1_cols = {c.name: c for c in schema1.columns}
        s2_cols = {c.name: c for c in schema2.columns}
        
        # Missing columns
        for col in s1_cols:
            if col not in s2_cols:
                diff["missing_in_2"].append(col)
        for col in s2_cols:
            if col not in s1_cols:
                diff["missing_in_1"].append(col)
                
        # Compare overlapping
        for col in set(s1_cols.keys()).intersection(set(s2_cols.keys())):
            c1 = s1_cols[col]
            c2 = s2_cols[col]
            
            if c1.dtype != c2.dtype:
                diff["type_mismatches"].append({
                    "column": col,
                    "schema1_type": c1.dtype,
                    "schema2_type": c2.dtype
                })
                
            # Basic constraints comparison
            constraints_diff = {}
            if c1.nullable != c2.nullable: constraints_diff["nullable"] = (c1.nullable, c2.nullable)
            if c1.unique != c2.unique: constraints_diff["unique"] = (c1.unique, c2.unique)
            if c1.min_value != c2.min_value: constraints_diff["min_value"] = (c1.min_value, c2.min_value)
            if c1.max_value != c2.max_value: constraints_diff["max_value"] = (c1.max_value, c2.max_value)
            
            if constraints_diff:
                diff["constraint_changes"].append({
                    "column": col,
                    "changes": constraints_diff
                })
                
        return diff
