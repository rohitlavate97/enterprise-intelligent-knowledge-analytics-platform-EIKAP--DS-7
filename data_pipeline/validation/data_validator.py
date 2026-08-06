"""
Data Validation module for EIKAP data pipeline.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
import re

from shared.logging import get_logger
from shared.exceptions import ValidationError


@dataclass
class ValidationResult:
    """Result of a data validation operation."""
    is_valid: bool = True
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> str:
        """Generate a formatted text report of the validation results."""
        lines = []
        lines.append("=== Validation Report ===")
        lines.append(f"Status: {'VALID' if self.is_valid else 'INVALID'}")
        
        if self.summary:
            lines.append("\n--- Summary ---")
            for k, v in self.summary.items():
                lines.append(f"{k}: {v}")
                
        if self.errors:
            lines.append("\n--- Errors ---")
            for i, err in enumerate(self.errors, 1):
                col = err.get('column', 'N/A')
                rule = err.get('rule', 'N/A')
                msg = err.get('message', '')
                lines.append(f"{i}. Column '{col}' (Rule: {rule}) - {msg}")
                
        if self.warnings:
            lines.append("\n--- Warnings ---")
            for i, warn in enumerate(self.warnings, 1):
                col = warn.get('column', 'N/A')
                rule = warn.get('rule', 'N/A')
                msg = warn.get('message', '')
                lines.append(f"{i}. Column '{col}' (Rule: {rule}) - {msg}")
                
        return "\n".join(lines)


class DataValidator:
    """Validator for data content rules."""
    
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def _create_result(self, is_valid: bool, errors: List[Dict], warnings: List[Dict]) -> ValidationResult:
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def validate_not_null(self, df: pd.DataFrame, columns: List[str], threshold: float = 0.0) -> ValidationResult:
        """
        Check that null percentage per column is below or equal to threshold.
        
        Args:
            df: DataFrame to validate.
            columns: List of columns to check.
            threshold: Maximum allowed fraction of nulls (0.0 to 1.0). Default 0.0 (no nulls).
        """
        errors = []
        total_rows = len(df)
        
        if total_rows == 0:
            return self._create_result(True, [], [])
            
        for col in columns:
            if col not in df.columns:
                errors.append({
                    "column": col,
                    "rule": "not_null",
                    "message": f"Column '{col}' not found in DataFrame.",
                    "severity": "error"
                })
                continue
                
            null_count = df[col].isnull().sum()
            null_fraction = null_count / total_rows
            
            if null_fraction > threshold:
                errors.append({
                    "column": col,
                    "rule": "not_null",
                    "message": f"Null fraction {null_fraction:.4f} exceeds threshold {threshold:.4f} "
                               f"({null_count} nulls out of {total_rows}).",
                    "severity": "error"
                })
                
        return self._create_result(len(errors) == 0, errors, [])

    def validate_unique(self, df: pd.DataFrame, columns: List[str]) -> ValidationResult:
        """
        Check that values in the specified columns are unique.
        
        Args:
            df: DataFrame to validate.
            columns: List of columns to check.
        """
        errors = []
        
        for col in columns:
            if col not in df.columns:
                errors.append({
                    "column": col,
                    "rule": "unique",
                    "message": f"Column '{col}' not found in DataFrame.",
                    "severity": "error"
                })
                continue
                
            duplicated = df[col].duplicated()
            dup_count = duplicated.sum()
            
            if dup_count > 0:
                errors.append({
                    "column": col,
                    "rule": "unique",
                    "message": f"Found {dup_count} duplicate values in column '{col}'.",
                    "severity": "error"
                })
                
        return self._create_result(len(errors) == 0, errors, [])

    def validate_range(self, df: pd.DataFrame, column: str, min_val: Any, max_val: Any) -> ValidationResult:
        """
        Check that numeric values in a column are within a specified range.
        
        Args:
            df: DataFrame to validate.
            column: Column to check.
            min_val: Minimum allowed value (inclusive).
            max_val: Maximum allowed value (inclusive).
        """
        errors = []
        
        if column not in df.columns:
            errors.append({
                "column": column,
                "rule": "range",
                "message": f"Column '{column}' not found in DataFrame.",
                "severity": "error"
            })
            return self._create_result(False, errors, [])
            
        # Ignore nulls for range validation
        valid_data = df[column].dropna()
        
        out_of_bounds = valid_data[(valid_data < min_val) | (valid_data > max_val)]
        count_out = len(out_of_bounds)
        
        if count_out > 0:
            errors.append({
                "column": column,
                "rule": "range",
                "message": f"Found {count_out} values outside range [{min_val}, {max_val}].",
                "severity": "error"
            })
            
        return self._create_result(len(errors) == 0, errors, [])

    def validate_regex(self, df: pd.DataFrame, column: str, pattern: str) -> ValidationResult:
        """
        Check that string values in a column match a regular expression.
        
        Args:
            df: DataFrame to validate.
            column: Column to check.
            pattern: Regex pattern string.
        """
        errors = []
        
        if column not in df.columns:
            errors.append({
                "column": column,
                "rule": "regex",
                "message": f"Column '{column}' not found in DataFrame.",
                "severity": "error"
            })
            return self._create_result(False, errors, [])
            
        valid_data = df[column].dropna().astype(str)
        regex = re.compile(pattern)
        
        # Finding elements that do NOT match
        mismatches = valid_data[~valid_data.apply(lambda x: bool(regex.match(x)))]
        count_mismatch = len(mismatches)
        
        if count_mismatch > 0:
            errors.append({
                "column": column,
                "rule": "regex",
                "message": f"Found {count_mismatch} values that do not match pattern '{pattern}'.",
                "severity": "error"
            })
            
        return self._create_result(len(errors) == 0, errors, [])

    def validate_dtype(self, df: pd.DataFrame, column: str, expected_dtype: str) -> ValidationResult:
        """
        Check that a column has the expected pandas dtype.
        
        Args:
            df: DataFrame to validate.
            column: Column to check.
            expected_dtype: String representation of expected dtype (e.g., 'int64').
        """
        errors = []
        
        if column not in df.columns:
            errors.append({
                "column": column,
                "rule": "dtype",
                "message": f"Column '{column}' not found in DataFrame.",
                "severity": "error"
            })
            return self._create_result(False, errors, [])
            
        actual_dtype = str(df[column].dtype)
        if actual_dtype != expected_dtype:
            # Handle some flexible mapping if needed, but strict string matching for now
            errors.append({
                "column": column,
                "rule": "dtype",
                "message": f"Expected dtype '{expected_dtype}', found '{actual_dtype}'.",
                "severity": "error"
            })
            
        return self._create_result(len(errors) == 0, errors, [])

    def validate_category(self, df: pd.DataFrame, column: str, allowed_values: List[Any]) -> ValidationResult:
        """
        Check that all values in a column are within a predefined list.
        
        Args:
            df: DataFrame to validate.
            column: Column to check.
            allowed_values: List of allowed values.
        """
        errors = []
        
        if column not in df.columns:
            errors.append({
                "column": column,
                "rule": "category",
                "message": f"Column '{column}' not found in DataFrame.",
                "severity": "error"
            })
            return self._create_result(False, errors, [])
            
        valid_data = df[column].dropna()
        invalid_values = valid_data[~valid_data.isin(allowed_values)]
        count_invalid = len(invalid_values)
        
        if count_invalid > 0:
            errors.append({
                "column": column,
                "rule": "category",
                "message": f"Found {count_invalid} values not in allowed list.",
                "severity": "error"
            })
            
        return self._create_result(len(errors) == 0, errors, [])

    def validate_all(self, df: pd.DataFrame, rules: List[Dict]) -> ValidationResult:
        """
        Run multiple validation rules and aggregate results.
        
        Args:
            df: DataFrame to validate.
            rules: List of rule dictionaries. 
                   e.g. {"type": "not_null", "columns": ["id"], "threshold": 0.0}
                   
        Returns:
            Aggregated ValidationResult.
        """
        all_errors = []
        all_warnings = []
        
        for rule in rules:
            rule_type = rule.get("type")
            try:
                if rule_type == "not_null":
                    res = self.validate_not_null(df, rule.get("columns", []), rule.get("threshold", 0.0))
                elif rule_type == "unique":
                    res = self.validate_unique(df, rule.get("columns", []))
                elif rule_type == "range":
                    res = self.validate_range(df, rule.get("column", ""), rule.get("min_val"), rule.get("max_val"))
                elif rule_type == "regex":
                    res = self.validate_regex(df, rule.get("column", ""), rule.get("pattern", ""))
                elif rule_type == "dtype":
                    res = self.validate_dtype(df, rule.get("column", ""), rule.get("expected_dtype", ""))
                elif rule_type == "category":
                    res = self.validate_category(df, rule.get("column", ""), rule.get("allowed_values", []))
                else:
                    self.logger.warning(f"Unknown validation rule type: {rule_type}")
                    continue
                    
                all_errors.extend(res.errors)
                all_warnings.extend(res.warnings)
            except Exception as e:
                self.logger.error(f"Error applying rule {rule_type}: {str(e)}")
                all_errors.append({
                    "column": rule.get("column", rule.get("columns", "unknown")),
                    "rule": rule_type,
                    "message": f"Rule execution failed: {str(e)}",
                    "severity": "error"
                })
                
        is_valid = len(all_errors) == 0
        summary = {
            "total_rows": len(df),
            "error_count": len(all_errors),
            "warning_count": len(all_warnings),
            "rules_applied": len(rules)
        }
        
        return ValidationResult(is_valid=is_valid, errors=all_errors, warnings=all_warnings, summary=summary)
