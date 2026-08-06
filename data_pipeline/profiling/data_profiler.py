import pandas as pd
import numpy as np
from typing import Dict, Any, List
from shared.logging import get_logger

class DataProfiler:
    def __init__(self):
        self.logger = get_logger(__name__)

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"error": "Empty DataFrame"}

        memory_usage_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
        overview = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_mb": memory_usage_mb,
            "dtypes_distribution": df.dtypes.astype(str).value_counts().to_dict()
        }

        numeric_stats = {}
        categorical_stats = {}
        datetime_stats = {}
        boolean_stats = {}
        
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                numeric_stats[col] = self._profile_numeric(series)
            elif pd.api.types.is_bool_dtype(series):
                boolean_stats[col] = self._profile_boolean(series)
            elif pd.api.types.is_datetime64_any_dtype(series):
                datetime_stats[col] = self._profile_datetime(series)
            else:
                categorical_stats[col] = self._profile_categorical(series)

        correlations = []
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            for i in range(len(upper_tri.columns)):
                for j in range(i):
                    val = upper_tri.iloc[j, i]
                    if not pd.isna(val) and val > 0.7:
                        correlations.append({
                            "feature1": upper_tri.columns[i],
                            "feature2": upper_tri.columns[j],
                            "correlation": float(val)
                        })

        missing_data = {}
        for col in df.columns:
            m_count = int(df[col].isna().sum())
            if m_count > 0:
                missing_data[col] = {
                    "count": m_count,
                    "percentage": (m_count / len(df)) * 100
                }

        constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
        
        high_cardinality = []
        potential_id_columns = []
        
        for col in df.columns:
            n_unique = df[col].nunique()
            if n_unique / len(df) > 0.95:
                high_cardinality.append(col)
            if n_unique == len(df) and (pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
                potential_id_columns.append(col)

        return {
            "overview": overview,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
            "datetime_stats": datetime_stats,
            "boolean_stats": boolean_stats,
            "correlations": correlations,
            "missing_data": missing_data,
            "constant_columns": constant_columns,
            "high_cardinality": high_cardinality,
            "potential_id_columns": potential_id_columns
        }

    def _profile_numeric(self, series: pd.Series) -> Dict[str, Any]:
        s = series.dropna()
        if s.empty:
            return {}
        return {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
            "skewness": float(s.skew()) if len(s) > 2 else 0.0,
            "kurtosis": float(s.kurtosis()) if len(s) > 3 else 0.0,
            "percentiles": s.quantile([0.05, 0.25, 0.50, 0.75, 0.95]).to_dict(),
            "zeros_count": int((s == 0).sum()),
            "negative_count": int((s < 0).sum())
        }

    def _profile_categorical(self, series: pd.Series) -> Dict[str, Any]:
        s = series.dropna()
        if s.empty:
            return {}
        val_counts = s.value_counts()
        probs = val_counts / len(s)
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        mode = s.mode()
        return {
            "unique_count": int(s.nunique()),
            "top_values": val_counts.head(5).to_dict(),
            "entropy": float(entropy),
            "mode": mode.iloc[0] if not mode.empty else None,
            "mode_frequency": int(val_counts.iloc[0]) if not val_counts.empty else 0
        }

    def _profile_datetime(self, series: pd.Series) -> Dict[str, Any]:
        s = series.dropna()
        if s.empty:
            return {}
        return {
            "min": str(s.min()),
            "max": str(s.max()),
            "range": str(s.max() - s.min()),
            "most_common_day_of_week": s.dt.dayofweek.mode().iloc[0] if not s.empty else None,
            "most_common_hour": s.dt.hour.mode().iloc[0] if not s.empty else None
        }

    def _profile_boolean(self, series: pd.Series) -> Dict[str, Any]:
        s = series.dropna()
        if s.empty:
            return {}
        true_count = int(s.sum())
        false_count = int(len(s) - true_count)
        return {
            "true_count": true_count,
            "false_count": false_count,
            "true_ratio": float(true_count / len(s))
        }

    def profile_column(self, series: pd.Series) -> Dict[str, Any]:
        if series.empty:
            return {"error": "Empty series"}
        
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            return self._profile_numeric(series)
        elif pd.api.types.is_bool_dtype(series):
            return self._profile_boolean(series)
        elif pd.api.types.is_datetime64_any_dtype(series):
            return self._profile_datetime(series)
        else:
            return self._profile_categorical(series)

    def compare_profiles(self, profile1: Dict[str, Any], profile2: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rows_diff": profile2.get("overview", {}).get("rows", 0) - profile1.get("overview", {}).get("rows", 0),
            "cols_diff": profile2.get("overview", {}).get("columns", 0) - profile1.get("overview", {}).get("columns", 0)
        }

    def to_report(self, profile: Dict[str, Any]) -> str:
        if "error" in profile:
            return profile["error"]
            
        overview = profile.get("overview", {})
        report = []
        report.append(f"Data Profile Report")
        report.append(f"===================")
        report.append(f"Rows: {overview.get('rows', 0)}")
        report.append(f"Columns: {overview.get('columns', 0)}")
        report.append(f"Memory Usage: {overview.get('memory_usage_mb', 0):.2f} MB")
        report.append(f"")
        report.append(f"Missing Data:")
        for col, stats in profile.get("missing_data", {}).items():
            report.append(f"  - {col}: {stats['count']} ({stats['percentage']:.2f}%)")
            
        return "\n".join(report)
