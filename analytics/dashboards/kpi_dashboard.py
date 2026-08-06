from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from shared.logging import get_logger

@dataclass
class KPIMetric:
    """Represents a single Key Performance Indicator metric."""
    name: str
    value: float
    previous_value: float
    change_pct: float
    target: float
    on_target: bool
    unit: str

class KPIDashboard:
    """Dashboard for calculating and summarizing executive KPIs."""
    
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def calculate_executive_kpis(self, df: pd.DataFrame) -> Dict[str, KPIMetric]:
        """Computes core SaaS/Enterprise KPIs from the given DataFrame."""
        self.logger.info("Calculating executive KPIs")
        kpis = {}
        if df.empty:
            self.logger.warning("Empty dataframe provided for KPI calculation.")
            return kpis
        
        def safe_sum(col: str) -> float:
            return float(df[col].sum()) if col in df.columns else 0.0

        mrr_val = safe_sum('revenue')
        mrr_prev = safe_sum('prev_revenue')
        mrr_change = ((mrr_val - mrr_prev) / mrr_prev * 100) if mrr_prev else 0.0
        mrr_target = 100000.0
        kpis['MRR'] = KPIMetric("Monthly Recurring Revenue", mrr_val, mrr_prev, mrr_change, mrr_target, mrr_val >= mrr_target, "$")

        arr_val = mrr_val * 12
        arr_prev = mrr_prev * 12
        kpis['ARR'] = KPIMetric("Annual Recurring Revenue", arr_val, arr_prev, mrr_change, mrr_target * 12, arr_val >= mrr_target * 12, "$")

        users = safe_sum('users')
        churned = safe_sum('churned_users')
        churn_val = (churned / users * 100) if users else 0.0
        churn_target = 5.0
        kpis['Churn Rate'] = KPIMetric("Churn Rate", churn_val, 0.0, 0.0, churn_target, churn_val <= churn_target, "%")

        spend = safe_sum('marketing_spend')
        new_users = safe_sum('new_users')
        cac_val = (spend / new_users) if new_users else 0.0
        cac_target = 100.0
        kpis['CAC'] = KPIMetric("Customer Acquisition Cost", cac_val, 0.0, 0.0, cac_target, cac_val <= cac_target, "$")

        arpu_val = (mrr_val / users) if users else 0.0
        arpu_target = 50.0
        kpis['ARPU'] = KPIMetric("Average Revenue Per User", arpu_val, 0.0, 0.0, arpu_target, arpu_val >= arpu_target, "$")

        ltv_val = (arpu_val / (churn_val / 100)) if churn_val > 0 else 0.0
        ltv_target = cac_target * 3
        kpis['LTV'] = KPIMetric("Customer Lifetime Value", ltv_val, 0.0, 0.0, ltv_target, ltv_val >= ltv_target, "$")

        ltv_cac_ratio = (ltv_val / cac_val) if cac_val > 0 else 0.0
        ltv_cac_target = 3.0
        kpis['LTV:CAC'] = KPIMetric("LTV to CAC Ratio", ltv_cac_ratio, 0.0, 0.0, ltv_cac_target, ltv_cac_ratio >= ltv_cac_target, "x")

        conversions = safe_sum('conversions')
        visitors = safe_sum('visitors')
        conv_val = (conversions / visitors * 100) if visitors else 0.0
        conv_target = 10.0
        kpis['Conversion Rate'] = KPIMetric("Conversion Rate", conv_val, 0.0, 0.0, conv_target, conv_val >= conv_target, "%")

        nps_val = float(df['nps_score'].mean()) if 'nps_score' in df.columns else 0.0
        nps_target = 30.0
        kpis['NPS'] = KPIMetric("Net Promoter Score", nps_val, 0.0, 0.0, nps_target, nps_val >= nps_target, "pt")

        return kpis

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, float]:
        """Simplified KPI calculation returning raw float values for test compatibility."""
        rev = float(df['revenue'].sum()) if 'revenue' in df.columns else 0.0
        users = float(df['user_id'].nunique()) if 'user_id' in df.columns else (float(df['users'].sum()) if 'users' in df.columns else 0.0)
        return {"revenue": rev, "users": users}

    def generate_kpi_summary(self, kpis: Dict[str, KPIMetric]) -> Dict[str, Any]:
        """Generates a summary dictionary of KPIs for API and UI rendering."""
        self.logger.info("Generating KPI summary")
        summary = {}
        for key, metric in kpis.items():
            summary[key] = {
                "name": metric.name,
                "value": round(metric.value, 2),
                "previous_value": round(metric.previous_value, 2),
                "change_pct": round(metric.change_pct, 2),
                "target": round(metric.target, 2),
                "on_target": metric.on_target,
                "unit": metric.unit,
                "display": f"{metric.value:.2f}{metric.unit} ({metric.change_pct:+.2f}%)"
            }
        return summary

    def get_kpi_trend(self, df: pd.DataFrame, metric_name: str, freq: str = "D") -> pd.DataFrame:
        """Aggregates a metric over time."""
        self.logger.info(f"Generating KPI trend for {metric_name} with frequency {freq}")
        if df.empty or 'date' not in df.columns or metric_name not in df.columns:
            self.logger.warning(f"Required columns missing or empty DataFrame for {metric_name} trend.")
            return pd.DataFrame()
            
        try:
            df_trend = df.copy()
            df_trend['date'] = pd.to_datetime(df_trend['date'])
            trend = df_trend.groupby(pd.Grouper(key='date', freq=freq))[metric_name].sum().reset_index()
            return trend
        except Exception as e:
            self.logger.error(f"Error computing KPI trend: {e}")
            return pd.DataFrame()
