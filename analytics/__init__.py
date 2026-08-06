"""Analytics package."""

from analytics.charts import (
    TrendChart,
    DistributionChart,
    CorrelationChart,
    FeatureImportanceChart,
    ChartExporter
)
from analytics.dashboards import (
    KPIDashboard,
    CustomerAnalytics,
    SalesAnalytics
)
from analytics.reports import ExecutiveReportGenerator
from analytics.realtime import KPIStreamer, LiveKPIPublisher
from analytics.module import BusinessAnalyticsModule

__all__ = [
    "TrendChart",
    "DistributionChart",
    "CorrelationChart",
    "FeatureImportanceChart",
    "ChartExporter",
    "KPIDashboard",
    "CustomerAnalytics",
    "SalesAnalytics",
    "ExecutiveReportGenerator",
    "KPIStreamer",
    "LiveKPIPublisher",
    "BusinessAnalyticsModule",
]
