import pytest
import pandas as pd
from analytics.dashboards import KPIDashboard, CustomerAnalytics, SalesAnalytics
from analytics.charts import ChartExporter

@pytest.fixture
def sample_kpi_data():
    return pd.DataFrame({
        'user_id': [1, 2, 3, 1, 2],
        'revenue': [100.0, 200.0, 150.0, 50.0, 300.0],
        'region': ['North', 'South', 'East', 'North', 'South'],
        'date': pd.date_range(start='2023-01-01', periods=5)
    })

@pytest.fixture
def kpi_dashboard():
    return KPIDashboard()

@pytest.fixture
def customer_analytics():
    return CustomerAnalytics()

@pytest.fixture
def sales_analytics():
    return SalesAnalytics()

@pytest.fixture
def chart_exporter():
    return ChartExporter()
