import pandas as pd
from analytics.charts import TrendChart, DistributionChart, CorrelationChart, FeatureImportanceChart

def test_plotly_line_trend(sample_kpi_data, chart_exporter):
    chart = TrendChart()
    fig = chart.plot_plotly(sample_kpi_data, 'date', 'revenue')
    assert fig is not None
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure)

def test_matplotlib_line_trend(sample_kpi_data, chart_exporter):
    chart = TrendChart()
    fig = chart.plot_matplotlib(sample_kpi_data, 'date', 'revenue')
    assert fig is not None
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)
    b64 = chart_exporter.export_base64(fig)
    assert b64.startswith('iVBORw0KGgo') or len(b64) > 0

def test_histogram_distribution(sample_kpi_data):
    chart = DistributionChart()
    fig = chart.plot_histogram(sample_kpi_data, 'revenue')
    assert fig is not None

def test_correlation_heatmap(sample_kpi_data):
    chart = CorrelationChart()
    numeric_df = sample_kpi_data[['user_id', 'revenue']]
    fig = chart.plot_heatmap(numeric_df)
    assert fig is not None

def test_feature_importance_plot():
    chart = FeatureImportanceChart()
    fig = chart.plot_importance({'age': 0.5, 'income': 0.3})
    assert fig is not None

def test_chart_exporter_base64(sample_kpi_data, chart_exporter):
    chart = TrendChart()
    fig = chart.plot_matplotlib(sample_kpi_data, 'date', 'revenue')
    b64 = chart_exporter.export_base64(fig)
    assert isinstance(b64, str)
    assert len(b64) > 0
