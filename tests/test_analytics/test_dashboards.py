def test_kpi_dashboard_calculation(kpi_dashboard, sample_kpi_data):
    kpis = kpi_dashboard.calculate_kpis(sample_kpi_data)
    assert kpis['revenue'] == 800.0
    assert kpis['users'] == 3.0

def test_rfm_customer_segmentation(customer_analytics, sample_kpi_data):
    df = customer_analytics.rfm_segmentation(sample_kpi_data)
    assert not df.empty
    assert 'segment' in df.columns
    assert len(df) == 3

def test_cohort_retention(customer_analytics, sample_kpi_data):
    df = customer_analytics.cohort_retention(sample_kpi_data)
    assert not df.empty
    assert 'retention' in df.columns

def test_sales_analytics_by_region(sales_analytics, sample_kpi_data):
    df = sales_analytics.sales_by_region(sample_kpi_data)
    assert len(df) == 3
    assert df[df['region'] == 'North']['revenue'].iloc[0] == 150.0

def test_sales_funnel_analysis(sales_analytics, sample_kpi_data):
    funnel = sales_analytics.sales_funnel_analysis(sample_kpi_data)
    assert funnel['leads'] == 1000.0
