from analytics.module import BusinessAnalyticsModule, AnalyticsInput, ModuleCategory, ModuleMaturity

def test_business_analytics_metadata():
    module = BusinessAnalyticsModule()
    assert module.metadata.name == "business_analytics"
    assert module.metadata.category == ModuleCategory.ANALYTICS
    assert module.metadata.maturity == ModuleMaturity.STANDARD

def test_business_analytics_predict():
    module = BusinessAnalyticsModule()
    input_data = AnalyticsInput(
        dataset_name="sales_db",
        metric_type="revenue",
        date_range={"start": "2023-01-01", "end": "2023-12-31"}
    )
    output = module.predict(input_data)
    
    assert output.kpis["total_revenue"] == 150000.0
    assert "trend_chart" in output.charts_base64
    assert output.execution_time_ms >= 0

def test_business_analytics_compliance_contract():
    # Simulating UniversalModuleContract check on BusinessAnalyticsModule
    module = BusinessAnalyticsModule()
    
    # Contract checks:
    assert hasattr(module, "metadata")
    assert hasattr(module, "train")
    assert hasattr(module, "predict")
    assert hasattr(module, "explain")
    assert hasattr(module, "evaluate")
    assert hasattr(module, "health_check")
    
    health = module.health_check()
    assert health["status"] == "healthy"
