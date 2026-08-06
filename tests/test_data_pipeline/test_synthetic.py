def test_churn_generator_shape(churn_generator):
    df = churn_generator.generate(100)
    assert len(df) == 100

def test_churn_generator_label_distribution(churn_generator):
    df = churn_generator.generate(1000)
    churn_rate = df["churned"].mean()
    assert 0.1 < churn_rate < 0.4

def test_churn_generator_reproducibility():
    from data_pipeline.synthetic.generators import CustomerChurnGenerator
    g1 = CustomerChurnGenerator(42)
    g2 = CustomerChurnGenerator(42)
    df1 = g1.generate(100)
    df2 = g2.generate(100)
    assert df1.equals(df2)

def test_fraud_generator_label_distribution(fraud_generator):
    df = fraud_generator.generate(1000)
    fraud_rate = df["is_fraud"].mean()
    assert 0.0 < fraud_rate < 0.1

def test_credit_risk_generator_correlations():
    from data_pipeline.synthetic.generators import CreditRiskGenerator
    g = CreditRiskGenerator(42)
    df = g.generate(100)
    assert "is_default" in df.columns

def test_sentiment_generator_text_content():
    from data_pipeline.synthetic.generators import SentimentAnalysisGenerator
    g = SentimentAnalysisGenerator(42)
    df = g.generate(10)
    assert len(df["text"].iloc[0]) > 0

def test_resume_generator_restricted_framing():
    from data_pipeline.synthetic.generators import ResumeScreeningGenerator
    g = ResumeScreeningGenerator(42)
    df = g.generate(10)
    assert df["recommendation"].isin(["shortlist", "reject", "review"]).all()

def test_all_generators_produce_valid_data():
    from data_pipeline.synthetic.generators import (
        CustomerChurnGenerator, FraudDetectionGenerator, CreditRiskGenerator,
        ProductRecommendationGenerator, SentimentAnalysisGenerator, CustomerSupportGenerator,
        DocumentSearchGenerator, ResumeScreeningGenerator, FinancialAnalyticsGenerator,
        ImageClassificationGenerator, OCRGenerator, KnowledgeAssistantGenerator,
        MultiDocQAGenerator, SalesAnalyticsGenerator, KPIDashboardGenerator
    )
    generators = [
        CustomerChurnGenerator, FraudDetectionGenerator, CreditRiskGenerator,
        ProductRecommendationGenerator, SentimentAnalysisGenerator, CustomerSupportGenerator,
        DocumentSearchGenerator, ResumeScreeningGenerator, FinancialAnalyticsGenerator,
        ImageClassificationGenerator, OCRGenerator, KnowledgeAssistantGenerator,
        MultiDocQAGenerator, SalesAnalyticsGenerator, KPIDashboardGenerator
    ]
    for gen_cls in generators:
        g = gen_cls(42)
        df = g.generate(5)
        assert len(df) == 5
