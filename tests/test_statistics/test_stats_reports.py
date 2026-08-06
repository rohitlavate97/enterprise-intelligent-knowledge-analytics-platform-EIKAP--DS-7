
from statistics import StatisticalReportGenerator

def test_generate_statistical_report():
    gen = StatisticalReportGenerator()
    report = gen.generate_statistical_report({"normality_results": "All good"})
    assert "Statistical Analysis Report" in report
    assert "All good" in report
