from analytics.reports import ExecutiveReportGenerator

def test_generate_markdown_report():
    gen = ExecutiveReportGenerator()
    report = gen.generate_markdown_report({"revenue": 1000}, "Test Report")
    assert "# Test Report" in report
    assert "revenue" in report

def test_generate_html_report():
    gen = ExecutiveReportGenerator()
    report = gen.generate_html_report({"users": 50}, "HTML Report")
    assert "HTML Report" in report
    assert "users" in report
