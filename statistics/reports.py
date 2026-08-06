from typing import Dict, Any

class StatisticalReportGenerator:
    def generate_statistical_report(self, results: Dict[str, Any]) -> str:
        report = "Statistical Analysis Report\\n===========================\\n"
        if "normality_results" in results:
            report += "Normality Results: " + str(results["normality_results"]) + "\\n"
        if "hypothesis_tests" in results:
            report += "Hypothesis Tests: " + str(results["hypothesis_tests"]) + "\\n"
        return report
