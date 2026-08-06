from typing import Dict, Any, Optional, Union
from shared.logging import get_logger
import pandas as pd

class ExecutiveReportGenerator:
    """Generates Markdown and HTML executive reports."""
    
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def generate_markdown_report(
        self,
        kpis: Dict[str, Any],
        sales_summary: Optional[Union[Dict[str, Any], str]] = None,
        customer_summary: Optional[Dict[str, Any]] = None,
        title: str = "Executive Business Performance Report"
    ) -> str:
        """Generates a comprehensive GitHub-flavored Markdown report."""
        self.logger.info("Generating Markdown report")
        
        # Support overload if title passed as 2nd param
        if isinstance(sales_summary, str):
            title = sales_summary
            sales_summary = {}
            customer_summary = {}
            
        sales_dict = sales_summary or {}
        customer_dict = customer_summary or {}
        
        md_lines = [
            f"# {title}",
            "",
            "## Executive Summary",
            "This report provides an overview of the core Key Performance Indicators (KPIs), sales performance, and customer analytics.",
            ""
        ]
        
        # KPI Section
        md_lines.append("## Core KPIs")
        md_lines.append("| Metric | Value | Target | Status |")
        md_lines.append("|---|---|---|---|")
        
        for kpi_name, data in kpis.items():
            if isinstance(data, dict):
                val = data.get("value", 0.0)
                target = data.get("target", 0.0)
                unit = data.get("unit", "")
                on_target = data.get("on_target", False)
                status_badge = "🟢 On Target" if on_target else "🔴 Needs Attention"
                md_lines.append(f"| **{kpi_name}** | {val:,.2f}{unit} | {target:,.2f}{unit} | {status_badge} |")
            else:
                md_lines.append(f"| **{kpi_name}** | {data} | - | 🟢 Recorded |")
            
        md_lines.append("")
        
        # Sales Summary
        md_lines.append("## Sales Analytics")
        md_lines.append("### Revenue by Region")
        for region, rev in sales_dict.get("revenue_by_region", {}).items():
            md_lines.append(f"- **{region}**: ${rev:,.2f}")
            
        md_lines.append("")
        md_lines.append("### Revenue by Category")
        for cat, rev in sales_dict.get("revenue_by_product_category", {}).items():
            md_lines.append(f"- **{cat}**: ${rev:,.2f}")
            
        md_lines.append("")
        
        # Customer Summary
        md_lines.append("## Customer Analytics")
        if "clv_distribution" in customer_dict:
            clv = customer_dict["clv_distribution"]
            md_lines.append("### Customer Lifetime Value (CLV)")
            md_lines.append(f"- **Average CLV**: ${clv.get('mean', 0.0):,.2f}")
            md_lines.append(f"- **Median CLV**: ${clv.get('median', 0.0):,.2f}")
            
        md_lines.append("")
        return "\n".join(md_lines)

    def generate_html_report(
        self,
        markdown_report: Union[str, Dict[str, Any]],
        chart_base64_images: Optional[Union[Dict[str, str], str]] = None
    ) -> str:
        """Wraps Markdown into a standalone HTML document."""
        self.logger.info("Generating HTML report from Markdown")

        if isinstance(markdown_report, dict):
            # Overload where dict + title is passed
            title = str(chart_base64_images) if isinstance(chart_base64_images, str) else "Executive Report"
            md_text = self.generate_markdown_report(markdown_report, title=title)
            charts_dict = None
        else:
            md_text = markdown_report
            charts_dict = chart_base64_images if isinstance(chart_base64_images, dict) else None
        
        try:
            import markdown
            html_content = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
        except ImportError:
            self.logger.warning("Markdown package not available, returning raw text inside HTML.")
            html_content = f"<pre>{md_text}</pre>"
            
        charts_html = ""
        if charts_dict:
            charts_html += "<div class='charts-container'>\n"
            for title_str, b64_img in charts_dict.items():
                charts_html += f"  <div class='chart'>\n    <h3>{title_str}</h3>\n    <img src='data:image/png;base64,{b64_img}' alt='{title_str}'/>\n  </div>\n"
            charts_html += "</div>\n"

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Executive Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 1000px;
        margin: 0 auto;
        padding: 20px;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 20px;
    }}
    th, td {{
        border: 1px solid #ddd;
        padding: 8px 12px;
        text-align: left;
    }}
    th {{
        background-color: #f4f4f4;
    }}
    h1, h2, h3 {{
        color: #2c3e50;
    }}
    .charts-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-top: 30px;
    }}
    .chart {{
        border: 1px solid #eee;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .chart img {{
        max-width: 100%;
        height: auto;
    }}
</style>
</head>
<body>
    <div class="report-content">
        {html_content}
    </div>
    {charts_html}
</body>
</html>"""
        
        return html_template
