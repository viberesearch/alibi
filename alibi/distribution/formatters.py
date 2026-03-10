"""Output format converters for distribution results.

Converts DistributionResult into various output formats:
- JSON: Structured data with indent
- Markdown: Tables and sections for Telegram/Obsidian
- CSV: Comma-separated values (TABULAR form only)
- HTML: Simple styled table
"""

from __future__ import annotations

import csv
import io
import json
from enum import Enum
from typing import Any

from alibi.distribution.forms import DistributionForm, DistributionResult


class OutputFormat(str, Enum):
    """Available output formats."""

    JSON = "json"
    MARKDOWN = "md"
    CSV = "csv"
    HTML = "html"


def format_output(result: DistributionResult, output_format: OutputFormat) -> str:
    """Format a DistributionResult into the requested output format.

    Args:
        result: The distribution result to format
        output_format: Target output format

    Returns:
        Formatted string in the requested format

    Raises:
        ValueError: If CSV format requested for non-TABULAR form
    """
    formatters = {
        OutputFormat.JSON: _format_json,
        OutputFormat.MARKDOWN: _format_markdown,
        OutputFormat.CSV: _format_csv,
        OutputFormat.HTML: _format_html,
    }

    formatter = formatters[output_format]
    return formatter(result)


def _format_json(result: DistributionResult) -> str:
    """Format as JSON with indentation."""
    payload = {
        "form": result.form.value,
        "title": result.title,
        "period": result.period,
        "data": result.data,
        "metadata": result.metadata,
    }
    return json.dumps(payload, indent=2, default=str)


def _format_markdown(result: DistributionResult) -> str:
    """Format as Markdown with tables and sections."""
    md_formatters = {
        DistributionForm.SUMMARY: _md_summary,
        DistributionForm.DETAILED: _md_detailed,
        DistributionForm.ANALYTICAL: _md_analytical,
        DistributionForm.TABULAR: _md_tabular,
    }

    formatter = md_formatters[result.form]
    return formatter(result)


def _md_summary(result: DistributionResult) -> str:
    """Markdown for SUMMARY form."""
    d = result.data
    period_str = _period_string(result.period)

    lines = [
        f"# {result.title}",
        f"*{period_str}*",
        "",
        "## Overview",
        "",
        f"- **Expenses**: {d['total_expenses']:,.2f}",
        f"- **Income**: {d['total_income']:,.2f}",
        f"- **Net**: {d['net']:+,.2f}",
        f"- **Transactions**: {d['transaction_count']}",
        "",
    ]

    if d.get("top_vendors"):
        lines.extend(
            [
                "## Top Vendors",
                "",
                "| Vendor | Total | Count |",
                "|--------|------:|------:|",
            ]
        )
        for v in d["top_vendors"]:
            lines.append(f"| {v['vendor']} | {v['total']:,.2f} | {v['count']} |")
        lines.append("")

    if d.get("top_categories"):
        lines.extend(
            [
                "## Top Categories",
                "",
                "| Category | Total |",
                "|----------|------:|",
            ]
        )
        for c in d["top_categories"]:
            lines.append(f"| {c['category']} | {c['total']:,.2f} |")
        lines.append("")

    return "\n".join(lines)


def _md_detailed(result: DistributionResult) -> str:
    """Markdown for DETAILED form."""
    d = result.data
    period_str = _period_string(result.period)

    lines = [
        f"# {result.title}",
        f"*{period_str}*",
        "",
        "## Overview",
        "",
        f"- **Expenses**: {d['total_expenses']:,.2f}",
        f"- **Income**: {d['total_income']:,.2f}",
        f"- **Net**: {d['net']:+,.2f}",
        f"- **Transactions**: {d['transaction_count']}",
        f"- **Artifacts Processed**: {d.get('artifacts_processed', 0)}",
        "",
    ]

    if d.get("all_vendors"):
        lines.extend(
            [
                "## All Vendors",
                "",
                "| Vendor | Total | Count |",
                "|--------|------:|------:|",
            ]
        )
        for v in d["all_vendors"]:
            lines.append(f"| {v['vendor']} | {v['total']:,.2f} | {v['count']} |")
        lines.append("")

    if d.get("all_categories"):
        lines.extend(
            [
                "## All Categories",
                "",
                "| Category | Total |",
                "|----------|------:|",
            ]
        )
        for c in d["all_categories"]:
            lines.append(f"| {c['category']} | {c['total']:,.2f} |")
        lines.append("")

    if d.get("line_items_by_category"):
        lines.extend(["## Line Items by Category", ""])
        for cat, items in d["line_items_by_category"].items():
            lines.extend(
                [
                    f"### {cat}",
                    "",
                    "| Item | Qty | Unit Price | Total |",
                    "|------|----:|----------:|------:|",
                ]
            )
            for item in items:
                up = f"{item['unit_price']:,.2f}" if item.get("unit_price") else "-"
                tp = f"{item['total_price']:,.2f}" if item.get("total_price") else "-"
                lines.append(
                    f"| {item['name']} | {item['quantity']:.0f} | {up} | {tp} |"
                )
            lines.append("")

    if d.get("monthly_trend"):
        lines.extend(
            [
                "## Monthly Trend",
                "",
                "| Month | Expenses | Income |",
                "|-------|--------:|-------:|",
            ]
        )
        for m in d["monthly_trend"]:
            lines.append(
                f"| {m['month']} | {m['expenses']:,.2f} | {m['income']:,.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def _md_analytical(result: DistributionResult) -> str:
    """Markdown for ANALYTICAL form."""
    d = result.data
    period_str = _period_string(result.period)

    lines = [
        f"# {result.title}",
        f"*{period_str}*",
        "",
    ]

    if d.get("spending_trend"):
        lines.extend(
            [
                "## Spending Trend",
                "",
                "| Period | Total | Avg |",
                "|--------|------:|----:|",
            ]
        )
        for t in d["spending_trend"]:
            lines.append(f"| {t['period']} | {t['total']:,.2f} | {t['avg']:,.2f} |")
        lines.append("")

    if d.get("category_distribution"):
        lines.extend(
            [
                "## Category Distribution",
                "",
                "| Category | Amount | % |",
                "|----------|-------:|--:|",
            ]
        )
        for c in d["category_distribution"]:
            lines.append(
                f"| {c['category']} | {c['amount']:,.2f} | {c['percentage']:.1f}% |"
            )
        lines.append("")

    if d.get("vendor_frequency"):
        lines.extend(
            [
                "## Vendor Frequency",
                "",
                "| Vendor | Count | Avg Amount | Pattern |",
                "|--------|------:|-----------:|---------|",
            ]
        )
        for v in d["vendor_frequency"]:
            lines.append(
                f"| {v['vendor']} | {v['count']} "
                f"| {v['avg_amount']:,.2f} | {v['trend']} |"
            )
        lines.append("")

    if d.get("anomalies"):
        lines.extend(
            [
                "## Anomalies",
                "",
                "| Date | Vendor | Amount | Reason |",
                "|------|--------|-------:|--------|",
            ]
        )
        for a in d["anomalies"]:
            lines.append(
                f"| {a['date']} | {a['vendor']} "
                f"| {a['amount']:,.2f} | {a['reason']} |"
            )
        lines.append("")

    if d.get("recurring"):
        lines.extend(
            [
                "## Recurring Expenses",
                "",
                "| Vendor | Avg Amount | Frequency |",
                "|--------|-----------:|-----------|",
            ]
        )
        for r in d["recurring"]:
            lines.append(
                f"| {r['vendor']} | {r['avg_amount']:,.2f} | {r['frequency']} |"
            )
        lines.append("")

    return "\n".join(lines)


def _md_tabular(result: DistributionResult) -> str:
    """Markdown for TABULAR form."""
    d = result.data
    headers = d.get("headers", [])
    rows = d.get("rows", [])

    if not headers:
        return "No data available."

    lines = [
        f"# {result.title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for row in rows:
        formatted = []
        for val in row:
            if isinstance(val, float):
                formatted.append(f"{val:,.2f}")
            else:
                formatted.append(str(val))
        lines.append("| " + " | ".join(formatted) + " |")

    lines.append("")
    return "\n".join(lines)


def _format_csv(result: DistributionResult) -> str:
    """Format as CSV (only valid for TABULAR form).

    Raises:
        ValueError: If result form is not TABULAR
    """
    if result.form != DistributionForm.TABULAR:
        raise ValueError(
            f"CSV format is only supported for TABULAR form, got {result.form.value}"
        )

    output = io.StringIO()
    writer = csv.writer(output)

    headers = result.data.get("headers", [])
    rows = result.data.get("rows", [])

    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)

    return output.getvalue()


def _format_html(result: DistributionResult) -> str:
    """Format as HTML with basic styling."""
    html_formatters: dict[DistributionForm, Any] = {
        DistributionForm.SUMMARY: _html_summary,
        DistributionForm.DETAILED: _html_summary,  # reuse summary layout
        DistributionForm.ANALYTICAL: _html_summary,
        DistributionForm.TABULAR: _html_tabular,
    }

    formatter = html_formatters[result.form]
    body = formatter(result)

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
body {{ font-family: sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #f5f5f5; }}
tr:nth-child(even) {{ background-color: #fafafa; }}
h1 {{ color: #333; }}
h2 {{ color: #555; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _html_summary(result: DistributionResult) -> str:
    """HTML body for summary/detailed/analytical forms."""
    d = result.data
    period_str = _period_string(result.period)

    parts = [f"<h1>{_escape_html(result.title)}</h1>", f"<p><em>{period_str}</em></p>"]

    # Render any dict values as key-value pairs
    for key, value in d.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            # Render list of dicts as table
            parts.append(f"<h2>{_escape_html(key.replace('_', ' ').title())}</h2>")
            parts.append(_dict_list_to_html_table(value))
        elif isinstance(value, dict):
            parts.append(f"<h2>{_escape_html(key.replace('_', ' ').title())}</h2>")
            parts.append("<ul>")
            for k, v in value.items():
                parts.append(
                    f"<li><strong>{_escape_html(str(k))}</strong>: {_escape_html(str(v))}</li>"
                )
            parts.append("</ul>")
        elif not isinstance(value, list):
            pass  # scalar values handled in overview below

    # Overview section for scalar values
    scalars = {k: v for k, v in d.items() if not isinstance(v, (list, dict))}
    if scalars:
        overview = "<h2>Overview</h2><ul>"
        for k, v in scalars.items():
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                overview += f"<li><strong>{label}</strong>: {v:,.2f}</li>"
            else:
                overview += f"<li><strong>{label}</strong>: {v}</li>"
        overview += "</ul>"
        # Insert after title
        parts.insert(2, overview)

    return "\n".join(parts)


def _html_tabular(result: DistributionResult) -> str:
    """HTML body for tabular form."""
    d = result.data
    headers = d.get("headers", [])
    rows = d.get("rows", [])

    parts = [f"<h1>{_escape_html(result.title)}</h1>"]
    parts.append("<table>")
    parts.append("<thead><tr>")
    for h in headers:
        parts.append(f"<th>{_escape_html(str(h))}</th>")
    parts.append("</tr></thead>")
    parts.append("<tbody>")
    for row in rows:
        parts.append("<tr>")
        for val in row:
            if isinstance(val, float):
                parts.append(f"<td>{val:,.2f}</td>")
            else:
                parts.append(f"<td>{_escape_html(str(val))}</td>")
        parts.append("</tr>")
    parts.append("</tbody>")
    parts.append("</table>")

    return "\n".join(parts)


def _dict_list_to_html_table(items: list[dict[str, Any]]) -> str:
    """Convert a list of dicts to an HTML table."""
    if not items:
        return "<p>No data.</p>"

    headers = list(items[0].keys())
    parts = ["<table>", "<thead><tr>"]
    for h in headers:
        parts.append(f"<th>{_escape_html(h.replace('_', ' ').title())}</th>")
    parts.append("</tr></thead>")
    parts.append("<tbody>")
    for item in items:
        parts.append("<tr>")
        for h in headers:
            val = item.get(h, "")
            if isinstance(val, float):
                parts.append(f"<td>{val:,.2f}</td>")
            else:
                parts.append(f"<td>{_escape_html(str(val))}</td>")
        parts.append("</tr>")
    parts.append("</tbody>")
    parts.append("</table>")
    return "\n".join(parts)


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _period_string(period: dict[str, str]) -> str:
    """Format period dict as readable string."""
    from_str = period.get("from", "all")
    to_str = period.get("to", "now")
    if from_str == "all" and to_str == "now":
        return "All time"
    if from_str == "all":
        return f"Up to {to_str}"
    if to_str == "now":
        return f"From {from_str}"
    return f"{from_str} to {to_str}"
