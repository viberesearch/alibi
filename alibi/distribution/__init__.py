"""Distribution module for audience-specific output forms.

Provides four output forms tailored to different audiences:
- SUMMARY: High-level spending overview (Telegram, quick CLI)
- DETAILED: Full breakdown with line items (Obsidian, PDF)
- ANALYTICAL: Patterns, trends, comparisons (MCP/Claude, CSV analysis)
- TABULAR: Spreadsheet-friendly flat data (CSV/XLSX export)
"""

from alibi.distribution.formatters import OutputFormat, format_output
from alibi.distribution.forms import DistributionForm, DistributionResult, distribute
from alibi.distribution.renderer import DistributionRenderer

__all__ = [
    "DistributionForm",
    "DistributionRenderer",
    "DistributionResult",
    "OutputFormat",
    "distribute",
    "format_output",
]
