"""High-level distribution renderer combining form and format.

Provides convenience methods for common audience-specific outputs:
- Telegram: Summary in Markdown
- Obsidian: Detailed in Markdown
- Export: Tabular in CSV
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alibi.distribution.formatters import OutputFormat, format_output
from alibi.distribution.forms import DistributionForm, distribute

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager


class DistributionRenderer:
    """Render distributions for different audiences."""

    def __init__(self, db: DatabaseManager) -> None:
        """Initialize renderer with database connection.

        Args:
            db: Database manager instance
        """
        self.db = db

    def render(
        self,
        form: DistributionForm,
        output_format: OutputFormat,
        date_from: str | None = None,
        date_to: str | None = None,
        space_id: str = "default",
    ) -> str:
        """Generate and format a distribution.

        Args:
            form: Distribution form (level of detail)
            output_format: Output format (json, md, csv, html)
            date_from: Start date filter
            date_to: End date filter
            space_id: Space to query

        Returns:
            Formatted string output
        """
        result = distribute(self.db, form, date_from, date_to, space_id)
        return format_output(result, output_format)

    def render_for_telegram(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        space_id: str = "default",
    ) -> str:
        """Shortcut: Summary in markdown format for Telegram.

        Args:
            date_from: Start date filter
            date_to: End date filter
            space_id: Space to query

        Returns:
            Markdown-formatted summary
        """
        return self.render(
            DistributionForm.SUMMARY,
            OutputFormat.MARKDOWN,
            date_from,
            date_to,
            space_id,
        )

    def render_for_obsidian(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        space_id: str = "default",
    ) -> str:
        """Shortcut: Detailed in markdown format for Obsidian.

        Args:
            date_from: Start date filter
            date_to: End date filter
            space_id: Space to query

        Returns:
            Markdown-formatted detailed report
        """
        return self.render(
            DistributionForm.DETAILED,
            OutputFormat.MARKDOWN,
            date_from,
            date_to,
            space_id,
        )

    def render_for_export(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        space_id: str = "default",
    ) -> str:
        """Shortcut: Tabular in CSV format for spreadsheets.

        Args:
            date_from: Start date filter
            date_to: End date filter
            space_id: Space to query

        Returns:
            CSV-formatted transaction data
        """
        return self.render(
            DistributionForm.TABULAR,
            OutputFormat.CSV,
            date_from,
            date_to,
            space_id,
        )
