"""MCP resource providers for alibi data.

Resources expose static or semi-static data through URI-based access.
Each resource function returns structured data as dicts/lists.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from alibi.config import get_config
from alibi.db.connection import DatabaseManager


def _get_db() -> DatabaseManager:
    """Get an initialized database manager."""
    config = get_config()
    db = DatabaseManager(config)
    if not db.is_initialized():
        db.initialize()
    return db


def get_recent_transactions(db: DatabaseManager) -> dict[str, Any]:
    """Get the last 30 days of transactions.

    Args:
        db: Database manager instance.

    Returns:
        Dict with date range and transaction list.
    """
    today = date.today()
    date_from = today - timedelta(days=30)

    rows = db.fetchall(
        """
        SELECT id, fact_type, vendor, total_amount, currency,
               event_date, status
        FROM facts
        WHERE event_date >= ?
        ORDER BY event_date DESC
        """,
        (date_from.isoformat(),),
    )

    return {
        "date_from": date_from.isoformat(),
        "date_to": today.isoformat(),
        "total": len(rows),
        "transactions": [dict(r) for r in rows],
    }


def get_monthly_report(db: DatabaseManager, year: int, month: int) -> dict[str, Any]:
    """Generate monthly report data.

    Args:
        db: Database manager instance.
        year: Report year.
        month: Report month (1-12).

    Returns:
        Dict with expenses, income, top vendors, and artifact count.
    """
    date_from = f"{year:04d}-{month:02d}-01"
    if month == 12:
        date_to = f"{year + 1:04d}-01-01"
    else:
        date_to = f"{year:04d}-{month + 1:02d}-01"

    # Total expenses
    total_row = db.fetchone(
        """SELECT COUNT(*) as count, SUM(CAST(total_amount AS REAL)) as total
           FROM facts
           WHERE event_date >= ? AND event_date < ?
                 AND fact_type IN ('purchase', 'subscription_payment')""",
        (date_from, date_to),
    )

    # Total income (refunds)
    income_row = db.fetchone(
        """SELECT COUNT(*) as count, SUM(CAST(total_amount AS REAL)) as total
           FROM facts
           WHERE event_date >= ? AND event_date < ?
                 AND fact_type = 'refund'""",
        (date_from, date_to),
    )

    # Top vendors
    vendor_rows = db.fetchall(
        """SELECT vendor, COUNT(*) as count,
                  SUM(CAST(total_amount AS REAL)) as total
           FROM facts
           WHERE event_date >= ? AND event_date < ?
                 AND fact_type IN ('purchase', 'subscription_payment')
                 AND vendor IS NOT NULL
           GROUP BY vendor ORDER BY total DESC LIMIT 10""",
        (date_from, date_to),
    )

    # Documents processed
    artifact_row = db.fetchone(
        """SELECT COUNT(*) as count
           FROM documents
           WHERE created_at >= ? AND created_at < ?""",
        (date_from, date_to),
    )

    return {
        "period": {"year": year, "month": month},
        "expenses": {
            "count": total_row["count"] if total_row else 0,
            "total": (round(total_row["total"] or 0, 2) if total_row else 0),
        },
        "income": {
            "count": income_row["count"] if income_row else 0,
            "total": (round(income_row["total"] or 0, 2) if income_row else 0),
        },
        "top_vendors": [dict(r) for r in vendor_rows],
        "artifacts_processed": (artifact_row["count"] if artifact_row else 0),
    }


def get_expiring_warranties(db: DatabaseManager) -> dict[str, Any]:
    """Get items with warranties expiring in the next 90 days.

    Args:
        db: Database manager instance.

    Returns:
        Dict with items whose warranties expire within 90 days.
    """
    today = date.today()
    cutoff = today + timedelta(days=90)

    rows = db.fetchall(
        """
        SELECT id, name, category, model, purchase_date,
               purchase_price, currency, warranty_expires,
               warranty_type, status
        FROM items
        WHERE warranty_expires IS NOT NULL
              AND warranty_expires >= ?
              AND warranty_expires <= ?
              AND status = 'active'
        ORDER BY warranty_expires ASC
        """,
        (today.isoformat(), cutoff.isoformat()),
    )

    return {
        "as_of": today.isoformat(),
        "cutoff_date": cutoff.isoformat(),
        "total": len(rows),
        "items": [dict(r) for r in rows],
    }


def register_resources(mcp: Any) -> None:
    """Register all resources on the MCP server instance.

    Args:
        mcp: The MCPServer instance to register resources on.
    """

    @mcp.resource("alibi://transactions/recent")  # type: ignore[misc,untyped-decorator]
    def recent_transactions() -> str:
        """Last 30 days of transactions."""
        import json

        db = _get_db()
        data = get_recent_transactions(db)
        return json.dumps(data, default=str)

    @mcp.resource("alibi://reports/monthly/{year}/{month}")  # type: ignore[misc,untyped-decorator]
    def monthly_report(year: int, month: int) -> str:
        """Monthly spending report for a given year and month."""
        import json

        db = _get_db()
        data = get_monthly_report(db, year, month)
        return json.dumps(data, default=str)

    @mcp.resource("alibi://items/warranties/expiring")  # type: ignore[misc,untyped-decorator]
    def expiring_warranties() -> str:
        """Items with warranties expiring in the next 90 days."""
        import json

        db = _get_db()
        data = get_expiring_warranties(db)
        return json.dumps(data, default=str)
