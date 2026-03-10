"""Distribution forms for audience-specific output.

Each form produces a different level of detail tailored to specific audiences:
- SUMMARY: High-level spending overview (Telegram, quick CLI)
- DETAILED: Full breakdown with line items (Obsidian, PDF)
- ANALYTICAL: Patterns, trends, comparisons (MCP/Claude, CSV analysis)
- TABULAR: Spreadsheet-friendly flat data (CSV/XLSX export)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager


class DistributionForm(str, Enum):
    """Available distribution forms."""

    SUMMARY = "summary"
    DETAILED = "detailed"
    ANALYTICAL = "analytical"
    TABULAR = "tabular"


@dataclass
class DistributionResult:
    """Result of a distribution query."""

    form: DistributionForm
    title: str
    period: dict[str, str]
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def distribute(
    db: DatabaseManager,
    form: DistributionForm,
    date_from: str | None = None,
    date_to: str | None = None,
    space_id: str = "default",
) -> DistributionResult:
    """Generate distribution data in the requested form.

    Args:
        db: Database manager instance
        form: Which distribution form to generate
        date_from: Start date filter (YYYY-MM-DD or YYYY-MM)
        date_to: End date filter (YYYY-MM-DD or YYYY-MM)
        space_id: Space to query

    Returns:
        DistributionResult with audience-appropriate data
    """
    handlers = {
        DistributionForm.SUMMARY: _build_summary,
        DistributionForm.DETAILED: _build_detailed,
        DistributionForm.ANALYTICAL: _build_analytical,
        DistributionForm.TABULAR: _build_tabular,
    }

    handler = handlers[form]
    data = handler(db, date_from, date_to, space_id)

    period = {
        "from": date_from or "all",
        "to": date_to or "now",
    }

    title_map = {
        DistributionForm.SUMMARY: "Spending Summary",
        DistributionForm.DETAILED: "Detailed Report",
        DistributionForm.ANALYTICAL: "Spending Analysis",
        DistributionForm.TABULAR: "Transaction Data",
    }

    return DistributionResult(
        form=form,
        title=title_map[form],
        period=period,
        data=data,
        metadata={
            "space_id": space_id,
            "form": form.value,
        },
    )


def _build_date_filter(
    date_from: str | None,
    date_to: str | None,
    date_column: str = "event_date",
) -> tuple[str, list[str]]:
    """Build SQL WHERE clause fragment for date filtering.

    Args:
        date_from: Start date (inclusive)
        date_to: End date (inclusive)
        date_column: Column name to filter on (default: event_date)

    Returns:
        Tuple of (SQL fragment, parameters list)
    """
    conditions: list[str] = []
    params: list[str] = []

    if date_from:
        conditions.append(f"{date_column} >= ?")
        params.append(date_from)
    if date_to:
        conditions.append(f"{date_column} <= ?")
        params.append(date_to)

    if conditions:
        return " AND " + " AND ".join(conditions), params
    return "", params


def _build_summary(
    db: DatabaseManager,
    date_from: str | None,
    date_to: str | None,
    space_id: str,
) -> dict[str, Any]:
    """Build SUMMARY form data - high-level overview."""
    date_sql, date_params = _build_date_filter(date_from, date_to)

    # Totals
    totals = db.fetchone(
        f"""
        SELECT
            COALESCE(SUM(CASE WHEN fact_type IN ('purchase', 'subscription_payment')
                         THEN CAST(total_amount AS REAL) ELSE 0 END), 0) as expenses,
            COALESCE(SUM(CASE WHEN fact_type = 'refund'
                         THEN CAST(total_amount AS REAL) ELSE 0 END), 0) as income,
            COUNT(*) as count
        FROM facts
        WHERE 1=1{date_sql}
        """,  # noqa: S608
        (*date_params,),
    )

    total_expenses = round(float(totals[0]), 2) if totals else 0.0
    total_income = round(float(totals[1]), 2) if totals else 0.0
    transaction_count = int(totals[2]) if totals else 0

    # Top 5 vendors by spending
    vendor_rows = db.fetchall(
        f"""
        SELECT
            COALESCE(vendor, 'Unknown') as vendor,
            SUM(CAST(total_amount AS REAL)) as total,
            COUNT(*) as count
        FROM facts
        WHERE fact_type IN ('purchase', 'subscription_payment'){date_sql}
        GROUP BY COALESCE(vendor, 'Unknown')
        ORDER BY total DESC
        LIMIT 5
        """,  # noqa: S608
        (*date_params,),
    )

    top_vendors = [
        {
            "vendor": row[0],
            "total": round(float(row[1]), 2),
            "count": int(row[2]),
        }
        for row in vendor_rows
    ]

    # Top 5 categories (from fact items)
    cat_date_sql, cat_date_params = _build_date_filter(
        date_from, date_to, "f.event_date"
    )
    category_rows = db.fetchall(
        f"""
        SELECT
            COALESCE(fi.category, 'uncategorized') as category,
            SUM(CAST(fi.total_price AS REAL)) as total
        FROM fact_items fi
        JOIN facts f ON fi.fact_id = f.id
        WHERE f.fact_type IN ('purchase', 'subscription_payment'){cat_date_sql}
        GROUP BY COALESCE(fi.category, 'uncategorized')
        ORDER BY total DESC
        LIMIT 5
        """,  # noqa: S608
        (*cat_date_params,),
    )

    top_categories = [
        {
            "category": row[0],
            "total": round(float(row[1]), 2),
        }
        for row in category_rows
    ]

    return {
        "total_expenses": total_expenses,
        "total_income": total_income,
        "net": round(total_income - total_expenses, 2),
        "transaction_count": transaction_count,
        "top_vendors": top_vendors,
        "top_categories": top_categories,
    }


def _build_detailed(
    db: DatabaseManager,
    date_from: str | None,
    date_to: str | None,
    space_id: str,
) -> dict[str, Any]:
    """Build DETAILED form data - full breakdown with line items."""
    # Start with summary data
    summary = _build_summary(db, date_from, date_to, space_id)
    date_sql, date_params = _build_date_filter(date_from, date_to)

    # All vendors (no limit)
    all_vendor_rows = db.fetchall(
        f"""
        SELECT
            COALESCE(vendor, 'Unknown') as vendor,
            SUM(CAST(total_amount AS REAL)) as total,
            COUNT(*) as count
        FROM facts
        WHERE fact_type IN ('purchase', 'subscription_payment'){date_sql}
        GROUP BY COALESCE(vendor, 'Unknown')
        ORDER BY total DESC
        """,  # noqa: S608
        (*date_params,),
    )

    all_vendors = [
        {
            "vendor": row[0],
            "total": round(float(row[1]), 2),
            "count": int(row[2]),
        }
        for row in all_vendor_rows
    ]

    # All categories (no limit)
    cat_date_sql, cat_date_params = _build_date_filter(
        date_from, date_to, "f.event_date"
    )
    all_category_rows = db.fetchall(
        f"""
        SELECT
            COALESCE(fi.category, 'uncategorized') as category,
            SUM(CAST(fi.total_price AS REAL)) as total
        FROM fact_items fi
        JOIN facts f ON fi.fact_id = f.id
        WHERE f.fact_type IN ('purchase', 'subscription_payment'){cat_date_sql}
        GROUP BY COALESCE(fi.category, 'uncategorized')
        ORDER BY total DESC
        """,  # noqa: S608
        (*cat_date_params,),
    )

    all_categories = [
        {
            "category": row[0],
            "total": round(float(row[1]), 2),
        }
        for row in all_category_rows
    ]

    # Line items by category
    li_date_sql, li_date_params = _build_date_filter(date_from, date_to, "f.event_date")
    line_item_rows = db.fetchall(
        f"""
        SELECT
            fi.category,
            fi.name,
            fi.quantity,
            fi.unit_price,
            fi.total_price
        FROM fact_items fi
        JOIN facts f ON fi.fact_id = f.id
        WHERE f.fact_type IN ('purchase', 'subscription_payment'){li_date_sql}
        ORDER BY fi.category, fi.name
        """,  # noqa: S608
        (*li_date_params,),
    )

    line_items_by_category: dict[str, list[dict[str, Any]]] = {}
    for row in line_item_rows:
        cat = row[0] or "uncategorized"
        if cat not in line_items_by_category:
            line_items_by_category[cat] = []
        line_items_by_category[cat].append(
            {
                "name": row[1],
                "quantity": float(row[2]) if row[2] else 1.0,
                "unit_price": round(float(row[3]), 2) if row[3] else None,
                "total_price": round(float(row[4]), 2) if row[4] else None,
                "currency": "EUR",
            }
        )

    # Documents processed
    doc_date_sql, doc_date_params = _build_date_filter(date_from, date_to, "created_at")
    artifact_row = db.fetchone(
        f"""
        SELECT COUNT(*) FROM documents
        WHERE 1=1{doc_date_sql}
        """,  # noqa: S608
        (*doc_date_params,),
    )
    artifacts_processed = int(artifact_row[0]) if artifact_row else 0

    # Monthly trend
    trend_rows = db.fetchall(
        f"""
        SELECT
            strftime('%Y-%m', event_date) as month,
            COALESCE(SUM(CASE WHEN fact_type IN ('purchase', 'subscription_payment')
                         THEN CAST(total_amount AS REAL) ELSE 0 END), 0) as expenses,
            COALESCE(SUM(CASE WHEN fact_type = 'refund'
                         THEN CAST(total_amount AS REAL) ELSE 0 END), 0) as income
        FROM facts
        WHERE 1=1{date_sql}
        GROUP BY strftime('%Y-%m', event_date)
        ORDER BY month
        """,  # noqa: S608
        (*date_params,),
    )

    monthly_trend = [
        {
            "month": row[0],
            "expenses": round(float(row[1]), 2),
            "income": round(float(row[2]), 2),
        }
        for row in trend_rows
    ]

    return {
        **summary,
        "all_vendors": all_vendors,
        "all_categories": all_categories,
        "line_items_by_category": line_items_by_category,
        "artifacts_processed": artifacts_processed,
        "monthly_trend": monthly_trend,
    }


def _build_analytical(
    db: DatabaseManager,
    date_from: str | None,
    date_to: str | None,
    space_id: str,
) -> dict[str, Any]:
    """Build ANALYTICAL form data - patterns and trends."""
    date_sql, date_params = _build_date_filter(date_from, date_to)

    # Spending trend by month
    trend_rows = db.fetchall(
        f"""
        SELECT
            strftime('%Y-%m', event_date) as period,
            SUM(CAST(total_amount AS REAL)) as total,
            AVG(CAST(total_amount AS REAL)) as avg
        FROM facts
        WHERE fact_type IN ('purchase', 'subscription_payment'){date_sql}
        GROUP BY strftime('%Y-%m', event_date)
        ORDER BY period
        """,  # noqa: S608
        (*date_params,),
    )

    spending_trend = [
        {
            "period": row[0],
            "total": round(float(row[1]), 2),
            "avg": round(float(row[2]), 2),
        }
        for row in trend_rows
    ]

    # Vendor frequency analysis
    vendor_rows = db.fetchall(
        f"""
        SELECT
            COALESCE(vendor, 'Unknown') as vendor,
            COUNT(*) as count,
            AVG(CAST(total_amount AS REAL)) as avg_amount,
            MIN(event_date) as first_seen,
            MAX(event_date) as last_seen
        FROM facts
        WHERE fact_type IN ('purchase', 'subscription_payment'){date_sql}
        GROUP BY COALESCE(vendor, 'Unknown')
        ORDER BY count DESC
        """,  # noqa: S608
        (*date_params,),
    )

    vendor_frequency = [
        {
            "vendor": row[0],
            "count": int(row[1]),
            "avg_amount": round(float(row[2]), 2),
            "trend": "recurring" if int(row[1]) >= 3 else "occasional",
        }
        for row in vendor_rows
    ]

    # Category distribution (percentages)
    cat_date_sql, cat_date_params = _build_date_filter(
        date_from, date_to, "f.event_date"
    )
    cat_rows = db.fetchall(
        f"""
        SELECT
            COALESCE(fi.category, 'uncategorized') as category,
            SUM(CAST(fi.total_price AS REAL)) as amount
        FROM fact_items fi
        JOIN facts f ON fi.fact_id = f.id
        WHERE f.fact_type IN ('purchase', 'subscription_payment'){cat_date_sql}
        GROUP BY COALESCE(fi.category, 'uncategorized')
        ORDER BY amount DESC
        """,  # noqa: S608
        (*cat_date_params,),
    )

    total_cat_amount = sum(float(row[1]) for row in cat_rows) if cat_rows else 0.0
    category_distribution = [
        {
            "category": row[0],
            "percentage": (
                round(float(row[1]) / total_cat_amount * 100, 1)
                if total_cat_amount > 0
                else 0.0
            ),
            "amount": round(float(row[1]), 2),
        }
        for row in cat_rows
    ]

    # Anomalies - facts significantly above average
    anomaly_rows = db.fetchall(
        f"""
        SELECT
            event_date,
            COALESCE(vendor, 'Unknown') as vendor,
            CAST(total_amount AS REAL) as amount
        FROM facts
        WHERE fact_type IN ('purchase', 'subscription_payment'){date_sql}
          AND CAST(total_amount AS REAL) > (
              SELECT AVG(CAST(total_amount AS REAL)) * 3
              FROM facts
              WHERE fact_type IN ('purchase', 'subscription_payment'){date_sql}
          )
        ORDER BY total_amount DESC
        LIMIT 10
        """,  # noqa: S608
        (*date_params, *date_params),
    )

    anomalies = [
        {
            "date": row[0],
            "vendor": row[1],
            "amount": round(float(row[2]), 2),
            "reason": "Significantly above average spending",
        }
        for row in anomaly_rows
    ]

    # Recurring vendors (3+ facts)
    recurring_rows = db.fetchall(
        f"""
        SELECT
            COALESCE(vendor, 'Unknown') as vendor,
            AVG(CAST(total_amount AS REAL)) as avg_amount,
            COUNT(*) as count
        FROM facts
        WHERE fact_type IN ('purchase', 'subscription_payment'){date_sql}
        GROUP BY COALESCE(vendor, 'Unknown')
        HAVING COUNT(*) >= 3
        ORDER BY count DESC
        """,  # noqa: S608
        (*date_params,),
    )

    recurring = [
        {
            "vendor": row[0],
            "avg_amount": round(float(row[1]), 2),
            "frequency": (
                "weekly"
                if int(row[2]) >= 12
                else "monthly" if int(row[2]) >= 3 else "occasional"
            ),
        }
        for row in recurring_rows
    ]

    return {
        "spending_trend": spending_trend,
        "vendor_frequency": vendor_frequency,
        "category_distribution": category_distribution,
        "anomalies": anomalies,
        "recurring": recurring,
    }


def _build_tabular(
    db: DatabaseManager,
    date_from: str | None,
    date_to: str | None,
    space_id: str,
) -> dict[str, Any]:
    """Build TABULAR form data - flat rows for spreadsheet export."""
    date_sql, date_params = _build_date_filter(date_from, date_to)

    rows = db.fetchall(
        f"""
        SELECT
            event_date,
            COALESCE(vendor, '') as vendor,
            COALESCE(fact_type, '') as fact_type,
            CAST(total_amount AS REAL) as amount,
            COALESCE(currency, 'EUR') as currency,
            COALESCE(
                (SELECT fi.category FROM fact_items fi
                 WHERE fi.fact_id = f.id
                 LIMIT 1),
                ''
            ) as category
        FROM facts f
        WHERE 1=1{date_sql}
        ORDER BY event_date DESC
        """,  # noqa: S608
        (*date_params,),
    )

    headers = [
        "date",
        "vendor",
        "fact_type",
        "amount",
        "currency",
        "category",
    ]

    data_rows = [
        [
            row[0],
            row[1],
            row[2],
            round(float(row[3]), 2),
            row[4],
            row[5],
        ]
        for row in rows
    ]

    return {
        "headers": headers,
        "rows": data_rows,
    }
