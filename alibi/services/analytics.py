"""Analytics service.

Unified interface to all analytics modules (spending, subscriptions,
anomalies, vendor analysis). Wraps existing implementations without
adding new logic.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from alibi.analytics import anomalies as _anomalies
from alibi.analytics import patterns as _patterns
from alibi.analytics import spending as _spending
from alibi.analytics import subscriptions as _subscriptions
from alibi.analytics import vendors as _vendors
from alibi.analytics.anomalies import SpendingAnomaly
from alibi.analytics.patterns import SpendingInsights
from alibi.analytics.spending import MonthlySpend, VendorSpend
from alibi.analytics.subscriptions import SubscriptionPattern
from alibi.analytics.vendors import VendorDeduplicationReport

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager

__all__ = [
    "spending_summary",
    "spending_by_vendor",
    "spending_by_month",
    "spending_analysis",
    "monthly_report",
    "detect_subscriptions",
    "get_upcoming_subscriptions",
    "detect_anomalies",
    "analyze_spending_patterns",
    "compare_periods",
    "vendor_report",
    "cloud_quality_report",
    "vendor_cloud_accuracy",
    "analyze_price_factors",
    "get_category_price_factors",
    "price_factor_summary",
]


def spending_summary(
    db: "DatabaseManager",
    period: str = "month",
    filters: dict[str, Any] | None = None,
) -> list[VendorSpend] | list[MonthlySpend]:
    """Return a spending summary for the requested period type.

    Args:
        db: Database manager.
        period: Aggregation axis. ``"month"`` groups by calendar month;
            ``"vendor"`` groups by vendor. Defaults to ``"month"``.
        filters: Optional filter dict. Recognised keys are ``date_from``
            (``date``), ``date_to`` (``date``), and ``limit`` (``int``,
            vendor period only).

    Returns:
        ``list[MonthlySpend]`` when period is ``"month"``;
        ``list[VendorSpend]`` when period is ``"vendor"``.

    Raises:
        ValueError: If *period* is not ``"month"`` or ``"vendor"``.
    """
    filters = filters or {}
    date_from: date | None = filters.get("date_from")
    date_to: date | None = filters.get("date_to")

    if period == "month":
        return _spending.spending_by_month(db, date_from=date_from, date_to=date_to)
    if period == "vendor":
        limit: int = filters.get("limit", 50)
        return _spending.spending_by_vendor(
            db, date_from=date_from, date_to=date_to, limit=limit
        )
    raise ValueError(f"Unknown period {period!r}. Use 'month' or 'vendor'.")


def spending_by_vendor(
    db: "DatabaseManager",
    filters: dict[str, Any] | None = None,
) -> list[VendorSpend]:
    """Return top vendors by total spend.

    Args:
        db: Database manager.
        filters: Optional filter dict. Recognised keys are ``date_from``
            (``date``), ``date_to`` (``date``), and ``limit`` (``int``).

    Returns:
        List of :class:`~alibi.analytics.spending.VendorSpend` sorted by
        total spend descending.
    """
    filters = filters or {}
    return _spending.spending_by_vendor(
        db,
        date_from=filters.get("date_from"),
        date_to=filters.get("date_to"),
        limit=filters.get("limit", 50),
    )


def spending_by_month(
    db: "DatabaseManager",
    filters: dict[str, Any] | None = None,
) -> list[MonthlySpend]:
    """Return monthly spending totals.

    Args:
        db: Database manager.
        filters: Optional filter dict. Recognised keys are ``date_from``
            (``date``) and ``date_to`` (``date``).

    Returns:
        List of :class:`~alibi.analytics.spending.MonthlySpend` sorted
        chronologically.
    """
    filters = filters or {}
    return _spending.spending_by_month(
        db,
        date_from=filters.get("date_from"),
        date_to=filters.get("date_to"),
    )


def spending_analysis(
    db: "DatabaseManager",
    group_by: str = "month",
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze spending patterns with flexible grouping.

    Args:
        db: Database manager.
        group_by: Grouping axis — ``"day"``, ``"week"``, ``"month"``,
            ``"vendor"``, or ``"category"``.
        filters: Optional filter dict. Recognised keys are ``date_from``
            (``str``) and ``date_to`` (``str``).

    Returns:
        Dict with ``group_by`` and ``data`` keys.

    Raises:
        ValueError: If *group_by* is not one of the allowed values.
    """
    allowed = {"day", "week", "month", "vendor", "category"}
    if group_by not in allowed:
        raise ValueError(
            f"Unknown group_by {group_by!r}. Use one of {sorted(allowed)}."
        )

    filters = filters or {}
    conditions = ["fact_type IN ('purchase', 'subscription_payment')"]
    params: list[Any] = []

    if filters.get("date_from"):
        conditions.append("event_date >= ?")
        params.append(str(filters["date_from"]))
    if filters.get("date_to"):
        conditions.append("event_date <= ?")
        params.append(str(filters["date_to"]))

    where = f" WHERE {' AND '.join(conditions)}"
    from_clause = "FROM facts"

    group_exprs = {
        "month": "strftime('%Y-%m', event_date)",
        "day": "strftime('%Y-%m-%d', event_date)",
        "week": "strftime('%Y-W%W', event_date)",
        "vendor": "COALESCE(vendor, 'Unknown')",
    }

    if group_by == "category":
        from_clause = "FROM facts JOIN fact_items fi ON facts.id = fi.fact_id"
        group_expr = "COALESCE(fi.category, 'Uncategorized')"
    else:
        group_expr = group_exprs[group_by]

    sql = f"""
        SELECT {group_expr} as period,
               COUNT(DISTINCT facts.id) as count,
               SUM(CAST(total_amount AS REAL)) as total,
               AVG(CAST(total_amount AS REAL)) as average,
               MIN(CAST(total_amount AS REAL)) as min_amount,
               MAX(CAST(total_amount AS REAL)) as max_amount
        {from_clause}{where}
        GROUP BY {group_expr}
        ORDER BY period DESC
    """  # noqa: S608

    rows = db.fetchall(sql, tuple(params))
    return {
        "group_by": group_by,
        "data": [dict(r) for r in rows],
    }


def monthly_report(
    db: "DatabaseManager",
    year: int,
    month: int,
) -> dict[str, Any]:
    """Generate a monthly spending report.

    Args:
        db: Database manager.
        year: Calendar year.
        month: Calendar month (1-12).

    Returns:
        Dict with period, expenses, income, top_vendors, artifacts_processed.
    """
    date_from = f"{year:04d}-{month:02d}-01"
    if month == 12:
        date_to = f"{year + 1:04d}-01-01"
    else:
        date_to = f"{year:04d}-{month + 1:02d}-01"

    total_row = db.fetchone(
        """SELECT COUNT(*) as count,
                  SUM(CAST(total_amount AS REAL)) as total
           FROM facts
           WHERE event_date >= ? AND event_date < ?
                 AND fact_type IN ('purchase', 'subscription_payment')""",
        (date_from, date_to),
    )

    income_row = db.fetchone(
        """SELECT COUNT(*) as count,
                  SUM(CAST(total_amount AS REAL)) as total
           FROM facts
           WHERE event_date >= ? AND event_date < ?
                 AND fact_type = 'refund'""",
        (date_from, date_to),
    )

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

    doc_row = db.fetchone(
        """SELECT COUNT(*) as count
           FROM documents
           WHERE created_at >= ? AND created_at < ?""",
        (date_from, date_to),
    )

    return {
        "period": {"year": year, "month": month},
        "expenses": {
            "count": total_row["count"] if total_row else 0,
            "total": round(total_row["total"] or 0, 2) if total_row else 0,
        },
        "income": {
            "count": income_row["count"] if income_row else 0,
            "total": round(income_row["total"] or 0, 2) if income_row else 0,
        },
        "top_vendors": [dict(r) for r in vendor_rows],
        "artifacts_processed": doc_row["count"] if doc_row else 0,
    }


def detect_subscriptions(
    db: "DatabaseManager",
    min_occurrences: int = 3,
    min_confidence: float = 0.5,
) -> list[SubscriptionPattern]:
    """Detect subscription and recurring-payment patterns.

    Args:
        db: Database manager.
        min_occurrences: Minimum occurrences to consider recurring.
        min_confidence: Minimum confidence threshold.

    Returns:
        List of :class:`~alibi.analytics.subscriptions.SubscriptionPattern`
        sorted by confidence descending.
    """
    return _subscriptions.detect_subscriptions(
        db, min_occurrences=min_occurrences, min_confidence=min_confidence
    )


def get_upcoming_subscriptions(
    patterns: list[SubscriptionPattern],
    days_ahead: int = 30,
) -> list[tuple[SubscriptionPattern, date]]:
    """Get subscription payments expected in the next N days.

    Args:
        patterns: Pre-computed subscription patterns.
        days_ahead: Number of days to look ahead.

    Returns:
        List of (pattern, expected_date) tuples.
    """
    return _subscriptions.get_upcoming_subscriptions(patterns, days_ahead=days_ahead)


def detect_anomalies(
    db: "DatabaseManager",
    lookback_days: int = 90,
    std_threshold: float = 2.0,
) -> list[SpendingAnomaly]:
    """Detect unusual spending patterns.

    Args:
        db: Database manager.
        lookback_days: Days of history for baseline.
        std_threshold: Standard deviations to flag as anomaly.

    Returns:
        List of :class:`~alibi.analytics.anomalies.SpendingAnomaly`
        sorted by severity descending.
    """
    return _anomalies.detect_anomalies(
        db,
        lookback_days=lookback_days,
        std_threshold=std_threshold,
    )


def analyze_spending_patterns(
    db: "DatabaseManager",
    months: int = 6,
) -> SpendingInsights:
    """Analyze spending patterns over time.

    Args:
        db: Database manager.
        months: Number of months to analyze.

    Returns:
        :class:`~alibi.analytics.patterns.SpendingInsights` with monthly
        trends, category trends, and savings rate.
    """
    return _patterns.analyze_spending_patterns(db, months=months)


def compare_periods(
    db: "DatabaseManager",
    period1_start: date,
    period1_end: date,
    period2_start: date,
    period2_end: date,
) -> dict[str, Any]:
    """Compare spending between two date ranges.

    Args:
        db: Database manager.
        period1_start: Start of first period.
        period1_end: End of first period.
        period2_start: Start of second period.
        period2_end: End of second period.

    Returns:
        Comparison dict with totals, differences, and category breakdowns.
    """
    return _patterns.compare_periods(
        db, period1_start, period1_end, period2_start, period2_end
    )


def vendor_report(
    db: "DatabaseManager",
) -> VendorDeduplicationReport:
    """Return a vendor alias / deduplication report.

    Args:
        db: Database manager.

    Returns:
        :class:`~alibi.analytics.vendors.VendorDeduplicationReport` with
        alias groups, total vendor counts, and unkeyed vendor names.
    """
    return _vendors.vendor_deduplication_report(db)


def verify_extractions(
    db: "DatabaseManager",
    doc_ids: list[str] | None = None,
    limit: int = 20,
    api_key: str | None = None,
) -> list[Any]:
    """Cross-validate extracted receipts via Gemini batch verification.

    Args:
        db: Database manager.
        doc_ids: Specific document IDs, or None for recent.
        limit: Max documents per batch.
        api_key: Optional API key override.

    Returns:
        List of VerificationResult.
    """
    from alibi.extraction.gemini_verifier import verify_documents

    return verify_documents(db, doc_ids=doc_ids, limit=limit, api_key=api_key)


def correction_confusion_matrix(
    db: "DatabaseManager",
    limit: int = 1000,
    min_count: int = 2,
) -> Any:
    """Build confusion matrix from user corrections.

    Args:
        db: Database manager.
        limit: Max events to analyze.
        min_count: Min occurrences to include.

    Returns:
        ConfusionMatrix.
    """
    from alibi.analytics.corrections import build_confusion_matrix

    return build_confusion_matrix(db, limit=limit, min_count=min_count)


def get_refinement_suggestions(
    db: "DatabaseManager",
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Get refinement suggestions from correction analysis.

    Args:
        db: Database manager.
        limit: Max events.

    Returns:
        List of suggestion dicts.
    """
    from alibi.analytics.corrections import get_refinement_suggestions as _get

    return _get(db, limit=limit)


def location_spending(
    db: "DatabaseManager",
    cluster_radius_m: float = 100.0,
) -> list[Any]:
    """Spending aggregated by location.

    Args:
        db: Database manager.
        cluster_radius_m: Cluster radius in meters.

    Returns:
        List of LocationSpending.
    """
    from alibi.analytics.location import spending_by_location

    return spending_by_location(db, cluster_radius_m=cluster_radius_m)


def vendor_branches(
    db: "DatabaseManager",
    vendor_key: str | None = None,
) -> list[Any]:
    """Compare vendor branches across locations.

    Args:
        db: Database manager.
        vendor_key: Filter to vendor, or None for all.

    Returns:
        List of VendorBranchComparison.
    """
    from alibi.analytics.location import vendor_branch_comparison

    return vendor_branch_comparison(db, vendor_key=vendor_key)


def nearby_vendors(
    db: "DatabaseManager",
    lat: float,
    lng: float,
    radius_m: float = 2000.0,
    limit: int = 10,
) -> list[Any]:
    """Suggest vendors near a location.

    Args:
        db: Database manager.
        lat: Latitude.
        lng: Longitude.
        radius_m: Search radius in meters.
        limit: Max suggestions.

    Returns:
        List of LocationSuggestion.
    """
    from alibi.analytics.location import nearby_vendor_suggestions

    return nearby_vendor_suggestions(db, lat, lng, radius_m=radius_m, limit=limit)


def cloud_quality_report(
    db: "DatabaseManager",
    limit_vendors: int = 20,
    limit_trends: int = 12,
) -> Any:
    """Build a comprehensive cloud formation quality report.

    Args:
        db: Database manager.
        limit_vendors: Max vendors in accuracy breakdown.
        limit_trends: Max monthly periods in trend data.

    Returns:
        CloudQualityReport.
    """
    from alibi.analytics.cloud_quality import build_quality_report

    return build_quality_report(
        db, limit_vendors=limit_vendors, limit_trends=limit_trends
    )


def vendor_cloud_accuracy(
    db: "DatabaseManager",
    vendor_key: str,
) -> Any:
    """Get cloud formation accuracy for a specific vendor.

    Args:
        db: Database manager.
        vendor_key: Vendor key to query.

    Returns:
        VendorAccuracy or None.
    """
    from alibi.analytics.cloud_quality import get_vendor_cloud_accuracy

    return get_vendor_cloud_accuracy(db, vendor_key)


def analyze_price_factors(
    db: "DatabaseManager",
    comparable_name: str | None = None,
    category: str | None = None,
    min_observations: int = 3,
) -> list[Any]:
    """Discover which product attributes influence price.

    For each product (grouped by comparable_name), computes the marginal
    price impact of attributes like product_variant, brand, and product
    annotations (organic, free-range, etc.).

    Args:
        db: Database manager.
        comparable_name: Analyze a specific product (or None for all).
        category: Filter to a category (e.g., "Dairy", "Eggs").
        min_observations: Minimum total observations to analyze a product.

    Returns:
        List of ProductPriceProfile sorted by total_observations descending.
    """
    from alibi.analytics.price_factors import analyze_price_factors as _analyze

    return _analyze(
        db,
        comparable_name=comparable_name,
        category=category,
        min_observations=min_observations,
    )


def get_category_price_factors(
    db: "DatabaseManager",
    category: str,
    min_observations: int = 5,
) -> list[Any]:
    """Get price factors aggregated across all products in a category.

    Args:
        db: Database manager.
        category: Category name (e.g., "Dairy").
        min_observations: Minimum observations per product.

    Returns:
        List of PriceFactor aggregated across all products in the category.
    """
    from alibi.analytics.price_factors import get_category_price_factors as _get

    return _get(db, category=category, min_observations=min_observations)


def price_factor_summary(
    db: "DatabaseManager",
    min_observations: int = 5,
) -> dict[str, Any]:
    """Get a high-level summary of price factors across all categories.

    Args:
        db: Database manager.
        min_observations: Minimum observations per product.

    Returns:
        Dict with categories, top_factors, and products_analyzed.
    """
    from alibi.analytics.price_factors import price_factor_summary as _summary

    return _summary(db, min_observations=min_observations)
