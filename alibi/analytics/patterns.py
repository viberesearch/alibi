"""Spending pattern analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager


@dataclass
class MonthlyTrend:
    """Monthly spending trend data."""

    month: str  # "YYYY-MM"
    total_expenses: Decimal
    total_income: Decimal
    net: Decimal
    transaction_count: int
    top_categories: list[tuple[str, Decimal]] = field(default_factory=list)


@dataclass
class CategoryTrend:
    """Trend data for a specific category."""

    category: str
    months: list[str]
    amounts: list[Decimal]
    trend_direction: str  # "increasing", "decreasing", "stable"
    avg_monthly: Decimal


@dataclass
class SpendingInsights:
    """Comprehensive spending insights."""

    period_start: date
    period_end: date
    monthly_trends: list[MonthlyTrend]
    category_trends: list[CategoryTrend]
    savings_rate: float
    biggest_increase_category: str | None
    biggest_decrease_category: str | None


def analyze_spending_patterns(
    db: "DatabaseManager",
    months: int = 6,
) -> SpendingInsights:
    """Analyze spending patterns over time.

    Note: Income tracking requires v2 income facts (not yet implemented).
    Currently only tracks expenses (purchases, subscriptions, refunds).

    Args:
        db: Database manager
        months: Number of months to analyze

    Returns:
        SpendingInsights with trends and analysis
    """
    today = date.today()
    period_end = today
    period_start = date(today.year, today.month, 1) - timedelta(days=months * 31)
    period_start = date(period_start.year, period_start.month, 1)

    # Get all facts in period
    rows = db.fetchall(
        """
        SELECT f.id, f.fact_type, f.total_amount, f.event_date, f.vendor,
               GROUP_CONCAT(DISTINCT fi.category) as categories
        FROM facts f
        LEFT JOIN fact_items fi ON f.id = fi.fact_id
        WHERE f.event_date >= ? AND f.event_date <= ?
        GROUP BY f.id
        ORDER BY f.event_date
        """,
        (period_start.isoformat(), period_end.isoformat()),
    )

    # Build monthly data
    monthly_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "expenses": Decimal("0"),
            "income": Decimal("0"),
            "count": 0,
            "categories": defaultdict(Decimal),
        }
    )

    category_monthly: dict[str, dict[str, Decimal]] = defaultdict(
        lambda: defaultdict(Decimal)
    )

    for row in rows:
        fact_type = row[1]
        amount = Decimal(str(row[2])) if row[2] else Decimal("0")
        txn_date = row[3]
        categories_str = row[5] or ""

        if isinstance(txn_date, str):
            txn_date = date.fromisoformat(txn_date)

        month_key = txn_date.strftime("%Y-%m")

        # Map fact_type to expense/income
        if fact_type in ("purchase", "subscription_payment"):
            monthly_data[month_key]["expenses"] += amount
        elif fact_type == "refund":
            monthly_data[month_key]["income"] += amount

        monthly_data[month_key]["count"] += 1

        # Extract category from fact_items
        category = _extract_category(categories_str)
        if category and fact_type in ("purchase", "subscription_payment"):
            monthly_data[month_key]["categories"][category] += amount
            category_monthly[category][month_key] += amount

    # Build monthly trends
    monthly_trends: list[MonthlyTrend] = []
    sorted_months = sorted(monthly_data.keys())

    for month in sorted_months:
        data = monthly_data[month]
        expenses = data["expenses"]
        income = data["income"]

        # Get top categories
        categories = data["categories"]
        top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]

        monthly_trends.append(
            MonthlyTrend(
                month=month,
                total_expenses=expenses,
                total_income=income,
                net=income - expenses,
                transaction_count=data["count"],
                top_categories=top_cats,
            )
        )

    # Build category trends
    category_trends: list[CategoryTrend] = []
    for category, month_amounts in category_monthly.items():
        trend = _calculate_category_trend(category, month_amounts, sorted_months)
        if trend:
            category_trends.append(trend)

    # Sort by average spending
    category_trends.sort(key=lambda x: x.avg_monthly, reverse=True)

    # Calculate savings rate
    total_income = sum(m.total_income for m in monthly_trends)
    total_expenses = sum(m.total_expenses for m in monthly_trends)
    savings_rate = 0.0
    if total_income > 0:
        savings_rate = float((total_income - total_expenses) / total_income)

    # Find biggest changes
    biggest_increase = None
    biggest_decrease = None
    max_increase = Decimal("-999999")
    max_decrease = Decimal("999999")

    for trend in category_trends:
        if len(trend.amounts) >= 2:
            change = trend.amounts[-1] - trend.amounts[0]
            if change > max_increase:
                max_increase = change
                biggest_increase = trend.category
            if change < max_decrease:
                max_decrease = change
                biggest_decrease = trend.category

    return SpendingInsights(
        period_start=period_start,
        period_end=period_end,
        monthly_trends=monthly_trends,
        category_trends=category_trends,
        savings_rate=round(savings_rate, 3),
        biggest_increase_category=biggest_increase,
        biggest_decrease_category=biggest_decrease,
    )


def _extract_category(categories: str | None) -> str | None:
    """Extract first category from comma-separated category string.

    In v2, fact_items store category directly (e.g. 'groceries'),
    not as tag paths (e.g. 'category/groceries').
    """
    if not categories:
        return None

    first = categories.split(",")[0].strip()
    return first if first else None


def _calculate_category_trend(
    category: str,
    month_amounts: dict[str, Decimal],
    all_months: list[str],
) -> CategoryTrend | None:
    """Calculate trend for a category."""
    amounts: list[Decimal] = []
    months_with_data: list[str] = []

    for month in all_months:
        amount = month_amounts.get(month, Decimal("0"))
        amounts.append(amount)
        months_with_data.append(month)

    if not amounts:
        return None

    avg_monthly = sum(amounts) / len(amounts)

    # Determine trend direction using simple linear regression
    trend_direction = "stable"
    if len(amounts) >= 2:
        # Calculate slope
        n = len(amounts)
        x_mean = (n - 1) / 2
        y_mean = float(sum(amounts)) / n

        numerator = sum((i - x_mean) * (float(amounts[i]) - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            slope = numerator / denominator
            # Normalize slope by average
            if y_mean > 0:
                normalized_slope = slope / y_mean
                if normalized_slope > 0.05:
                    trend_direction = "increasing"
                elif normalized_slope < -0.05:
                    trend_direction = "decreasing"

    return CategoryTrend(
        category=category,
        months=months_with_data,
        amounts=amounts,
        trend_direction=trend_direction,
        avg_monthly=Decimal(str(round(float(avg_monthly), 2))),
    )


def compare_periods(
    db: "DatabaseManager",
    period1_start: date,
    period1_end: date,
    period2_start: date,
    period2_end: date,
) -> dict[str, Any]:
    """Compare spending between two periods.

    Args:
        db: Database manager
        period1_start: Start of first period
        period1_end: End of first period
        period2_start: Start of second period
        period2_end: End of second period

    Returns:
        Dict with comparison metrics
    """

    def get_period_stats(start: date, end: date) -> dict[str, Any]:
        """Get statistics for a period."""
        rows = db.fetchall(
            """
            SELECT f.fact_type, f.total_amount, f.vendor,
                   GROUP_CONCAT(DISTINCT fi.category) as categories
            FROM facts f
            LEFT JOIN fact_items fi ON f.id = fi.fact_id
            WHERE f.event_date >= ? AND f.event_date <= ?
            GROUP BY f.id
            """,
            (start.isoformat(), end.isoformat()),
        )

        stats: dict[str, Any] = {
            "total_expenses": Decimal("0"),
            "total_income": Decimal("0"),
            "transaction_count": 0,
            "categories": defaultdict(Decimal),
            "vendors": defaultdict(Decimal),
        }

        for row in rows:
            fact_type = row[0]
            amount = Decimal(str(row[1])) if row[1] else Decimal("0")
            vendor = row[2] or "Unknown"
            categories_str = row[3] or ""

            stats["transaction_count"] += 1

            if fact_type in ("purchase", "subscription_payment"):
                stats["total_expenses"] += amount
                stats["vendors"][vendor] += amount
                category = _extract_category(categories_str)
                if category:
                    stats["categories"][category] += amount
            elif fact_type == "refund":
                stats["total_income"] += amount

        return stats

    period1 = get_period_stats(period1_start, period1_end)
    period2 = get_period_stats(period2_start, period2_end)

    # Calculate changes
    def pct_change(old: Decimal, new: Decimal) -> float:
        if old == 0:
            return 100.0 if new > 0 else 0.0
        return round(float((new - old) / old * 100), 1)

    expense_change = pct_change(period1["total_expenses"], period2["total_expenses"])
    income_change = pct_change(period1["total_income"], period2["total_income"])

    # Category changes
    all_categories = set(period1["categories"].keys()) | set(
        period2["categories"].keys()
    )
    category_changes: list[dict[str, Any]] = []

    for cat in all_categories:
        old_val = period1["categories"].get(cat, Decimal("0"))
        new_val = period2["categories"].get(cat, Decimal("0"))
        change = pct_change(old_val, new_val)
        category_changes.append(
            {
                "category": cat,
                "period1": float(old_val),
                "period2": float(new_val),
                "change_pct": change,
            }
        )

    category_changes.sort(key=lambda x: abs(x["change_pct"]), reverse=True)

    return {
        "period1": {
            "start": period1_start.isoformat(),
            "end": period1_end.isoformat(),
            "total_expenses": float(period1["total_expenses"]),
            "total_income": float(period1["total_income"]),
            "transaction_count": period1["transaction_count"],
        },
        "period2": {
            "start": period2_start.isoformat(),
            "end": period2_end.isoformat(),
            "total_expenses": float(period2["total_expenses"]),
            "total_income": float(period2["total_income"]),
            "transaction_count": period2["transaction_count"],
        },
        "changes": {
            "expense_change_pct": expense_change,
            "income_change_pct": income_change,
            "category_changes": category_changes[:10],
        },
    }
