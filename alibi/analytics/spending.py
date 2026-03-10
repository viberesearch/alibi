"""V2 spending analytics from facts.

Provides spending trends, vendor concentration, item frequency,
and seasonal patterns based on the v2 atom-cloud-fact data model.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager


@dataclass
class VendorSpend:
    """Spending summary for a single vendor."""

    vendor: str
    vendor_key: str | None
    total: Decimal
    count: int
    avg_amount: Decimal
    first_date: date
    last_date: date
    share_pct: float = 0.0


@dataclass
class MonthlySpend:
    """Spending totals for a single month."""

    month: str  # YYYY-MM
    total: Decimal
    count: int
    avg_amount: Decimal


@dataclass
class ItemFrequency:
    """Purchase frequency for a product."""

    name: str
    name_normalized: str
    count: int
    total_spent: Decimal
    avg_price: Decimal
    vendors: list[str] = field(default_factory=list)


@dataclass
class SeasonalPattern:
    """Spending pattern by month-of-year."""

    month_number: int  # 1-12
    month_name: str
    avg_spend: Decimal
    total_years: int
    deviation_from_mean: float  # Percentage above/below annual average


def spending_by_vendor(
    db: "DatabaseManager",
    date_from: date | None = None,
    date_to: date | None = None,
    limit: int = 50,
) -> list[VendorSpend]:
    """Get top vendors by total spend.

    Args:
        db: Database manager
        date_from: Start date filter
        date_to: End date filter
        limit: Max vendors to return

    Returns:
        List of VendorSpend sorted by total descending
    """
    from alibi.db import v2_store

    facts = v2_store.list_facts(
        db, date_from=date_from, date_to=date_to, fact_type="purchase", limit=5000
    )

    if not facts:
        return []

    # Group by vendor_key or vendor name
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in facts:
        key = f.get("vendor_key") or f.get("vendor") or "Unknown"
        groups[key].append(f)

    grand_total = Decimal("0")
    results: list[VendorSpend] = []

    for key, vendor_facts in groups.items():
        amounts = []
        dates = []
        vendor_name = "Unknown"

        for f in vendor_facts:
            if f.get("total_amount") is not None:
                amounts.append(Decimal(str(f["total_amount"])))
            event_date = f.get("event_date")
            if isinstance(event_date, str):
                event_date = date.fromisoformat(event_date)
            if event_date:
                dates.append(event_date)
            if f.get("vendor"):
                vendor_name = f["vendor"]

        if not amounts or not dates:
            continue

        total = sum(amounts, Decimal("0"))
        grand_total += total

        results.append(
            VendorSpend(
                vendor=vendor_name,
                vendor_key=key if key != vendor_name else None,
                total=total,
                count=len(amounts),
                avg_amount=Decimal(str(round(float(total) / len(amounts), 2))),
                first_date=min(dates),
                last_date=max(dates),
            )
        )

    # Calculate share percentages
    if grand_total > 0:
        for vs in results:
            vs.share_pct = round(float(vs.total / grand_total * 100), 1)

    results.sort(key=lambda x: x.total, reverse=True)
    return results[:limit]


def spending_by_month(
    db: "DatabaseManager",
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[MonthlySpend]:
    """Get monthly spending totals.

    Args:
        db: Database manager
        date_from: Start date filter
        date_to: End date filter

    Returns:
        List of MonthlySpend sorted chronologically
    """
    from alibi.db import v2_store

    facts = v2_store.list_facts(
        db, date_from=date_from, date_to=date_to, fact_type="purchase", limit=10000
    )

    if not facts:
        return []

    monthly: dict[str, list[Decimal]] = defaultdict(list)

    for f in facts:
        if f.get("total_amount") is None:
            continue

        event_date = f.get("event_date")
        if isinstance(event_date, str):
            event_date = date.fromisoformat(event_date)
        if not event_date:
            continue

        month_key = f"{event_date.year}-{event_date.month:02d}"
        monthly[month_key].append(Decimal(str(f["total_amount"])))

    results: list[MonthlySpend] = []
    for month_key in sorted(monthly.keys()):
        amounts = monthly[month_key]
        total = sum(amounts, Decimal("0"))
        results.append(
            MonthlySpend(
                month=month_key,
                total=total,
                count=len(amounts),
                avg_amount=Decimal(str(round(float(total) / len(amounts), 2))),
            )
        )

    return results


def item_frequency(
    db: "DatabaseManager",
    date_from: date | None = None,
    date_to: date | None = None,
    limit: int = 50,
) -> list[ItemFrequency]:
    """Get most frequently purchased items.

    Args:
        db: Database manager
        date_from: Start date filter
        date_to: End date filter
        limit: Max items to return

    Returns:
        List of ItemFrequency sorted by count descending
    """
    from alibi.db import v2_store

    facts = v2_store.list_facts(
        db, date_from=date_from, date_to=date_to, fact_type="purchase", limit=5000
    )

    if not facts:
        return []

    # Aggregate by normalized item name
    items: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "name": "",
            "count": 0,
            "total_spent": Decimal("0"),
            "vendors": set(),
        }
    )

    for f in facts:
        fact_items = v2_store.get_fact_items(db, f["id"])
        vendor = f.get("vendor") or "Unknown"

        for item in fact_items:
            norm_name = (item.get("name_normalized") or item.get("name") or "").lower()
            if not norm_name:
                continue

            entry = items[norm_name]
            entry["name"] = item.get("name") or norm_name
            entry["count"] += 1
            if item.get("total_price") is not None:
                entry["total_spent"] += Decimal(str(item["total_price"]))
            entry["vendors"].add(vendor)

    results: list[ItemFrequency] = []
    for norm_name, data in items.items():
        total = data["total_spent"]
        count = data["count"]
        results.append(
            ItemFrequency(
                name=data["name"],
                name_normalized=norm_name,
                count=count,
                total_spent=total,
                avg_price=(
                    Decimal(str(round(float(total) / count, 2)))
                    if count > 0 and total > 0
                    else Decimal("0")
                ),
                vendors=sorted(data["vendors"]),
            )
        )

    results.sort(key=lambda x: x.count, reverse=True)
    return results[:limit]


def seasonal_patterns(
    db: "DatabaseManager",
    min_years: int = 1,
) -> list[SeasonalPattern]:
    """Detect seasonal spending patterns by month-of-year.

    Calculates average spending per calendar month across all years,
    then identifies months that deviate significantly from the mean.

    Args:
        db: Database manager
        min_years: Minimum years of data required

    Returns:
        List of SeasonalPattern for months 1-12
    """
    from alibi.db import v2_store

    facts = v2_store.list_facts(db, fact_type="purchase", limit=50000)

    if not facts:
        return []

    MONTH_NAMES = [
        "",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    # Accumulate by (year, month)
    year_month_totals: dict[tuple[int, int], Decimal] = defaultdict(Decimal)
    years_seen: set[int] = set()

    for f in facts:
        if f.get("total_amount") is None:
            continue
        event_date = f.get("event_date")
        if isinstance(event_date, str):
            event_date = date.fromisoformat(event_date)
        if not event_date:
            continue

        year_month_totals[(event_date.year, event_date.month)] += Decimal(
            str(f["total_amount"])
        )
        years_seen.add(event_date.year)

    total_years = len(years_seen)
    if total_years < min_years:
        return []

    # Calculate monthly averages across years
    month_totals: dict[int, Decimal] = defaultdict(Decimal)
    for (_, month), total in year_month_totals.items():
        month_totals[month] += total

    monthly_avgs = {
        m: Decimal(str(round(float(total) / total_years, 2)))
        for m, total in month_totals.items()
    }

    # Calculate overall monthly average
    all_avgs = list(monthly_avgs.values())
    if not all_avgs:
        return []
    overall_avg = sum(all_avgs, Decimal("0")) / len(all_avgs)

    results: list[SeasonalPattern] = []
    for month_num in range(1, 13):
        avg = monthly_avgs.get(month_num, Decimal("0"))
        deviation = 0.0
        if overall_avg > 0:
            deviation = round(
                float((avg - overall_avg) / overall_avg * Decimal("100")), 1
            )

        results.append(
            SeasonalPattern(
                month_number=month_num,
                month_name=MONTH_NAMES[month_num],
                avg_spend=avg,
                total_years=total_years,
                deviation_from_mean=deviation,
            )
        )

    return results
