"""Spending anomaly detection."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager

from alibi.analytics.patterns import _extract_category
from alibi.normalizers.vendors import normalize_vendor


@dataclass
class SpendingAnomaly:
    """A detected spending anomaly."""

    fact_id: str
    vendor: str
    amount: Decimal
    date: date
    anomaly_type: str  # "high_amount", "unusual_vendor", "unusual_category"
    severity: float  # 0-1
    explanation: str


def detect_anomalies(
    db: "DatabaseManager",
    lookback_days: int = 90,
    std_threshold: float = 2.0,
) -> list[SpendingAnomaly]:
    """Detect unusual spending patterns.

    Detection methods:
    1. High amount: Facts > threshold std dev above category mean
    2. Unusual vendor: First fact with a vendor AND high amount
    3. Unusual category: Significant spending in a category with little history

    Args:
        db: Database manager
        lookback_days: Days of history to analyze
        std_threshold: Number of std devs to flag as anomaly

    Returns:
        List of SpendingAnomaly sorted by severity
    """
    today = date.today()
    start_date = today - timedelta(days=lookback_days)

    # Get all facts (purchases/subscriptions) in lookback period
    rows = db.fetchall(
        """
        SELECT f.id, f.vendor, f.total_amount, f.event_date,
               GROUP_CONCAT(DISTINCT fi.category) as categories
        FROM facts f
        LEFT JOIN fact_items fi ON f.id = fi.fact_id
        WHERE f.fact_type IN ('purchase', 'subscription_payment')
              AND f.event_date >= ?
        GROUP BY f.id
        ORDER BY f.event_date
        """,
        (start_date.isoformat(),),
    )

    if not rows:
        return []

    # Build statistics
    category_stats: dict[str, list[float]] = defaultdict(list)
    vendor_history: dict[str, list[tuple[str, float, date]]] = defaultdict(list)
    all_amounts: list[float] = []

    transactions: list[tuple[str, str, Decimal, date, str | None]] = []

    for row in rows:
        txn_id = row[0]
        vendor = row[1] or "Unknown"
        amount = Decimal(str(row[2])) if row[2] else Decimal("0")
        txn_date = row[3]
        categories = row[4]

        if isinstance(txn_date, str):
            txn_date = date.fromisoformat(txn_date)

        amount_float = float(amount)
        all_amounts.append(amount_float)

        # Track by category (from fact_items)
        category = _extract_category(categories) if categories else None
        if category:
            category_stats[category].append(amount_float)

        # Track by vendor
        normalized_vendor = normalize_vendor(vendor)
        vendor_history[normalized_vendor].append((txn_id, amount_float, txn_date))

        transactions.append((txn_id, vendor, amount, txn_date, category))

    if not all_amounts:
        return []

    # Calculate overall statistics
    overall_mean = statistics.mean(all_amounts)
    overall_std = statistics.stdev(all_amounts) if len(all_amounts) > 1 else 0

    anomalies: list[SpendingAnomaly] = []

    for txn_id, vendor, amount, txn_date, category in transactions:
        amount_float = float(amount)

        # Check 1: High amount relative to category
        if category and len(category_stats[category]) > 2:
            cat_amounts = category_stats[category]
            cat_mean = statistics.mean(cat_amounts)
            cat_std = statistics.stdev(cat_amounts) if len(cat_amounts) > 1 else 0

            if cat_std > 0:
                z_score = (amount_float - cat_mean) / cat_std
                if z_score > std_threshold:
                    severity = min(1.0, z_score / 5.0)
                    anomalies.append(
                        SpendingAnomaly(
                            fact_id=txn_id,
                            vendor=vendor,
                            amount=amount,
                            date=txn_date,
                            anomaly_type="high_amount",
                            severity=round(severity, 2),
                            explanation=(
                                f"Amount is {z_score:.1f} std devs above "
                                f"average for {category} ({cat_mean:.2f})"
                            ),
                        )
                    )
                    continue

        # Check 2: High amount relative to overall spending
        if overall_std > 0:
            z_score = (amount_float - overall_mean) / overall_std
            if z_score > std_threshold:
                # Check if this is a new vendor
                normalized = normalize_vendor(vendor)
                vendor_txns = vendor_history.get(normalized, [])

                # Find if this is the first transaction with this vendor
                first_txn = (
                    min(vendor_txns, key=lambda x: x[2]) if vendor_txns else None
                )

                if first_txn and first_txn[0] == txn_id and len(vendor_txns) == 1:
                    # New vendor with high amount
                    severity = min(1.0, z_score / 4.0)
                    anomalies.append(
                        SpendingAnomaly(
                            fact_id=txn_id,
                            vendor=vendor,
                            amount=amount,
                            date=txn_date,
                            anomaly_type="unusual_vendor",
                            severity=round(severity, 2),
                            explanation=(
                                f"First transaction with '{vendor}' and amount "
                                f"is {z_score:.1f} std devs above average "
                                f"({overall_mean:.2f})"
                            ),
                        )
                    )
                    continue

                # Just high amount
                severity = min(1.0, z_score / 5.0)
                anomalies.append(
                    SpendingAnomaly(
                        fact_id=txn_id,
                        vendor=vendor,
                        amount=amount,
                        date=txn_date,
                        anomaly_type="high_amount",
                        severity=round(severity, 2),
                        explanation=(
                            f"Amount is {z_score:.1f} std devs above "
                            f"overall average ({overall_mean:.2f})"
                        ),
                    )
                )

    # Sort by severity
    return sorted(anomalies, key=lambda a: a.severity, reverse=True)
