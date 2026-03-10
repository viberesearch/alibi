"""Subscription/recurring payment detection from v2 facts."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager

from alibi.normalizers.vendors import normalize_vendor_slug as normalize_vendor_name

# Standard subscription periods: (target_days, tolerance_days)
SUBSCRIPTION_PERIODS: dict[str, tuple[int, int]] = {
    "weekly": (7, 2),
    "biweekly": (14, 3),
    "monthly": (30, 4),
    "quarterly": (91, 10),
    "semi_annual": (182, 14),
    "annual": (365, 21),
}


@dataclass
class SubscriptionPattern:
    """A detected subscription pattern from v2 facts."""

    vendor: str
    vendor_normalized: str
    avg_amount: Decimal
    period_type: str  # "weekly", "monthly", "quarterly", "annual", "irregular"
    frequency_days: int
    confidence: float
    last_date: date
    next_expected: date
    occurrences: int
    amount_variance: float
    fact_ids: list[str] = field(default_factory=list)


def detect_subscriptions(
    db: "DatabaseManager",
    min_occurrences: int = 3,
    amount_tolerance: float = 0.10,
    min_confidence: float = 0.5,
) -> list[SubscriptionPattern]:
    """Detect subscription patterns from v2 facts.

    Algorithm:
    1. Query all purchase facts grouped by vendor
    2. For each vendor, cluster by similar amounts
    3. Calculate interval statistics and classify period type
    4. Compute confidence based on regularity
    5. Return patterns above min_confidence

    Args:
        db: Database manager
        min_occurrences: Minimum facts to consider recurring
        amount_tolerance: Maximum % variance in amounts (default 10%)
        min_confidence: Minimum confidence score (0-1)

    Returns:
        List of SubscriptionPattern sorted by confidence
    """
    from alibi.db.v2_store import get_facts_grouped_by_vendor

    vendor_facts = get_facts_grouped_by_vendor(db, fact_type="purchase")

    if not vendor_facts:
        return []

    # Facts are already grouped by vendor_key (or vendor name fallback).
    # Build entry tuples for clustering.
    vendor_groups: dict[str, list[tuple[str, Decimal, date, str]]] = defaultdict(list)
    for group_key, facts in vendor_facts.items():
        for f in facts:
            amount = Decimal(str(f["total_amount"])) if f.get("total_amount") else None
            event_date = f.get("event_date")
            if isinstance(event_date, str):
                event_date = date.fromisoformat(event_date)

            vendor_name = f.get("vendor") or group_key
            if amount is not None and event_date is not None:
                vendor_groups[group_key].append(
                    (vendor_name, amount, event_date, f["id"])
                )

    patterns: list[SubscriptionPattern] = []

    for group_key, entries in vendor_groups.items():
        if len(entries) < min_occurrences:
            continue

        # Derive normalized display name from the most common vendor name
        name_counts: dict[str, int] = defaultdict(int)
        for vendor_name, _, _, _ in entries:
            name_counts[vendor_name] += 1
        display_name = max(name_counts, key=lambda v: name_counts[v])
        normalized_name = normalize_vendor_name(display_name)

        # Cluster by similar amounts
        clusters = _cluster_by_amount(entries, amount_tolerance)

        for cluster in clusters:
            if len(cluster) < min_occurrences:
                continue

            pattern = _analyze_cluster(normalized_name, cluster, min_confidence)
            if pattern:
                patterns.append(pattern)

    return sorted(patterns, key=lambda p: p.confidence, reverse=True)


def _cluster_by_amount(
    entries: list[tuple[str, Decimal, date, str]],
    tolerance: float,
) -> list[list[tuple[str, Decimal, date, str]]]:
    """Cluster entries by similar amounts."""
    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda x: x[1])

    clusters: list[list[tuple[str, Decimal, date, str]]] = []
    current: list[tuple[str, Decimal, date, str]] = [sorted_entries[0]]
    cluster_avg = float(sorted_entries[0][1])

    for entry in sorted_entries[1:]:
        amount = float(entry[1])
        if cluster_avg > 0:
            diff_pct = abs(amount - cluster_avg) / cluster_avg
            if diff_pct <= tolerance:
                current.append(entry)
                cluster_avg = sum(float(e[1]) for e in current) / len(current)
            else:
                if current:
                    clusters.append(current)
                current = [entry]
                cluster_avg = amount
        else:
            current.append(entry)

    if current:
        clusters.append(current)

    return clusters


def _classify_period(avg_interval: float) -> tuple[str, float]:
    """Classify an average interval into a subscription period type.

    Returns (period_name, period_confidence) where period_confidence
    indicates how well the interval matches a known period.
    """
    best_match = "irregular"
    best_score = 0.0

    for name, (target, tolerance) in SUBSCRIPTION_PERIODS.items():
        deviation = abs(avg_interval - target)
        if deviation <= tolerance:
            score = 1.0 - (deviation / tolerance)
            if score > best_score:
                best_match = name
                best_score = score

    return best_match, best_score


def _analyze_cluster(
    normalized_vendor: str,
    cluster: list[tuple[str, Decimal, date, str]],
    min_confidence: float,
) -> SubscriptionPattern | None:
    """Analyze a cluster of facts to detect a subscription pattern."""
    if len(cluster) < 2:
        return None

    sorted_cluster = sorted(cluster, key=lambda x: x[2])

    # Calculate intervals
    intervals: list[int] = []
    for i in range(1, len(sorted_cluster)):
        days = (sorted_cluster[i][2] - sorted_cluster[i - 1][2]).days
        if days > 0:
            intervals.append(days)

    if not intervals:
        return None

    avg_interval = statistics.mean(intervals)
    std_dev = statistics.stdev(intervals) if len(intervals) > 1 else 0

    # Base confidence from interval regularity
    if avg_interval > 0:
        regularity_conf = max(0.0, 1.0 - (std_dev / avg_interval))
    else:
        regularity_conf = 0.0

    # Period classification bonus
    period_type, period_conf = _classify_period(avg_interval)

    # Combined confidence: regularity (60%) + period match (30%) + occurrences (10%)
    occurrence_bonus = min(0.1, (len(sorted_cluster) - 2) * 0.02)
    confidence = (regularity_conf * 0.6) + (period_conf * 0.3) + occurrence_bonus

    if confidence < min_confidence:
        return None

    # Amount statistics
    amounts = [float(e[1]) for e in sorted_cluster]
    avg_amount = Decimal(str(round(statistics.mean(amounts), 2)))
    amount_variance = 0.0
    if len(amounts) > 1 and avg_amount > 0:
        amount_variance = statistics.stdev(amounts) / float(avg_amount)

    # Most common original vendor name
    vendor_counts: dict[str, int] = defaultdict(int)
    for vendor, _, _, _ in cluster:
        vendor_counts[vendor] += 1
    original_vendor = max(vendor_counts, key=vendor_counts.get)  # type: ignore[arg-type]

    last_date = sorted_cluster[-1][2]
    next_expected = last_date + timedelta(days=int(round(avg_interval)))
    fact_ids = [e[3] for e in sorted_cluster]

    return SubscriptionPattern(
        vendor=original_vendor,
        vendor_normalized=normalized_vendor,
        avg_amount=avg_amount,
        period_type=period_type,
        frequency_days=int(round(avg_interval)),
        confidence=round(confidence, 2),
        last_date=last_date,
        next_expected=next_expected,
        occurrences=len(sorted_cluster),
        amount_variance=round(amount_variance, 3),
        fact_ids=fact_ids,
    )


def mark_subscriptions(
    db: "DatabaseManager",
    pattern: SubscriptionPattern,
) -> int:
    """Update fact_type to subscription_payment for all facts in a pattern.

    Returns count of updated facts.
    """
    from alibi.db.v2_store import update_fact_type

    count = 0
    for fact_id in pattern.fact_ids:
        update_fact_type(db, fact_id, "subscription_payment")
        count += 1

    return count


def get_upcoming_subscriptions(
    patterns: list[SubscriptionPattern],
    days_ahead: int = 30,
) -> list[tuple[SubscriptionPattern, date]]:
    """Get subscription payments expected in next N days.

    Args:
        patterns: Pre-computed patterns
        days_ahead: Number of days to look ahead

    Returns:
        List of (pattern, expected_date) tuples sorted by date
    """
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)

    upcoming: list[tuple[SubscriptionPattern, date]] = []

    for pattern in patterns:
        check_date = pattern.next_expected

        while check_date < today:
            check_date += timedelta(days=pattern.frequency_days)

        while check_date <= cutoff:
            upcoming.append((pattern, check_date))
            check_date += timedelta(days=pattern.frequency_days)

    return sorted(upcoming, key=lambda x: x[1])
