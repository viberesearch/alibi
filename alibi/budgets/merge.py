"""Merge logic for budget scenarios.

Implements a simplified ma-engine tree pattern:
- Trunk (actual): Real spending from transactions (immutable historical data)
- Base (projected): Inertial projection based on recurring patterns
- Branches (scenarios): "What if" budgets with delta-only storage

Overrides replace actuals for matching (category, period) pairs.
Non-overridden categories pass through from actuals unchanged.
"""

from decimal import Decimal
from typing import Optional

from alibi.budgets.models import BudgetComparison, BudgetEntry


def merge_entries(
    actual: list[BudgetEntry],
    overrides: list[BudgetEntry],
) -> list[BudgetEntry]:
    """Merge actual entries with scenario overrides.

    Overrides replace actuals for matching (category, period) pairs.
    Categories present only in actuals pass through unchanged.
    Categories present only in overrides are added as new entries.

    Args:
        actual: Base entries (typically from real transaction data).
        overrides: Scenario-specific overrides (delta-only storage).

    Returns:
        Merged list of BudgetEntry objects.
    """
    # Build lookup: (category, period) -> entry
    merged: dict[tuple[str, str], BudgetEntry] = {}

    for entry in actual:
        key = (entry.category, entry.period)
        merged[key] = entry

    # Overrides replace matching keys or add new ones
    for entry in overrides:
        key = (entry.category, entry.period)
        merged[key] = entry

    # Return sorted by (period, category) for deterministic output
    return sorted(merged.values(), key=lambda e: (e.period, e.category))


def compute_variance(
    base: list[BudgetEntry],
    compare: list[BudgetEntry],
) -> list[BudgetComparison]:
    """Compute per-category variance between two entry sets.

    For each (category, period) present in either set, computes:
    - variance = compare_amount - base_amount
    - variance_pct = (variance / base_amount) * 100 if base_amount != 0

    Args:
        base: Base scenario entries.
        compare: Comparison scenario entries.

    Returns:
        List of BudgetComparison objects sorted by (period, category).
    """
    base_map: dict[tuple[str, str], Decimal] = {}
    compare_map: dict[tuple[str, str], Decimal] = {}

    for entry in base:
        key = (entry.category, entry.period)
        base_map[key] = base_map.get(key, Decimal("0")) + entry.amount

    for entry in compare:
        key = (entry.category, entry.period)
        compare_map[key] = compare_map.get(key, Decimal("0")) + entry.amount

    # Union of all keys
    all_keys = sorted(set(base_map.keys()) | set(compare_map.keys()))

    comparisons: list[BudgetComparison] = []
    for category, period in all_keys:
        base_amount = base_map.get((category, period), Decimal("0"))
        compare_amount = compare_map.get((category, period), Decimal("0"))
        variance = compare_amount - base_amount

        variance_pct: Optional[Decimal] = None
        if base_amount != Decimal("0"):
            # Round to 2 decimal places for readability
            variance_pct = (variance / base_amount * Decimal("100")).quantize(
                Decimal("0.01")
            )

        comparisons.append(
            BudgetComparison(
                category=category,
                period=period,
                base_amount=base_amount,
                compare_amount=compare_amount,
                variance=variance,
                variance_pct=variance_pct,
            )
        )

    return comparisons
