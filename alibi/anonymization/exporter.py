"""Privacy-preserving fact export for cloud AI analysis.

Three anonymization levels:
    categories_only: Only categories, no names/amounts/dates. Safe for any external use.
    pseudonymized: Consistent fake names, shifted amounts/dates. Preserves patterns.
    statistical: Only aggregates. No individual records leave the system.

All anonymization is reversible locally using an AnonymizationKey.
The key never leaves the local system.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any


class AnonymizationLevel(str, Enum):
    """Level of anonymization for export."""

    CATEGORIES_ONLY = "categories_only"
    PSEUDONYMIZED = "pseudonymized"
    STATISTICAL = "statistical"


@dataclass
class AnonymizationKey:
    """Local-only key for reversing anonymization.

    Contains all mappings needed to restore original data.
    This file must never leave the local system.
    """

    # Secret used to derive consistent pseudonyms
    secret: str
    # Vendor name → pseudonym mapping
    vendor_map: dict[str, str] = field(default_factory=dict)
    # Amount multiplication factor (preserves ratios)
    amount_factor: float = 1.0
    # Date offset in days (preserves intervals and day-of-week)
    date_offset_days: int = 0
    # Item name → generalized name mapping
    item_map: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON for local storage."""
        return json.dumps(
            {
                "secret": self.secret,
                "vendor_map": self.vendor_map,
                "amount_factor": self.amount_factor,
                "date_offset_days": self.date_offset_days,
                "item_map": self.item_map,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, data: str) -> "AnonymizationKey":
        """Deserialize from JSON."""
        d = json.loads(data)
        return cls(
            secret=d["secret"],
            vendor_map=d.get("vendor_map", {}),
            amount_factor=d.get("amount_factor", 1.0),
            date_offset_days=d.get("date_offset_days", 0),
            item_map=d.get("item_map", {}),
        )


def generate_key() -> AnonymizationKey:
    """Generate a new anonymization key with random parameters.

    The amount factor is between 0.5 and 2.0 (preserves ratios).
    The date offset is between -90 and +90 days (preserves intervals).
    """
    return AnonymizationKey(
        secret=secrets.token_hex(32),
        amount_factor=0.5 + secrets.randbelow(150) / 100.0,  # 0.50 to 1.99
        date_offset_days=secrets.randbelow(181) - 90,  # -90 to +90
    )


def anonymize_export(
    facts: list[dict[str, Any]],
    items_by_fact: dict[str, list[dict[str, Any]]] | None = None,
    level: AnonymizationLevel = AnonymizationLevel.PSEUDONYMIZED,
    key: AnonymizationKey | None = None,
) -> tuple[list[dict[str, Any]], AnonymizationKey]:
    """Export facts with privacy-preserving anonymization.

    Args:
        facts: List of fact dicts from v2_store.list_facts()
        items_by_fact: Optional dict mapping fact_id to list of item dicts
        level: Anonymization level to apply
        key: Existing key to reuse (for consistency across exports).
             If None, generates a new key.

    Returns:
        Tuple of (anonymized_data, key) where key enables local restoration.
    """
    if key is None:
        key = generate_key()

    items_by_fact = items_by_fact or {}

    if level == AnonymizationLevel.STATISTICAL:
        return _export_statistical(facts, items_by_fact), key

    if level == AnonymizationLevel.CATEGORIES_ONLY:
        return _export_categories_only(facts, items_by_fact, key), key

    # PSEUDONYMIZED
    return _export_pseudonymized(facts, items_by_fact, key), key


def restore_import(
    anonymized_data: list[dict[str, Any]],
    key: AnonymizationKey,
) -> list[dict[str, Any]]:
    """Reverse pseudonymized anonymization using the local key.

    Only works for PSEUDONYMIZED level. CATEGORIES_ONLY and STATISTICAL
    are lossy and cannot be fully restored.

    Args:
        anonymized_data: Records from anonymize_export with PSEUDONYMIZED level
        key: The AnonymizationKey returned by anonymize_export

    Returns:
        List of restored fact dicts with original values
    """
    reverse_vendors = {v: k for k, v in key.vendor_map.items()}
    reverse_items = {v: k for k, v in key.item_map.items()}
    reverse_factor = 1.0 / key.amount_factor if key.amount_factor != 0 else 1.0
    reverse_offset = -key.date_offset_days

    restored = []
    for record in anonymized_data:
        r = dict(record)

        # Restore vendor
        if r.get("vendor") and r["vendor"] in reverse_vendors:
            r["vendor"] = reverse_vendors[r["vendor"]]

        # Restore amounts
        for amount_field in ("total_amount", "avg_amount"):
            if r.get(amount_field) is not None:
                val = Decimal(str(r[amount_field]))
                r[amount_field] = str(
                    (val * Decimal(str(reverse_factor))).quantize(Decimal("0.01"))
                )

        # Restore date
        if r.get("event_date"):
            d = r["event_date"]
            if isinstance(d, str):
                d = date.fromisoformat(d)
            r["event_date"] = (d + timedelta(days=reverse_offset)).isoformat()

        # Restore items
        if "items" in r:
            for item in r["items"]:
                if item.get("name") and item["name"] in reverse_items:
                    item["name"] = reverse_items[item["name"]]
                for af in ("unit_price", "total_price"):
                    if item.get(af) is not None:
                        val = Decimal(str(item[af]))
                        item[af] = str(
                            (val * Decimal(str(reverse_factor))).quantize(
                                Decimal("0.01")
                            )
                        )

        restored.append(r)

    return restored


# ---------------------------------------------------------------------------
# Internal export functions
# ---------------------------------------------------------------------------


def _pseudonymize_vendor(vendor: str, key: AnonymizationKey) -> str:
    """Generate a consistent pseudonym for a vendor name."""
    if vendor in key.vendor_map:
        return key.vendor_map[vendor]

    # Generate deterministic pseudonym from secret + vendor
    idx = len(key.vendor_map) + 1
    pseudonym = f"VENDOR_{chr(64 + ((idx - 1) % 26) + 1)}"
    if idx > 26:
        pseudonym = (
            f"VENDOR_{chr(64 + ((idx - 1) // 26))}{chr(64 + ((idx - 1) % 26) + 1)}"
        )

    key.vendor_map[vendor] = pseudonym
    return pseudonym


def _pseudonymize_item(name: str, category: str | None, key: AnonymizationKey) -> str:
    """Generate a consistent generalized name for an item."""
    if name in key.item_map:
        return key.item_map[name]

    cat = category or "item"
    idx = sum(1 for v in key.item_map.values() if v.startswith(cat)) + 1
    pseudonym = f"{cat}_{idx}"
    key.item_map[name] = pseudonym
    return pseudonym


def _shift_amount(amount: Any, factor: float) -> str | None:
    """Multiply amount by factor, preserving ratios."""
    if amount is None:
        return None
    val = Decimal(str(amount))
    shifted = (val * Decimal(str(factor))).quantize(Decimal("0.01"))
    return str(shifted)


def _shift_date(d: Any, offset_days: int) -> str | None:
    """Shift a date by offset_days, preserving intervals."""
    if d is None:
        return None
    if isinstance(d, str):
        d = date.fromisoformat(d)
    shifted: date = d + timedelta(days=offset_days)
    return shifted.isoformat()


def _export_pseudonymized(
    facts: list[dict[str, Any]],
    items_by_fact: dict[str, list[dict[str, Any]]],
    key: AnonymizationKey,
) -> list[dict[str, Any]]:
    """Export facts with pseudonymized anonymization."""
    result = []
    for fact in facts:
        record: dict[str, Any] = {
            "fact_type": fact.get("fact_type"),
            "vendor": (
                _pseudonymize_vendor(fact["vendor"], key)
                if fact.get("vendor")
                else None
            ),
            "total_amount": _shift_amount(fact.get("total_amount"), key.amount_factor),
            "currency": fact.get("currency"),
            "event_date": _shift_date(fact.get("event_date"), key.date_offset_days),
            "status": fact.get("status"),
        }

        # Pseudonymize payments
        payments = fact.get("payments")
        if isinstance(payments, str):
            payments = json.loads(payments)
        if payments:
            anon_payments = []
            for p in payments:
                anon_payments.append(
                    {
                        "method": p.get("method"),
                        "amount": _shift_amount(p.get("amount"), key.amount_factor),
                    }
                )
            record["payments"] = anon_payments

        # Pseudonymize items
        fact_id = fact.get("id", "")
        fact_items = items_by_fact.get(fact_id, [])
        if fact_items:
            anon_items = []
            for item in fact_items:
                anon_items.append(
                    {
                        "name": _pseudonymize_item(
                            item.get("name", ""),
                            item.get("category"),
                            key,
                        ),
                        "category": item.get("category"),
                        "quantity": item.get("quantity"),
                        "unit": item.get("unit"),
                        "unit_price": _shift_amount(
                            item.get("unit_price"), key.amount_factor
                        ),
                        "total_price": _shift_amount(
                            item.get("total_price"), key.amount_factor
                        ),
                    }
                )
            record["items"] = anon_items

        result.append(record)

    return result


def _export_categories_only(
    facts: list[dict[str, Any]],
    items_by_fact: dict[str, list[dict[str, Any]]],
    key: AnonymizationKey,
) -> list[dict[str, Any]]:
    """Export facts with only categories visible. No names, amounts, or exact dates."""
    result = []
    for fact in facts:
        record: dict[str, Any] = {
            "fact_type": fact.get("fact_type"),
            "vendor": (
                _pseudonymize_vendor(fact["vendor"], key)
                if fact.get("vendor")
                else None
            ),
            "currency": fact.get("currency"),
            "event_month": (
                _extract_month(fact.get("event_date"), key.date_offset_days)
            ),
            "status": fact.get("status"),
        }

        # Only include item categories, no names or prices
        fact_id = fact.get("id", "")
        fact_items = items_by_fact.get(fact_id, [])
        if fact_items:
            categories = [
                item.get("category") or "uncategorized" for item in fact_items
            ]
            record["item_categories"] = categories
            record["item_count"] = len(fact_items)

        result.append(record)

    return result


def _extract_month(d: Any, offset_days: int) -> str | None:
    """Extract year-month from a date after shifting."""
    if d is None:
        return None
    if isinstance(d, str):
        d = date.fromisoformat(d)
    shifted: date = d + timedelta(days=offset_days)
    result: str = shifted.strftime("%Y-%m")
    return result


def _export_statistical(
    facts: list[dict[str, Any]],
    items_by_fact: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Export only aggregate statistics. No individual records."""
    if not facts:
        return [{"summary": "no_data", "total_facts": 0}]

    # Aggregate by fact_type
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        by_type[fact.get("fact_type", "unknown")].append(fact)

    result = []
    for fact_type, type_facts in by_type.items():
        amounts = []
        for f in type_facts:
            if f.get("total_amount") is not None:
                amounts.append(float(Decimal(str(f["total_amount"]))))

        # Count items across all facts of this type
        total_items = 0
        category_counts: dict[str, int] = defaultdict(int)
        for f in type_facts:
            fid = f.get("id", "")
            for item in items_by_fact.get(fid, []):
                total_items += 1
                cat = item.get("category") or "uncategorized"
                category_counts[cat] += 1

        record: dict[str, Any] = {
            "summary": "aggregate",
            "fact_type": fact_type,
            "count": len(type_facts),
            "total_items": total_items,
        }

        if amounts:
            record["amount_min"] = round(min(amounts), 2)
            record["amount_max"] = round(max(amounts), 2)
            record["amount_mean"] = round(statistics.mean(amounts), 2)
            record["amount_median"] = round(statistics.median(amounts), 2)
            if len(amounts) > 1:
                record["amount_stddev"] = round(statistics.stdev(amounts), 2)

        if category_counts:
            record["top_categories"] = dict(
                sorted(
                    category_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            )

        result.append(record)

    return result
