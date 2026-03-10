"""Vendor deduplication report from v2 facts.

Identifies vendors with the same vendor_key but different display names,
and vendors that may be duplicates based on name similarity.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager


@dataclass
class VendorAlias:
    """A group of names that share the same vendor_key."""

    vendor_key: str
    names: list[str]  # Distinct display names, most frequent first
    fact_count: int
    total_amount: Decimal


@dataclass
class VendorDeduplicationReport:
    """Results of vendor deduplication analysis."""

    aliases: list[VendorAlias]
    total_vendors: int
    vendors_with_aliases: int
    unkeyed_vendors: list[str]  # Vendors with no vendor_key


def vendor_deduplication_report(
    db: "DatabaseManager",
) -> VendorDeduplicationReport:
    """Build a report of vendor aliases (same vendor_key, different names).

    This report powers vendor alias management by showing which display
    names map to the same underlying entity via vendor_key.

    Returns:
        VendorDeduplicationReport with alias groups and statistics
    """
    rows = db.fetchall(
        """
        SELECT vendor, vendor_key, total_amount
        FROM facts
        WHERE vendor IS NOT NULL
        ORDER BY vendor_key, vendor
        """,
        (),
    )

    if not rows:
        return VendorDeduplicationReport(
            aliases=[],
            total_vendors=0,
            vendors_with_aliases=0,
            unkeyed_vendors=[],
        )

    # Group by vendor_key
    keyed: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"names": defaultdict(int), "total": Decimal("0"), "count": 0}
    )
    unkeyed: set[str] = set()
    all_vendors: set[str] = set()

    for row in rows:
        vendor = row["vendor"]
        vendor_key = row["vendor_key"]
        amount = (
            Decimal(str(row["total_amount"])) if row["total_amount"] else Decimal("0")
        )
        all_vendors.add(vendor)

        if vendor_key:
            entry = keyed[vendor_key]
            entry["names"][vendor] += 1
            entry["total"] += amount
            entry["count"] += 1
        else:
            unkeyed.add(vendor)

    # Build alias groups (only for keys with 2+ distinct names)
    aliases: list[VendorAlias] = []
    for vk, data in keyed.items():
        names_dict = data["names"]
        if len(names_dict) >= 2:
            # Sort by frequency descending
            sorted_names = sorted(names_dict, key=lambda n: names_dict[n], reverse=True)
            aliases.append(
                VendorAlias(
                    vendor_key=vk,
                    names=sorted_names,
                    fact_count=data["count"],
                    total_amount=data["total"],
                )
            )

    aliases.sort(key=lambda a: a.fact_count, reverse=True)

    return VendorDeduplicationReport(
        aliases=aliases,
        total_vendors=len(all_vendors),
        vendors_with_aliases=sum(len(a.names) for a in aliases),
        unkeyed_vendors=sorted(unkeyed),
    )
