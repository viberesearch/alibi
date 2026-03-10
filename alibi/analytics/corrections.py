"""Correction analytics: confusion matrix from user corrections.

Analyzes correction history to detect systematic extraction errors
and identify categories/vendors needing cloud refinement.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CategoryConfusion:
    """Confusion between two categories."""

    original: str
    corrected: str
    count: int
    examples: list[dict[str, str]] = field(default_factory=list)


@dataclass
class VendorCorrectionStats:
    """Correction statistics for a vendor."""

    vendor_key: str
    vendor_name: str
    total_corrections: int
    field_corrections: dict[str, int] = field(default_factory=dict)
    category_changes: list[CategoryConfusion] = field(default_factory=list)


@dataclass
class ConfusionMatrix:
    """Full confusion matrix from corrections."""

    total_corrections: int = 0
    category_confusions: list[CategoryConfusion] = field(default_factory=list)
    vendor_stats: list[VendorCorrectionStats] = field(default_factory=list)
    top_corrected_fields: dict[str, int] = field(default_factory=dict)
    refinement_candidates: list[str] = field(default_factory=list)


def _load_correction_events(db: Any, limit: int = 1000) -> list[dict[str, Any]]:
    """Load correction events from the database."""
    events: list[dict[str, Any]] = []

    # 1. Items with user confirmations (enrichment corrections)
    rows = db.fetchall(
        """
        SELECT fi.id, fi.name, fi.brand, fi.category, fi.enrichment_source,
               fi.enrichment_confidence, f.vendor_key,
               COALESCE(
                   (SELECT im.value FROM identity_members im
                    JOIN identities i ON im.identity_id = i.id
                    WHERE i.entity_type = 'vendor'
                    AND im.member_type = 'name'
                    AND im.identity_id = (
                        SELECT identity_id FROM identity_members
                        WHERE member_type = 'vendor_key'
                        AND value = f.vendor_key
                        LIMIT 1
                    )
                    LIMIT 1),
                   f.vendor_key
               ) as vendor_name
        FROM fact_items fi
        JOIN facts f ON fi.fact_id = f.id
        WHERE fi.enrichment_source = 'user_confirmed'
        ORDER BY fi.id
        LIMIT ?
        """,
        (limit,),
    )
    for row in rows:
        events.append(
            {
                "type": "enrichment_correction",
                "item_id": row[0],
                "item_name": row[1],
                "brand": row[2],
                "category": row[3],
                "source": row[4],
                "confidence": row[5],
                "vendor_key": row[6],
                "vendor_name": row[7],
            }
        )

    # 2. Items with low-confidence enrichment
    rows = db.fetchall(
        """
        SELECT fi.id, fi.name, fi.category, fi.enrichment_source,
               fi.enrichment_confidence, f.vendor_key,
               COALESCE(
                   (SELECT im.value FROM identity_members im
                    JOIN identities i ON im.identity_id = i.id
                    WHERE i.entity_type = 'vendor'
                    AND im.member_type = 'name'
                    AND im.identity_id = (
                        SELECT identity_id FROM identity_members
                        WHERE member_type = 'vendor_key'
                        AND value = f.vendor_key
                        LIMIT 1
                    )
                    LIMIT 1),
                   f.vendor_key
               ) as vendor_name
        FROM fact_items fi
        JOIN facts f ON fi.fact_id = f.id
        WHERE fi.enrichment_source IN ('cloud_refined', 'llm_inference')
        AND fi.enrichment_confidence < 0.7
        ORDER BY fi.enrichment_confidence ASC
        LIMIT ?
        """,
        (limit,),
    )
    for row in rows:
        events.append(
            {
                "type": "low_confidence",
                "item_id": row[0],
                "item_name": row[1],
                "category": row[2],
                "source": row[3],
                "confidence": row[4],
                "vendor_key": row[5],
                "vendor_name": row[6],
            }
        )

    return events


def build_confusion_matrix(
    db: Any,
    limit: int = 1000,
    min_count: int = 2,
) -> ConfusionMatrix:
    """Build confusion matrix from correction history.

    Args:
        db: DatabaseManager.
        limit: Max correction events to analyze.
        min_count: Minimum occurrences to include in results.

    Returns:
        ConfusionMatrix with category confusions, vendor stats,
        and refinement candidates.
    """
    events = _load_correction_events(db, limit=limit)
    if not events:
        return ConfusionMatrix()

    matrix = ConfusionMatrix(total_corrections=len(events))

    # Count corrections by field
    field_counts: Counter[str] = Counter()
    for e in events:
        if e["type"] == "enrichment_correction":
            if e.get("category"):
                field_counts["category"] += 1
            if e.get("brand"):
                field_counts["brand"] += 1
        elif e["type"] == "low_confidence":
            field_counts["low_confidence_enrichment"] += 1

    matrix.top_corrected_fields = dict(field_counts.most_common(10))

    # Category confusion: track which categories get corrections per vendor
    category_by_vendor: dict[str, Counter[str]] = defaultdict(Counter)
    for e in events:
        cat = e.get("category")
        vk = e.get("vendor_key")
        if cat and vk:
            category_by_vendor[vk][cat] += 1

    # Find categories with conflicting assignments
    confusions = []
    for _vk, cats in category_by_vendor.items():
        if len(cats) >= 2:
            items = cats.most_common()
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if items[i][1] >= min_count or items[j][1] >= min_count:
                        confusions.append(
                            CategoryConfusion(
                                original=items[i][0],
                                corrected=items[j][0],
                                count=items[i][1] + items[j][1],
                            )
                        )

    matrix.category_confusions = sorted(confusions, key=lambda x: x.count, reverse=True)

    # Vendor correction stats
    vendor_corrections: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "fields": Counter(), "name": ""}
    )
    for e in events:
        vk = e.get("vendor_key")
        if vk:
            vendor_corrections[vk]["count"] += 1
            vendor_corrections[vk]["name"] = e.get("vendor_name", vk)
            vendor_corrections[vk]["fields"][e["type"]] += 1

    vendor_stats = []
    for vk, data in vendor_corrections.items():
        if data["count"] >= min_count:
            vendor_stats.append(
                VendorCorrectionStats(
                    vendor_key=vk,
                    vendor_name=data["name"],
                    total_corrections=data["count"],
                    field_corrections=dict(data["fields"]),
                )
            )

    matrix.vendor_stats = sorted(
        vendor_stats, key=lambda x: x.total_corrections, reverse=True
    )

    # Refinement candidates: categories with high correction rates
    cat_total: Counter[str] = Counter()
    cat_corrections: Counter[str] = Counter()

    all_items = db.fetchall(
        "SELECT category, COUNT(*) FROM fact_items "
        "WHERE category IS NOT NULL GROUP BY category",
        (),
    )
    for row in all_items:
        cat_total[row[0]] = row[1]

    for e in events:
        cat = e.get("category")
        if cat:
            cat_corrections[cat] += 1

    candidates = []
    for cat, corrections in cat_corrections.items():
        total = cat_total.get(cat, 0)
        if total > 0 and corrections >= min_count:
            rate = corrections / total
            if rate >= 0.1:  # 10%+ correction rate
                candidates.append(cat)

    matrix.refinement_candidates = sorted(candidates)

    return matrix


def get_refinement_suggestions(
    db: Any,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Get actionable refinement suggestions from correction analysis."""
    matrix = build_confusion_matrix(db, limit=limit)
    suggestions: list[dict[str, Any]] = []

    for conf in matrix.category_confusions[:10]:
        suggestions.append(
            {
                "type": "category_confusion",
                "priority": "high" if conf.count >= 5 else "medium",
                "category_a": conf.original,
                "category_b": conf.corrected,
                "count": conf.count,
                "action": (
                    f"Review items categorized as '{conf.original}' or "
                    f"'{conf.corrected}' -- frequent confusion"
                ),
            }
        )

    for vs in matrix.vendor_stats[:5]:
        suggestions.append(
            {
                "type": "vendor_corrections",
                "priority": "high" if vs.total_corrections >= 10 else "medium",
                "vendor": vs.vendor_name,
                "vendor_key": vs.vendor_key,
                "correction_count": vs.total_corrections,
                "action": (
                    f"Vendor '{vs.vendor_name}' has {vs.total_corrections} "
                    f"corrections -- consider reprocessing or template update"
                ),
            }
        )

    for cat in matrix.refinement_candidates:
        suggestions.append(
            {
                "type": "refinement_candidate",
                "priority": "medium",
                "category": cat,
                "action": (
                    f"Category '{cat}' has high correction rate "
                    f"-- run cloud refinement"
                ),
            }
        )

    return sorted(
        suggestions,
        key=lambda x: (0 if x["priority"] == "high" else 1, -x.get("count", 0)),
    )
