"""Cloud formation learning — weight adjustment from correction history.

Records feature vectors from bundle moves (merge/split) and uses the
accumulated data to adjust cloud formation scoring weights per vendor
pair. Also provides warm-start logic for new vendors based on POS
provider similarity.

The cloud_correction_history table (migration 032) is the backing store.
Each row captures the observable features at the moment a user invoked
move_bundle(), together with a flag indicating whether the move was a
false-positive split or a deliberate merge.

Weight adjustments are applied as multipliers on top of the default
confidence thresholds defined in formation.py. A multiplier of 1.15
means "use a threshold 15% higher than the default for this vendor".
All multipliers are clamped to [1 - MAX_DEVIATION, 1 + MAX_DEVIATION].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

# Minimum corrections before adjusting weights for a vendor pair
_MIN_CORRECTIONS = 3

# Per-correction step size for nudging weight multipliers
_LEARNING_RATE = 0.1

# Maximum deviation from 1.0 in either direction
_MAX_WEIGHT_DEVIATION = 0.3


@dataclass
class CorrectionFeatureVector:
    """Feature vector captured when a bundle is moved between clouds."""

    vendor_key_a: str | None
    vendor_key_b: str | None
    vendor_similarity: float
    amount_diff: float
    date_diff_days: int
    location_distance: float | None
    was_false_positive: bool
    source_bundle_type: str
    target_bundle_type: str | None
    item_overlap: float


def record_correction(
    db: DatabaseManager,
    feature: CorrectionFeatureVector,
) -> None:
    """Persist a correction feature vector to cloud_correction_history."""
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO cloud_correction_history "
            "(vendor_key_a, vendor_key_b, vendor_similarity, amount_diff, "
            "date_diff_days, location_distance, was_false_positive, "
            "source_bundle_type, target_bundle_type, item_overlap) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                feature.vendor_key_a,
                feature.vendor_key_b,
                feature.vendor_similarity,
                feature.amount_diff,
                feature.date_diff_days,
                feature.location_distance,
                1 if feature.was_false_positive else 0,
                feature.source_bundle_type,
                feature.target_bundle_type,
                feature.item_overlap,
            ),
        )
    logger.debug(
        "Recorded correction: vendor_a=%s vendor_b=%s fp=%s",
        feature.vendor_key_a,
        feature.vendor_key_b,
        feature.was_false_positive,
    )


def get_weight_adjustments(
    db: DatabaseManager,
    vendor_key: str | None = None,
) -> dict[str, float]:
    """Compute weight adjustment multipliers from correction history.

    Returns dict mapping weight names to multipliers, e.g.
    {"vendor": 1.2, "amount": 0.9}. Empty dict = use defaults.
    """
    if vendor_key:
        rows = db.fetchall(
            "SELECT vendor_similarity, amount_diff, date_diff_days, "
            "item_overlap, was_false_positive "
            "FROM cloud_correction_history "
            "WHERE vendor_key_a = ? OR vendor_key_b = ? "
            "ORDER BY created_at DESC LIMIT 50",
            (vendor_key, vendor_key),
        )
    else:
        rows = db.fetchall(
            "SELECT vendor_similarity, amount_diff, date_diff_days, "
            "item_overlap, was_false_positive "
            "FROM cloud_correction_history "
            "ORDER BY created_at DESC LIMIT 200",
        )

    if len(rows) < _MIN_CORRECTIONS:
        return {}

    false_positives = [r for r in rows if r["was_false_positive"]]
    true_merges = [r for r in rows if not r["was_false_positive"]]

    adjustments: dict[str, float] = {}

    if false_positives:
        fp_count = len(false_positives)
        avg_vendor_sim = sum(r["vendor_similarity"] for r in false_positives) / fp_count
        avg_amount_diff = sum(r["amount_diff"] for r in false_positives) / fp_count

        if avg_vendor_sim < 0.5 and avg_amount_diff < 1.0:
            adjustments["vendor"] = min(
                1.0 + _LEARNING_RATE * fp_count,
                1.0 + _MAX_WEIGHT_DEVIATION,
            )
            adjustments["amount"] = max(
                1.0 - _LEARNING_RATE * fp_count * 0.5,
                1.0 - _MAX_WEIGHT_DEVIATION,
            )

        avg_date_diff = sum(r["date_diff_days"] for r in false_positives) / fp_count
        if avg_date_diff > 3:
            adjustments["date"] = min(
                1.0 + _LEARNING_RATE * fp_count * 0.5,
                1.0 + _MAX_WEIGHT_DEVIATION,
            )

    if true_merges:
        avg_item_overlap = sum(r["item_overlap"] for r in true_merges) / len(
            true_merges
        )
        if avg_item_overlap > 0.3:
            adjustments["item_overlap"] = min(
                1.0 + _LEARNING_RATE * 2,
                1.0 + _MAX_WEIGHT_DEVIATION,
            )

    return adjustments


def get_false_positive_pairs(
    db: DatabaseManager,
    min_count: int = 2,
) -> list[tuple[str, str, int]]:
    """Find vendor pairs that are repeatedly incorrectly merged."""
    rows = db.fetchall(
        "SELECT vendor_key_a, vendor_key_b, COUNT(*) as cnt "
        "FROM cloud_correction_history "
        "WHERE was_false_positive = 1 "
        "AND vendor_key_a IS NOT NULL "
        "AND vendor_key_b IS NOT NULL "
        "GROUP BY vendor_key_a, vendor_key_b "
        "HAVING cnt >= ? "
        "ORDER BY cnt DESC",
        (min_count,),
    )
    return [(r["vendor_key_a"], r["vendor_key_b"], r["cnt"]) for r in rows]


def is_known_false_positive_pair(
    db: DatabaseManager,
    vendor_key_a: str | None,
    vendor_key_b: str | None,
    threshold: int = 2,
) -> bool:
    """Return True if these two vendors are a known false-positive pair.

    Checks both orderings of the vendor keys.
    """
    if not vendor_key_a or not vendor_key_b:
        return False

    row = db.fetchone(
        "SELECT COUNT(*) as cnt FROM cloud_correction_history "
        "WHERE was_false_positive = 1 "
        "AND ((vendor_key_a = ? AND vendor_key_b = ?) "
        "  OR (vendor_key_a = ? AND vendor_key_b = ?))",
        (vendor_key_a, vendor_key_b, vendor_key_b, vendor_key_a),
    )
    return row is not None and int(row["cnt"]) >= threshold


def get_pos_provider_weights(
    db: DatabaseManager,
    pos_provider: str,
) -> dict[str, float] | None:
    """Derive weight adjustments for a new vendor from POS-provider history.

    When a vendor with no correction history appears, if it uses a known
    POS provider, aggregate corrections from all vendors with that POS
    provider as a warm start.
    """
    vendor_rows = db.fetchall(
        "SELECT DISTINCT im.value AS vendor_key "
        "FROM identity_members im "
        "JOIN identities i ON im.identity_id = i.id "
        "WHERE i.entity_type = 'vendor' "
        "AND im.member_type = 'vendor_key' "
        "AND i.metadata LIKE ?",
        (f'%"pos_provider": "{pos_provider}"%',),
    )

    if not vendor_rows:
        return None

    vendor_keys = [r["vendor_key"] for r in vendor_rows]

    all_corrections: list[Any] = []
    for vk in vendor_keys:
        rows = db.fetchall(
            "SELECT vendor_similarity, amount_diff, date_diff_days, "
            "item_overlap, was_false_positive "
            "FROM cloud_correction_history "
            "WHERE vendor_key_a = ? OR vendor_key_b = ?",
            (vk, vk),
        )
        all_corrections.extend(rows)

    if len(all_corrections) < _MIN_CORRECTIONS:
        return None

    false_positives = [r for r in all_corrections if r["was_false_positive"]]
    true_merges = [r for r in all_corrections if not r["was_false_positive"]]

    adjustments: dict[str, float] = {}

    if false_positives:
        fp_count = len(false_positives)
        avg_vendor_sim = sum(r["vendor_similarity"] for r in false_positives) / fp_count
        avg_amount_diff = sum(r["amount_diff"] for r in false_positives) / fp_count
        if avg_vendor_sim < 0.5 and avg_amount_diff < 1.0:
            adjustments["vendor"] = min(
                1.0 + _LEARNING_RATE * fp_count,
                1.0 + _MAX_WEIGHT_DEVIATION,
            )
            adjustments["amount"] = max(
                1.0 - _LEARNING_RATE * fp_count * 0.5,
                1.0 - _MAX_WEIGHT_DEVIATION,
            )

    if true_merges:
        avg_item_overlap = sum(r["item_overlap"] for r in true_merges) / len(
            true_merges
        )
        if avg_item_overlap > 0.3:
            adjustments["item_overlap"] = min(
                1.0 + _LEARNING_RATE * 2,
                1.0 + _MAX_WEIGHT_DEVIATION,
            )

    return adjustments if adjustments else None


def get_correction_stats(db: DatabaseManager) -> dict[str, Any]:
    """Return summary statistics of cloud formation correction history."""
    total_row = db.fetchone("SELECT COUNT(*) AS cnt FROM cloud_correction_history")
    fp_row = db.fetchone(
        "SELECT COUNT(*) AS cnt FROM cloud_correction_history "
        "WHERE was_false_positive = 1"
    )
    merge_row = db.fetchone(
        "SELECT COUNT(*) AS cnt FROM cloud_correction_history "
        "WHERE was_false_positive = 0"
    )

    top_rows = db.fetchall(
        "SELECT COALESCE(vendor_key_a, vendor_key_b) AS vendor_key, "
        "COUNT(*) AS cnt "
        "FROM cloud_correction_history "
        "WHERE vendor_key_a IS NOT NULL OR vendor_key_b IS NOT NULL "
        "GROUP BY vendor_key "
        "ORDER BY cnt DESC "
        "LIMIT 10"
    )

    return {
        "total_corrections": int(total_row["cnt"]) if total_row else 0,
        "false_positives": int(fp_row["cnt"]) if fp_row else 0,
        "true_merges": int(merge_row["cnt"]) if merge_row else 0,
        "top_corrected_vendors": [(r["vendor_key"], int(r["cnt"])) for r in top_rows],
        "adjusted_weights": get_weight_adjustments(db),
    }
