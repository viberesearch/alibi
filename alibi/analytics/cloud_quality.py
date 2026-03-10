"""Cloud formation quality metrics.

Computes accuracy and health metrics for the cloud formation system
using correction_events, cloud_correction_history, and cloud/fact tables.

Metrics include:
- Overall split/merge rates (how often clouds are corrected)
- Per-vendor accuracy (which vendors cause the most cloud issues)
- Match type effectiveness (which scoring signals work best)
- Cloud size distribution (how many bundles per cloud)
- Correction trends over time
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class VendorAccuracy:
    """Cloud formation accuracy for a single vendor."""

    vendor_key: str
    vendor_name: str | None
    total_clouds: int
    corrected_clouds: int
    accuracy_rate: float  # 1.0 - (corrected / total)
    false_positives: int  # Wrongly merged (then split)
    correction_count: int  # Total corrections on this vendor's facts


@dataclass
class MatchTypeStats:
    """Effectiveness of a cloud match type."""

    match_type: str
    total_uses: int
    avg_confidence: float
    correction_rate: float  # How often this match type leads to corrections


@dataclass
class CloudSizeDistribution:
    """Distribution of bundle counts per cloud."""

    single_bundle: int  # Clouds with 1 bundle
    two_bundles: int  # Clouds with 2 bundles
    three_plus: int  # Clouds with 3+ bundles
    max_bundles: int  # Largest cloud
    avg_bundles: float  # Average bundles per cloud


@dataclass
class CorrectionTrend:
    """Correction rate for a time period."""

    period: str  # YYYY-MM
    total_corrections: int
    false_positives: int
    true_merges: int
    facts_created: int
    correction_rate: float  # corrections / facts


@dataclass
class CloudQualityReport:
    """Comprehensive cloud formation quality report."""

    total_clouds: int
    total_facts: int
    total_corrections: int
    overall_accuracy: float  # 1.0 - (corrected_clouds / total_clouds)
    false_positive_rate: float  # false_positives / total_corrections
    vendor_accuracy: list[VendorAccuracy] = field(default_factory=list)
    match_type_stats: list[MatchTypeStats] = field(default_factory=list)
    size_distribution: CloudSizeDistribution | None = None
    trends: list[CorrectionTrend] = field(default_factory=list)
    top_false_positive_pairs: list[tuple[str, str, int]] = field(default_factory=list)


def build_quality_report(
    db: DatabaseManager,
    limit_vendors: int = 20,
    limit_trends: int = 12,
) -> CloudQualityReport:
    """Build a comprehensive cloud formation quality report.

    Args:
        db: Database connection.
        limit_vendors: Max vendors in accuracy breakdown.
        limit_trends: Max monthly periods in trend data.

    Returns:
        CloudQualityReport with all metrics.
    """
    conn = db.get_connection()

    total_clouds = _count(conn, "clouds")
    total_facts = _count(conn, "facts")
    total_corrections = _count(conn, "cloud_correction_history")

    # Corrected clouds: clouds that had bundles moved away
    corrected_clouds = _count_corrected_clouds(conn)

    overall_accuracy = (
        1.0 - (corrected_clouds / total_clouds) if total_clouds > 0 else 1.0
    )

    # False positive rate from cloud_correction_history
    fp_count = _count_where(conn, "cloud_correction_history", "was_false_positive = 1")
    fp_rate = fp_count / total_corrections if total_corrections > 0 else 0.0

    report = CloudQualityReport(
        total_clouds=total_clouds,
        total_facts=total_facts,
        total_corrections=total_corrections,
        overall_accuracy=overall_accuracy,
        false_positive_rate=fp_rate,
    )

    report.vendor_accuracy = _vendor_accuracy(conn, limit_vendors)
    report.match_type_stats = _match_type_stats(conn)
    report.size_distribution = _cloud_size_distribution(conn)
    report.trends = _correction_trends(conn, limit_trends)
    report.top_false_positive_pairs = _top_false_positive_pairs(conn)

    return report


def get_vendor_cloud_accuracy(
    db: DatabaseManager,
    vendor_key: str,
) -> VendorAccuracy | None:
    """Get cloud formation accuracy for a specific vendor.

    Args:
        db: Database connection.
        vendor_key: Vendor key to query.

    Returns:
        VendorAccuracy or None if vendor has no clouds.
    """
    conn = db.get_connection()

    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM facts WHERE vendor_key = ?",
        (vendor_key,),
    ).fetchone()
    total_clouds = row["cnt"] if row else 0

    if total_clouds == 0:
        return None

    # Get vendor name
    name_row = conn.execute(
        "SELECT canonical_name FROM identities i "
        "JOIN identity_members im ON i.id = im.identity_id "
        "WHERE im.member_type = 'vendor_key' AND im.value = ? "
        "LIMIT 1",
        (vendor_key,),
    ).fetchone()
    vendor_name = name_row["canonical_name"] if name_row else None

    # Count corrections on this vendor's facts
    correction_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM correction_events "
        "WHERE entity_type = 'fact' "
        "AND entity_id IN (SELECT id FROM facts WHERE vendor_key = ?)",
        (vendor_key,),
    ).fetchone()["cnt"]

    # Count bundle moves (false positives)
    fp_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM cloud_correction_history "
        "WHERE was_false_positive = 1 "
        "AND (vendor_key_a = ? OR vendor_key_b = ?)",
        (vendor_key, vendor_key),
    ).fetchone()["cnt"]

    # Corrected clouds for this vendor
    corrected = conn.execute(
        "SELECT COUNT(DISTINCT ce.entity_id) as cnt "
        "FROM correction_events ce "
        "WHERE ce.entity_type = 'bundle' AND ce.field = 'cloud_id' "
        "AND ce.entity_id IN ("
        "  SELECT bundle_id FROM cloud_bundles cb "
        "  JOIN facts f ON cb.cloud_id = f.cloud_id "
        "  WHERE f.vendor_key = ?"
        ")",
        (vendor_key,),
    ).fetchone()["cnt"]

    accuracy = 1.0 - (corrected / total_clouds) if total_clouds > 0 else 1.0

    return VendorAccuracy(
        vendor_key=vendor_key,
        vendor_name=vendor_name,
        total_clouds=total_clouds,
        corrected_clouds=corrected,
        accuracy_rate=accuracy,
        false_positives=fp_count,
        correction_count=correction_count,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count(conn: Any, table: str) -> int:
    """Count rows in a table."""
    row = conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()  # noqa: S608
    return row["cnt"] if row else 0


def _count_where(conn: Any, table: str, where: str) -> int:
    """Count rows matching a WHERE clause."""
    row = conn.execute(
        f"SELECT COUNT(*) as cnt FROM {table} WHERE {where}"  # noqa: S608
    ).fetchone()
    return row["cnt"] if row else 0


def _count_corrected_clouds(conn: Any) -> int:
    """Count clouds that had bundles moved away (corrections)."""
    row = conn.execute(
        "SELECT COUNT(DISTINCT old_value) as cnt "
        "FROM correction_events "
        "WHERE entity_type = 'bundle' AND field = 'cloud_id'"
    ).fetchone()
    return row["cnt"] if row else 0


def _vendor_accuracy(conn: Any, limit: int) -> list[VendorAccuracy]:
    """Per-vendor cloud accuracy, sorted by correction count desc."""
    rows = conn.execute(
        "SELECT f.vendor_key, "
        "  COUNT(DISTINCT f.id) as total_clouds, "
        "  i.canonical_name as vendor_name "
        "FROM facts f "
        "LEFT JOIN identity_members im "
        "  ON im.member_type = 'vendor_key' AND im.value = f.vendor_key "
        "LEFT JOIN identities i ON im.identity_id = i.id "
        "WHERE f.vendor_key IS NOT NULL "
        "GROUP BY f.vendor_key "
        "ORDER BY total_clouds DESC "
        "LIMIT ?",
        (limit,),
    ).fetchall()

    results = []
    for row in rows:
        vk = row["vendor_key"]

        correction_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM correction_events "
            "WHERE entity_type = 'fact' "
            "AND entity_id IN (SELECT id FROM facts WHERE vendor_key = ?)",
            (vk,),
        ).fetchone()["cnt"]

        fp_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM cloud_correction_history "
            "WHERE was_false_positive = 1 "
            "AND (vendor_key_a = ? OR vendor_key_b = ?)",
            (vk, vk),
        ).fetchone()["cnt"]

        total = row["total_clouds"]
        corrected = min(correction_count, total)
        accuracy = 1.0 - (corrected / total) if total > 0 else 1.0

        results.append(
            VendorAccuracy(
                vendor_key=vk,
                vendor_name=row["vendor_name"],
                total_clouds=total,
                corrected_clouds=corrected,
                accuracy_rate=accuracy,
                false_positives=fp_count,
                correction_count=correction_count,
            )
        )

    # Sort by correction count descending (worst first)
    results.sort(key=lambda v: v.correction_count, reverse=True)
    return results


def _match_type_stats(conn: Any) -> list[MatchTypeStats]:
    """Effectiveness of each match type in cloud_bundles."""
    rows = conn.execute(
        "SELECT match_type, "
        "  COUNT(*) as total_uses, "
        "  AVG(match_confidence) as avg_confidence "
        "FROM cloud_bundles "
        "GROUP BY match_type "
        "ORDER BY total_uses DESC"
    ).fetchall()

    results = []
    for row in rows:
        mt = row["match_type"]

        # Count how many bundles with this match_type were later moved
        corrected = conn.execute(
            "SELECT COUNT(*) as cnt FROM correction_events ce "
            "WHERE ce.entity_type = 'bundle' AND ce.field = 'cloud_id' "
            "AND ce.entity_id IN ("
            "  SELECT bundle_id FROM cloud_bundles WHERE match_type = ?"
            ")",
            (mt,),
        ).fetchone()["cnt"]

        total = row["total_uses"]
        correction_rate = corrected / total if total > 0 else 0.0

        results.append(
            MatchTypeStats(
                match_type=mt,
                total_uses=total,
                avg_confidence=round(row["avg_confidence"] or 0.0, 3),
                correction_rate=round(correction_rate, 3),
            )
        )

    return results


def _cloud_size_distribution(conn: Any) -> CloudSizeDistribution:
    """Distribution of bundle counts per cloud."""
    rows = conn.execute(
        "SELECT cloud_id, COUNT(*) as bundle_count "
        "FROM cloud_bundles "
        "GROUP BY cloud_id"
    ).fetchall()

    single = 0
    two = 0
    three_plus = 0
    max_b = 0
    total_bundles = 0

    for row in rows:
        bc = row["bundle_count"]
        total_bundles += bc
        if bc > max_b:
            max_b = bc
        if bc == 1:
            single += 1
        elif bc == 2:
            two += 1
        else:
            three_plus += 1

    total_clouds = len(rows)
    avg = total_bundles / total_clouds if total_clouds > 0 else 0.0

    return CloudSizeDistribution(
        single_bundle=single,
        two_bundles=two,
        three_plus=three_plus,
        max_bundles=max_b,
        avg_bundles=round(avg, 2),
    )


def _correction_trends(conn: Any, limit: int) -> list[CorrectionTrend]:
    """Monthly correction trends."""
    # Get monthly correction counts
    correction_rows = conn.execute(
        "SELECT strftime('%Y-%m', created_at) as period, "
        "  COUNT(*) as total "
        "FROM cloud_correction_history "
        "GROUP BY period "
        "ORDER BY period DESC "
        "LIMIT ?",
        (limit,),
    ).fetchall()

    if not correction_rows:
        return []

    # Get monthly false positive counts
    fp_rows = conn.execute(
        "SELECT strftime('%Y-%m', created_at) as period, "
        "  COUNT(*) as cnt "
        "FROM cloud_correction_history "
        "WHERE was_false_positive = 1 "
        "GROUP BY period"
    ).fetchall()
    fp_map = {r["period"]: r["cnt"] for r in fp_rows}

    # Get monthly fact creation counts
    fact_rows = conn.execute(
        "SELECT strftime('%Y-%m', created_at) as period, "
        "  COUNT(*) as cnt "
        "FROM facts "
        "GROUP BY period"
    ).fetchall()
    fact_map = {r["period"]: r["cnt"] for r in fact_rows}

    results = []
    for row in correction_rows:
        period = row["period"]
        total = row["total"]
        fp = fp_map.get(period, 0)
        facts = fact_map.get(period, 0)
        rate = total / facts if facts > 0 else 0.0

        results.append(
            CorrectionTrend(
                period=period,
                total_corrections=total,
                false_positives=fp,
                true_merges=total - fp,
                facts_created=facts,
                correction_rate=round(rate, 3),
            )
        )

    # Return chronological order
    results.reverse()
    return results


def _top_false_positive_pairs(
    conn: Any,
    limit: int = 10,
) -> list[tuple[str, str, int]]:
    """Top vendor pairs that are frequently wrongly merged."""
    rows = conn.execute(
        "SELECT vendor_key_a, vendor_key_b, COUNT(*) as cnt "
        "FROM cloud_correction_history "
        "WHERE was_false_positive = 1 "
        "AND vendor_key_a IS NOT NULL "
        "AND vendor_key_b IS NOT NULL "
        "GROUP BY vendor_key_a, vendor_key_b "
        "HAVING cnt >= 2 "
        "ORDER BY cnt DESC "
        "LIMIT ?",
        (limit,),
    ).fetchall()

    return [(r["vendor_key_a"], r["vendor_key_b"], r["cnt"]) for r in rows]
