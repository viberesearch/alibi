"""Tests for cloud formation quality metrics."""

from __future__ import annotations

import os
import uuid

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.analytics.cloud_quality import (
    CloudQualityReport,
    CloudSizeDistribution,
    CorrectionTrend,
    MatchTypeStats,
    VendorAccuracy,
    build_quality_report,
    get_vendor_cloud_accuracy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_cloud_with_fact(  # type: ignore[no-untyped-def]
    db, vendor_key=None, vendor="Test Vendor"
) -> tuple[str, str, str]:
    """Create a cloud + fact + bundle + cloud_bundles entry."""
    conn = db.get_connection()
    cloud_id = f"cloud-{uuid.uuid4().hex[:8]}"
    fact_id = f"fact-{uuid.uuid4().hex[:8]}"
    bundle_id = f"bundle-{uuid.uuid4().hex[:8]}"
    doc_id = f"doc-{uuid.uuid4().hex[:8]}"

    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) "
        "VALUES (?, ?, ?)",
        (doc_id, f"/tmp/{doc_id}.jpg", f"hash-{doc_id}"),
    )
    conn.execute(
        "INSERT INTO clouds (id, status, confidence) " "VALUES (?, 'collapsed', 0.85)",
        (cloud_id,),
    )
    conn.execute(
        "INSERT INTO bundles (id, document_id, bundle_type) " "VALUES (?, ?, 'basket')",
        (bundle_id, doc_id),
    )
    conn.execute(
        "INSERT INTO cloud_bundles (cloud_id, bundle_id, match_type, match_confidence) "
        "VALUES (?, ?, 'exact_amount', 0.9)",
        (cloud_id, bundle_id),
    )
    conn.execute(
        "INSERT INTO facts "
        "(id, cloud_id, fact_type, vendor, vendor_key, "
        "total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', ?, ?, 10.0, 'EUR', '2026-01-15')",
        (fact_id, cloud_id, vendor, vendor_key),
    )
    conn.commit()
    return cloud_id, fact_id, bundle_id


def _seed_cloud_with_bundles(  # type: ignore[no-untyped-def]
    db, vendor_key=None, match_types=None, vendor="Test"
) -> tuple[str, str, list[str]]:
    """Create a cloud with multiple bundles."""
    conn = db.get_connection()
    cloud_id = f"cloud-{uuid.uuid4().hex[:8]}"
    fact_id = f"fact-{uuid.uuid4().hex[:8]}"

    conn.execute(
        "INSERT INTO clouds (id, status, confidence) " "VALUES (?, 'collapsed', 0.85)",
        (cloud_id,),
    )

    match_types = match_types or ["exact_amount", "vendor+date"]
    bundle_ids = []
    for mt in match_types:
        bundle_id = f"bundle-{uuid.uuid4().hex[:8]}"
        doc_id = f"doc-{uuid.uuid4().hex[:8]}"
        conn.execute(
            "INSERT OR IGNORE INTO documents (id, file_path, file_hash) "
            "VALUES (?, ?, ?)",
            (doc_id, f"/tmp/{doc_id}.jpg", f"hash-{doc_id}"),
        )
        conn.execute(
            "INSERT INTO bundles (id, document_id, bundle_type) "
            "VALUES (?, ?, 'basket')",
            (bundle_id, doc_id),
        )
        conn.execute(
            "INSERT INTO cloud_bundles "
            "(cloud_id, bundle_id, match_type, match_confidence) "
            "VALUES (?, ?, ?, 0.85)",
            (cloud_id, bundle_id, mt),
        )
        bundle_ids.append(bundle_id)

    conn.execute(
        "INSERT INTO facts "
        "(id, cloud_id, fact_type, vendor, vendor_key, "
        "total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', ?, ?, 10.0, 'EUR', '2026-01-15')",
        (fact_id, cloud_id, vendor, vendor_key),
    )
    conn.commit()
    return cloud_id, fact_id, bundle_ids


def _seed_correction_event(  # type: ignore[no-untyped-def]
    db, entity_type, entity_id, field, old_value, new_value
) -> None:
    """Insert a correction event."""
    conn = db.get_connection()
    conn.execute(
        "INSERT INTO correction_events "
        "(id, entity_type, entity_id, field, old_value, new_value, source) "
        "VALUES (?, ?, ?, ?, ?, ?, 'test')",
        (str(uuid.uuid4()), entity_type, entity_id, field, old_value, new_value),
    )
    conn.commit()


def _seed_cloud_correction(  # type: ignore[no-untyped-def]
    db,
    vendor_key_a=None,
    vendor_key_b=None,
    was_false_positive=0,
    vendor_similarity=0.8,
) -> None:
    """Insert a cloud_correction_history entry."""
    conn = db.get_connection()
    conn.execute(
        "INSERT INTO cloud_correction_history "
        "(vendor_key_a, vendor_key_b, vendor_similarity, "
        "amount_diff, date_diff_days, was_false_positive, item_overlap) "
        "VALUES (?, ?, ?, 0.5, 1, ?, 0.3)",
        (vendor_key_a, vendor_key_b, vendor_similarity, was_false_positive),
    )
    conn.commit()


# ===========================================================================
# TestBuildQualityReport
# ===========================================================================


class TestBuildQualityReport:
    def test_empty_db(self, db):
        report = build_quality_report(db)
        assert isinstance(report, CloudQualityReport)
        assert report.total_clouds == 0
        assert report.total_facts == 0
        assert report.total_corrections == 0
        assert report.overall_accuracy == 1.0
        assert report.false_positive_rate == 0.0

    def test_basic_metrics(self, db):
        _seed_cloud_with_fact(db, vendor_key="vk-a")
        _seed_cloud_with_fact(db, vendor_key="vk-b")

        report = build_quality_report(db)
        assert report.total_clouds == 2
        assert report.total_facts == 2
        assert report.overall_accuracy == 1.0

    def test_with_corrections(self, db):
        _, fact_id, bundle_id = _seed_cloud_with_fact(db, vendor_key="vk-a")

        # Simulate a bundle move correction
        _seed_correction_event(
            db, "bundle", bundle_id, "cloud_id", "old-cloud", "new-cloud"
        )
        _seed_cloud_correction(db, "vk-a", "vk-b", was_false_positive=1)

        report = build_quality_report(db)
        assert report.total_corrections == 1
        assert report.false_positive_rate == 1.0

    def test_size_distribution(self, db):
        # Cloud with 1 bundle
        _seed_cloud_with_fact(db, vendor_key="vk-a")
        # Cloud with 2 bundles
        _seed_cloud_with_bundles(
            db, vendor_key="vk-b", match_types=["exact_amount", "vendor+date"]
        )

        report = build_quality_report(db)
        sd = report.size_distribution
        assert isinstance(sd, CloudSizeDistribution)
        assert sd.single_bundle == 1
        assert sd.two_bundles == 1
        assert sd.three_plus == 0
        assert sd.max_bundles == 2

    def test_match_type_stats(self, db):
        _seed_cloud_with_bundles(
            db,
            vendor_key="vk-a",
            match_types=["exact_amount", "exact_amount", "vendor+date"],
        )

        report = build_quality_report(db)
        assert len(report.match_type_stats) >= 1

        exact = next(
            (m for m in report.match_type_stats if m.match_type == "exact_amount"),
            None,
        )
        assert exact is not None
        assert exact.total_uses == 2
        assert isinstance(exact, MatchTypeStats)

    def test_vendor_accuracy_breakdown(self, db):
        _seed_cloud_with_fact(db, vendor_key="vk-a", vendor="Vendor A")
        _seed_cloud_with_fact(db, vendor_key="vk-a", vendor="Vendor A")
        _seed_cloud_with_fact(db, vendor_key="vk-b", vendor="Vendor B")

        report = build_quality_report(db)
        assert len(report.vendor_accuracy) >= 2

    def test_false_positive_pairs(self, db):
        _seed_cloud_with_fact(db)
        _seed_cloud_correction(db, "vk-a", "vk-b", was_false_positive=1)
        _seed_cloud_correction(db, "vk-a", "vk-b", was_false_positive=1)
        _seed_cloud_correction(db, "vk-a", "vk-b", was_false_positive=1)

        report = build_quality_report(db)
        assert len(report.top_false_positive_pairs) >= 1
        assert report.top_false_positive_pairs[0][2] >= 2

    def test_trends(self, db):
        _seed_cloud_with_fact(db, vendor_key="vk-a")
        _seed_cloud_correction(db, "vk-a", "vk-b")

        report = build_quality_report(db)
        assert len(report.trends) >= 1
        assert isinstance(report.trends[0], CorrectionTrend)


# ===========================================================================
# TestGetVendorCloudAccuracy
# ===========================================================================


class TestGetVendorCloudAccuracy:
    def test_unknown_vendor_returns_none(self, db):
        assert get_vendor_cloud_accuracy(db, "nonexistent") is None

    def test_vendor_with_no_corrections(self, db):
        _seed_cloud_with_fact(db, vendor_key="vk-clean")
        _seed_cloud_with_fact(db, vendor_key="vk-clean")

        result = get_vendor_cloud_accuracy(db, "vk-clean")
        assert result is not None
        assert isinstance(result, VendorAccuracy)
        assert result.total_clouds == 2
        assert result.correction_count == 0
        assert result.accuracy_rate == 1.0

    def test_vendor_with_corrections(self, db):
        _, fact_id, _ = _seed_cloud_with_fact(db, vendor_key="vk-bad")
        _seed_correction_event(db, "fact", fact_id, "vendor", "Wrong", "Right")

        result = get_vendor_cloud_accuracy(db, "vk-bad")
        assert result is not None
        assert result.correction_count == 1

    def test_vendor_with_false_positives(self, db):
        _seed_cloud_with_fact(db, vendor_key="vk-fp")
        _seed_cloud_correction(db, "vk-fp", "vk-other", was_false_positive=1)

        result = get_vendor_cloud_accuracy(db, "vk-fp")
        assert result is not None
        assert result.false_positives == 1
