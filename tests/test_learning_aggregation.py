"""Tests for cross-session learning aggregation and maintenance."""

from __future__ import annotations

import json
import os
import uuid

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.maintenance.learning_aggregation import (
    DataQualityReport,
    MaintenanceReport,
    deduplicate_identity_members,
    delete_garbage_items,
    mark_stale_templates,
    prune_confidence_history,
    recalculate_template_reliability,
    remove_orphaned_members,
    run_full_maintenance,
    stamp_extraction_source,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_vendor_with_template(
    db,
    vendor_name,
    template_dict,
    vendor_key=None,
    identity_id=None,
):
    """Create a vendor identity with template metadata."""
    conn = db.get_connection()
    identity_id = identity_id or f"id-{vendor_name.lower().replace(' ', '-')}"
    metadata = json.dumps({"template": template_dict})
    conn.execute(
        "INSERT INTO identities (id, entity_type, canonical_name, metadata, active) "
        "VALUES (?, 'vendor', ?, ?, 1)",
        (identity_id, vendor_name, metadata),
    )
    if vendor_key:
        conn.execute(
            "INSERT INTO identity_members "
            "(id, identity_id, member_type, value, source) "
            "VALUES (?, ?, 'vendor_key', ?, 'test')",
            (str(uuid.uuid4()), identity_id, vendor_key),
        )
    conn.commit()
    return identity_id


def _seed_fact_for_vendor(db, fact_id, vendor_key):
    """Create a minimal fact for correction event testing."""
    conn = db.get_connection()
    doc_id = f"doc-{fact_id}"
    cloud_id = f"cloud-{fact_id}"
    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) "
        "VALUES (?, ?, ?)",
        (doc_id, f"/tmp/{doc_id}.jpg", f"hash-{doc_id}"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')",
        (cloud_id,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO facts "
        "(id, cloud_id, fact_type, vendor, vendor_key, "
        "total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', 'Test', ?, 10.0, 'EUR', '2026-01-01')",
        (fact_id, cloud_id, vendor_key),
    )
    conn.commit()


def _seed_correction_event(db, entity_id, field, old_value, new_value):
    """Insert a correction event."""
    conn = db.get_connection()
    conn.execute(
        "INSERT INTO correction_events "
        "(id, entity_type, entity_id, field, old_value, new_value, source) "
        "VALUES (?, 'fact', ?, ?, ?, ?, 'test')",
        (str(uuid.uuid4()), entity_id, field, old_value, new_value),
    )
    conn.commit()


def _seed_fact_item(
    db,
    name,
    *,
    brand=None,
    category=None,
    enrichment_source=None,
    enrichment_confidence=None,
    fact_id=None,
):
    """Insert a fact (with supporting cloud/document/atom) and a fact_item.

    Returns (fact_id, item_id).
    """
    conn = db.get_connection()
    fact_id = fact_id or str(uuid.uuid4())
    cloud_id = f"cloud-{fact_id}"
    doc_id = f"doc-{fact_id}"
    atom_id = f"atom-{fact_id}"
    item_id = str(uuid.uuid4())

    conn.execute(
        "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')",
        (cloud_id,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) "
        "VALUES (?, ?, ?)",
        (doc_id, f"/tmp/{doc_id}.jpg", f"hash-{doc_id}"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO facts "
        "(id, cloud_id, fact_type, vendor, vendor_key, "
        "total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', 'Test', 'vk-garbage-test', 10.0, 'EUR', '2026-01-01')",
        (fact_id, cloud_id),
    )
    conn.execute(
        "INSERT OR IGNORE INTO atoms (id, document_id, atom_type, data) "
        "VALUES (?, ?, 'item', '{}')",
        (atom_id, doc_id),
    )
    conn.execute(
        "INSERT INTO fact_items "
        "(id, fact_id, atom_id, name, brand, category, "
        " enrichment_source, enrichment_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            item_id,
            fact_id,
            atom_id,
            name,
            brand,
            category,
            enrichment_source,
            enrichment_confidence,
        ),
    )
    conn.commit()
    return fact_id, item_id


# ===========================================================================
# TestRecalculateTemplateReliability
# ===========================================================================


class TestRecalculateTemplateReliability:
    def test_no_templates_returns_zero(self, db):
        result = recalculate_template_reliability(db)
        assert result == 0

    def test_template_with_corrections_updates_common_fixes(self, db):
        vendor_key = "vat-test-vendor"
        identity_id = _seed_vendor_with_template(
            db,
            "Test Vendor",
            {"layout_type": "standard", "success_count": 5, "common_fixes": {}},
            vendor_key=vendor_key,
        )

        # Create fact and correction events
        _seed_fact_for_vendor(db, "fact-1", vendor_key)
        _seed_correction_event(db, "fact-1", "total_amount", "10.0", "15.0")
        _seed_correction_event(db, "fact-1", "total_amount", "15.0", "12.0")
        _seed_correction_event(db, "fact-1", "vendor", "Wrong", "Correct")

        result = recalculate_template_reliability(db)
        assert result == 1

        # Verify common_fixes updated
        conn = db.get_connection()
        row = conn.execute(
            "SELECT metadata FROM identities WHERE id = ?", (identity_id,)
        ).fetchone()
        metadata = json.loads(row["metadata"])
        template = metadata["template"]
        assert template["common_fixes"]["total_amount"] == 2
        assert template["common_fixes"]["vendor"] == 1

    def test_high_correction_rate_marks_stale(self, db):
        vendor_key = "vat-stale-vendor"
        identity_id = _seed_vendor_with_template(
            db,
            "Stale Vendor",
            {"layout_type": "standard", "success_count": 3, "common_fixes": {}},
            vendor_key=vendor_key,
        )

        # Create 1 fact with many corrections (>30% rate)
        _seed_fact_for_vendor(db, "fact-stale", vendor_key)
        for i in range(5):
            _seed_correction_event(db, "fact-stale", f"field_{i}", "old", "new")

        recalculate_template_reliability(db)

        conn = db.get_connection()
        row = conn.execute(
            "SELECT metadata FROM identities WHERE id = ?", (identity_id,)
        ).fetchone()
        metadata = json.loads(row["metadata"])
        assert metadata["template"].get("stale") is True

    def test_template_without_vendor_key_skipped(self, db):
        """Templates with no vendor_key member are skipped gracefully."""
        _seed_vendor_with_template(
            db,
            "No VK Vendor",
            {"layout_type": "standard", "success_count": 3},
            vendor_key=None,
        )
        result = recalculate_template_reliability(db)
        assert result == 0


# ===========================================================================
# TestMarkStaleTemplates
# ===========================================================================


class TestMarkStaleTemplates:
    def test_no_templates_returns_zero(self, db):
        assert mark_stale_templates(db) == 0

    def test_recent_template_not_marked_stale(self, db):
        from datetime import datetime, timezone

        _seed_vendor_with_template(
            db,
            "Recent Vendor",
            {
                "layout_type": "standard",
                "success_count": 5,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
        )
        assert mark_stale_templates(db, max_age_days=90) == 0

    def test_old_template_marked_stale(self, db):
        from datetime import datetime, timezone, timedelta

        old_date = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        identity_id = _seed_vendor_with_template(
            db,
            "Old Vendor",
            {
                "layout_type": "standard",
                "success_count": 5,
                "last_updated": old_date,
            },
        )

        assert mark_stale_templates(db, max_age_days=90) == 1

        conn = db.get_connection()
        row = conn.execute(
            "SELECT metadata FROM identities WHERE id = ?", (identity_id,)
        ).fetchone()
        metadata = json.loads(row["metadata"])
        assert metadata["template"]["stale"] is True

    def test_no_last_updated_marked_stale(self, db):
        _seed_vendor_with_template(
            db,
            "No Date Vendor",
            {"layout_type": "standard", "success_count": 3},
        )
        assert mark_stale_templates(db) == 1

    def test_already_stale_not_double_counted(self, db):
        _seed_vendor_with_template(
            db,
            "Already Stale",
            {"layout_type": "standard", "success_count": 5, "stale": True},
        )
        assert mark_stale_templates(db) == 0

    def test_zero_success_count_skipped(self, db):
        _seed_vendor_with_template(
            db,
            "Zero Count",
            {"layout_type": "standard", "success_count": 0},
        )
        assert mark_stale_templates(db) == 0


# ===========================================================================
# TestPruneConfidenceHistory
# ===========================================================================


class TestPruneConfidenceHistory:
    def test_short_history_not_pruned(self, db):
        _seed_vendor_with_template(
            db,
            "Short History",
            {
                "layout_type": "standard",
                "success_count": 5,
                "confidence_history": [0.9, 0.85, 0.88],
            },
        )
        assert prune_confidence_history(db, max_entries=20) == 0

    def test_long_history_truncated(self, db):
        long_history = [0.8 + i * 0.005 for i in range(30)]
        identity_id = _seed_vendor_with_template(
            db,
            "Long History",
            {
                "layout_type": "standard",
                "success_count": 10,
                "confidence_history": long_history,
            },
        )

        assert prune_confidence_history(db, max_entries=10) == 1

        conn = db.get_connection()
        row = conn.execute(
            "SELECT metadata FROM identities WHERE id = ?", (identity_id,)
        ).fetchone()
        metadata = json.loads(row["metadata"])
        history = metadata["template"]["confidence_history"]
        assert len(history) == 10
        # Should be the last 10 entries
        assert history == long_history[-10:]

    def test_no_history_unchanged(self, db):
        _seed_vendor_with_template(
            db,
            "No History",
            {"layout_type": "standard", "success_count": 3},
        )
        assert prune_confidence_history(db) == 0


# ===========================================================================
# TestDeduplicateIdentityMembers
# ===========================================================================


class TestDeduplicateIdentityMembers:
    def test_no_duplicates_returns_zero(self, db):
        conn = db.get_connection()
        identity_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO identities (id, entity_type, canonical_name, active) "
            "VALUES (?, 'vendor', 'Unique', 1)",
            (identity_id,),
        )
        conn.execute(
            "INSERT INTO identity_members "
            "(id, identity_id, member_type, value, source) "
            "VALUES (?, ?, 'name', 'Unique Store', 'test')",
            (str(uuid.uuid4()), identity_id),
        )
        conn.commit()
        assert deduplicate_identity_members(db) == 0

    def test_duplicate_members_removed(self, db):
        """Duplicates created by dropping UNIQUE constraint are cleaned up."""
        conn = db.get_connection()
        identity_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO identities (id, entity_type, canonical_name, active) "
            "VALUES (?, 'vendor', 'DupVendor', 1)",
            (identity_id,),
        )
        # Drop UNIQUE constraint temporarily to simulate legacy data
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "CREATE TABLE identity_members_tmp AS " "SELECT * FROM identity_members"
        )
        conn.execute("DROP TABLE identity_members")
        conn.execute(
            "CREATE TABLE identity_members ("
            "  id TEXT PRIMARY KEY,"
            "  identity_id TEXT NOT NULL,"
            "  member_type TEXT NOT NULL,"
            "  value TEXT NOT NULL,"
            "  source TEXT DEFAULT 'extraction',"
            "  created_at DATETIME DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        conn.execute(
            "INSERT INTO identity_members " "SELECT * FROM identity_members_tmp"
        )
        conn.execute("DROP TABLE identity_members_tmp")

        # Now insert duplicates without UNIQUE constraint
        for _ in range(3):
            conn.execute(
                "INSERT INTO identity_members "
                "(id, identity_id, member_type, value, source) "
                "VALUES (?, ?, 'name', 'Same Name', 'test')",
                (str(uuid.uuid4()), identity_id),
            )
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()

        removed = deduplicate_identity_members(db)
        assert removed == 2  # keep 1, remove 2

        # Verify only 1 remains
        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM identity_members "
            "WHERE identity_id = ? AND member_type = 'name' "
            "AND value = 'Same Name'",
            (identity_id,),
        ).fetchone()["cnt"]
        assert count == 1

    def test_different_member_types_not_affected(self, db):
        conn = db.get_connection()
        identity_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO identities (id, entity_type, canonical_name, active) "
            "VALUES (?, 'vendor', 'MultiType', 1)",
            (identity_id,),
        )
        conn.execute(
            "INSERT INTO identity_members "
            "(id, identity_id, member_type, value, source) "
            "VALUES (?, ?, 'name', 'Store', 'test')",
            (str(uuid.uuid4()), identity_id),
        )
        conn.execute(
            "INSERT INTO identity_members "
            "(id, identity_id, member_type, value, source) "
            "VALUES (?, ?, 'normalized_name', 'store', 'test')",
            (str(uuid.uuid4()), identity_id),
        )
        conn.commit()
        assert deduplicate_identity_members(db) == 0


# ===========================================================================
# TestRemoveOrphanedMembers
# ===========================================================================


class TestRemoveOrphanedMembers:
    def test_no_orphans_returns_zero(self, db):
        assert remove_orphaned_members(db) == 0

    def test_orphaned_members_removed(self, db):
        conn = db.get_connection()
        # Disable FK to simulate orphaned data from legacy/migration
        conn.execute("PRAGMA foreign_keys = OFF")
        fake_identity_id = "non-existent-identity-id"
        conn.execute(
            "INSERT INTO identity_members "
            "(id, identity_id, member_type, value, source) "
            "VALUES (?, ?, 'name', 'Orphan', 'test')",
            (str(uuid.uuid4()), fake_identity_id),
        )
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()

        removed = remove_orphaned_members(db)
        assert removed == 1


# ===========================================================================
# TestRunFullMaintenance
# ===========================================================================


class TestRunFullMaintenance:
    def test_runs_all_operations(self, db):
        """Full maintenance returns aggregate report."""
        report = run_full_maintenance(db)
        assert isinstance(report, MaintenanceReport)
        assert report.templates_recalculated == 0
        assert report.templates_marked_stale == 0
        assert report.templates_pruned == 0
        assert report.members_deduplicated == 0
        assert report.orphaned_members_removed == 0

    def test_combined_maintenance(self, db):
        """Maintenance finds and fixes multiple issues."""
        # Create a vendor with long history (will be pruned)
        _seed_vendor_with_template(
            db,
            "Prunable Vendor",
            {
                "layout_type": "standard",
                "success_count": 5,
                "confidence_history": [0.85] * 30,
                "last_updated": "2025-01-01T00:00:00+00:00",  # old
            },
        )

        # Create orphaned member (disable FK to simulate legacy data)
        conn = db.get_connection()
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "INSERT INTO identity_members "
            "(id, identity_id, member_type, value, source) "
            "VALUES (?, 'ghost-identity', 'name', 'Ghost', 'test')",
            (str(uuid.uuid4()),),
        )
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()

        report = run_full_maintenance(db, max_history_entries=10, stale_days=90)

        assert report.templates_pruned >= 1
        assert report.templates_marked_stale >= 1
        assert report.orphaned_members_removed >= 1


# ===========================================================================
# TestDeleteGarbageItems
# ===========================================================================


class TestDeleteGarbageItems:
    def test_delete_garbage_items_exact_names(self, db):
        """Items with exact garbage names are deleted."""
        _, purchase_id = _seed_fact_item(db, "PURCHASE")
        _, tip_id = _seed_fact_item(db, "TIP")
        _, kapta_id = _seed_fact_item(db, "KAPTA")
        _, syndao_id = _seed_fact_item(db, "ΣYNDAO")
        _, synoao_id = _seed_fact_item(db, "ΣYNOAO")

        report = delete_garbage_items(db)

        assert isinstance(report, DataQualityReport)
        assert report.units_fixed == 5

        conn = db.get_connection()
        for item_id in (purchase_id, tip_id, kapta_id, syndao_id, synoao_id):
            row = conn.execute(
                "SELECT id FROM fact_items WHERE id = ?", (item_id,)
            ).fetchone()
            assert row is None, f"Expected {item_id!r} to be deleted"

    def test_delete_garbage_items_like_patterns(self, db):
        """Items matching LIKE patterns are deleted."""
        _, vat_id = _seed_fact_item(db, "ΦΠA 24% ΦΠA 24%")
        _, mult_id = _seed_fact_item(db, "3 × 1.50")
        _, geo_id = _seed_fact_item(db, "Βορείουνατολικός Τομέας Αθηνών")
        _, long_id = _seed_fact_item(db, "X" * 201)

        report = delete_garbage_items(db)

        assert report.units_fixed == 4

        conn = db.get_connection()
        for item_id in (vat_id, mult_id, geo_id, long_id):
            row = conn.execute(
                "SELECT id FROM fact_items WHERE id = ?", (item_id,)
            ).fetchone()
            assert row is None, f"Expected {item_id!r} to be deleted"

    def test_delete_garbage_items_preserves_real(self, db):
        """Garbage items are deleted while legitimate product items survive."""
        _, real_id = _seed_fact_item(db, "Organic Milk 1L", brand="Alpro")
        _, another_real_id = _seed_fact_item(db, "Bread 500g")
        _, garbage_id = _seed_fact_item(db, "PURCHASE")
        _, short_no_meta_id = _seed_fact_item(db, "A")  # <=2 chars, no brand/category
        # Short name WITH brand — should NOT be deleted
        _, short_with_brand_id = _seed_fact_item(db, "AB", brand="ACME")

        report = delete_garbage_items(db)

        # Only the two true garbage items should be removed
        assert report.units_fixed == 2

        conn = db.get_connection()
        # Real items survive
        for item_id in (real_id, another_real_id, short_with_brand_id):
            row = conn.execute(
                "SELECT id FROM fact_items WHERE id = ?", (item_id,)
            ).fetchone()
            assert row is not None, f"Expected {item_id!r} to be preserved"

        # Garbage items are gone
        for item_id in (garbage_id, short_no_meta_id):
            row = conn.execute(
                "SELECT id FROM fact_items WHERE id = ?", (item_id,)
            ).fetchone()
            assert row is None, f"Expected {item_id!r} to be deleted"


# ===========================================================================
# TestStampExtractionSource
# ===========================================================================


class TestStampExtractionSource:
    def test_stamp_extraction_source_basic(self, db):
        """Items with brand+category but no enrichment_source get stamped."""
        _, item_id = _seed_fact_item(
            db, "Greek Yoghurt", brand="Fage", category="Dairy"
        )

        report = stamp_extraction_source(db)

        assert isinstance(report, DataQualityReport)
        assert report.units_fixed == 1

        conn = db.get_connection()
        row = conn.execute(
            "SELECT enrichment_source, enrichment_confidence "
            "FROM fact_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        assert row["enrichment_source"] == "extraction"
        assert abs(row["enrichment_confidence"] - 0.70) < 0.001

    def test_stamp_extraction_source_skips_already_stamped(self, db):
        """Items that already have an enrichment_source are not overwritten."""
        _, off_id = _seed_fact_item(
            db,
            "Olive Oil",
            brand="Minerva",
            category="Condiments",
            enrichment_source="openfoodfacts",
            enrichment_confidence=0.95,
        )
        _, user_id = _seed_fact_item(
            db,
            "Pasta",
            brand="Barilla",
            category="Dry Goods",
            enrichment_source="user_confirmed",
            enrichment_confidence=1.0,
        )

        report = stamp_extraction_source(db)

        assert report.units_fixed == 0

        conn = db.get_connection()
        row = conn.execute(
            "SELECT enrichment_source, enrichment_confidence "
            "FROM fact_items WHERE id = ?",
            (off_id,),
        ).fetchone()
        assert row["enrichment_source"] == "openfoodfacts"
        assert abs(row["enrichment_confidence"] - 0.95) < 0.001

    def test_stamp_extraction_source_skips_incomplete(self, db):
        """Items missing brand or category are not stamped."""
        # Has brand but no category
        _, brand_only_id = _seed_fact_item(db, "Mystery Item", brand="SomeBrand")
        # Has category but no brand
        _, cat_only_id = _seed_fact_item(db, "Unknown Brand Chips", category="Snacks")
        # Has neither brand nor category
        _, bare_id = _seed_fact_item(db, "Plain Item")

        report = stamp_extraction_source(db)

        assert report.units_fixed == 0

        conn = db.get_connection()
        for item_id in (brand_only_id, cat_only_id, bare_id):
            row = conn.execute(
                "SELECT enrichment_source FROM fact_items WHERE id = ?",
                (item_id,),
            ).fetchone()
            assert row["enrichment_source"] is None
