"""Tests for correction event log and extraction metrics adaptive learning."""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

import pytest

from alibi.db import v2_store
from alibi.db.models import (
    CloudStatus,
    Fact,
    FactItem,
    FactStatus,
    FactType,
    TaxType,
    UnitType,
)
from alibi.extraction.templates import (
    VendorTemplate,
    record_extraction_observation,
    template_to_hints,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cloud(db_manager, cloud_id: str | None = None) -> str:
    cid = cloud_id or str(uuid.uuid4())
    with db_manager.transaction() as cursor:
        cursor.execute(
            "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
            (cid, CloudStatus.COLLAPSED.value, 1.0),
        )
    return cid


def _make_fact(
    db_manager,
    cloud_id: str,
    vendor: str = "Test Vendor",
    vendor_key: str | None = None,
    fact_id: str | None = None,
) -> str:
    fid = fact_id or str(uuid.uuid4())
    fact = Fact(
        id=fid,
        cloud_id=cloud_id,
        fact_type=FactType.PURCHASE,
        vendor=vendor,
        vendor_key=vendor_key,
        total_amount=Decimal("10.00"),
        currency="EUR",
        event_date=date(2024, 1, 15),
        status=FactStatus.CONFIRMED,
    )
    v2_store.store_fact(db_manager, fact, [])
    return fid


def _make_fact_item(db_manager, fact_id: str, name: str = "Widget") -> str:
    """Insert a fact item with a stub atom to satisfy the FK constraint on atom_id."""
    import json

    item_id = str(uuid.uuid4())
    atom_id = str(uuid.uuid4())

    # fact_items.atom_id FK → atoms.id; insert a minimal atom stub first.
    # atoms.document_id FK → documents.id; use a stub document too.
    doc_id = str(uuid.uuid4())
    with db_manager.transaction() as cursor:
        cursor.execute(
            "INSERT OR IGNORE INTO documents "
            "(id, file_path, file_hash, source, user_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (doc_id, f"/tmp/{doc_id}.jpg", doc_id[:16], "test", "system"),
        )
        cursor.execute(
            "INSERT OR IGNORE INTO atoms "
            "(id, document_id, atom_type, data, confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            (atom_id, doc_id, "item", json.dumps({"name": name}), 1.0),
        )
        cursor.execute(
            "INSERT OR IGNORE INTO fact_items "
            "(id, fact_id, atom_id, name, name_normalized, "
            "quantity, unit, unit_price, total_price, "
            "brand, category, comparable_unit_price, comparable_unit, "
            "barcode, unit_quantity, tax_rate, tax_type, "
            "enrichment_source, enrichment_confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item_id,
                fact_id,
                atom_id,
                name,
                None,
                1.0,
                UnitType.PIECE.value,
                None,
                5.0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                TaxType.NONE.value,
                None,
                None,
            ),
        )
    return item_id


# ---------------------------------------------------------------------------
# TestCorrectionEventsDB — integration tests against real in-memory SQLite
# ---------------------------------------------------------------------------


class TestCorrectionEventsDB:
    def test_record_and_retrieve_by_entity(self, db_manager):
        """Record a correction event then fetch it back by entity."""
        entity_id = str(uuid.uuid4())

        event_id = v2_store.record_correction_event(
            db_manager,
            entity_type="fact",
            entity_id=entity_id,
            field="vendor",
            old_value="Old Corp",
            new_value="New Corp",
            source="api",
            user_id="user-1",
        )

        rows = v2_store.get_corrections_by_entity(db_manager, "fact", entity_id)

        assert len(rows) == 1
        row = rows[0]
        assert row["id"] == event_id
        assert row["entity_type"] == "fact"
        assert row["entity_id"] == entity_id
        assert row["field"] == "vendor"
        assert row["old_value"] == "Old Corp"
        assert row["new_value"] == "New Corp"
        assert row["source"] == "api"
        assert row["user_id"] == "user-1"
        assert row["created_at"] is not None

    def test_retrieve_by_field(self, db_manager):
        """get_corrections_by_field returns all events for that field across entities."""
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        id_c = str(uuid.uuid4())

        v2_store.record_correction_event(
            db_manager, "fact", id_a, "vendor", "A Old", "A New", "api"
        )
        v2_store.record_correction_event(
            db_manager, "fact", id_b, "vendor", "B Old", "B New", "api"
        )
        # Different field — should NOT appear in vendor results
        v2_store.record_correction_event(
            db_manager, "fact", id_c, "amount", "1.00", "2.00", "api"
        )

        vendor_rows = v2_store.get_corrections_by_field(db_manager, "vendor")
        amount_rows = v2_store.get_corrections_by_field(db_manager, "amount")

        assert len(vendor_rows) == 2
        assert all(r["field"] == "vendor" for r in vendor_rows)
        entity_ids = {r["entity_id"] for r in vendor_rows}
        assert entity_ids == {id_a, id_b}

        assert len(amount_rows) == 1
        assert amount_rows[0]["entity_id"] == id_c

    def test_correction_rate_no_corrections(self, db_manager):
        """A vendor with facts but zero corrections has rate 0.0."""
        vendor_key = "CY10000001A"
        cloud_id = _make_cloud(db_manager)
        _make_fact(db_manager, cloud_id, vendor_key=vendor_key)

        result = v2_store.get_correction_rate(db_manager, vendor_key)

        assert result["total_facts"] == 1
        assert result["corrected_facts"] == 0
        assert result["rate"] == 0.0

    def test_correction_rate_with_corrections(self, db_manager):
        """Vendor with 2 facts where 1 is corrected → rate 0.5."""
        vendor_key = "CY10000002B"
        cloud_a = _make_cloud(db_manager)
        cloud_b = _make_cloud(db_manager)
        fact_a = _make_fact(db_manager, cloud_a, vendor_key=vendor_key)
        _make_fact(db_manager, cloud_b, vendor_key=vendor_key)

        # Correct only fact_a
        v2_store.record_correction_event(
            db_manager,
            entity_type="fact",
            entity_id=fact_a,
            field="vendor",
            old_value="Old Name",
            new_value="New Name",
            source="api",
        )

        result = v2_store.get_correction_rate(db_manager, vendor_key)

        assert result["total_facts"] == 2
        assert result["corrected_facts"] == 1
        assert result["rate"] == pytest.approx(0.5)

    def test_corrections_by_vendor(self, db_manager):
        """get_corrections_by_vendor returns events for facts and fact_items of a vendor."""
        vendor_key = "CY10000003C"
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id, vendor_key=vendor_key)
        item_id = _make_fact_item(db_manager, fact_id, name="Apple Juice")

        # Correction on the fact itself
        v2_store.record_correction_event(
            db_manager, "fact", fact_id, "vendor", "Old", "New", "api"
        )
        # Correction on the item
        v2_store.record_correction_event(
            db_manager, "fact_item", item_id, "category", None, "Beverage", "api"
        )
        # Correction on a different vendor's fact — should NOT appear
        other_cloud = _make_cloud(db_manager)
        other_fact = _make_fact(db_manager, other_cloud, vendor_key="CY99999999Z")
        v2_store.record_correction_event(
            db_manager, "fact", other_fact, "vendor", "X", "Y", "api"
        )

        rows = v2_store.get_corrections_by_vendor(db_manager, vendor_key)

        assert len(rows) == 2
        entity_ids = {r["entity_id"] for r in rows}
        assert fact_id in entity_ids
        assert item_id in entity_ids
        assert other_fact not in entity_ids

    def test_old_new_value_serialized_as_string(self, db_manager):
        """Numeric values for old/new are stored and retrieved as strings."""
        entity_id = str(uuid.uuid4())

        v2_store.record_correction_event(
            db_manager,
            entity_type="fact",
            entity_id=entity_id,
            field="amount",
            old_value=10,
            new_value=Decimal("99.95"),
            source="api",
        )

        rows = v2_store.get_corrections_by_entity(db_manager, "fact", entity_id)
        assert len(rows) == 1
        assert rows[0]["old_value"] == "10"
        assert rows[0]["new_value"] == "99.95"

    def test_none_values_stored_as_null(self, db_manager):
        """None old/new values are stored as SQL NULL (not the string 'None')."""
        entity_id = str(uuid.uuid4())

        v2_store.record_correction_event(
            db_manager,
            entity_type="fact",
            entity_id=entity_id,
            field="category",
            old_value=None,
            new_value="Grocery",
            source="api",
        )

        rows = v2_store.get_corrections_by_entity(db_manager, "fact", entity_id)
        assert len(rows) == 1
        assert rows[0]["old_value"] is None
        assert rows[0]["new_value"] == "Grocery"


# ---------------------------------------------------------------------------
# TestCorrectionLoggingIntegration — services actually record corrections
# ---------------------------------------------------------------------------


class TestCorrectionLoggingIntegration:
    def test_update_fact_records_correction(self, db_manager):
        """services.correction.update_fact writes a correction_events row."""
        from alibi.services import correction as svc

        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id, vendor="Original Vendor")

        result = svc.update_fact(db_manager, fact_id, {"vendor": "Updated Vendor"})

        assert result is True
        rows = v2_store.get_corrections_by_entity(db_manager, "fact", fact_id)
        assert len(rows) == 1
        assert rows[0]["field"] == "vendor"
        assert rows[0]["old_value"] == "Original Vendor"
        assert rows[0]["new_value"] == "Updated Vendor"
        assert rows[0]["source"] == "system"

    def test_correct_vendor_records_correction(self, db_manager):
        """services.correction.correct_vendor writes a correction_events row."""
        from alibi.services import correction as svc

        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id, vendor="Old Supermarket")

        result = svc.correct_vendor(db_manager, fact_id, "New Supermarket")

        assert result is True
        rows = v2_store.get_corrections_by_entity(db_manager, "fact", fact_id)
        assert len(rows) == 1
        assert rows[0]["field"] == "vendor"
        assert rows[0]["old_value"] == "Old Supermarket"
        # new_value is the normalized form
        assert rows[0]["new_value"] == "New Supermarket"

    def test_update_fact_item_records_correction(self, db_manager):
        """services.correction.update_fact_item writes a correction_events row."""
        from alibi.services import correction as svc

        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(db_manager, fact_id, name="Milk")

        result = svc.update_fact_item(db_manager, item_id, {"brand": "Alpro"})

        assert result is True
        rows = v2_store.get_corrections_by_entity(db_manager, "fact_item", item_id)
        assert len(rows) == 1
        assert rows[0]["field"] == "brand"
        assert rows[0]["old_value"] is None
        assert rows[0]["new_value"] == "Alpro"
        assert rows[0]["entity_type"] == "fact_item"

    def test_no_correction_when_value_unchanged(self, db_manager):
        """Updating a field to the same value does not record a correction event."""
        from alibi.services import correction as svc

        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id, vendor="Same Vendor")

        # Update vendor to identical value
        result = svc.update_fact(db_manager, fact_id, {"vendor": "Same Vendor"})

        assert result is True
        rows = v2_store.get_corrections_by_entity(db_manager, "fact", fact_id)
        # No correction event since value did not change
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# TestExtractionMetrics — VendorTemplate adaptive learning
# ---------------------------------------------------------------------------


class TestExtractionMetrics:
    def test_record_observation_appends_confidence(self):
        """New confidence value is appended to confidence_history."""
        template = VendorTemplate(success_count=3)

        updated = record_extraction_observation(template, confidence=0.85)

        assert 0.85 in updated.confidence_history
        assert len(updated.confidence_history) == 1

    def test_confidence_history_capped_at_20(self):
        """After 25 observations, only the most recent 20 are kept."""
        template = VendorTemplate(success_count=5)

        for i in range(25):
            template = record_extraction_observation(template, confidence=0.9)

        assert len(template.confidence_history) == 20

    def test_adaptive_threshold_high(self):
        """10+ high-confidence docs (avg > 0.95) → threshold set to 0.95."""
        template = VendorTemplate(success_count=10)
        for _ in range(10):
            template = record_extraction_observation(template, confidence=0.97)

        assert template.adaptive_skip_threshold == pytest.approx(0.95)

    def test_adaptive_threshold_low(self):
        """10+ low-confidence docs (avg < 0.70) → threshold set to 0.7."""
        template = VendorTemplate(success_count=10)
        for _ in range(10):
            template = record_extraction_observation(template, confidence=0.55)

        assert template.adaptive_skip_threshold == pytest.approx(0.7)

    def test_adaptive_threshold_none(self):
        """Average in the 0.70–0.95 band → threshold is None."""
        template = VendorTemplate(success_count=10)
        for _ in range(10):
            template = record_extraction_observation(template, confidence=0.80)

        assert template.adaptive_skip_threshold is None

    def test_ocr_tier_memory(self):
        """preferred_ocr_tier tracks the maximum (worst) tier seen."""
        template = VendorTemplate(success_count=5)

        t1 = record_extraction_observation(template, confidence=0.9, ocr_tier=1)
        assert t1.preferred_ocr_tier == 1

        t2 = record_extraction_observation(t1, confidence=0.9, ocr_tier=3)
        assert t2.preferred_ocr_tier == 3

        # Lower tier after the high one should not replace it
        t3 = record_extraction_observation(t2, confidence=0.9, ocr_tier=2)
        assert t3.preferred_ocr_tier == 3

    def test_needs_rotation_latched(self):
        """Once needs_rotation is True it stays True regardless of subsequent calls."""
        template = VendorTemplate(success_count=5)

        t1 = record_extraction_observation(template, confidence=0.9, was_rotated=True)
        assert t1.needs_rotation is True

        t2 = record_extraction_observation(t1, confidence=0.9, was_rotated=False)
        assert t2.needs_rotation is True

    def test_common_fixes_accumulated(self):
        """Fix counts accumulate across multiple observations."""
        template = VendorTemplate(success_count=5)

        t1 = record_extraction_observation(
            template, confidence=0.8, fixes_applied=["date_format"]
        )
        t2 = record_extraction_observation(
            t1, confidence=0.8, fixes_applied=["date_format", "currency"]
        )
        t3 = record_extraction_observation(
            t2, confidence=0.8, fixes_applied=["currency"]
        )

        assert t3.common_fixes["date_format"] == 2
        assert t3.common_fixes["currency"] == 2

    def test_template_to_hints_with_unreliable_fields(self):
        """Fields with >= 5 corrections appear in hints.unreliable_fields."""
        template = VendorTemplate(
            success_count=5,
            currency="EUR",
            layout_type="columnar",
            common_fixes={"date_format": 5, "vendor_name": 3, "total_amount": 7},
        )

        hints = template_to_hints(template, vendor_name="Test Shop")

        assert hints.unreliable_fields is not None
        assert "date_format" in hints.unreliable_fields
        assert "total_amount" in hints.unreliable_fields
        # vendor_name only 3 — below threshold
        assert "vendor_name" not in hints.unreliable_fields

    def test_template_roundtrip_with_new_fields(self):
        """to_dict/from_dict preserves all adaptive learning fields."""
        original = VendorTemplate(
            layout_type="columnar",
            currency="EUR",
            pos_provider="JCC",
            success_count=12,
            gemini_bootstrapped=True,
            language="el",
            has_barcodes=True,
            has_unit_quantities=False,
            typical_item_count=8,
            confidence_history=[0.85, 0.90, 0.92],
            adaptive_skip_threshold=0.95,
            preferred_ocr_tier=1,
            needs_rotation=True,
            common_fixes={"date_format": 3, "currency": 6},
        )

        d = original.to_dict()
        restored = VendorTemplate.from_dict(d)

        assert restored.layout_type == original.layout_type
        assert restored.currency == original.currency
        assert restored.pos_provider == original.pos_provider
        assert restored.success_count == original.success_count
        assert restored.gemini_bootstrapped == original.gemini_bootstrapped
        assert restored.language == original.language
        assert restored.has_barcodes == original.has_barcodes
        assert restored.has_unit_quantities == original.has_unit_quantities
        assert restored.typical_item_count == original.typical_item_count
        assert restored.confidence_history == original.confidence_history
        assert restored.adaptive_skip_threshold == pytest.approx(
            original.adaptive_skip_threshold
        )
        assert restored.preferred_ocr_tier == original.preferred_ocr_tier
        assert restored.needs_rotation == original.needs_rotation
        assert restored.common_fixes == original.common_fixes

    def test_empty_template_roundtrip(self):
        """Default VendorTemplate round-trips without serializing optional fields."""
        template = VendorTemplate()

        d = template.to_dict()
        restored = VendorTemplate.from_dict(d)

        # Only mandatory keys are present in dict
        assert set(d.keys()) == {"layout_type", "success_count"}

        assert restored.layout_type == "standard"
        assert restored.success_count == 0
        assert restored.currency is None
        assert restored.pos_provider is None
        assert restored.gemini_bootstrapped is False
        assert restored.language is None
        assert restored.has_barcodes is None
        assert restored.has_unit_quantities is None
        assert restored.typical_item_count is None
        assert restored.confidence_history == []
        assert restored.adaptive_skip_threshold is None
        assert restored.preferred_ocr_tier is None
        assert restored.needs_rotation is False
        assert restored.common_fixes == {}
