"""Tests for alibi.services.enrichment_review — enrichment feedback loop."""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

import pytest

from alibi.db import v2_store
from alibi.db.models import (
    Cloud,
    CloudStatus,
    Fact,
    FactItem,
    FactStatus,
    FactType,
)
from alibi.services import enrichment_review as svc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cloud(db) -> str:
    cloud_id = str(uuid.uuid4())
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
            (cloud_id, CloudStatus.COLLAPSED.value, 1.0),
        )
    return cloud_id


def _make_fact(db, cloud_id: str, vendor: str = "Test Vendor") -> str:
    fact_id = str(uuid.uuid4())
    fact = Fact(
        id=fact_id,
        cloud_id=cloud_id,
        fact_type=FactType.PURCHASE,
        vendor=vendor,
        vendor_key=None,
        total_amount=Decimal("25.00"),
        currency="EUR",
        event_date=date(2024, 6, 1),
        status=FactStatus.CONFIRMED,
    )
    v2_store.store_fact(db, fact, [])
    return fact_id


def _make_doc_and_atom(db) -> str:
    """Create a minimal document + atom pair and return the atom_id."""
    doc_id = str(uuid.uuid4())
    atom_id = str(uuid.uuid4())
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
            (doc_id, f"/tmp/test-{doc_id[:8]}.jpg", doc_id),
        )
        cursor.execute(
            "INSERT INTO atoms (id, document_id, atom_type, data) VALUES (?, ?, ?, ?)",
            (atom_id, doc_id, "item", "{}"),
        )
    return atom_id


def _make_fact_item(
    db,
    fact_id: str,
    name: str = "Olive Oil",
    brand: str | None = "BestBrand",
    category: str | None = "oils",
    enrichment_source: str | None = "open_food_facts",
    enrichment_confidence: float | None = 0.6,
) -> str:
    atom_id = _make_doc_and_atom(db)
    item_id = str(uuid.uuid4())
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO fact_items
                (id, fact_id, atom_id, name, brand, category,
                 enrichment_source, enrichment_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
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
    return item_id


# ---------------------------------------------------------------------------
# get_review_queue
# ---------------------------------------------------------------------------


class TestGetReviewQueue:
    def test_returns_low_confidence_items(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)

        # Low confidence — should appear
        low_id = _make_fact_item(
            db_manager,
            fact_id,
            name="Olive Oil",
            enrichment_source="open_food_facts",
            enrichment_confidence=0.5,
        )
        # High confidence — should NOT appear
        _make_fact_item(
            db_manager,
            fact_id,
            name="Tomatoes",
            enrichment_source="open_food_facts",
            enrichment_confidence=0.95,
        )

        result = svc.get_review_queue(db_manager, threshold=0.8)

        ids = [r["id"] for r in result]
        assert low_id in ids
        assert all(
            r["id"] != low_id or r["enrichment_confidence"] < 0.8 for r in result
        )

    def test_excludes_items_without_enrichment(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)

        # No enrichment at all — should NOT appear
        _make_fact_item(
            db_manager,
            fact_id,
            name="Plain Item",
            brand=None,
            category=None,
            enrichment_source=None,
            enrichment_confidence=None,
        )

        result = svc.get_review_queue(db_manager, threshold=0.8)
        assert result == []

    def test_ordered_worst_first(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)

        _make_fact_item(
            db_manager,
            fact_id,
            name="A",
            enrichment_source="open_food_facts",
            enrichment_confidence=0.7,
        )
        _make_fact_item(
            db_manager,
            fact_id,
            name="B",
            enrichment_source="open_food_facts",
            enrichment_confidence=0.3,
        )
        _make_fact_item(
            db_manager,
            fact_id,
            name="C",
            enrichment_source="open_food_facts",
            enrichment_confidence=0.5,
        )

        result = svc.get_review_queue(db_manager, threshold=0.8)
        confidences = [r["enrichment_confidence"] for r in result]
        assert confidences == sorted(confidences)

    def test_includes_vendor_from_fact(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id, vendor="Super Market")

        _make_fact_item(
            db_manager,
            fact_id,
            name="Bread",
            enrichment_source="llm",
            enrichment_confidence=0.4,
        )

        result = svc.get_review_queue(db_manager, threshold=0.8)
        assert len(result) == 1
        assert result[0]["vendor"] == "Super Market"

    def test_respects_limit(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)

        for i in range(10):
            _make_fact_item(
                db_manager,
                fact_id,
                name=f"Item {i}",
                enrichment_source="open_food_facts",
                enrichment_confidence=0.1 + i * 0.05,
            )

        result = svc.get_review_queue(db_manager, threshold=0.8, limit=3)
        assert len(result) == 3

    def test_threshold_boundary(self, db_manager):
        """Items at exactly the threshold should NOT be returned (strict <)."""
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)

        # Exactly at threshold
        _make_fact_item(
            db_manager,
            fact_id,
            name="Exact",
            enrichment_source="open_food_facts",
            enrichment_confidence=0.8,
        )

        result = svc.get_review_queue(db_manager, threshold=0.8)
        assert result == []


# ---------------------------------------------------------------------------
# confirm_enrichment
# ---------------------------------------------------------------------------


class TestConfirmEnrichment:
    def test_sets_user_confirmed_and_full_confidence(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(
            db_manager,
            fact_id,
            enrichment_source="open_food_facts",
            enrichment_confidence=0.5,
        )

        ok = svc.confirm_enrichment(db_manager, item_id)

        assert ok is True
        row = db_manager.fetchone(
            "SELECT enrichment_source, enrichment_confidence FROM fact_items WHERE id = ?",
            (item_id,),
        )
        assert row["enrichment_source"] == "user_confirmed"
        assert row["enrichment_confidence"] == pytest.approx(1.0)

    def test_updates_brand_when_provided(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(
            db_manager,
            fact_id,
            brand="OldBrand",
            enrichment_source="open_food_facts",
            enrichment_confidence=0.4,
        )

        ok = svc.confirm_enrichment(db_manager, item_id, brand="CorrectBrand")

        assert ok is True
        row = db_manager.fetchone(
            "SELECT brand, enrichment_source, enrichment_confidence FROM fact_items WHERE id = ?",
            (item_id,),
        )
        assert row["brand"] == "CorrectBrand"
        assert row["enrichment_source"] == "user_confirmed"

    def test_updates_category_when_provided(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(
            db_manager,
            fact_id,
            category="wrong_cat",
            enrichment_source="llm",
            enrichment_confidence=0.3,
        )

        ok = svc.confirm_enrichment(db_manager, item_id, category="dairy")

        assert ok is True
        row = db_manager.fetchone(
            "SELECT category FROM fact_items WHERE id = ?",
            (item_id,),
        )
        assert row["category"] == "Dairy"

    def test_updates_both_brand_and_category(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(
            db_manager,
            fact_id,
            brand="WrongBrand",
            category="wrong_cat",
            enrichment_source="open_food_facts",
            enrichment_confidence=0.55,
        )

        ok = svc.confirm_enrichment(
            db_manager, item_id, brand="RealBrand", category="beverages"
        )

        assert ok is True
        row = db_manager.fetchone(
            "SELECT brand, category, enrichment_source, enrichment_confidence "
            "FROM fact_items WHERE id = ?",
            (item_id,),
        )
        assert row["brand"] == "RealBrand"
        assert row["category"] == "Beverages"
        assert row["enrichment_source"] == "user_confirmed"
        assert row["enrichment_confidence"] == pytest.approx(1.0)

    def test_returns_false_for_missing_item(self, db_manager):
        ok = svc.confirm_enrichment(db_manager, str(uuid.uuid4()))
        assert ok is False

    def test_confirmed_item_no_longer_in_review_queue(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(
            db_manager,
            fact_id,
            enrichment_source="open_food_facts",
            enrichment_confidence=0.5,
        )

        svc.confirm_enrichment(db_manager, item_id)

        queue = svc.get_review_queue(db_manager, threshold=0.8)
        ids = [r["id"] for r in queue]
        assert item_id not in ids


# ---------------------------------------------------------------------------
# reject_enrichment
# ---------------------------------------------------------------------------


class TestRejectEnrichment:
    def test_clears_enrichment_fields_to_null(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(
            db_manager,
            fact_id,
            brand="SomeBrand",
            category="food",
            enrichment_source="open_food_facts",
            enrichment_confidence=0.6,
        )

        ok = svc.reject_enrichment(db_manager, item_id)

        assert ok is True
        row = db_manager.fetchone(
            "SELECT brand, category, enrichment_source, enrichment_confidence "
            "FROM fact_items WHERE id = ?",
            (item_id,),
        )
        assert row["brand"] is None
        assert row["category"] is None
        assert row["enrichment_source"] is None
        assert row["enrichment_confidence"] is None

    def test_returns_false_for_missing_item(self, db_manager):
        ok = svc.reject_enrichment(db_manager, str(uuid.uuid4()))
        assert ok is False

    def test_rejected_item_not_in_review_queue(self, db_manager):
        """After rejection the item has no enrichment_source, so queue excludes it."""
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(
            db_manager,
            fact_id,
            enrichment_source="open_food_facts",
            enrichment_confidence=0.4,
        )

        svc.reject_enrichment(db_manager, item_id)

        queue = svc.get_review_queue(db_manager, threshold=0.8)
        ids = [r["id"] for r in queue]
        assert item_id not in ids

    def test_preserves_name_field(self, db_manager):
        """Rejecting enrichment must not affect the item name."""
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(
            db_manager,
            fact_id,
            name="Special Olive Oil",
            enrichment_source="llm",
            enrichment_confidence=0.3,
        )

        svc.reject_enrichment(db_manager, item_id)

        row = db_manager.fetchone(
            "SELECT name FROM fact_items WHERE id = ?",
            (item_id,),
        )
        assert row["name"] == "Special Olive Oil"


# ---------------------------------------------------------------------------
# get_review_stats
# ---------------------------------------------------------------------------


class TestGetReviewStats:
    def test_counts_by_source(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)

        for _ in range(3):
            _make_fact_item(
                db_manager,
                fact_id,
                enrichment_source="open_food_facts",
                enrichment_confidence=0.9,
            )
        for _ in range(2):
            _make_fact_item(
                db_manager,
                fact_id,
                enrichment_source="llm",
                enrichment_confidence=0.5,
            )

        stats = svc.get_review_stats(db_manager)

        by_source = {
            row["enrichment_source"]: row["count"] for row in stats["by_source"]
        }
        assert by_source.get("open_food_facts") == 3
        assert by_source.get("llm") == 2

    def test_total_enriched_and_avg_confidence(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)

        _make_fact_item(
            db_manager,
            fact_id,
            enrichment_source="open_food_facts",
            enrichment_confidence=0.8,
        )
        _make_fact_item(
            db_manager,
            fact_id,
            enrichment_source="open_food_facts",
            enrichment_confidence=0.6,
        )
        # Unenriched — should not count
        _make_fact_item(
            db_manager,
            fact_id,
            enrichment_source=None,
            enrichment_confidence=None,
        )

        stats = svc.get_review_stats(db_manager)

        assert stats["total_enriched"] == 2
        assert stats["avg_confidence"] == pytest.approx(0.7)

    def test_pending_review_count(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)

        # Two items below 0.8 threshold — pending
        _make_fact_item(
            db_manager,
            fact_id,
            enrichment_source="llm",
            enrichment_confidence=0.5,
        )
        _make_fact_item(
            db_manager,
            fact_id,
            enrichment_source="llm",
            enrichment_confidence=0.3,
        )
        # One item above threshold — not pending
        _make_fact_item(
            db_manager,
            fact_id,
            enrichment_source="open_food_facts",
            enrichment_confidence=0.95,
        )

        stats = svc.get_review_stats(db_manager)
        assert stats["pending_review"] == 2

    def test_empty_db_returns_zeros(self, db_manager):
        stats = svc.get_review_stats(db_manager)

        assert stats["total_enriched"] == 0
        assert stats["pending_review"] == 0
        assert stats["by_source"] == []

    def test_user_confirmed_counted_separately(self, db_manager):
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)

        item_id = _make_fact_item(
            db_manager,
            fact_id,
            enrichment_source="open_food_facts",
            enrichment_confidence=0.5,
        )
        svc.confirm_enrichment(db_manager, item_id)

        stats = svc.get_review_stats(db_manager)

        by_source = {
            row["enrichment_source"]: row["count"] for row in stats["by_source"]
        }
        assert by_source.get("user_confirmed") == 1
        # user_confirmed has confidence=1.0, so not pending
        assert stats["pending_review"] == 0
