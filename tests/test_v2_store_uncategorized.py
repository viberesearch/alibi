"""Tests for v2_store.list_fact_items_uncategorized()."""

from __future__ import annotations

import os
from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.db import v2_store
from alibi.db.connection import DatabaseManager
from alibi.db.models import (
    Cloud,
    CloudBundle,
    CloudMatchType,
    CloudStatus,
    Document,
    Fact,
    FactStatus,
    FactType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_document(db: DatabaseManager) -> Document:
    doc = Document(
        id=str(uuid4()),
        file_path="/test/receipt.jpg",
        file_hash=str(uuid4()),
    )
    v2_store.store_document(db, doc)
    return doc


def _make_bundle(db: DatabaseManager, doc: Document) -> str:
    from alibi.db.models import Bundle, BundleType

    bundle = Bundle(
        id=str(uuid4()),
        document_id=doc.id,
        bundle_type=BundleType.BASKET,
    )
    v2_store.store_bundle(db, bundle, [])
    return bundle.id


def _make_fact(db: DatabaseManager) -> tuple[Fact, str]:
    """Create a minimal fact; returns (fact, doc_id)."""
    cloud = Cloud(id=str(uuid4()), status=CloudStatus.COLLAPSED)
    doc = _make_document(db)
    bundle_id = _make_bundle(db, doc)
    cb = CloudBundle(
        cloud_id=cloud.id,
        bundle_id=bundle_id,
        match_type=CloudMatchType.MANUAL,
        match_confidence=Decimal("1.0"),
    )
    v2_store.store_cloud(db, cloud, cb)

    fact = Fact(
        id=str(uuid4()),
        cloud_id=cloud.id,
        fact_type=FactType.PURCHASE,
        vendor="Test Store",
        vendor_key=None,
        total_amount=Decimal("25.00"),
        currency="EUR",
        event_date=date(2025, 1, 15),
        status=FactStatus.CONFIRMED,
    )
    v2_store.store_fact(db, fact, [])
    return fact, doc.id


def _add_fact_item(
    db: DatabaseManager,
    fact_id: str,
    name: str,
    category: str | None,
    doc_id: str = "doc-test",
) -> str:
    """Insert an atom + fact_item row; returns the new item id."""
    item_id = str(uuid4())
    atom_id = str(uuid4())
    conn = db.get_connection()
    # fact_items.atom_id FK references atoms.id
    conn.execute(
        "INSERT INTO atoms (id, document_id, atom_type, data) VALUES (?, ?, ?, ?)",
        (atom_id, doc_id, "item", "{}"),
    )
    conn.execute(
        """INSERT INTO fact_items
           (id, fact_id, atom_id, name, quantity, unit_price, total_price, category)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            item_id,
            fact_id,
            atom_id,
            name,
            "1",
            "5.00",
            "5.00",
            category,
        ),
    )
    conn.commit()
    return item_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestListFactItemsUncategorized:
    def test_returns_items_without_category(self, db: DatabaseManager) -> None:
        fact, doc_id = _make_fact(db)
        item_no_cat_id = _add_fact_item(db, fact.id, "bread", None, doc_id=doc_id)
        _add_fact_item(db, fact.id, "milk", "dairy", doc_id=doc_id)

        result = v2_store.list_fact_items_uncategorized(db, limit=100)

        ids = [r["id"] for r in result]
        assert item_no_cat_id in ids
        # The categorized item should NOT appear
        assert not any(r.get("category") for r in result)

    def test_returns_empty_when_all_categorized(self, db: DatabaseManager) -> None:
        fact, doc_id = _make_fact(db)
        _add_fact_item(db, fact.id, "bread", "bakery", doc_id=doc_id)
        _add_fact_item(db, fact.id, "milk", "dairy", doc_id=doc_id)

        result = v2_store.list_fact_items_uncategorized(db, limit=100)

        assert result == []

    def test_returns_empty_when_no_items(self, db: DatabaseManager) -> None:
        result = v2_store.list_fact_items_uncategorized(db, limit=100)
        assert result == []

    def test_respects_limit_parameter(self, db: DatabaseManager) -> None:
        fact, doc_id = _make_fact(db)
        for i in range(10):
            _add_fact_item(db, fact.id, f"item{i}", None, doc_id=doc_id)

        result = v2_store.list_fact_items_uncategorized(db, limit=3)

        assert len(result) == 3

    def test_empty_string_category_also_returned(self, db: DatabaseManager) -> None:
        """Items with category='' (empty string) count as uncategorized."""
        fact, doc_id = _make_fact(db)
        item_empty_id = _add_fact_item(db, fact.id, "unknown item", "", doc_id=doc_id)

        result = v2_store.list_fact_items_uncategorized(db, limit=100)

        ids = [r["id"] for r in result]
        assert item_empty_id in ids

    def test_result_contains_expected_fields(self, db: DatabaseManager) -> None:
        fact, doc_id = _make_fact(db)
        item_id = _add_fact_item(db, fact.id, "bread", None, doc_id=doc_id)

        result = v2_store.list_fact_items_uncategorized(db, limit=100)

        assert len(result) == 1
        row = result[0]
        assert "id" in row
        assert "fact_id" in row
        assert "name" in row
        assert row["id"] == item_id
        assert row["fact_id"] == fact.id
        assert row["name"] == "bread"

    def test_default_limit_is_100(self, db: DatabaseManager) -> None:
        """Calling without limit argument uses default of 100."""
        fact, doc_id = _make_fact(db)
        for i in range(5):
            _add_fact_item(db, fact.id, f"item{i}", None, doc_id=doc_id)

        result = v2_store.list_fact_items_uncategorized(db)

        assert len(result) == 5

    def test_mixed_none_and_empty_both_returned(self, db: DatabaseManager) -> None:
        fact, doc_id = _make_fact(db)
        item_none_id = _add_fact_item(db, fact.id, "item_null", None, doc_id=doc_id)
        item_empty_id = _add_fact_item(db, fact.id, "item_empty", "", doc_id=doc_id)

        result = v2_store.list_fact_items_uncategorized(db, limit=100)

        ids = {r["id"] for r in result}
        assert item_none_id in ids
        assert item_empty_id in ids
