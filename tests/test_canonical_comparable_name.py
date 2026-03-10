"""Tests for get_canonical_comparable_name() cascade."""

from __future__ import annotations

import os
import tempfile

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.config import Config, reset_config
from alibi.db.connection import DatabaseManager
from alibi.db import v2_store


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """Create a fresh database with full schema."""
    reset_config()
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    config = Config(db_path=db_path, _env_file=None)
    manager = DatabaseManager(config)
    if not manager.is_initialized():
        manager.initialize()
    yield manager
    manager.close()
    os.unlink(db_path)


def _seed_item(
    db: DatabaseManager,
    *,
    doc_id: str,
    atom_id: str,
    cloud_id: str,
    fact_id: str,
    item_id: str,
    vendor_key: str = "CY12345678X",
    item_name: str = "Γάλα Πλήρες",
    barcode: str | None = None,
    brand: str | None = None,
    comparable_name: str | None = None,
) -> None:
    """Insert a minimal document -> atom -> cloud -> fact -> fact_item chain."""
    conn = db.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
        (doc_id, f"/tmp/{doc_id}.jpg", f"hash-{doc_id}"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO atoms (id, document_id, atom_type, data) "
        "VALUES (?, ?, 'item', '{}')",
        (atom_id, doc_id),
    )
    conn.execute(
        "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')",
        (cloud_id,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO facts "
        "(id, cloud_id, fact_type, vendor, vendor_key, "
        "total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', 'TestShop', ?, 10.0, 'EUR', '2026-01-15')",
        (fact_id, cloud_id, vendor_key),
    )
    conn.execute(
        "INSERT OR IGNORE INTO fact_items "
        "(id, fact_id, atom_id, name, barcode, brand, comparable_name) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (item_id, fact_id, atom_id, item_name, barcode, brand, comparable_name),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Tests for get_canonical_comparable_name
# ---------------------------------------------------------------------------


class TestGetCanonicalComparableName:
    def test_level1_same_vendor_name_match(self, db: DatabaseManager) -> None:
        """Level 1: same item name + same vendor_key returns at 0.95."""
        _seed_item(
            db,
            doc_id="d1",
            atom_id="a1",
            cloud_id="c1",
            fact_id="f1",
            item_id="i1",
            vendor_key="CY12345678X",
            item_name="Γάλα Πλήρες",
            comparable_name="Full Fat Milk",
        )

        result = v2_store.get_canonical_comparable_name(
            db, item_name="Γάλα Πλήρες", vendor_key="CY12345678X"
        )

        assert result is not None
        assert result["comparable_name"] == "Full Fat Milk"
        assert result["confidence"] == 0.95
        assert result["source"] == "vendor_history"

    def test_level2_same_brand(self, db: DatabaseManager) -> None:
        """Level 2: same brand across vendors returns at 0.85."""
        _seed_item(
            db,
            doc_id="d2",
            atom_id="a2",
            cloud_id="c2",
            fact_id="f2",
            item_id="i2",
            vendor_key="OTHER_VENDOR",
            item_name="Γάλα Πλήρες",
            brand="ΔΕΛΤΑ",
            comparable_name="Full Fat Milk",
        )

        result = v2_store.get_canonical_comparable_name(
            db,
            item_name="Γάλα Πλήρες",
            vendor_key="DIFFERENT_VENDOR",
            brand="ΔΕΛΤΑ",
        )

        assert result is not None
        assert result["comparable_name"] == "Full Fat Milk"
        assert result["confidence"] == 0.85
        assert result["source"] == "brand_history"

    def test_level3_name_only(self, db: DatabaseManager) -> None:
        """Level 3: any vendor, name match only returns at 0.75."""
        _seed_item(
            db,
            doc_id="d3",
            atom_id="a3",
            cloud_id="c3",
            fact_id="f3",
            item_id="i3",
            vendor_key="SOME_VENDOR",
            item_name="Γάλα Πλήρες",
            comparable_name="Full Fat Milk",
        )

        result = v2_store.get_canonical_comparable_name(db, item_name="Γάλα Πλήρες")

        assert result is not None
        assert result["comparable_name"] == "Full Fat Milk"
        assert result["confidence"] == 0.75
        assert result["source"] == "name_history"

    def test_no_match_returns_none(self, db: DatabaseManager) -> None:
        """No historical data returns None."""
        result = v2_store.get_canonical_comparable_name(db, item_name="Unknown Product")
        assert result is None

    def test_barcode_match(self, db: DatabaseManager) -> None:
        """Barcode-based match works regardless of name."""
        _seed_item(
            db,
            doc_id="d4",
            atom_id="a4",
            cloud_id="c4",
            fact_id="f4",
            item_id="i4",
            vendor_key="CY99999999Z",
            item_name="Γάλα Πλήρες",
            barcode="5290004000123",
            comparable_name="Full Fat Milk",
        )

        result = v2_store.get_canonical_comparable_name(
            db,
            item_name="Different Name",
            barcode="5290004000123",
            vendor_key="CY99999999Z",
        )

        assert result is not None
        assert result["comparable_name"] == "Full Fat Milk"

    def test_null_comparable_name_ignored(self, db: DatabaseManager) -> None:
        """Items with NULL comparable_name are not returned."""
        _seed_item(
            db,
            doc_id="d5",
            atom_id="a5",
            cloud_id="c5",
            fact_id="f5",
            item_id="i5",
            vendor_key="CY12345678X",
            item_name="Γάλα Πλήρες",
            comparable_name=None,
        )

        result = v2_store.get_canonical_comparable_name(
            db, item_name="Γάλα Πλήρες", vendor_key="CY12345678X"
        )

        assert result is None

    def test_most_frequent_wins(self, db: DatabaseManager) -> None:
        """When multiple comparable_names exist, the most frequent wins."""
        for i in range(3):
            _seed_item(
                db,
                doc_id=f"d-freq-a{i}",
                atom_id=f"a-freq-a{i}",
                cloud_id=f"c-freq-a{i}",
                fact_id=f"f-freq-a{i}",
                item_id=f"i-freq-a{i}",
                vendor_key="CY12345678X",
                item_name="Γάλα Πλήρες",
                comparable_name="Full Fat Milk",
            )
        _seed_item(
            db,
            doc_id="d-freq-b",
            atom_id="a-freq-b",
            cloud_id="c-freq-b",
            fact_id="f-freq-b",
            item_id="i-freq-b",
            vendor_key="CY12345678X",
            item_name="Γάλα Πλήρες",
            comparable_name="Whole Milk",
        )

        result = v2_store.get_canonical_comparable_name(
            db, item_name="Γάλα Πλήρες", vendor_key="CY12345678X"
        )

        assert result is not None
        assert result["comparable_name"] == "Full Fat Milk"
