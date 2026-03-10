"""Tests for get_canonical_unit_quantity() and ProductMatch unit_quantity/unit fields."""

from __future__ import annotations

import os
import tempfile

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.config import Config, reset_config
from alibi.db.connection import DatabaseManager
from alibi.db import v2_store
from alibi.enrichment.product_resolver import ProductMatch, resolve_product


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
    item_name: str = "Olive Oil",
    barcode: str | None = None,
    brand: str | None = None,
    category: str | None = None,
    unit_quantity: float | None = None,
    unit: str = "piece",
) -> None:
    """Insert a minimal document -> atom -> cloud -> fact -> fact_item chain."""
    conn = db.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
        (doc_id, f"/tmp/{doc_id}.jpg", f"hash-{doc_id}"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO atoms (id, document_id, atom_type, data) VALUES (?, ?, 'item', '{}')",
        (atom_id, doc_id),
    )
    conn.execute(
        "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')",
        (cloud_id,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO facts "
        "(id, cloud_id, fact_type, vendor, vendor_key, total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', 'TestShop', ?, 10.0, 'EUR', '2026-01-15')",
        (fact_id, cloud_id, vendor_key),
    )
    conn.execute(
        "INSERT OR IGNORE INTO fact_items "
        "(id, fact_id, atom_id, name, barcode, brand, category, quantity, unit, unit_quantity) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
        (
            item_id,
            fact_id,
            atom_id,
            item_name,
            barcode,
            brand,
            category,
            unit,
            unit_quantity,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Tests for get_canonical_unit_quantity
# ---------------------------------------------------------------------------


class TestGetCanonicalUnitQuantity:
    def test_level1_same_vendor_name_match(self, db: DatabaseManager) -> None:
        """Level 1: same item name + same vendor_key returns unit_quantity at 0.95."""
        _seed_item(
            db,
            doc_id="d1",
            atom_id="a1",
            cloud_id="c1",
            fact_id="f1",
            item_id="i1",
            vendor_key="CY12345678X",
            item_name="Olive Oil",
            unit_quantity=0.75,
            unit="l",
        )

        result = v2_store.get_canonical_unit_quantity(
            db, item_name="Olive Oil", vendor_key="CY12345678X"
        )

        assert result is not None
        assert result["unit_quantity"] == 0.75
        assert result["unit"] == "l"
        assert result["confidence"] == 0.95
        assert result["source"] == "vendor_history"

    def test_level1_same_vendor_barcode_match(self, db: DatabaseManager) -> None:
        """Level 1: barcode + same vendor_key returns unit_quantity at 0.95."""
        _seed_item(
            db,
            doc_id="d1",
            atom_id="a1",
            cloud_id="c1",
            fact_id="f1",
            item_id="i1",
            vendor_key="CY12345678X",
            item_name="Sparkling Water",
            barcode="5449000000996",
            unit_quantity=1.5,
            unit="l",
        )

        result = v2_store.get_canonical_unit_quantity(
            db,
            item_name="Sparkling Water",
            barcode="5449000000996",
            vendor_key="CY12345678X",
        )

        assert result is not None
        assert result["unit_quantity"] == 1.5
        assert result["confidence"] == 0.95
        assert result["source"] == "vendor_history"

    def test_level2_same_brand_different_vendor(self, db: DatabaseManager) -> None:
        """Level 2: same brand, different vendor returns unit_quantity at 0.85."""
        _seed_item(
            db,
            doc_id="d1",
            atom_id="a1",
            cloud_id="c1",
            fact_id="f1",
            item_id="i1",
            vendor_key="CY11111111A",
            item_name="Feta Cheese",
            brand="Pittas",
            unit_quantity=200.0,
            unit="g",
        )

        # Query with different vendor but same brand
        result = v2_store.get_canonical_unit_quantity(
            db,
            item_name="Feta Cheese",
            vendor_key="CY99999999Z",
            brand="Pittas",
        )

        assert result is not None
        assert result["unit_quantity"] == 200.0
        assert result["unit"] == "g"
        assert result["confidence"] == 0.85
        assert result["source"] == "brand_history"

    def test_level3_any_vendor_name_match(self, db: DatabaseManager) -> None:
        """Level 3: same name, no vendor/brand, returns unit_quantity at 0.75."""
        _seed_item(
            db,
            doc_id="d1",
            atom_id="a1",
            cloud_id="c1",
            fact_id="f1",
            item_id="i1",
            vendor_key="CY11111111A",
            item_name="Orange Juice",
            unit_quantity=1.0,
            unit="l",
        )

        # Query without vendor_key or brand
        result = v2_store.get_canonical_unit_quantity(
            db,
            item_name="Orange Juice",
        )

        assert result is not None
        assert result["unit_quantity"] == 1.0
        assert result["confidence"] == 0.75
        assert result["source"] == "name_history"

    def test_barcode_lookup_no_vendor(self, db: DatabaseManager) -> None:
        """Barcode match alone (level 3) returns result."""
        _seed_item(
            db,
            doc_id="d1",
            atom_id="a1",
            cloud_id="c1",
            fact_id="f1",
            item_id="i1",
            vendor_key="CY11111111A",
            item_name="Pasta",
            barcode="8001120919533",
            unit_quantity=500.0,
            unit="g",
        )

        result = v2_store.get_canonical_unit_quantity(
            db,
            item_name="Different Name But Same Barcode",
            barcode="8001120919533",
        )

        assert result is not None
        assert result["unit_quantity"] == 500.0

    def test_no_match_returns_none(self, db: DatabaseManager) -> None:
        """Item with no historical data returns None."""
        result = v2_store.get_canonical_unit_quantity(
            db,
            item_name="Nonexistent Product",
            vendor_key="CY12345678X",
        )

        assert result is None

    def test_null_unit_quantity_not_returned(self, db: DatabaseManager) -> None:
        """Items stored without unit_quantity are not returned."""
        _seed_item(
            db,
            doc_id="d1",
            atom_id="a1",
            cloud_id="c1",
            fact_id="f1",
            item_id="i1",
            vendor_key="CY12345678X",
            item_name="Bread",
            unit_quantity=None,
        )

        result = v2_store.get_canonical_unit_quantity(
            db,
            item_name="Bread",
            vendor_key="CY12345678X",
        )

        assert result is None

    def test_most_frequent_value_wins(self, db: DatabaseManager) -> None:
        """When multiple unit_quantity values exist, the most frequent wins."""
        # Two records with 0.5, one record with 1.0
        for i in range(2):
            _seed_item(
                db,
                doc_id=f"d{i}",
                atom_id=f"a{i}",
                cloud_id=f"c{i}",
                fact_id=f"f{i}",
                item_id=f"i{i}",
                vendor_key="CY12345678X",
                item_name="Yogurt",
                unit_quantity=0.5,
                unit="kg",
            )
        _seed_item(
            db,
            doc_id="d9",
            atom_id="a9",
            cloud_id="c9",
            fact_id="f9",
            item_id="i9",
            vendor_key="CY12345678X",
            item_name="Yogurt",
            unit_quantity=1.0,
            unit="kg",
        )

        result = v2_store.get_canonical_unit_quantity(
            db,
            item_name="Yogurt",
            vendor_key="CY12345678X",
        )

        assert result is not None
        assert result["unit_quantity"] == 0.5

    def test_level1_preferred_over_level3(self, db: DatabaseManager) -> None:
        """Level 1 result is returned even when level 3 data also exists."""
        # Level 3 data: different vendor
        _seed_item(
            db,
            doc_id="d1",
            atom_id="a1",
            cloud_id="c1",
            fact_id="f1",
            item_id="i1",
            vendor_key="CY11111111A",
            item_name="Halloumi",
            unit_quantity=250.0,
            unit="g",
        )
        # Level 1 data: same vendor, different quantity
        _seed_item(
            db,
            doc_id="d2",
            atom_id="a2",
            cloud_id="c2",
            fact_id="f2",
            item_id="i2",
            vendor_key="CY99999999Z",
            item_name="Halloumi",
            unit_quantity=200.0,
            unit="g",
        )

        result = v2_store.get_canonical_unit_quantity(
            db,
            item_name="Halloumi",
            vendor_key="CY99999999Z",
        )

        assert result is not None
        assert result["unit_quantity"] == 200.0
        assert result["source"] == "vendor_history"

    def test_case_insensitive_name_match(self, db: DatabaseManager) -> None:
        """Name matching is case-insensitive."""
        _seed_item(
            db,
            doc_id="d1",
            atom_id="a1",
            cloud_id="c1",
            fact_id="f1",
            item_id="i1",
            vendor_key="CY12345678X",
            item_name="OLIVE OIL",
            unit_quantity=0.75,
            unit="l",
        )

        result = v2_store.get_canonical_unit_quantity(
            db,
            item_name="olive oil",
            vendor_key="CY12345678X",
        )

        assert result is not None
        assert result["unit_quantity"] == 0.75

    def test_brand_none_skips_level2(self, db: DatabaseManager) -> None:
        """When brand is None, level 2 is skipped and falls through to level 3."""
        _seed_item(
            db,
            doc_id="d1",
            atom_id="a1",
            cloud_id="c1",
            fact_id="f1",
            item_id="i1",
            vendor_key="CY11111111A",
            item_name="Butter",
            brand="Lurpak",
            unit_quantity=250.0,
            unit="g",
        )

        # brand=None means level 2 skip; no level 1 match; level 3 matches by name
        result = v2_store.get_canonical_unit_quantity(
            db,
            item_name="Butter",
            vendor_key="CY99999999Z",
            brand=None,
        )

        assert result is not None
        assert result["confidence"] == 0.75
        assert result["source"] == "name_history"


# ---------------------------------------------------------------------------
# Tests for ProductMatch unit_quantity/unit fields
# ---------------------------------------------------------------------------


class TestProductMatchUnitQuantity:
    def test_product_match_has_unit_quantity_field(self) -> None:
        """ProductMatch dataclass includes unit_quantity and unit fields."""
        pm = ProductMatch(
            brand="Lurpak",
            category="Dairy",
            matched_name="Butter",
            similarity=1.0,
            source="exact_match",
            same_vendor=True,
            unit_quantity=250.0,
            unit="g",
        )
        assert pm.unit_quantity == 250.0
        assert pm.unit == "g"

    def test_product_match_defaults_none(self) -> None:
        """ProductMatch unit_quantity and unit default to None."""
        pm = ProductMatch(
            brand="Lurpak",
            category="Dairy",
            matched_name="Butter",
            similarity=1.0,
            source="exact_match",
        )
        assert pm.unit_quantity is None
        assert pm.unit is None

    def test_resolve_product_carries_unit_quantity(self, db: DatabaseManager) -> None:
        """resolve_product() returns unit_quantity from enriched candidates."""
        _seed_item(
            db,
            doc_id="d1",
            atom_id="a1",
            cloud_id="c1",
            fact_id="f1",
            item_id="i1",
            vendor_key="CY12345678X",
            item_name="Halloumi Cheese",
            brand="Pittas",
            category="Dairy",
            unit_quantity=200.0,
            unit="g",
        )

        match = resolve_product(db, "Halloumi Cheese", vendor_key="CY12345678X")

        assert match is not None
        assert match.unit_quantity == 200.0
        assert match.unit == "g"

    def test_resolve_product_unit_quantity_none_when_not_stored(
        self, db: DatabaseManager
    ) -> None:
        """resolve_product() returns None unit_quantity when not in DB."""
        _seed_item(
            db,
            doc_id="d1",
            atom_id="a1",
            cloud_id="c1",
            fact_id="f1",
            item_id="i1",
            vendor_key="CY12345678X",
            item_name="Plain Bread",
            brand="Local",
            category="Bakery",
            unit_quantity=None,
        )

        match = resolve_product(db, "Plain Bread", vendor_key="CY12345678X")

        assert match is not None
        assert match.unit_quantity is None
