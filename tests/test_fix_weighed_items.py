"""Tests for fix_weighed_item_units data quality function."""

from __future__ import annotations

import json
import os
import uuid
from datetime import date
from decimal import Decimal

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.db import v2_store
from alibi.db.connection import DatabaseManager
from alibi.db.models import (
    CloudStatus,
    Fact,
    FactStatus,
    FactType,
    TaxType,
    UnitType,
)
from alibi.maintenance.learning_aggregation import fix_weighed_item_units


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cloud(db: DatabaseManager, cloud_id: str | None = None) -> str:
    cid = cloud_id or str(uuid.uuid4())
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
            (cid, CloudStatus.COLLAPSED.value, 1.0),
        )
    return cid


def _make_fact(
    db: DatabaseManager,
    cloud_id: str,
    vendor: str = "Test Vendor",
) -> str:
    fid = str(uuid.uuid4())
    fact = Fact(
        id=fid,
        cloud_id=cloud_id,
        fact_type=FactType.PURCHASE,
        vendor=vendor,
        total_amount=Decimal("10.00"),
        currency="EUR",
        event_date=date(2025, 1, 15),
        status=FactStatus.CONFIRMED,
    )
    v2_store.store_fact(db, fact, [])
    return fid


def _make_fact_item(
    db: DatabaseManager,
    fact_id: str,
    name: str = "Test Item",
    unit: str = "pcs",
    unit_quantity: float | None = None,
    unit_price: float | None = None,
    total_price: float | None = None,
) -> str:
    """Insert a fact_item with a stub document and atom to satisfy FK constraints."""
    item_id = str(uuid.uuid4())
    atom_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    with db.transaction() as cursor:
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
                unit,
                unit_price,
                total_price,
                None,
                None,
                None,
                None,
                None,
                unit_quantity,
                None,
                TaxType.NONE.value,
                None,
                None,
            ),
        )
    return item_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fix_pcs_to_kg(db: DatabaseManager) -> None:
    """Item with unit='pcs', fractional unit_quantity, high unit_price should have
    unit changed to 'kg' when unit_quantity * unit_price ≈ total_price."""
    cloud_id = _make_cloud(db)
    fact_id = _make_fact(db, cloud_id)
    item_id = _make_fact_item(
        db,
        fact_id,
        name="Chicken Breast",
        unit="pcs",
        unit_quantity=0.25,
        unit_price=20.0,
        total_price=5.0,
    )

    report = fix_weighed_item_units(db)

    assert report.units_fixed == 1
    assert report.unit_quantities_backfilled == 0

    conn = db.get_connection()
    row = conn.execute(
        "SELECT unit FROM fact_items WHERE id = ?", (item_id,)
    ).fetchone()
    assert row["unit"] == "kg"


def test_skip_regular_pcs(db: DatabaseManager) -> None:
    """Item with unit='pcs' and NULL unit_quantity (whole item) should NOT be changed."""
    cloud_id = _make_cloud(db)
    fact_id = _make_fact(db, cloud_id)
    item_id = _make_fact_item(
        db,
        fact_id,
        name="Bottled Water",
        unit="pcs",
        unit_quantity=None,
        unit_price=3.0,
        total_price=3.0,
    )

    report = fix_weighed_item_units(db)

    assert report.units_fixed == 0
    assert report.unit_quantities_backfilled == 0

    conn = db.get_connection()
    row = conn.execute(
        "SELECT unit FROM fact_items WHERE id = ?", (item_id,)
    ).fetchone()
    assert row["unit"] == "pcs"


def test_idempotent(db: DatabaseManager) -> None:
    """Running fix_weighed_item_units twice produces 0 changes on second run."""
    cloud_id = _make_cloud(db)
    fact_id = _make_fact(db, cloud_id)
    _make_fact_item(
        db,
        fact_id,
        name="Minced Meat",
        unit="pcs",
        unit_quantity=0.5,
        unit_price=12.0,
        total_price=6.0,
    )

    first_report = fix_weighed_item_units(db)
    assert first_report.units_fixed == 1

    second_report = fix_weighed_item_units(db)
    assert second_report.units_fixed == 0
    assert second_report.unit_quantities_backfilled == 0
    assert second_report.total == 0


def test_backfill_unit_quantity(db: DatabaseManager) -> None:
    """Item with unit='kg', NULL unit_quantity, unit_price != total_price gets
    unit_quantity backfilled as total_price / unit_price."""
    cloud_id = _make_cloud(db)
    fact_id = _make_fact(db, cloud_id)
    item_id = _make_fact_item(
        db,
        fact_id,
        name="Olive Oil",
        unit="kg",
        unit_quantity=None,
        unit_price=10.0,
        total_price=3.5,
    )

    report = fix_weighed_item_units(db)

    assert report.unit_quantities_backfilled == 1
    assert report.units_fixed == 0

    conn = db.get_connection()
    row = conn.execute(
        "SELECT unit_quantity FROM fact_items WHERE id = ?", (item_id,)
    ).fetchone()
    # 3.5 / 10.0 = 0.35, rounded to 3dp
    assert row["unit_quantity"] == pytest.approx(0.35, abs=0.001)


def test_fix_pcs_to_kg_fractional_above_one(db: DatabaseManager) -> None:
    """Item with unit='pcs' and fractional unit_quantity > 1 (e.g., 1.94kg chicken)
    should be fixed to unit='kg' when price math matches."""
    cloud_id = _make_cloud(db)
    fact_id = _make_fact(db, cloud_id)
    item_id = _make_fact_item(
        db,
        fact_id,
        name="Anna Chicken Whole Cut",
        unit="pcs",
        unit_quantity=1.94,
        unit_price=4.95,
        total_price=9.60,
    )

    report = fix_weighed_item_units(db)

    assert report.units_fixed == 1

    conn = db.get_connection()
    row = conn.execute(
        "SELECT unit FROM fact_items WHERE id = ?", (item_id,)
    ).fetchone()
    assert row["unit"] == "kg"


def test_skip_integer_uq(db: DatabaseManager) -> None:
    """Item with integer unit_quantity should NOT be converted (could be multi-buy)."""
    cloud_id = _make_cloud(db)
    fact_id = _make_fact(db, cloud_id)
    item_id = _make_fact_item(
        db,
        fact_id,
        name="Yoghurt 1kg",
        unit="pcs",
        unit_quantity=2.0,
        unit_price=5.25,
        total_price=10.50,
    )

    report = fix_weighed_item_units(db)

    assert report.units_fixed == 0

    conn = db.get_connection()
    row = conn.execute(
        "SELECT unit FROM fact_items WHERE id = ?", (item_id,)
    ).fetchone()
    assert row["unit"] == "pcs"


def test_skip_ambiguous_kg(db: DatabaseManager) -> None:
    """Item with unit='kg', NULL unit_quantity, and unit_price == total_price should
    NOT be backfilled (ambiguous: could be 1kg at that price)."""
    cloud_id = _make_cloud(db)
    fact_id = _make_fact(db, cloud_id)
    item_id = _make_fact_item(
        db,
        fact_id,
        name="Potatoes",
        unit="kg",
        unit_quantity=None,
        unit_price=5.0,
        total_price=5.0,
    )

    report = fix_weighed_item_units(db)

    assert report.unit_quantities_backfilled == 0
    assert report.units_fixed == 0

    conn = db.get_connection()
    row = conn.execute(
        "SELECT unit_quantity FROM fact_items WHERE id = ?", (item_id,)
    ).fetchone()
    assert row["unit_quantity"] is None
