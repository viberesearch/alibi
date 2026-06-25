"""Tests for the deterministic comparable-price recompute pass."""

from __future__ import annotations

import os
from decimal import Decimal

os.environ["ALIBI_TESTING"] = "1"

from alibi.enrichment.comparable_prices import (
    recompute_for_row,
    recompute_pending_comparable_prices,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_fact_item(db, item_id: str, name: str, **cols) -> None:
    """Insert a fact_item (with chain) carrying the given column values."""
    doc_id = f"doc-{item_id}"
    atom_id = f"atom-{item_id}"
    cloud_id = f"cloud-{item_id}"
    fact_id = f"fact-{item_id}"
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
        "(id, cloud_id, fact_type, vendor, total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', 'Store', 10.0, 'EUR', '2026-01-01')",
        (fact_id, cloud_id),
    )
    keys = ["id", "fact_id", "atom_id", "name", "quantity"]
    vals = [item_id, fact_id, atom_id, name, cols.pop("quantity", 1)]
    for k, v in cols.items():
        keys.append(k)
        vals.append(v)
    placeholders = ", ".join("?" for _ in keys)
    conn.execute(
        f"INSERT OR IGNORE INTO fact_items ({', '.join(keys)}) "  # noqa: S608
        f"VALUES ({placeholders})",
        tuple(vals),
    )
    conn.commit()


def _get(db, item_id):
    return db.fetchone(
        "SELECT unit, unit_quantity, comparable_unit, comparable_unit_price "
        "FROM fact_items WHERE id = ?",
        (item_id,),
    )


# ===========================================================================
# TestRecomputeForRow (pure)
# ===========================================================================


class TestRecomputeForRow:
    def test_recovers_volume_from_name(self):
        # 2L olive oil for 9.00 -> stored as pcs; recover -> 4.50 EUR/L.
        changes = recompute_for_row(
            {
                "id": "x",
                "name": "CALABRIA OLIVE OIL 2L",
                "quantity": 1,
                "unit": "pcs",
                "unit_quantity": None,
                "total_price": "9.00",
                "comparable_unit": "pcs",
                "comparable_unit_price": "9.00",
            }
        )
        assert changes is not None
        assert changes["unit"] == "l"
        assert changes["unit_quantity"] == "2"
        assert changes["comparable_unit"] == "l"
        assert changes["comparable_unit_price"] == "4.50"

    def test_recovers_weight_from_name(self):
        # 450g pasta for 1.80 -> 4.00 EUR/kg.
        changes = recompute_for_row(
            {
                "id": "x",
                "name": "PASTA 450G",
                "quantity": 1,
                "unit": "pcs",
                "unit_quantity": None,
                "total_price": "1.80",
                "comparable_unit": "pcs",
                "comparable_unit_price": "1.80",
            }
        )
        assert changes is not None
        assert changes["comparable_unit"] == "kg"
        assert changes["comparable_unit_price"] == "4.00"

    def test_count_item_without_size_left_alone(self):
        # No parseable size, already pcs with matching price -> no change.
        changes = recompute_for_row(
            {
                "id": "x",
                "name": "TOMATOES",
                "quantity": 1,
                "unit": "pcs",
                "unit_quantity": None,
                "total_price": "2.00",
                "comparable_unit": "pcs",
                "comparable_unit_price": "2.00",
            }
        )
        assert changes is None

    def test_sized_unit_not_overridden(self):
        # Already kg with a quantity: do not let a stray name token clobber it.
        changes = recompute_for_row(
            {
                "id": "x",
                "name": "TVOROG 430G",
                "quantity": 1,
                "unit": "kg",
                "unit_quantity": "0.43",
                "total_price": "11.00",
                "comparable_unit": "kg",
                "comparable_unit_price": "25.58",
            }
        )
        # comparable_unit already kg and correct -> no-op.
        assert changes is None

    def test_implausible_price_skipped(self):
        # Corrupt total_price must not be written.
        changes = recompute_for_row(
            {
                "id": "x",
                "name": "WATER 500ML",
                "quantity": 1,
                "unit": "pcs",
                "unit_quantity": None,
                "total_price": "9911.96",
                "comparable_unit": "pcs",
                "comparable_unit_price": "9911.96",
            }
        )
        assert changes is None

    def test_price_above_ceiling_skipped(self):
        # A normalised price above _MAX_PLAUSIBLE_UNIT_PRICE (200) is rejected:
        # 50 EUR for a 100 ml bottle -> 500 EUR/L, which we refuse to write.
        changes = recompute_for_row(
            {
                "id": "x",
                "name": "MYSTERY OIL 100ML",
                "quantity": 1,
                "unit": None,
                "unit_quantity": None,
                "total_price": "50.00",
                "comparable_unit": None,
                "comparable_unit_price": None,
            }
        )
        assert changes is None

    def test_price_just_below_ceiling_written(self):
        # 15 EUR for a 100 ml bottle -> 150 EUR/L, under the 200 ceiling, so it
        # is recovered from the name and written.
        changes = recompute_for_row(
            {
                "id": "x",
                "name": "PREMIUM OIL 100ML",
                "quantity": 1,
                "unit": "pcs",
                "unit_quantity": None,
                "total_price": "15.00",
                "comparable_unit": "pcs",
                "comparable_unit_price": "15.00",
            }
        )
        assert changes is not None
        assert changes["comparable_unit"] == "l"
        assert Decimal(changes["comparable_unit_price"]) == Decimal("150")

    def test_no_total_price_skipped(self):
        assert (
            recompute_for_row(
                {"id": "x", "name": "MILK 1L", "quantity": 1, "total_price": None}
            )
            is None
        )


# ===========================================================================
# TestRecomputePending (DB)
# ===========================================================================


class TestRecomputePending:
    def test_recovers_and_writes(self, db):
        _seed_fact_item(
            db,
            "oil",
            "CALABRIA OLIVE OIL 2L",
            unit="pcs",
            total_price=9.00,
            comparable_unit="pcs",
            comparable_unit_price=9.00,
        )
        results = recompute_pending_comparable_prices(db, limit=50)
        assert any(r.changed for r in results)
        row = _get(db, "oil")
        assert row["comparable_unit"] == "l"
        assert float(row["comparable_unit_price"]) == 4.50
        assert row["unit"] == "l"
        assert float(row["unit_quantity"]) == 2.0

    def test_idempotent_rerun_is_noop(self, db):
        _seed_fact_item(
            db,
            "oil",
            "CALABRIA OLIVE OIL 2L",
            unit="pcs",
            total_price=9.00,
            comparable_unit="pcs",
            comparable_unit_price=9.00,
        )
        first = recompute_pending_comparable_prices(db, limit=50)
        assert sum(1 for r in first if r.changed) == 1
        second = recompute_pending_comparable_prices(db, limit=50)
        assert sum(1 for r in second if r.changed) == 0

    def test_count_item_unchanged(self, db):
        _seed_fact_item(
            db,
            "tom",
            "TOMATOES",
            unit="pcs",
            total_price=2.00,
            comparable_unit="pcs",
            comparable_unit_price=2.00,
        )
        results = recompute_pending_comparable_prices(db, limit=50)
        assert all(not r.changed for r in results)
        row = _get(db, "tom")
        assert row["comparable_unit"] == "pcs"
