"""Tests for alibi.services.snapshot — out-of-band edit detection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from alibi.db.connection import DatabaseManager
from alibi.services import snapshot as snap_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_fact_items(db: DatabaseManager, items: list[dict]) -> None:
    """Insert minimal document + atom + cloud + fact + fact_items for testing."""
    conn = db.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) "
        "VALUES ('doc-1', '/tmp/test.jpg', 'hash-1')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO atoms (id, document_id, atom_type, data) "
        "VALUES ('atom-1', 'doc-1', 'item', '{}')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO clouds (id, status) VALUES ('cloud-1', 'collapsed')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO facts "
        "(id, cloud_id, fact_type, vendor, total_amount, currency, event_date) "
        "VALUES ('fact-1', 'cloud-1', 'purchase', 'TestShop', 10.0, 'EUR', '2025-01-01')"
    )
    for item in items:
        cols = ["id", "fact_id", "atom_id", "name", "quantity", "total_price"]
        vals = [
            item["id"],
            item.get("fact_id", "fact-1"),
            item.get("atom_id", "atom-1"),
            item.get("name", "Item"),
            item.get("quantity", 1),
            item.get("total_price", 5.0),
        ]
        extras = [
            "name_normalized",
            "comparable_name",
            "brand",
            "category",
            "barcode",
            "unit",
            "unit_quantity",
            "enrichment_source",
            "enrichment_confidence",
        ]
        for col in extras:
            if col in item:
                cols.append(col)
                vals.append(item[col])

        placeholders = ", ".join("?" * len(cols))
        col_str = ", ".join(cols)
        conn.execute(
            f"INSERT INTO fact_items ({col_str}) VALUES ({placeholders})",
            tuple(vals),
        )
    conn.commit()


@pytest.fixture(autouse=True)
def _isolate_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect snapshot file to tmp_path so tests don't touch real ~/.alibi."""
    snap_dir = tmp_path / ".alibi"
    snap_file = snap_dir / "corrections_snapshot.json"
    monkeypatch.setattr(snap_mod, "SNAPSHOT_DIR", snap_dir)
    monkeypatch.setattr(snap_mod, "SNAPSHOT_FILE", snap_file)


# ---------------------------------------------------------------------------
# take_snapshot
# ---------------------------------------------------------------------------


class TestTakeSnapshot:
    def test_creates_snapshot_file(self, db: DatabaseManager):
        _seed_fact_items(db, [{"id": "item-1", "name": "Milk", "brand": "Alpro"}])

        count = snap_mod.take_snapshot(db)
        assert count == 1
        assert snap_mod.SNAPSHOT_FILE.exists()

        data = json.loads(snap_mod.SNAPSHOT_FILE.read_text())
        assert "item-1" in data
        assert data["item-1"]["brand"] == "Alpro"

    def test_raises_if_snapshot_exists(self, db: DatabaseManager):
        _seed_fact_items(db, [{"id": "item-1"}])
        snap_mod.take_snapshot(db)

        with pytest.raises(FileExistsError):
            snap_mod.take_snapshot(db)

    def test_empty_db(self, db: DatabaseManager):
        count = snap_mod.take_snapshot(db)
        assert count == 0


# ---------------------------------------------------------------------------
# load_snapshot / delete_snapshot
# ---------------------------------------------------------------------------


class TestSnapshotCRUD:
    def test_load_returns_none_when_missing(self):
        assert snap_mod.load_snapshot() is None

    def test_load_returns_data(self, db: DatabaseManager):
        _seed_fact_items(db, [{"id": "item-1", "name": "Bread"}])
        snap_mod.take_snapshot(db)

        data = snap_mod.load_snapshot()
        assert data is not None
        assert "item-1" in data

    def test_delete_returns_false_when_missing(self):
        assert snap_mod.delete_snapshot() is False

    def test_delete_removes_file(self, db: DatabaseManager):
        _seed_fact_items(db, [{"id": "item-1"}])
        snap_mod.take_snapshot(db)
        assert snap_mod.SNAPSHOT_FILE.exists()

        assert snap_mod.delete_snapshot() is True
        assert not snap_mod.SNAPSHOT_FILE.exists()


# ---------------------------------------------------------------------------
# detect_changes
# ---------------------------------------------------------------------------


class TestDetectChanges:
    def test_no_changes(self, db: DatabaseManager):
        _seed_fact_items(db, [{"id": "item-1", "name": "Milk", "brand": "Alpro"}])
        snap_mod.take_snapshot(db)

        snapshot = snap_mod.load_snapshot()
        changes = snap_mod.detect_changes(db, snapshot)
        assert changes == []

    def test_detects_field_change(self, db: DatabaseManager):
        _seed_fact_items(db, [{"id": "item-1", "name": "Milk", "brand": "Alpro"}])
        snap_mod.take_snapshot(db)

        # Simulate TablePlus edit
        with db.transaction() as conn:
            conn.execute(
                "UPDATE fact_items SET brand = ? WHERE id = ?",
                ("Oatly", "item-1"),
            )

        snapshot = snap_mod.load_snapshot()
        changes = snap_mod.detect_changes(db, snapshot)
        assert len(changes) == 1
        assert changes[0]["field"] == "brand"
        assert changes[0]["old_value"] == "Alpro"
        assert changes[0]["new_value"] == "Oatly"

    def test_detects_multiple_fields(self, db: DatabaseManager):
        _seed_fact_items(
            db,
            [{"id": "item-1", "name": "Milk", "brand": "Alpro", "category": "Dairy"}],
        )
        snap_mod.take_snapshot(db)

        with db.transaction() as conn:
            conn.execute(
                "UPDATE fact_items SET brand = ?, category = ? WHERE id = ?",
                ("Oatly", "Beverages", "item-1"),
            )

        snapshot = snap_mod.load_snapshot()
        changes = snap_mod.detect_changes(db, snapshot)
        fields = {c["field"] for c in changes}
        assert fields == {"brand", "category"}

    def test_none_vs_empty_treated_equal(self, db: DatabaseManager):
        """None and '' should not be flagged as a change."""
        _seed_fact_items(db, [{"id": "item-1", "name": "Milk"}])
        snap_mod.take_snapshot(db)

        # brand is NULL in DB; snapshot stores it as None
        # Simulate setting brand to empty string
        with db.transaction() as conn:
            conn.execute(
                "UPDATE fact_items SET brand = '' WHERE id = ?",
                ("item-1",),
            )

        snapshot = snap_mod.load_snapshot()
        changes = snap_mod.detect_changes(db, snapshot)
        assert changes == []

    def test_ignores_deleted_items(self, db: DatabaseManager):
        _seed_fact_items(db, [{"id": "item-1", "name": "Milk"}])
        snap_mod.take_snapshot(db)

        with db.transaction() as conn:
            conn.execute("DELETE FROM fact_items WHERE id = ?", ("item-1",))

        snapshot = snap_mod.load_snapshot()
        changes = snap_mod.detect_changes(db, snapshot)
        assert changes == []

    def test_multiple_items(self, db: DatabaseManager):
        _seed_fact_items(
            db,
            [
                {"id": "item-1", "name": "Milk", "brand": "Alpro"},
                {"id": "item-2", "name": "Bread", "brand": "Artisan"},
            ],
        )
        snap_mod.take_snapshot(db)

        with db.transaction() as conn:
            conn.execute(
                "UPDATE fact_items SET brand = ? WHERE id = ?",
                ("Oatly", "item-1"),
            )
            # item-2 unchanged

        snapshot = snap_mod.load_snapshot()
        changes = snap_mod.detect_changes(db, snapshot)
        assert len(changes) == 1
        assert changes[0]["item_id"] == "item-1"


# ---------------------------------------------------------------------------
# apply_changes
# ---------------------------------------------------------------------------


class TestApplyChanges:
    def test_records_correction_events(self, db: DatabaseManager):
        _seed_fact_items(db, [{"id": "item-1", "name": "Milk", "brand": "Alpro"}])
        snap_mod.take_snapshot(db)

        with db.transaction() as conn:
            conn.execute(
                "UPDATE fact_items SET brand = ? WHERE id = ?",
                ("Oatly", "item-1"),
            )

        snapshot = snap_mod.load_snapshot()
        changes = snap_mod.detect_changes(db, snapshot)
        recorded = snap_mod.apply_changes(db, changes, source="tableplus")
        assert recorded == 1

        # Verify correction_events table
        events = db.fetchall(
            "SELECT * FROM correction_events WHERE entity_id = ?",
            ("item-1",),
        )
        assert len(events) >= 1
        evt = events[0]
        assert evt["entity_type"] == "fact_item"
        assert evt["field"] == "brand"
        assert evt["source"] == "tableplus"

    def test_stamps_user_confirmed(self, db: DatabaseManager):
        _seed_fact_items(
            db,
            [
                {
                    "id": "item-1",
                    "name": "Milk",
                    "brand": "Alpro",
                    "enrichment_source": "openfoodfacts",
                }
            ],
        )
        snap_mod.take_snapshot(db)

        with db.transaction() as conn:
            conn.execute(
                "UPDATE fact_items SET brand = ? WHERE id = ?",
                ("Oatly", "item-1"),
            )

        snapshot = snap_mod.load_snapshot()
        changes = snap_mod.detect_changes(db, snapshot)
        snap_mod.apply_changes(db, changes)

        row = db.fetchone("SELECT * FROM fact_items WHERE id = ?", ("item-1",))
        assert row["enrichment_source"] == "user_confirmed"
        assert row["enrichment_confidence"] == 1.0

    def test_sibling_propagation(self, db: DatabaseManager):
        _seed_fact_items(
            db,
            [
                {
                    "id": "item-1",
                    "name": "Milk 1L",
                    "name_normalized": "milk",
                    "brand": "Alpro",
                },
                {
                    "id": "item-2",
                    "name": "Milk 500ml",
                    "name_normalized": "milk",
                    "brand": None,
                },
            ],
        )
        snap_mod.take_snapshot(db)

        with db.transaction() as conn:
            conn.execute(
                "UPDATE fact_items SET brand = ? WHERE id = ?",
                ("Oatly", "item-1"),
            )

        snapshot = snap_mod.load_snapshot()
        changes = snap_mod.detect_changes(db, snapshot)
        snap_mod.apply_changes(db, changes)

        # Sibling should get the brand propagated
        sibling = db.fetchone("SELECT * FROM fact_items WHERE id = ?", ("item-2",))
        assert sibling["brand"] == "Oatly"
        assert sibling["enrichment_source"] == "sibling_propagation"

    def test_apply_returns_count(self, db: DatabaseManager):
        _seed_fact_items(
            db,
            [{"id": "item-1", "name": "Milk", "brand": "A", "category": "Dairy"}],
        )
        snap_mod.take_snapshot(db)

        with db.transaction() as conn:
            conn.execute(
                "UPDATE fact_items SET brand = ?, category = ? WHERE id = ?",
                ("B", "Beverages", "item-1"),
            )

        snapshot = snap_mod.load_snapshot()
        changes = snap_mod.detect_changes(db, snapshot)
        recorded = snap_mod.apply_changes(db, changes)
        assert recorded == 2  # brand + category

    def test_empty_changes(self, db: DatabaseManager):
        recorded = snap_mod.apply_changes(db, [])
        assert recorded == 0
