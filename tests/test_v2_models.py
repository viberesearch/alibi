"""Tests for v2 models, enums, and migration system."""

import sqlite3
from collections.abc import Generator
from decimal import Decimal
from pathlib import Path

import pytest

from alibi.config import Config
from alibi.db.connection import DatabaseManager
from alibi.db.migrate import (
    _discover_migrations,
    auto_migrate,
    get_current_version,
    get_pending_migrations,
    migrate_down,
    migrate_up,
)
from alibi.db.models import (
    Artifact,
    DocumentStatus,
    DocumentType,
    DataType,
    DisplayType,
    FieldType,
    RecordType,
    TaxType,
    Tier,
    UnitType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Generator[Path, None, None]:
    path = tmp_path / "test_v2.db"
    yield path


@pytest.fixture
def db_manager(temp_db_path: Path) -> Generator[DatabaseManager, None, None]:
    config = Config(db_path=temp_db_path)
    manager = DatabaseManager(config)
    yield manager
    manager.close()


@pytest.fixture
def v1_conn(tmp_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Create a v1 database with all tables needed for migration testing."""
    db_path = tmp_path / "v1_test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT NOT NULL,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE spaces (id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT,
                             owner_id TEXT REFERENCES users(id),
                             created_at DATETIME DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE tags (id TEXT PRIMARY KEY, space_id TEXT, name TEXT NOT NULL,
                          path TEXT NOT NULL, type TEXT, color TEXT,
                          parent_id TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE artifacts (id TEXT PRIMARY KEY, space_id TEXT, type TEXT,
                               file_path TEXT NOT NULL, file_hash TEXT NOT NULL,
                               perceptual_hash TEXT, vendor TEXT, vendor_id TEXT,
                               document_id TEXT, document_date DATE,
                               amount DECIMAL(10,2), currency TEXT DEFAULT 'EUR',
                               raw_text TEXT, extracted_data JSON, status TEXT,
                               created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                               modified_at DATETIME,
                               created_by TEXT REFERENCES users(id));
        CREATE TABLE transactions (id TEXT PRIMARY KEY, space_id TEXT, type TEXT,
                                  vendor TEXT, description TEXT,
                                  amount DECIMAL NOT NULL,
                                  currency TEXT DEFAULT 'EUR',
                                  transaction_date DATE NOT NULL,
                                  payment_method TEXT, card_last4 TEXT,
                                  account_reference TEXT, status TEXT,
                                  note_path TEXT,
                                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                                  modified_at DATETIME, created_by TEXT);
        CREATE TABLE items (id TEXT PRIMARY KEY, space_id TEXT, name TEXT,
                           category TEXT, model TEXT, serial_number TEXT,
                           purchase_date DATE, purchase_price DECIMAL,
                           current_value DECIMAL, currency TEXT DEFAULT 'EUR',
                           status TEXT, warranty_expires DATE, warranty_type TEXT,
                           insurance_covered BOOLEAN, note_path TEXT,
                           created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                           modified_at DATETIME, created_by TEXT);
        CREATE TABLE line_items (id TEXT PRIMARY KEY,
                                artifact_id TEXT REFERENCES artifacts(id),
                                transaction_id TEXT REFERENCES transactions(id),
                                name TEXT NOT NULL,
                                quantity DECIMAL(10,3) DEFAULT 1,
                                unit_price DECIMAL(10,2),
                                total_price DECIMAL(10,2), category TEXT,
                                item_id TEXT REFERENCES items(id),
                                created_at DATETIME DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY,
                                     applied_at DATETIME DEFAULT CURRENT_TIMESTAMP);
        INSERT INTO schema_version (version) VALUES (1);
        """
    )
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestRecordType:
    def test_all_values(self) -> None:
        assert len(RecordType) == 15

    def test_payment(self) -> None:
        assert RecordType.PAYMENT.value == "payment"

    def test_purchase(self) -> None:
        assert RecordType.PURCHASE.value == "purchase"

    def test_from_string(self) -> None:
        assert RecordType("invoice") == RecordType.INVOICE

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError):
            RecordType("nonexistent")


class TestDataType:
    def test_values(self) -> None:
        assert DataType.ACTUAL.value == "actual"
        assert DataType.PROJECTED.value == "projected"
        assert DataType.TARGET.value == "target"

    def test_count(self) -> None:
        assert len(DataType) == 3


class TestUnitType:
    def test_all_values(self) -> None:
        assert len(UnitType) == 16

    def test_weight_units(self) -> None:
        assert UnitType.GRAM.value == "g"
        assert UnitType.KILOGRAM.value == "kg"
        assert UnitType.POUND.value == "lb"
        assert UnitType.OUNCE.value == "oz"

    def test_volume_units(self) -> None:
        assert UnitType.MILLILITER.value == "ml"
        assert UnitType.LITER.value == "l"
        assert UnitType.GALLON.value == "gal"

    def test_piece_default(self) -> None:
        assert UnitType.PIECE.value == "pcs"


class TestTaxType:
    def test_values(self) -> None:
        assert TaxType.VAT.value == "vat"
        assert TaxType.SALES_TAX.value == "sales_tax"
        assert TaxType.GST.value == "gst"
        assert TaxType.EXEMPT.value == "exempt"
        assert TaxType.INCLUDED.value == "included"
        assert TaxType.NONE.value == "none"


class TestTier:
    def test_values(self) -> None:
        assert Tier.T0.value == "0"
        assert Tier.T4.value == "4"

    def test_count(self) -> None:
        assert len(Tier) == 5


class TestDisplayType:
    def test_values(self) -> None:
        assert DisplayType.EXACT.value == "exact"
        assert DisplayType.MASKED.value == "masked"
        assert DisplayType.HIDDEN.value == "hidden"

    def test_count(self) -> None:
        assert len(DisplayType) == 5


class TestFieldType:
    def test_values(self) -> None:
        assert FieldType.CURRENCY.value == "currency"
        assert FieldType.WEIGHT.value == "weight"
        assert FieldType.VOLUME.value == "volume"
        assert FieldType.ENERGY.value == "energy"

    def test_count(self) -> None:
        assert len(FieldType) == 13


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestArtifactRecordType:
    def test_artifact_with_record_type(self) -> None:
        a = Artifact(
            id="art-1",
            space_id="sp-1",
            type=DocumentType.RECEIPT,
            record_type=RecordType.PURCHASE,
            file_path="inbox/receipt.jpg",
            file_hash="abc123",
        )
        assert a.record_type == RecordType.PURCHASE

    def test_artifact_without_record_type(self) -> None:
        a = Artifact(
            id="art-2",
            space_id="sp-1",
            type=DocumentType.RECEIPT,
            file_path="inbox/receipt.jpg",
            file_hash="abc123",
        )
        assert a.record_type is None


# ---------------------------------------------------------------------------
# Migration Tests
# ---------------------------------------------------------------------------


class TestMigrationDiscovery:
    def test_discover_finds_002(self) -> None:
        migrations = _discover_migrations()
        versions = [v for v, _, _, _ in migrations]
        assert 2 in versions

    def test_discover_has_down(self) -> None:
        migrations = _discover_migrations()
        m002 = [m for m in migrations if m[0] == 2][0]
        assert m002[3] is not None  # down_path exists


class TestMigrationUp:
    def test_get_version_v1(self, v1_conn: sqlite3.Connection) -> None:
        assert get_current_version(v1_conn) == 1

    def test_pending_from_v1(self, v1_conn: sqlite3.Connection) -> None:
        pending = get_pending_migrations(v1_conn)
        assert len(pending) >= 1
        assert pending[0][0] == 2

    def test_migrate_up(self, v1_conn: sqlite3.Connection) -> None:
        applied = migrate_up(v1_conn)
        assert applied == 34  # migrations 002-035 (34 files, no 001)
        assert get_current_version(v1_conn) == 35

    def test_migrate_up_idempotent(self, v1_conn: sqlite3.Connection) -> None:
        migrate_up(v1_conn)
        applied = migrate_up(v1_conn)
        assert applied == 0

    def test_v1_tables_dropped_after_migrate(self, v1_conn: sqlite3.Connection) -> None:
        migrate_up(v1_conn)
        cursor = v1_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        # v1 tables should be gone
        for dropped in [
            "artifacts",
            "transactions",
            "line_items",
            "transaction_artifacts",
            "transaction_tags",
            "item_artifacts",
            "item_transactions",
            "duplicate_log",
            "provenance",
        ]:
            assert dropped not in tables, f"Table '{dropped}' should be dropped"
        # v2 junction tables should exist
        assert "item_documents" in tables
        assert "item_facts" in tables

    def test_budgets_table_created(self, v1_conn: sqlite3.Connection) -> None:
        migrate_up(v1_conn)
        cursor = v1_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='budgets'"
        )
        assert cursor.fetchone() is not None

    def test_masking_snapshots_table_created(self, v1_conn: sqlite3.Connection) -> None:
        migrate_up(v1_conn)
        cursor = v1_conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='masking_snapshots'"
        )
        assert cursor.fetchone() is not None

    def test_migrate_up_with_target(self, v1_conn: sqlite3.Connection) -> None:
        applied = migrate_up(v1_conn, target=2)
        assert applied == 1
        assert get_current_version(v1_conn) == 2


class TestMigrationDown:
    def test_migrate_down(self, v1_conn: sqlite3.Connection) -> None:
        migrate_up(v1_conn)
        assert get_current_version(v1_conn) == 35
        reverted = migrate_down(v1_conn, target=1)
        assert reverted == 34  # reverts 035-002
        assert get_current_version(v1_conn) == 1

    def test_v1_tables_restored_after_down(self, v1_conn: sqlite3.Connection) -> None:
        migrate_up(v1_conn)
        migrate_down(v1_conn, target=1)
        cursor = v1_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        # v1 tables should be restored (empty)
        assert "line_items" in tables
        assert "artifacts" in tables
        assert "transactions" in tables

    def test_down_noop_if_at_target(self, v1_conn: sqlite3.Connection) -> None:
        reverted = migrate_down(v1_conn, target=1)
        assert reverted == 0

    def test_roundtrip(self, v1_conn: sqlite3.Connection) -> None:
        """Up then down then up again."""
        migrate_up(v1_conn)
        migrate_down(v1_conn, target=1)
        applied = migrate_up(v1_conn)
        assert applied == 34  # re-applies 002-035
        assert get_current_version(v1_conn) == 35


class TestAutoMigrate:
    def test_auto_migrate_on_v1(self, v1_conn: sqlite3.Connection) -> None:
        applied = auto_migrate(v1_conn)
        assert applied >= 1

    def test_auto_migrate_noop_on_current(self, v1_conn: sqlite3.Connection) -> None:
        auto_migrate(v1_conn)
        applied = auto_migrate(v1_conn)
        assert applied == 0


class TestFreshInstall:
    def test_fresh_install_schema_version(self, db_manager: DatabaseManager) -> None:
        db_manager.initialize()
        assert db_manager.get_schema_version() == 35

    def test_fresh_install_has_all_tables(self, db_manager: DatabaseManager) -> None:
        db_manager.initialize()
        rows = db_manager.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in rows]
        for t in [
            "budgets",
            "masking_snapshots",
            "item_documents",
            "item_facts",
            "user_contacts",
        ]:
            assert t in tables
        # v1 tables should not be present
        for t in ["artifacts", "transactions", "line_items", "provenance"]:
            assert t not in tables
        # tags/consumers tables should not be present
        for t in [
            "tags",
            "item_tags",
            "vendor_patterns",
            "consumers",
            "line_item_allocations",
        ]:
            assert t not in tables, f"Table '{t}' should not exist"

    def test_fresh_install_fact_items_columns(
        self, db_manager: DatabaseManager
    ) -> None:
        db_manager.initialize()
        cursor = db_manager.execute("PRAGMA table_info(fact_items)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "name" in columns
        assert "category" in columns
        assert "total_price" in columns


class TestDataIntegrity:
    def test_insert_v2_fact_item(self, db_manager: DatabaseManager) -> None:
        db_manager.initialize()
        conn = db_manager.get_connection()
        # Create required parent records
        conn.execute(
            "INSERT INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
            ("doc-test", "/test.jpg", "hash1"),
        )
        conn.execute(
            "INSERT INTO clouds (id, status) VALUES (?, 'collapsed')",
            ("cloud-test",),
        )
        conn.execute(
            "INSERT INTO facts (id, cloud_id, fact_type, vendor, total_amount, "
            "currency, event_date, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "fact-test",
                "cloud-test",
                "purchase",
                "Shop",
                "2.84",
                "EUR",
                "2026-01-15",
                "confirmed",
            ),
        )
        conn.execute(
            "INSERT INTO atoms (id, document_id, atom_type, data) "
            "VALUES (?, ?, ?, ?)",
            ("atom-test", "doc-test", "item", "{}"),
        )
        conn.execute(
            """INSERT INTO fact_items
               (id, fact_id, atom_id, name, quantity, unit,
                unit_price, total_price, category)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "fi-test-1",
                "fact-test",
                "atom-test",
                "Fresh Milk",
                "1.5",
                "l",
                "1.89",
                "2.84",
                "dairy",
            ),
        )
        conn.commit()
        row = db_manager.fetchone(
            "SELECT * FROM fact_items WHERE id = ?", ("fi-test-1",)
        )
        assert row is not None
        assert row["name"] == "Fresh Milk"
        assert row["category"] == "dairy"
