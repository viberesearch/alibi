"""Tests for database functionality."""

from collections.abc import Generator
from pathlib import Path

import pytest

from alibi.config import Config
from alibi.db.connection import DatabaseManager


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary database path (file does not exist yet)."""
    path = tmp_path / "test_alibi.db"
    yield path
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def db_manager(temp_db_path: Path) -> Generator[DatabaseManager, None, None]:
    """Create a database manager with temporary database."""
    config = Config(db_path=temp_db_path)
    manager = DatabaseManager(config)
    yield manager
    manager.close()


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    def test_initialize_creates_database(self, db_manager: DatabaseManager) -> None:
        """Test that initialize creates the database file."""
        assert not db_manager.db_path.exists()
        db_manager.initialize()
        assert db_manager.db_path.exists()

    def test_is_initialized_false_before_init(
        self, db_manager: DatabaseManager
    ) -> None:
        """Test that is_initialized returns False before initialization."""
        assert not db_manager.is_initialized()

    def test_is_initialized_true_after_init(self, db_manager: DatabaseManager) -> None:
        """Test that is_initialized returns True after initialization."""
        db_manager.initialize()
        assert db_manager.is_initialized()

    def test_schema_version_set_after_init(self, db_manager: DatabaseManager) -> None:
        """Test that schema version is set after initialization."""
        db_manager.initialize()
        version = db_manager.get_schema_version()
        assert version >= 10

    def test_tables_created(self, db_manager: DatabaseManager) -> None:
        """Test that all expected tables are created."""
        db_manager.initialize()

        expected_tables = [
            "users",
            "spaces",
            "space_members",
            "items",
            "item_documents",
            "item_facts",
            "schema_version",
            "budgets",
            "budget_entries",
            "masking_snapshots",
            # v2 Atom-Cloud-Fact tables
            "documents",
            "atoms",
            "bundles",
            "bundle_atoms",
            "clouds",
            "cloud_bundles",
            "facts",
            "fact_items",
            # Identity + annotations
            "identities",
            "identity_members",
            "annotations",
            # User contacts
            "user_contacts",
        ]

        # Get list of tables from database
        rows = db_manager.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in rows]

        for table in expected_tables:
            assert table in table_names, f"Table '{table}' not found"

    def test_foreign_keys_enabled(self, db_manager: DatabaseManager) -> None:
        """Test that foreign keys are enabled."""
        db_manager.initialize()
        row = db_manager.fetchone("PRAGMA foreign_keys")
        assert row is not None
        assert row[0] == 1

    def test_get_stats_returns_seeded_counts(self, db_manager: DatabaseManager) -> None:
        """Test that get_stats returns seeded counts after initialization."""
        db_manager.initialize()
        stats = db_manager.get_stats()

        assert stats["documents"] == 0
        assert stats["facts"] == 0
        assert stats["clouds"] == 0

    def test_transaction_context_manager_commits(
        self, db_manager: DatabaseManager
    ) -> None:
        """Test that transaction context manager commits on success."""
        db_manager.initialize()

        with db_manager.transaction() as cursor:
            cursor.execute(
                "INSERT INTO users (id, name) VALUES (?, ?)",
                ("test-user-1", "Test User"),
            )

        # Verify the insert persisted
        row = db_manager.fetchone(
            "SELECT name FROM users WHERE id = ?", ("test-user-1",)
        )
        assert row is not None
        assert row[0] == "Test User"

    def test_transaction_context_manager_rollback_on_error(
        self, db_manager: DatabaseManager
    ) -> None:
        """Test that transaction context manager rolls back on error."""
        db_manager.initialize()

        with pytest.raises(ValueError):
            with db_manager.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO users (id, name) VALUES (?, ?)",
                    ("test-user-2", "Test User 2"),
                )
                raise ValueError("Test error")

        # Verify the insert was rolled back
        row = db_manager.fetchone(
            "SELECT name FROM users WHERE id = ?", ("test-user-2",)
        )
        assert row is None

    def test_insert_and_query_user(self, db_manager: DatabaseManager) -> None:
        """Test inserting and querying a user."""
        db_manager.initialize()

        # Insert a user
        db_manager.execute(
            "INSERT INTO users (id, name) VALUES (?, ?)",
            ("user-123", "John Doe"),
        )
        db_manager.get_connection().commit()

        # Query the user
        row = db_manager.fetchone(
            "SELECT id, name FROM users WHERE id = ?", ("user-123",)
        )
        assert row is not None
        assert row["id"] == "user-123"
        assert row["name"] == "John Doe"

    def test_insert_space_with_foreign_key(self, db_manager: DatabaseManager) -> None:
        """Test inserting a space with foreign key to user."""
        db_manager.initialize()

        # Insert a user first
        db_manager.execute(
            "INSERT INTO users (id, name) VALUES (?, ?)",
            ("owner-1", "Owner"),
        )

        # Insert a space
        db_manager.execute(
            "INSERT INTO spaces (id, name, type, owner_id) VALUES (?, ?, ?, ?)",
            ("space-1", "My Space", "private", "owner-1"),
        )
        db_manager.get_connection().commit()

        # Query the space
        row = db_manager.fetchone("SELECT * FROM spaces WHERE id = ?", ("space-1",))
        assert row is not None
        assert row["name"] == "My Space"
        assert row["type"] == "private"
        assert row["owner_id"] == "owner-1"

    def test_foreign_key_constraint_enforced(self, db_manager: DatabaseManager) -> None:
        """Test that foreign key constraints are enforced."""
        db_manager.initialize()

        # Try to insert a space with non-existent owner
        with pytest.raises(Exception):  # sqlite3.IntegrityError
            db_manager.execute(
                "INSERT INTO spaces (id, name, type, owner_id) VALUES (?, ?, ?, ?)",
                ("space-bad", "Bad Space", "private", "non-existent-user"),
            )
            db_manager.get_connection().commit()


class TestConfig:
    """Tests for Config class."""

    def test_default_db_path(self) -> None:
        """Test default database path."""
        config = Config()
        assert config.db_path == Path("data/alibi.db")

    def test_default_currency(self) -> None:
        """Test default currency."""
        config = Config()
        assert config.default_currency == "EUR"

    def test_custom_db_path(self, temp_db_path: Path) -> None:
        """Test custom database path."""
        config = Config(db_path=temp_db_path)
        assert config.db_path == temp_db_path

    def test_get_absolute_db_path_relative(self) -> None:
        """Test get_absolute_db_path with relative path."""
        config = Config(db_path=Path("data/test.db"))
        abs_path = config.get_absolute_db_path()
        assert abs_path.is_absolute()
        assert abs_path.name == "test.db"

    def test_get_absolute_db_path_absolute(self, temp_db_path: Path) -> None:
        """Test get_absolute_db_path with absolute path."""
        config = Config(db_path=temp_db_path)
        abs_path = config.get_absolute_db_path()
        assert abs_path == temp_db_path

    def test_validate_paths_no_vault(self) -> None:
        """Test validate_paths with no vault set."""
        config = Config()
        errors = config.validate_paths()
        assert len(errors) == 0

    def test_validate_paths_invalid_vault(self) -> None:
        """Test validate_paths with invalid vault path."""
        config = Config(vault_path=Path("/nonexistent/vault/path"))
        errors = config.validate_paths()
        assert len(errors) == 1
        assert "does not exist" in errors[0]


class TestGetStats:
    """Regression tests for get_stats (audit fix)."""

    def test_get_stats_returns_v2_tables(self, db_manager: DatabaseManager) -> None:
        """get_stats() returns v2 table names: documents, facts, etc."""
        db_manager.initialize()
        stats = db_manager.get_stats()

        # v2 tables must be present
        assert "documents" in stats
        assert "atoms" in stats
        assert "bundles" in stats
        assert "clouds" in stats
        assert "facts" in stats
        assert "fact_items" in stats

        # v1 tables no longer present
        assert "artifacts" not in stats
        assert "transactions" not in stats

        # All counts should be 0 on fresh DB
        for count in stats.values():
            assert count == 0
