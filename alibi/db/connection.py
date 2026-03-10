"""Database connection management for Alibi."""

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Generator, Optional

from alibi.config import Config, get_config


# ---------------------------------------------------------------------------
# Register explicit adapters/converters to replace the deprecated defaults
# removed in Python 3.12.  This silences the DeprecationWarning triggered by
# sqlite3.PARSE_DECLTYPES.
# ---------------------------------------------------------------------------


def _adapt_date(val: date) -> str:
    return val.isoformat()


def _adapt_datetime(val: datetime) -> str:
    return val.isoformat()


def _convert_date(val: bytes) -> date:
    return date.fromisoformat(val.decode())


def _convert_datetime(val: bytes) -> datetime:
    return datetime.fromisoformat(val.decode())


sqlite3.register_adapter(date, _adapt_date)
sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("date", _convert_date)
sqlite3.register_converter("datetime", _convert_datetime)
sqlite3.register_converter("timestamp", _convert_datetime)


class DatabaseManager:
    """Manages SQLite database connections and initialization."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize database manager with configuration."""
        self.config = config or get_config()
        self._connection: Optional[sqlite3.Connection] = None

    @property
    def db_path(self) -> Path:
        """Get the database file path."""
        return self.config.get_absolute_db_path()

    def _get_schema_sql(self) -> str:
        """Load the schema SQL from file."""
        schema_path = Path(__file__).parent / "schema.sql"
        return schema_path.read_text()

    def initialize(self) -> None:
        """Initialize the database with schema and seed default data."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = self.get_connection()
        conn.executescript(self._get_schema_sql())

        # Seed default user and space (required by processing pipeline)
        conn.execute(
            "INSERT OR IGNORE INTO users (id, name) VALUES ('system', 'System')"
        )
        conn.execute(
            "INSERT OR IGNORE INTO spaces (id, name, type, owner_id) "
            "VALUES ('default', 'Default', 'private', 'system')"
        )
        conn.commit()

    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                check_same_thread=False,
            )
            # Enable foreign keys and WAL mode for concurrent reads
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            # Use Row factory for dict-like access
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database transactions."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Execute a SQL statement."""
        conn = self.get_connection()
        return conn.execute(sql, params)

    def executemany(
        self, sql: str, params_list: list[tuple[Any, ...]]
    ) -> sqlite3.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        conn = self.get_connection()
        return conn.executemany(sql, params_list)

    def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> Optional[sqlite3.Row]:
        """Execute a query and fetch one result."""
        cursor = self.execute(sql, params)
        result: Optional[sqlite3.Row] = cursor.fetchone()
        return result

    def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Execute a query and fetch all results."""
        cursor = self.execute(sql, params)
        return cursor.fetchall()

    def get_schema_version(self) -> int:
        """Get the current schema version."""
        try:
            row = self.fetchone("SELECT MAX(version) FROM schema_version")
            if row and row[0]:
                return int(row[0])
            return 0
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return 0

    def is_initialized(self) -> bool:
        """Check if the database has been initialized."""
        if not self.db_path.exists():
            return False
        try:
            version = self.get_schema_version()
            return version > 0
        except Exception:
            return False

    def get_table_count(self, table_name: str) -> int:
        """Get the count of rows in a table."""
        row = self.fetchone(f"SELECT COUNT(*) FROM {table_name}")  # noqa: S608
        return int(row[0]) if row else 0

    def get_stats(self) -> dict[str, int]:
        """Get database statistics."""
        tables = [
            "documents",
            "atoms",
            "bundles",
            "clouds",
            "facts",
            "fact_items",
        ]
        stats = {}
        for table in tables:
            try:
                stats[table] = self.get_table_count(table)
            except sqlite3.OperationalError:
                stats[table] = 0
        return stats


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def reset_db() -> None:
    """Reset the global database manager (for testing)."""
    global _db_manager
    if _db_manager is not None:
        _db_manager.close()
        _db_manager = None
