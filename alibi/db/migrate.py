"""Database migration system for Alibi.

Tracks applied migrations in the schema_version table and auto-applies
pending migrations on startup. Supports reversible up/down migrations.

Migration files follow the naming convention:
    {version}_{name}.sql       - Up migration
    {version}_{name}_down.sql  - Down migration (optional)
"""

import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _discover_migrations() -> list[tuple[int, str, Path, Optional[Path]]]:
    """Discover migration files in the migrations directory.

    Returns list of (version, name, up_path, down_path) sorted by version.
    """
    migrations: dict[int, tuple[str, Path, Optional[Path]]] = {}

    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        # Match pattern: 002_v2_models.sql or 002_v2_models_down.sql
        match = re.match(r"^(\d+)_(.+?)(?:_down)?\.sql$", path.name)
        if not match:
            continue

        version = int(match.group(1))
        name = match.group(2)
        is_down = path.name.endswith("_down.sql")

        if version not in migrations:
            migrations[version] = (name, path, None)

        if is_down:
            existing = migrations[version]
            migrations[version] = (existing[0], existing[1], path)
        else:
            existing_entry = migrations.get(version)
            if existing_entry is not None and existing_entry[2]:
                # Preserve down path
                migrations[version] = (name, path, existing_entry[2])
            else:
                migrations[version] = (name, path, None)

    return [
        (version, name, up_path, down_path)
        for version, (name, up_path, down_path) in sorted(migrations.items())
    ]


def get_current_version(conn: sqlite3.Connection) -> int:
    """Get the current schema version from the database."""
    try:
        cursor = conn.execute("SELECT MAX(version) FROM schema_version")
        row = cursor.fetchone()
        if row and row[0]:
            return int(row[0])
        return 0
    except sqlite3.OperationalError:
        return 0


def get_pending_migrations(
    conn: sqlite3.Connection,
) -> list[tuple[int, str, Path, Optional[Path]]]:
    """Get migrations that haven't been applied yet."""
    current = get_current_version(conn)
    all_migrations = _discover_migrations()
    return [(v, n, up, down) for v, n, up, down in all_migrations if v > current]


def apply_migration(conn: sqlite3.Connection, version: int, up_path: Path) -> None:
    """Apply a single up migration."""
    logger.info("Applying migration %d: %s", version, up_path.stem)
    sql = up_path.read_text()
    conn.executescript(sql)
    logger.info("Migration %d applied successfully", version)


def revert_migration(conn: sqlite3.Connection, version: int, down_path: Path) -> None:
    """Revert a single migration using its down script."""
    logger.info("Reverting migration %d: %s", version, down_path.stem)
    sql = down_path.read_text()
    conn.executescript(sql)
    logger.info("Migration %d reverted successfully", version)


def migrate_up(conn: sqlite3.Connection, target: Optional[int] = None) -> int:
    """Apply all pending migrations up to target version.

    Args:
        conn: SQLite connection.
        target: Optional max version to migrate to. If None, applies all.

    Returns:
        Number of migrations applied.
    """
    pending = get_pending_migrations(conn)
    applied = 0

    for version, name, up_path, _down_path in pending:
        if target is not None and version > target:
            break
        apply_migration(conn, version, up_path)
        applied += 1

    return applied


def migrate_down(conn: sqlite3.Connection, target: int) -> int:
    """Revert migrations down to target version.

    Args:
        conn: SQLite connection.
        target: Target version to revert to (exclusive - migrations > target are reverted).

    Returns:
        Number of migrations reverted.
    """
    current = get_current_version(conn)
    if current <= target:
        return 0

    all_migrations = _discover_migrations()
    # Get migrations to revert, in reverse order
    to_revert = [
        (v, n, up, down)
        for v, n, up, down in reversed(all_migrations)
        if v > target and v <= current
    ]

    reverted = 0
    for version, name, _up_path, down_path in to_revert:
        if down_path is None:
            raise ValueError(
                f"Migration {version} ({name}) has no down script; "
                f"cannot revert past version {version}"
            )
        revert_migration(conn, version, down_path)
        reverted += 1

    return reverted


def auto_migrate(conn: sqlite3.Connection) -> int:
    """Auto-apply pending migrations. Called on startup.

    Returns:
        Number of migrations applied.
    """
    pending = get_pending_migrations(conn)
    if not pending:
        return 0

    versions = [v for v, _, _, _ in pending]
    logger.info(
        "Found %d pending migration(s): %s",
        len(pending),
        ", ".join(str(v) for v in versions),
    )
    return migrate_up(conn)
