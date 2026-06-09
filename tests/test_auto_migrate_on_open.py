"""An existing, stale database is migrated to head when first opened."""

from __future__ import annotations

from pathlib import Path

import pytest

from alibi.config import Config
from alibi.db.connection import DatabaseManager
from alibi.db.migrate import _discover_migrations, revert_migration


def _roll_back_to(manager: DatabaseManager, target: int) -> None:
    """Revert the manager's DB down to ``target`` using the down-migrations."""
    conn = manager.get_connection()
    downs = {v: down for v, _, _, down in _discover_migrations()}
    current = manager.get_schema_version()
    for version in range(current, target, -1):
        down = downs.get(version)
        assert down is not None, f"missing down-migration for v{version}"
        revert_migration(conn, version, Path(down))
    conn.commit()


def _head_version() -> int:
    return max(v for v, *_ in _discover_migrations())


@pytest.fixture
def stale_db_path(tmp_path: Path) -> Path:
    """A real DB initialised at head, then rolled back two versions."""
    path = tmp_path / "stale.db"
    seed = DatabaseManager(Config(db_path=path))
    seed.initialize()
    _roll_back_to(seed, _head_version() - 2)
    assert seed.get_schema_version() == _head_version() - 2
    seed.close()
    return path


def test_opening_stale_db_migrates_to_head(stale_db_path: Path) -> None:
    manager = DatabaseManager(Config(db_path=stale_db_path))
    manager.get_connection()  # first open triggers the auto-migrate
    assert manager.get_schema_version() == _head_version()
    manager.close()


def test_auto_migrate_takes_a_backup(stale_db_path: Path) -> None:
    from_version = _head_version() - 2
    manager = DatabaseManager(Config(db_path=stale_db_path))
    manager.get_connection()
    backup = stale_db_path.with_name(
        stale_db_path.name + f".bak_premigrate_v{from_version}"
    )
    assert backup.exists(), "expected a pre-migration backup snapshot"
    # The backup preserves the pre-migration version.
    snap = DatabaseManager(Config(db_path=backup))
    # Reading the snapshot must not itself migrate it.
    import os

    os.environ["ALIBI_AUTO_MIGRATE"] = "0"
    try:
        assert snap.get_schema_version() == from_version
    finally:
        os.environ.pop("ALIBI_AUTO_MIGRATE", None)
        snap.close()
    manager.close()


def test_opt_out_env_var_disables_migration(stale_db_path: Path) -> None:
    import os

    os.environ["ALIBI_AUTO_MIGRATE"] = "0"
    try:
        manager = DatabaseManager(Config(db_path=stale_db_path))
        manager.get_connection()
        assert manager.get_schema_version() == _head_version() - 2  # untouched
    finally:
        os.environ.pop("ALIBI_AUTO_MIGRATE", None)
        manager.close()


def test_current_db_is_a_noop(tmp_path: Path) -> None:
    path = tmp_path / "fresh.db"
    manager = DatabaseManager(Config(db_path=path))
    manager.initialize()  # built from schema.sql, already at head
    manager.get_connection()
    assert manager.get_schema_version() == _head_version()
    # No pre-migration backup should be created for an already-current DB.
    assert not list(tmp_path.glob("*.bak_premigrate_*"))
    manager.close()
