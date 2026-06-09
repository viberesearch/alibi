"""Guard: ``schema.sql`` must match the migration chain.

A fresh install builds its database from ``alibi/db/schema.sql`` (a snapshot of
the head schema), while an existing install is brought to head by running the
incremental ``alibi/db/migrations/*.sql`` chain. These two paths MUST agree, or
the two populations silently diverge.

``schema.sql`` is no longer hand-maintained: ``scripts/generate_schema.py``
builds it FROM ``alibi/db/baseline_v1.sql`` plus the migration chain, so the two
cannot drift by construction. These tests defend that contract from two sides:

* :func:`test_schema_is_freshly_generated` — the committed ``schema.sql`` equals
  what the generator produces right now. This catches a migration added without
  regenerating, or a hand-edit of the snapshot. (Same logic as the generator's
  ``--check`` mode, which CI also runs.)
* the structural tests — a database built from ``schema.sql`` and one built from
  ``baseline_v1.sql`` + the chain have identical tables and columns. Belt to the
  generator's braces, and a clearer failure if either path is somehow malformed.

The legacy ``items`` precision/nullability drift and the dead ``space_members``
table were reconciled by migration 043, so there is no longer any allow-list.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from alibi.config import Config
from alibi.db.connection import DatabaseManager
from alibi.db.migrate import migrate_up

_BASELINE_V1 = (
    Path(__file__).resolve().parent.parent / "alibi" / "db" / "baseline_v1.sql"
).read_text()


def _user_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    )
    return {r[0] for r in rows}


def _columns(conn: sqlite3.Connection, table: str) -> set[tuple[str, str, int, int]]:
    """Return {(name, upper-cased declared type, notnull, pk)} for a table.

    Column order (cid) and default value are intentionally excluded: ALTER TABLE
    ADD COLUMN appends columns, so the migration build orders post-v1 columns
    differently from the inline schema.sql, which is not a real divergence.
    """
    return {
        (row[1], (row[2] or "").upper(), row[3], row[5])
        for row in conn.execute(f"PRAGMA table_info({table})")  # noqa: S608
    }


@pytest.fixture
def schema_conn(tmp_path: Path) -> sqlite3.Connection:
    """A database built from schema.sql (the fresh-install path)."""
    manager = DatabaseManager(Config(db_path=tmp_path / "from_schema.db"))
    manager.initialize()
    return manager.get_connection()


@pytest.fixture
def migrated_conn() -> sqlite3.Connection:
    """A database built by running the full migration chain off the v1 baseline."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(_BASELINE_V1)
    migrate_up(conn)
    return conn


def test_schema_is_freshly_generated() -> None:
    """The committed schema.sql is exactly what generate_schema.py produces.

    This is the anti-drift guarantee: a migration added without regenerating, or
    a hand-edit of the snapshot, makes the two differ and fails here (and in the
    generator's ``--check`` CI step).
    """
    import sys

    scripts = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import generate_schema
    finally:
        sys.path.remove(str(scripts))

    conn = generate_schema._build_head_db()
    rendered = generate_schema.render_schema(conn)
    committed = generate_schema._SCHEMA.read_text()
    assert committed == rendered, (
        "schema.sql is out of sync with the migration chain. "
        "Run: uv run python scripts/generate_schema.py"
    )


def test_no_schema_only_tables(
    schema_conn: sqlite3.Connection, migrated_conn: sqlite3.Connection
) -> None:
    """Every table in schema.sql is also produced by the migrations."""
    only_in_schema = _user_tables(schema_conn) - _user_tables(migrated_conn)
    assert not only_in_schema, (
        "Tables defined in schema.sql but created by no migration: "
        f"{sorted(only_in_schema)}."
    )


def test_no_migration_only_tables(
    schema_conn: sqlite3.Connection, migrated_conn: sqlite3.Connection
) -> None:
    """Every table the migrations create is also in the schema.sql snapshot."""
    only_in_migrations = _user_tables(migrated_conn) - _user_tables(schema_conn)
    assert not only_in_migrations, (
        "Tables created by a migration but missing from schema.sql: "
        f"{sorted(only_in_migrations)}. Regenerate schema.sql."
    )


def test_common_tables_have_matching_columns(
    schema_conn: sqlite3.Connection, migrated_conn: sqlite3.Connection
) -> None:
    """Columns of every shared table match between the two build paths."""
    common = _user_tables(schema_conn) & _user_tables(migrated_conn)
    mismatches: dict[str, dict[str, list[tuple[str, str, int, int]]]] = {}
    for table in sorted(common):
        schema_cols = _columns(schema_conn, table)
        migrated_cols = _columns(migrated_conn, table)
        if schema_cols != migrated_cols:
            mismatches[table] = {
                "schema_only": sorted(schema_cols - migrated_cols),
                "migration_only": sorted(migrated_cols - schema_cols),
            }
    assert not mismatches, (
        "schema.sql and the migration chain disagree on columns "
        f"(regenerate schema.sql):\n{mismatches}"
    )


def test_fact_items_enrichment_columns_present_both_ways(
    schema_conn: sqlite3.Connection, migrated_conn: sqlite3.Connection
) -> None:
    """Explicit check on the enrichment-sentinel columns (fast-fail clarity).

    These are the most recently added, most drift-prone columns. The generic
    ``test_common_tables_have_matching_columns`` covers every column including
    any added after this list; this one just fails loudly on the usual suspect.
    """
    expected = {"attributes", "unit_enriched", "comparable_name_enriched"}
    for label, conn in (("schema.sql", schema_conn), ("migrations", migrated_conn)):
        names = {c[0] for c in _columns(conn, "fact_items")}
        missing = expected - names
        assert not missing, f"fact_items via {label} is missing {sorted(missing)}"
