#!/usr/bin/env python
"""Generate ``alibi/db/schema.sql`` from the migration chain.

``schema.sql`` is the fresh-install snapshot of the head schema; the
``migrations/*.sql`` chain brings an existing database to the same head. The two
must agree, or fresh and migrated installs silently diverge. Rather than keep
``schema.sql`` hand-maintained and *check* it against the chain (the old guard),
this builds it FROM the chain so they cannot drift:

    fresh in-memory DB  ->  apply baseline_v1.sql  ->  migrate_up()  ->  dump

The dump is the ``CREATE`` statements of every surviving table / index / trigger
/ view (FTS5 shadow tables excluded — they follow their virtual table), with
``IF NOT EXISTS`` restored, followed by the ``schema_version`` seed rows so a
fresh install records itself at head and ``auto_migrate`` is a no-op.

Usage:
    uv run python scripts/generate_schema.py            # rewrite schema.sql
    uv run python scripts/generate_schema.py --check     # exit 1 if stale

The ``--check`` mode is what CI runs: it regenerates in memory and fails if the
committed ``schema.sql`` differs, so adding a migration without regenerating (or
hand-editing the snapshot) is caught.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

_DB_DIR = Path(__file__).resolve().parent.parent / "alibi" / "db"
_BASELINE = _DB_DIR / "baseline_v1.sql"
_SCHEMA = _DB_DIR / "schema.sql"

_HEADER = """\
-- ===========================================================================
-- Alibi database schema (HEAD) — GENERATED FILE, DO NOT EDIT BY HAND.
--
-- Produced by scripts/generate_schema.py from alibi/db/baseline_v1.sql plus the
-- alibi/db/migrations/*.sql chain. To change the schema, add a migration and
-- regenerate (`uv run python scripts/generate_schema.py`); CI fails if this file
-- is out of sync with the chain (`--check`).
-- ===========================================================================
"""

# Restore "IF NOT EXISTS" (sqlite_master stores the resolved CREATE without it),
# matching the long-standing schema.sql style and keeping the script idempotent.
_CREATE_RE = re.compile(
    r"^CREATE\s+(VIRTUAL\s+TABLE|TABLE|UNIQUE\s+INDEX|INDEX|TRIGGER|VIEW)\s+"
    r"(?!IF\s+NOT\s+EXISTS)",
    re.IGNORECASE,
)


def _build_head_db() -> sqlite3.Connection:
    """Apply the baseline then the full migration chain to a fresh in-memory DB."""
    from alibi.db.migrate import migrate_up

    conn = sqlite3.connect(":memory:")
    conn.executescript(_BASELINE.read_text())
    migrate_up(conn)
    return conn


def _shadow_prefixes(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Name prefixes of FTS5 shadow tables, derived from the virtual tables."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND sql LIKE 'CREATE VIRTUAL TABLE%'"
    )
    return tuple(f"{r[0]}_" for r in rows)


def _add_if_not_exists(sql: str) -> str:
    return _CREATE_RE.sub(lambda m: m.group(0) + "IF NOT EXISTS ", sql, count=1)


def render_schema(conn: sqlite3.Connection) -> str:
    """Render the deterministic schema.sql text for a head database."""
    shadows = _shadow_prefixes(conn)
    objects = conn.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
        "ORDER BY rowid"
    ).fetchall()

    parts: list[str] = [_HEADER]
    for _type, name, sql in objects:
        if any(name.startswith(p) for p in shadows):
            continue  # FTS5 shadow table — created with its virtual table
        parts.append(_add_if_not_exists(sql.strip()) + ";")

    versions = [
        r[0]
        for r in conn.execute("SELECT version FROM schema_version ORDER BY version")
    ]
    seeds = "\n".join(
        f"INSERT OR IGNORE INTO schema_version (version) VALUES ({v});"
        for v in versions
    )
    parts.append(
        "-- Record every applied migration so a fresh install is at head.\n" + seeds
    )

    return "\n\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if schema.sql is out of sync (do not write).",
    )
    args = parser.parse_args()

    conn = _build_head_db()
    rendered = render_schema(conn)

    if args.check:
        current = _SCHEMA.read_text() if _SCHEMA.exists() else ""
        if current != rendered:
            sys.stderr.write(
                "schema.sql is out of sync with the migration chain. "
                "Run: uv run python scripts/generate_schema.py\n"
            )
            return 1
        sys.stdout.write("schema.sql is in sync with the migration chain.\n")
        return 0

    _SCHEMA.write_text(rendered)
    sys.stdout.write(f"Wrote {_SCHEMA.relative_to(_DB_DIR.parent.parent)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
