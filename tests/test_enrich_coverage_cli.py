"""CLI tests for ``lt enrich coverage`` — the --check gate and --json output."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

os.environ["ALIBI_TESTING"] = "1"

from click.testing import CliRunner

from alibi.commands.enrich import enrich


def _seed(db, item_id: str, name: str = "ITEM", **cols) -> None:
    """Insert a fact_item with its supporting chain."""
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
    keys = ["id", "fact_id", "atom_id", "name"]
    vals = [item_id, fact_id, atom_id, name]
    for k, v in cols.items():
        keys.append(k)
        vals.append(v)
    ph = ", ".join("?" for _ in keys)
    conn.execute(
        f"INSERT OR IGNORE INTO fact_items ({', '.join(keys)}) VALUES ({ph})",  # noqa: S608
        tuple(vals),
    )
    conn.commit()


def _fully_enriched(db, item_id: str) -> None:
    """Seed a row that needs nothing (every field filled or marked)."""
    _seed(
        db,
        item_id,
        comparable_name="thing",
        comparable_name_enriched=1,
        unit_quantity=1.0,
        unit_enriched=1,
        category_path="other",
        category_taxonomy_version=1,
        attributes='{"state": "fresh"}',
        state_enriched=1,
    )


def _invoke(db, args):
    with patch("alibi.commands.enrich.get_db", return_value=db):
        return CliRunner().invoke(enrich, ["coverage", *args])


class TestCheck:
    def test_check_fails_when_pending(self, db):
        _seed(db, "p")  # bare row: pending on every field
        result = _invoke(db, ["--check", "--stragglers", "0"])
        assert result.exit_code == 1
        assert "FAILED" in result.output

    def test_check_passes_at_full_coverage(self, db):
        _fully_enriched(db, "done")
        result = _invoke(db, ["--check", "--stragglers", "0"])
        assert result.exit_code == 0
        assert "passed" in result.output

    def test_max_pending_tolerates(self, db):
        _seed(db, "p")  # 1 pending per field
        # threshold 1 tolerates it -> pass
        assert _invoke(db, ["--check", "--max-pending", "1"]).exit_code == 0
        # threshold 0 does not -> fail
        assert _invoke(db, ["--check", "--max-pending", "0"]).exit_code == 1

    def test_no_check_never_fails(self, db):
        _seed(db, "p")
        result = _invoke(db, [])  # no --check -> always exit 0
        assert result.exit_code == 0


class TestJson:
    def test_json_shape_and_ok_flag(self, db):
        _fully_enriched(db, "done")
        result = _invoke(db, ["--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["ok"] is True
        assert payload["total_pending"] == 0
        assert {f["field"] for f in payload["fields"]} == {
            "comparable_name",
            "unit_quantity",
            "category",
            "attributes",
            "state",
        }

    def test_json_check_exits_nonzero_but_still_emits(self, db):
        _seed(db, "p")
        result = _invoke(db, ["--json", "--check"])
        assert result.exit_code == 1
        payload = json.loads(result.output)  # JSON still emitted before exit
        assert payload["ok"] is False
        assert payload["total_pending"] > 0
