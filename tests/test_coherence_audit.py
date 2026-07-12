"""Tests for the human-gated enrichment coherence audit."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.config import Config, reset_config
from alibi.db.connection import DatabaseManager
from alibi.db.models import CloudStatus, Fact, FactStatus, FactType
from alibi.db import v2_store
from alibi.enrichment.coherence_audit import (
    CoherenceFinding,
    apply_coherence_fixes,
    audit_coherence,
    load_approved_findings,
    write_findings_yaml,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    reset_config()
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    config = Config(db_path=db_path, _env_file=None)
    manager = DatabaseManager(config)
    if not manager.is_initialized():
        manager.initialize()
    yield manager
    manager.close()
    os.unlink(db_path)


def _seed_item(
    db: DatabaseManager,
    name: str,
    comparable_name: str | None,
    category_path: str | None,
    enrichment_source: str | None = "combined",
    vendor: str = "Test Vendor",
) -> str:
    cloud_id = str(uuid.uuid4())
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
            (cloud_id, CloudStatus.COLLAPSED.value, 1.0),
        )
    fact = Fact(
        id=str(uuid.uuid4()),
        cloud_id=cloud_id,
        fact_type=FactType.PURCHASE,
        vendor=vendor,
        total_amount=Decimal("10.0"),
        currency="EUR",
        event_date=date(2026, 1, 15),
        status=FactStatus.CONFIRMED,
    )
    v2_store.store_fact(db, fact, [])
    doc_id = str(uuid.uuid4())
    atom_id = str(uuid.uuid4())
    item_id = str(uuid.uuid4())
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
            (doc_id, f"{item_id}.jpg", item_id[:16]),
        )
        cursor.execute(
            "INSERT INTO atoms (id, document_id, atom_type, data) "
            "VALUES (?, ?, 'item', ?)",
            (atom_id, doc_id, json.dumps({"name": name})),
        )
        cursor.execute(
            "INSERT INTO fact_items "
            "(id, fact_id, atom_id, name, comparable_name, category_path, "
            " total_price, enrichment_source) "
            "VALUES (?, ?, ?, ?, ?, ?, 10.0, ?)",
            (
                item_id,
                fact.id,
                atom_id,
                name,
                comparable_name,
                category_path,
                enrichment_source,
            ),
        )
    return item_id


def _llm_answer(entries: list[dict[str, Any]]):
    """Build a call_enrichment_llm stub returning the given item entries."""

    def _fake(prompt: str, **kwargs: Any) -> list[Any]:
        return entries

    return _fake


# ---------------------------------------------------------------------------
# audit_coherence
# ---------------------------------------------------------------------------


def test_audit_flags_incoherent_item(db: DatabaseManager) -> None:
    item_id = _seed_item(
        db, "Lauretana VAP - 750", "lauretana vap wine", "food > beverages > alcohol"
    )
    _seed_item(db, "Cassius Tea 500мл", "tea", "food > beverages > water")

    entries = [
        {
            "idx": 1,
            "coherent": False,
            "reason": "Lauretana is mineral water, not wine",
            "suggested_comparable_name": "mineral water",
            "suggested_category_path": "food > beverages > water",
        },
        {"idx": 2, "coherent": True},
    ]
    with patch(
        "alibi.enrichment.coherence_audit.call_enrichment_llm",
        side_effect=_llm_answer(entries),
    ):
        findings = audit_coherence(db)

    assert len(findings) == 1
    f = findings[0]
    assert f.item_id == item_id
    assert f.suggested_comparable_name == "mineral water"
    assert f.suggested_category_path == "food > beverages > water"
    assert f.approved is False


def test_audit_skips_user_confirmed(db: DatabaseManager) -> None:
    _seed_item(
        db,
        "Lauretana VAP - 750",
        "mineral water",
        "food > beverages > water",
        enrichment_source="user_confirmed",
    )
    with patch("alibi.enrichment.coherence_audit.call_enrichment_llm") as mock_llm:
        findings = audit_coherence(db)
    assert findings == []
    mock_llm.assert_not_called()


def test_audit_drops_invented_category_path(db: DatabaseManager) -> None:
    """Suggested paths not already in the DB taxonomy are nulled, not kept."""
    _seed_item(db, "Ancor's Oasis", "cheese", "food > dairy > cheese")
    entries = [
        {
            "idx": 1,
            "coherent": False,
            "reason": "a beverage, not cheese",
            "suggested_comparable_name": "ancor's oasis",
            "suggested_category_path": "food > made-up > nonsense",
        },
    ]
    with patch(
        "alibi.enrichment.coherence_audit.call_enrichment_llm",
        side_effect=_llm_answer(entries),
    ):
        findings = audit_coherence(db)
    assert len(findings) == 1
    assert findings[0].suggested_category_path is None
    assert findings[0].suggested_comparable_name == "ancor's oasis"


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


def test_yaml_round_trip_gates_on_approved(tmp_path: Path) -> None:
    findings = [
        CoherenceFinding(
            item_id="item-1",
            name="Lauretana VAP - 750",
            vendor="MM",
            comparable_name="lauretana vap wine",
            category_path="food > beverages > alcohol",
            suggested_comparable_name="mineral water",
            suggested_category_path="food > beverages > water",
            reason="mineral water, not wine",
        ),
        CoherenceFinding(
            item_id="item-2",
            name="Ancor's Oasis",
            vendor="MM",
            comparable_name="cheese",
            category_path="food > dairy > cheese",
            suggested_comparable_name="ancor's oasis",
            suggested_category_path=None,
            reason="a beverage",
        ),
    ]
    out = tmp_path / "findings.yaml"
    write_findings_yaml(findings, out, generated="2026-07-12")

    # Nothing approved yet
    assert load_approved_findings(out) == []

    # Approve only the first
    text = out.read_text(encoding="utf-8")
    text = text.replace("approved: false", "approved: true", 1)
    out.write_text(text, encoding="utf-8")

    approved = load_approved_findings(out)
    assert len(approved) == 1
    assert approved[0].item_id == "item-1"
    assert approved[0].suggested_comparable_name == "mineral water"


def test_load_rejects_malformed_file(tmp_path: Path) -> None:
    out = tmp_path / "bad.yaml"
    out.write_text("just a string", encoding="utf-8")
    with pytest.raises(ValueError):
        load_approved_findings(out)


# ---------------------------------------------------------------------------
# apply_coherence_fixes
# ---------------------------------------------------------------------------


def test_apply_updates_only_suggested_fields(db: DatabaseManager) -> None:
    item_id = _seed_item(
        db, "Lauretana VAP - 750", "lauretana vap wine", "food > beverages > alcohol"
    )
    finding = CoherenceFinding(
        item_id=item_id,
        name="Lauretana VAP - 750",
        vendor="Test Vendor",
        comparable_name="lauretana vap wine",
        category_path="food > beverages > alcohol",
        suggested_comparable_name="mineral water",
        suggested_category_path=None,
        reason="mineral water, not wine",
        approved=True,
    )
    result = apply_coherence_fixes(db, [finding])

    assert len(result.applied) == 1
    conn = db.get_connection()
    row = conn.execute(
        "SELECT comparable_name, category_path, enrichment_source, "
        "enrichment_confidence FROM fact_items WHERE id = ?",
        (item_id,),
    ).fetchone()
    assert row["comparable_name"] == "mineral water"
    # Not suggested -> untouched
    assert row["category_path"] == "food > beverages > alcohol"
    assert row["enrichment_source"] == "user_confirmed"
    assert row["enrichment_confidence"] == 1.0


def test_apply_skips_missing_item_and_empty_suggestions(db: DatabaseManager) -> None:
    gone = CoherenceFinding(
        item_id=str(uuid.uuid4()),
        name="Ghost",
        vendor=None,
        comparable_name=None,
        category_path=None,
        suggested_comparable_name="anything",
        suggested_category_path=None,
        reason=None,
        approved=True,
    )
    no_suggestion = CoherenceFinding(
        item_id=str(uuid.uuid4()),
        name="No-op",
        vendor=None,
        comparable_name=None,
        category_path=None,
        suggested_comparable_name=None,
        suggested_category_path=None,
        reason="flagged but nothing to change",
        approved=True,
    )
    result = apply_coherence_fixes(db, [gone, no_suggestion])
    assert result.applied == []
    assert result.rebuilt_stars == 0
