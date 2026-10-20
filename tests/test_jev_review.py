"""Tests for the opt-in hosted Jev category review."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.config import Config, reset_config
from alibi.db import v2_store
from alibi.db.connection import DatabaseManager
from alibi.db.models import CloudStatus, Fact, FactStatus, FactType
from alibi.enrichment import taxonomy
from alibi.enrichment.coherence_audit import load_approved_findings
from alibi.enrichment.jev_review import (
    JevError,
    decide,
    review_categories,
    reviewed_marker,
    taxonomy_levels,
    write_review_file,
)

MODEL = "jev-1.14.0"


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


def _cfg(**overrides: Any) -> Config:
    values: dict[str, Any] = {"jev_enabled": True, "jev_api_key": "test-key"}
    values.update(overrides)
    return Config(_env_file=None, **values)


def _seed_item(
    db: DatabaseManager,
    name: str,
    category_path: str,
    comparable_name: str | None = None,
    enrichment_source: str | None = "llm_category",
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
        vendor="Test Vendor",
        total_amount=Decimal("10.0"),
        currency="EUR",
        event_date=date(2026, 1, 15),
        status=FactStatus.CONFIRMED,
    )
    v2_store.store_fact(db, fact, [])
    doc_id, atom_id, item_id = (str(uuid.uuid4()) for _ in range(3))
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


class FakeJev:
    """MockTransport handler answering by receipt line: name -> (path, conf)."""

    def __init__(self, answers: dict[str, tuple[str, float]], served: str = MODEL):
        self.answers = answers
        self.served = served
        self.requests: list[dict[str, Any]] = []
        branches, children = taxonomy_levels()
        self.branch_of = {leaf: b for b, leaves in children.items() for leaf in leaves}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        out: dict[str, Any] = {}
        for key, question in body["questions"].items():
            path, conf = self.answers[body["state"][key]["receipt_line"]]
            options = question["criteria"]
            choice = path if path in options else self.branch_of[path]
            out[key] = {"type": "choice", "choice": choice, "confidence": conf}
        return httpx.Response(200, json={"model": self.served, "answers": out})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def _row(db: DatabaseManager, item_id: str) -> dict[str, Any]:
    cur = db.get_connection().execute(
        "SELECT category_path, category, enrichment_source, category_reviewed_by "
        "FROM fact_items WHERE id = ?",
        (item_id,),
    )
    return dict(cur.fetchone())


def test_decide_rules() -> None:
    cfg = _cfg()
    dairy, cheese = "food > dairy", "food > dairy > cheese"
    assert decide(cheese, cheese, 0.99, cfg) == "agree"
    assert decide(cheese, dairy, 0.99, cfg) == "agree"  # Jev coarser
    assert decide(dairy, cheese, 0.95, cfg) == "deepen"
    assert decide(dairy, cheese, 0.85, cfg) == "agree"  # below refine threshold
    assert decide(cheese, "food > bakery", 0.85, cfg) == "flag"
    assert decide(cheese, "food > bakery", 0.5, cfg) == "ignore"


def test_taxonomy_levels_fit_choice_limit() -> None:
    branches, children = taxonomy_levels()
    assert len(branches) <= 26
    assert all(len(leaves) <= 26 for leaves in children.values())
    assert sorted(p for leaves in children.values() for p in leaves) == sorted(
        taxonomy.VALID_PATHS
    )


def test_groups_by_product_skips_decided_rows_and_marks(db: DatabaseManager) -> None:
    a = _seed_item(db, "GOUDA 200G", "food > dairy > cheese", "gouda cheese")
    b = _seed_item(db, "GOUDA SLICES", "food > dairy > cheese", "gouda cheese")
    confirmed = _seed_item(
        db, "MILK", "food > dairy > milk", enrichment_source="user_confirmed"
    )
    sibling = _seed_item(
        db, "MILK 2", "food > dairy > milk", enrichment_source="sibling_propagation"
    )
    fake = FakeJev(
        {
            "GOUDA 200G": ("food > dairy > cheese", 0.99),
            "GOUDA SLICES": ("food > dairy > cheese", 0.99),
        }
    )

    result = review_categories(db, cfg=_cfg(), client=fake.client())

    assert result.groups_reviewed == 1 and result.items_marked == 2
    assert len(fake.requests) == 2  # branch call + leaf call
    assert len(fake.requests[0]["state"]) == 1
    marker = reviewed_marker(MODEL)
    assert _row(db, a)["category_reviewed_by"] == marker
    assert _row(db, b)["category_reviewed_by"] == marker
    assert _row(db, confirmed)["category_reviewed_by"] is None
    assert _row(db, sibling)["category_reviewed_by"] is None

    again = review_categories(db, cfg=_cfg(), client=fake.client())
    assert again.groups_reviewed == 0 and len(fake.requests) == 2


def test_deepen_is_a_finding_while_auto_apply_off(db: DatabaseManager) -> None:
    item = _seed_item(db, "KASSERI", "food > dairy")
    fake = FakeJev({"KASSERI": ("food > dairy > cheese", 0.95)})
    result = review_categories(db, cfg=_cfg(), client=fake.client())
    assert result.refined_items == 0
    assert [f.suggested_category_path for f in result.findings] == [
        "food > dairy > cheese"
    ]
    assert _row(db, item)["category_path"] == "food > dairy"


def test_deepen_applied_to_every_row_with_auto_apply(db: DatabaseManager) -> None:
    items = [_seed_item(db, "KASSERI", "food > dairy", "kasseri") for _ in range(2)]
    fake = FakeJev({"KASSERI": ("food > dairy > cheese", 0.95)})
    result = review_categories(db, cfg=_cfg(jev_auto_apply=True), client=fake.client())
    assert result.refined_items == 2 and result.findings == []
    for item in items:
        row = _row(db, item)
        assert row["category_path"] == "food > dairy > cheese"
        assert row["category"].lower() == taxonomy.leaf_of("food > dairy > cheese")
        assert row["enrichment_source"] == "jev_refine"


def test_conflict_flagged_only_above_threshold(db: DatabaseManager) -> None:
    _seed_item(db, "MAGGI BOUILLON", "food > dairy > cheese")
    _seed_item(db, "ODD LINE", "food > dairy > cheese")
    fake = FakeJev(
        {
            "MAGGI BOUILLON": ("food > pantry > condiments_sauces", 0.9),
            "ODD LINE": ("household", 0.4),
        }
    )
    result = review_categories(db, cfg=_cfg(), client=fake.client())
    assert [f.name for f in result.findings] == ["MAGGI BOUILLON"]
    assert "conflicts" in (result.findings[0].reason or "")


def test_model_mismatch_aborts_without_marking(db: DatabaseManager) -> None:
    item = _seed_item(db, "GOUDA", "food > dairy > cheese")
    fake = FakeJev({"GOUDA": ("food > dairy > cheese", 0.99)}, served="jev-1.13.0")
    with pytest.raises(JevError, match="served"):
        review_categories(db, cfg=_cfg(), client=fake.client())
    assert _row(db, item)["category_reviewed_by"] is None


def test_missing_key_raises(db: DatabaseManager) -> None:
    with pytest.raises(JevError, match="JEV_API_KEY"):
        review_categories(db, cfg=_cfg(jev_api_key=None))


def test_review_file_round_trips(db: DatabaseManager, tmp_path: Path) -> None:
    _seed_item(db, "MAGGI BOUILLON", "food > dairy > cheese")
    fake = FakeJev({"MAGGI BOUILLON": ("food > pantry > condiments_sauces", 0.9)})
    result = review_categories(db, cfg=_cfg(), client=fake.client())
    path = write_review_file(result.findings, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Jev category review")
    assert load_approved_findings(path) == []  # nothing approved by default
    path.write_text(text.replace("approved: false", "approved: true"), encoding="utf-8")
    approved = load_approved_findings(path)
    assert approved[0].suggested_category_path == "food > pantry > condiments_sauces"
