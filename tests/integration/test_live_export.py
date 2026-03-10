"""Live end-to-end analytics export integration test.

Run only when the analytics-stack server is up:
    uv run pytest tests/integration/test_live_export.py -v

Skip automatically when analytics-stack is unreachable.
Requires the alibi DB to have at least one fact.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from alibi.db.connection import DatabaseManager
from alibi.services.export_analytics import (
    build_export_payload,
    push_to_analytics_stack,
)

ANALYTICS_URL = "http://127.0.0.1:8070"


def _analytics_available() -> bool:
    try:
        r = httpx.get(f"{ANALYTICS_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


analytics_available = pytest.mark.skipif(
    not _analytics_available(),
    reason="Analytics-stack not running at http://127.0.0.1:8070",
)


@pytest.fixture(scope="module")
def db() -> DatabaseManager:
    manager = DatabaseManager()
    if not manager.is_initialized():
        manager.initialize()
    return manager


def _sql(query: str) -> dict[str, Any]:
    """Execute a SQL query against analytics-stack and return parsed response."""
    r = httpx.post(
        f"{ANALYTICS_URL}/v1/sql",
        json={"sql": query},
        timeout=10,
    )
    r.raise_for_status()
    result: dict[str, Any] = r.json()
    return result


# ---------------------------------------------------------------------------
# Pre-export: build payload and sanity-check
# ---------------------------------------------------------------------------


@pytest.mark.integration
@analytics_available
def test_build_payload_has_facts(db: DatabaseManager) -> None:
    """Payload built from live DB has at least one fact."""
    payload = build_export_payload(db)
    assert len(payload["facts"]) > 0, "No facts in DB — cannot test export"
    assert len(payload["documents"]) > 0


@pytest.mark.integration
@analytics_available
def test_payload_facts_have_provenance(db: DatabaseManager) -> None:
    """Every fact in the payload has source and user_id fields."""
    payload = build_export_payload(db)
    for fact in payload["facts"]:
        assert "source" in fact, f"Fact {fact['id']} missing source field"
        assert "user_id" in fact, f"Fact {fact['id']} missing user_id field"


@pytest.mark.integration
@analytics_available
def test_payload_documents_have_provenance(db: DatabaseManager) -> None:
    """Every document in the payload has source and user_id."""
    payload = build_export_payload(db)
    for doc in payload["documents"]:
        assert "source" in doc
        assert "user_id" in doc
        assert doc["source"] is not None, f"Doc {doc['id']} has NULL source"


# ---------------------------------------------------------------------------
# Live export: push to analytics-stack
# ---------------------------------------------------------------------------


@pytest.mark.integration
@analytics_available
def test_push_succeeds(db: DatabaseManager) -> None:
    """push_to_analytics_stack returns status=ok against live server."""
    result = push_to_analytics_stack(db, ANALYTICS_URL)
    assert result["status"] == "ok"
    assert result["http_status"] == 200
    assert result["facts_count"] > 0


@pytest.mark.integration
@analytics_available
def test_facts_arrived_in_postgres(db: DatabaseManager) -> None:
    """After export, alibi_facts table has rows matching the payload count."""
    payload = build_export_payload(db)
    push_to_analytics_stack(db, ANALYTICS_URL)

    resp = _sql("SELECT count(*) as cnt FROM datasets.alibi_facts")
    pg_count = resp["rows"][0][0]
    assert pg_count == len(payload["facts"])


@pytest.mark.integration
@analytics_available
def test_fact_items_arrived_in_postgres(db: DatabaseManager) -> None:
    """After export, alibi_fact_items table has correct row count."""
    payload = build_export_payload(db)
    push_to_analytics_stack(db, ANALYTICS_URL)

    resp = _sql("SELECT count(*) as cnt FROM datasets.alibi_fact_items")
    pg_count = resp["rows"][0][0]
    assert pg_count == len(payload["fact_items"])


@pytest.mark.integration
@analytics_available
def test_documents_arrived_in_postgres(db: DatabaseManager) -> None:
    """After export, alibi_documents table has correct row count."""
    payload = build_export_payload(db)
    push_to_analytics_stack(db, ANALYTICS_URL)

    resp = _sql("SELECT count(*) as cnt FROM datasets.alibi_documents")
    pg_count = resp["rows"][0][0]
    assert pg_count == len(payload["documents"])


@pytest.mark.integration
@analytics_available
def test_provenance_in_postgres(db: DatabaseManager) -> None:
    """Facts in PostgreSQL have non-NULL source and user_id."""
    push_to_analytics_stack(db, ANALYTICS_URL)

    resp = _sql(
        "SELECT source, user_id, count(*) as cnt "
        "FROM datasets.alibi_facts "
        "GROUP BY source, user_id "
        "ORDER BY cnt DESC"
    )
    assert len(resp["rows"]) > 0
    # Every group should have non-null source
    for row in resp["rows"]:
        source, user_id, cnt = row
        assert source is not None, f"NULL source for {cnt} facts"
        assert user_id is not None, f"NULL user_id for {cnt} facts"


@pytest.mark.integration
@analytics_available
def test_documents_provenance_in_postgres(db: DatabaseManager) -> None:
    """Documents in PostgreSQL have non-NULL source and user_id."""
    push_to_analytics_stack(db, ANALYTICS_URL)

    resp = _sql(
        "SELECT source, user_id, count(*) as cnt "
        "FROM datasets.alibi_documents "
        "GROUP BY source, user_id "
        "ORDER BY cnt DESC"
    )
    assert len(resp["rows"]) > 0
    for row in resp["rows"]:
        source, user_id, cnt = row
        assert source is not None, f"NULL source for {cnt} documents"


@pytest.mark.integration
@analytics_available
def test_idempotent_reexport(db: DatabaseManager) -> None:
    """Exporting twice doesn't double the row count (full-replace semantics)."""
    push_to_analytics_stack(db, ANALYTICS_URL)
    resp1 = _sql("SELECT count(*) FROM datasets.alibi_facts")
    count1 = resp1["rows"][0][0]

    push_to_analytics_stack(db, ANALYTICS_URL)
    resp2 = _sql("SELECT count(*) FROM datasets.alibi_facts")
    count2 = resp2["rows"][0][0]

    assert count1 == count2, f"Re-export changed count: {count1} -> {count2}"
