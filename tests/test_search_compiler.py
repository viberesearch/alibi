"""Tests for Search-as-Code search-plan compilation (services/search_compiler.py).

Pure tests (no live LLM) cover grounding / validation / deterministic translation
against a synthetic facet registry and a stub llm_fn. One integration test uses the
real SQLite ``db`` fixture (FK toggled off for direct inserts) to prove end-to-end
filtering and injection-safe parameter binding.
"""

from __future__ import annotations

import json

import pytest

from alibi.services import search_compiler as sc


def _ctx() -> dict:
    return {
        "schema_version": "0.1",
        "facets": {
            "fact_types": ["purchase", "refund"],
            "vendors": ["Alphamega", "Public", "Shell"],
            "categories": ["Beverages", "Dairy", "Transport"],
            "date_min": "2026-01-01",
            "date_max": "2026-06-01",
            "amount_min": 1.0,
            "amount_max": 500.0,
            "count": 3,
        },
    }


def _stub(plan: dict):
    state = {"calls": 0}

    def fn(system: str, user: str) -> str:
        state["calls"] += 1
        return json.dumps(plan)

    fn.state = state  # type: ignore[attr-defined]
    return fn


# --- deterministic translation ------------------------------------------------


def test_plan_to_search_params_deterministic():
    plan = {
        "schema_version": "0.1",
        "query_text": "milk",
        "filters": {
            "vendor": "Alphamega",
            "fact_type": "purchase",
            "category": "Dairy",
            "date_from": "2026-01-01",
            "date_to": "2026-03-31",
            "min_amount": 5,
            "max_amount": 100,
        },
        "limit": 25,
    }
    params = sc.plan_to_search_params(plan)
    assert params == {
        "filters": {
            "vendor": "Alphamega",
            "fact_type": "purchase",
            "date_from": "2026-01-01",
            "date_to": "2026-03-31",
            "min_amount": 5.0,
            "max_amount": 100.0,
        },
        "category": "Dairy",
        "query_text": "milk",
        "limit": 25,
    }
    assert sc.plan_to_search_params(plan) == params  # deterministic


# --- grounding: unknown facet values are flagged, never silently mapped --------


def test_grounding_flags_hallucinated_fact_type():
    ctx = _ctx()
    hallucinated = {
        "schema_version": "0.1",
        "query_text": "leases",
        "filters": {"fact_type": "lease"},  # not a real fact_type
        "limit": 10,
        "flags": [],
    }
    fn = _stub(hallucinated)
    plan = sc.compile_nl_to_plan("lease payments", ctx, llm_fn=fn)
    assert fn.state["calls"] == 1
    assert any("lease" in f and "fact_type" in f for f in plan["flags"])


def test_grounding_flags_hallucinated_category():
    flags = sc.validate_plan(
        {"schema_version": "0.1", "filters": {"category": "Electronics"}}, _ctx()
    )
    assert any("Electronics" in f for f in flags)


def test_validate_raises_on_bad_date():
    with pytest.raises(ValueError, match="date_from"):
        sc.validate_plan(
            {"schema_version": "0.1", "filters": {"date_from": "nope"}}, _ctx()
        )


def test_validate_raises_on_bad_amount():
    with pytest.raises(ValueError, match="min_amount"):
        sc.validate_plan(
            {"schema_version": "0.1", "filters": {"min_amount": "lots"}}, _ctx()
        )


def test_validate_raises_on_bad_limit():
    with pytest.raises(ValueError, match="limit"):
        sc.validate_plan({"schema_version": "0.1", "limit": 0}, _ctx())


# --- reproducibility ----------------------------------------------------------


def test_reproducibility_zero_llm_on_replan():
    ctx = _ctx()
    valid = {
        "schema_version": "0.1",
        "query_text": "groceries",
        "filters": {"fact_type": "purchase"},
        "limit": 5,
        "flags": [],
    }
    fn = _stub(valid)
    plan = sc.compile_nl_to_plan("grocery purchases", ctx, llm_fn=fn)
    assert fn.state["calls"] == 1
    p1 = sc.plan_to_search_params(plan)
    p2 = sc.plan_to_search_params(plan)
    assert p1 == p2
    assert fn.state["calls"] == 1  # re-translation touched no LLM


# --- integration against real SQLite -----------------------------------------


def _seed(db) -> None:
    """Insert a couple of facts + items directly (FK off to skip the atom chain)."""
    db.execute("PRAGMA foreign_keys = OFF")
    db.execute(
        "INSERT INTO facts (id, cloud_id, fact_type, vendor, total_amount, "
        "currency, event_date, status) VALUES "
        "('f1','c1','purchase','Alphamega',42.50,'EUR','2026-02-01','confirmed'),"
        "('f2','c2','refund','Shell',12.00,'EUR','2026-03-01','confirmed')"
    )
    db.execute(
        "INSERT INTO fact_items (id, fact_id, atom_id, name, category) VALUES "
        "('i1','f1','a1','Milk 1L','Dairy'),"
        "('i2','f2','a2','Fuel','Transport')"
    )
    db.get_connection().commit()


def test_build_context_and_execute_end_to_end(db):
    _seed(db)
    ctx = sc.build_search_context(db)
    assert "purchase" in ctx["facets"]["fact_types"]
    assert "Dairy" in ctx["facets"]["categories"]
    assert "Alphamega" in ctx["facets"]["vendors"]

    plan = {
        "schema_version": "0.1",
        "query_text": "",
        "filters": {"fact_type": "purchase", "category": "Dairy"},
        "limit": 10,
        "flags": [],
    }
    result = sc.execute_plan(plan, db, context=ctx)
    assert not result["flags"]
    ids = [f["id"] for f in result["facts"]]
    assert ids == ["f1"]  # the refund (Shell/Transport) is filtered out


def test_execute_injection_safe(db):
    _seed(db)
    # A malicious vendor value is bound as a parameter, not interpolated -> no error,
    # simply matches nothing.
    plan = {
        "schema_version": "0.1",
        "query_text": "",
        "filters": {"vendor": "x' OR '1'='1"},
        "limit": 10,
        "flags": [],
    }
    result = sc.execute_plan(plan, db)
    assert result["total"] == 0


# --- live-path client timeout (env-configurable, no live HTTP) -----------------


class _FakeResp:
    @staticmethod
    def raise_for_status() -> None:
        return None

    @staticmethod
    def json() -> dict:
        return {"message": {"content": "{}"}}


class _CaptureClient:
    """Stub httpx.Client that records the timeout instead of opening a socket."""

    captured: dict = {}

    def __init__(self, *, timeout):
        _CaptureClient.captured["timeout"] = timeout

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json):  # noqa: A002 - mirror httpx.Client signature
        return _FakeResp()


def _run_default_llm_fn(monkeypatch) -> float:
    import httpx

    _CaptureClient.captured = {}
    monkeypatch.setattr(httpx, "Client", _CaptureClient)
    sc._default_llm_fn("sys", "user")
    return _CaptureClient.captured["timeout"]


def test_structure_timeout_default_is_600(monkeypatch):
    monkeypatch.delenv("ALIBI_STRUCTURE_TIMEOUT", raising=False)
    assert _run_default_llm_fn(monkeypatch) == 600.0


def test_structure_timeout_env_override(monkeypatch):
    monkeypatch.setenv("ALIBI_STRUCTURE_TIMEOUT", "42")
    assert _run_default_llm_fn(monkeypatch) == 42.0
