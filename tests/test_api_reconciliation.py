"""Tests for the reconciliation API endpoint."""

from __future__ import annotations

import json
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from alibi.api.app import create_app
from alibi.api.deps import get_database
from alibi.db.connection import DatabaseManager


@pytest.fixture
def client(db_manager: DatabaseManager) -> Generator[TestClient, None, None]:
    app = create_app()
    app.dependency_overrides[get_database] = lambda: db_manager
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _seed(db: DatabaseManager) -> None:
    """Seed four facts, one per coverage class."""
    conn = db.get_connection()
    conn.execute(
        "INSERT INTO documents (id, file_path, file_hash) VALUES "
        "('doc-r', '/r.jpg', 'h-r')"
    )

    def fact(fid, vendor, total, payments=None, country="CY"):
        conn.execute("INSERT INTO clouds (id, status) VALUES (?, 'collapsed')", (fid,))
        conn.execute(
            "INSERT INTO facts (id, cloud_id, fact_type, vendor, total_amount, "
            "currency, country, event_date, status, payments) "
            "VALUES (?, ?, 'purchase', ?, ?, 'EUR', ?, '2026-02-01', 'confirmed', ?)",
            (fid, fid, vendor, total, country, payments),
        )

    def item(iid, fid, total_price):
        conn.execute(
            "INSERT INTO atoms (id, document_id, atom_type, data) "
            "VALUES (?, 'doc-r', 'item', '{}')",
            (iid,),
        )
        conn.execute(
            "INSERT INTO fact_items (id, fact_id, atom_id, name, total_price) "
            "VALUES (?, ?, ?, 'X', ?)",
            (iid, fid, iid, total_price),
        )

    # matched: items + payment, amounts agree
    fact("m", "LIDL", "10.00", payments=json.dumps([{"amount": "10.00"}]))
    item("m-i1", "m", "6.00")
    item("m-i2", "m", "4.00")
    # items_only: receipt, no payment
    fact("io", "PAPAS", "5.00")
    item("io-i1", "io", "5.00")
    # payment_only: card slip, no items
    fact("po", "LIDL", "20.00", payments=json.dumps([{"amount": "20.00"}]))
    # empty: neither layer
    fact("e", "GHOST", "0.00")
    conn.commit()


class TestReconciliationEndpoint:
    def test_summary_classifies_all(self, client, db_manager):
        _seed(db_manager)
        resp = client.get("/api/v1/reconciliation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"] == {
            "matched": 1,
            "items_only": 1,
            "payment_only": 1,
            "empty": 1,
            "total": 4,
        }

    def test_payment_amount_and_coverage_fields(self, client, db_manager):
        _seed(db_manager)
        data = client.get("/api/v1/reconciliation").json()
        by_id = {t["id"]: t for t in data["transactions"]}
        assert by_id["m"]["coverage"] == "matched"
        assert by_id["m"]["payment_amount"] == 10.0
        assert by_id["m"]["items_sum"] == 10.0
        assert by_id["po"]["coverage"] == "payment_only"
        assert by_id["po"]["payment_amount"] == 20.0
        assert by_id["po"]["n_items"] == 0

    def test_coverage_worklist_filter(self, client, db_manager):
        _seed(db_manager)
        data = client.get("/api/v1/reconciliation?coverage=payment_only").json()
        assert [t["id"] for t in data["transactions"]] == ["po"]
        assert data["summary"]["total"] == 1

    def test_vendor_filter_reuses_a_axis(self, client, db_manager):
        _seed(db_manager)
        data = client.get("/api/v1/reconciliation?vendor=lidl").json()
        ids = {t["id"] for t in data["transactions"]}
        assert ids == {"m", "po"}

    def test_invalid_coverage_rejected(self, client, db_manager):
        _seed(db_manager)
        resp = client.get("/api/v1/reconciliation?coverage=bogus")
        assert resp.status_code == 422
