"""Tests for fact CLI commands and API endpoints.

Tests the exposed interfaces for fact inspection and correction:
- CLI: `lt facts list`, `lt facts inspect`, `lt facts clouds`, etc.
- API: GET /api/v1/facts, POST /api/v1/facts/move-bundle, etc.
"""

from collections.abc import Generator
from datetime import date
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from alibi.api.app import create_app
from alibi.api.deps import get_database
from alibi.db import v2_store
from alibi.db.connection import DatabaseManager
from alibi.db.models import (
    Atom,
    AtomType,
    Bundle,
    BundleAtom,
    BundleAtomRole,
    BundleType,
    Cloud,
    CloudBundle,
    CloudMatchType,
    CloudStatus,
    Document,
    Fact,
    FactItem,
    FactStatus,
    FactType,
    TaxType,
    UnitType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_db(db: DatabaseManager) -> dict[str, object]:
    """Seed DB with a fact/cloud/bundle/atom structure."""
    # Create user (required for API auth)
    db.execute(
        "INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)",
        ("user-1", "Test User"),
    )
    db.get_connection().commit()

    doc_id = str(uuid4())
    doc = Document(
        id=doc_id,
        file_path="receipt.jpg",
        file_hash="abc123",
        raw_extraction={"vendor": "Test Store"},
    )
    v2_store.store_document(db, doc)

    atoms = [
        Atom(
            id=str(uuid4()),
            document_id=doc_id,
            atom_type=AtomType.VENDOR,
            data={"name": "Test Store"},
        ),
        Atom(
            id=str(uuid4()),
            document_id=doc_id,
            atom_type=AtomType.AMOUNT,
            data={"value": 25.50, "currency": "EUR", "semantic_type": "total"},
        ),
        Atom(
            id=str(uuid4()),
            document_id=doc_id,
            atom_type=AtomType.ITEM,
            data={"name": "Milk", "quantity": 2, "unit": "pcs", "total_price": 3.0},
        ),
    ]
    v2_store.store_atoms(db, atoms)

    bundle_id = str(uuid4())
    bundle = Bundle(id=bundle_id, document_id=doc_id, bundle_type=BundleType.BASKET)
    bundle_atoms = [
        BundleAtom(
            bundle_id=bundle_id, atom_id=atoms[0].id, role=BundleAtomRole.VENDOR_INFO
        ),
        BundleAtom(bundle_id=bundle_id, atom_id=atoms[1].id, role=BundleAtomRole.TOTAL),
        BundleAtom(
            bundle_id=bundle_id, atom_id=atoms[2].id, role=BundleAtomRole.BASKET_ITEM
        ),
    ]
    v2_store.store_bundle(db, bundle, bundle_atoms)

    cloud_id = str(uuid4())
    cloud = Cloud(id=cloud_id, status=CloudStatus.COLLAPSED)
    link = CloudBundle(
        cloud_id=cloud_id,
        bundle_id=bundle_id,
        match_type=CloudMatchType.EXACT_AMOUNT,
        match_confidence=Decimal("1.0"),
    )
    v2_store.store_cloud(db, cloud, link)

    fact_id = str(uuid4())
    fact = Fact(
        id=fact_id,
        cloud_id=cloud_id,
        fact_type=FactType.PURCHASE,
        vendor="Test Store",
        total_amount=Decimal("25.50"),
        currency="EUR",
        event_date=date(2025, 3, 15),
        status=FactStatus.CONFIRMED,
    )
    item = FactItem(
        id=str(uuid4()),
        fact_id=fact_id,
        atom_id=atoms[2].id,
        name="Milk",
        quantity=Decimal("2"),
        unit=UnitType.PIECE,
        total_price=Decimal("3.0"),
        tax_type=TaxType.VAT,
    )
    v2_store.store_fact(db, fact, [item])

    return {
        "doc_id": doc_id,
        "atoms": atoms,
        "bundle_id": bundle_id,
        "cloud_id": cloud_id,
        "fact_id": fact_id,
    }


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def client(db: DatabaseManager) -> Generator[TestClient, None, None]:
    """Create a test client with DB dependency override."""
    app = create_app()

    def override_get_database() -> DatabaseManager:
        return db

    app.dependency_overrides[get_database] = override_get_database
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLIFactsList:
    """Tests for `lt facts list`."""

    def test_list_empty(self, runner: CliRunner, db: DatabaseManager):
        """Empty DB shows no facts."""
        from alibi.cli import cli

        import alibi.commands.facts as cli_module

        original = cli_module.get_db  # type: ignore[attr-defined]
        cli_module.get_db = lambda: db  # type: ignore[attr-defined]
        try:
            result = runner.invoke(cli, ["facts", "list"])
            assert result.exit_code == 0
        finally:
            cli_module.get_db = original  # type: ignore[attr-defined]

    def test_list_with_facts(
        self,
        runner: CliRunner,
        db: DatabaseManager,
        seeded_db: dict[str, Any],
    ):
        """Shows facts in a table."""
        from alibi.cli import cli

        import alibi.commands.facts as cli_module

        original = cli_module.get_db  # type: ignore[attr-defined]
        cli_module.get_db = lambda: db  # type: ignore[attr-defined]
        try:
            result = runner.invoke(cli, ["facts", "list"])
            assert result.exit_code == 0
            assert "Test Store" in result.output
            assert "purchase" in result.output
        finally:
            cli_module.get_db = original  # type: ignore[attr-defined]


class TestCLIFactsInspect:
    """Tests for `lt facts inspect`."""

    def test_inspect_shows_tree(
        self,
        runner: CliRunner,
        db: DatabaseManager,
        seeded_db: dict[str, Any],
    ):
        """Inspect displays fact + cloud + bundle + atom tree."""
        from alibi.cli import cli

        import alibi.commands.facts as cli_module

        original = cli_module.get_db  # type: ignore[attr-defined]
        cli_module.get_db = lambda: db  # type: ignore[attr-defined]
        try:
            result = runner.invoke(cli, ["facts", "inspect", seeded_db["fact_id"]])
            assert result.exit_code == 0
            assert "Test Store" in result.output
            assert "Milk" in result.output
            assert "receipt.jpg" in result.output
        finally:
            cli_module.get_db = original  # type: ignore[attr-defined]

    def test_inspect_prefix(
        self,
        runner: CliRunner,
        db: DatabaseManager,
        seeded_db: dict[str, Any],
    ):
        """Inspect works with ID prefix."""
        from alibi.cli import cli

        import alibi.commands.facts as cli_module

        original = cli_module.get_db  # type: ignore[attr-defined]
        cli_module.get_db = lambda: db  # type: ignore[attr-defined]
        try:
            prefix = seeded_db["fact_id"][:8]
            result = runner.invoke(cli, ["facts", "inspect", prefix])
            assert result.exit_code == 0
            assert "Test Store" in result.output
        finally:
            cli_module.get_db = original  # type: ignore[attr-defined]


class TestCLIFactsClouds:
    """Tests for `lt facts clouds`."""

    def test_clouds_list(
        self, runner: CliRunner, db: DatabaseManager, seeded_db: dict[str, Any]
    ):
        """Lists clouds with summary."""
        from alibi.cli import cli

        import alibi.commands.facts as cli_module

        original = cli_module.get_db  # type: ignore[attr-defined]
        cli_module.get_db = lambda: db  # type: ignore[attr-defined]
        try:
            result = runner.invoke(cli, ["facts", "clouds"])
            assert result.exit_code == 0
            assert "collapsed" in result.output or "Clouds" in result.output
        finally:
            cli_module.get_db = original  # type: ignore[attr-defined]


class TestCLIFactsUnassigned:
    """Tests for `lt facts unassigned`."""

    def test_no_unassigned(
        self,
        runner: CliRunner,
        db: DatabaseManager,
        seeded_db: dict[str, Any],
    ):
        """No unassigned bundles shows success."""
        from alibi.cli import cli

        import alibi.commands.facts as cli_module

        original = cli_module.get_db  # type: ignore[attr-defined]
        cli_module.get_db = lambda: db  # type: ignore[attr-defined]
        try:
            result = runner.invoke(cli, ["facts", "unassigned"])
            assert result.exit_code == 0
            assert "No unassigned" in result.output
        finally:
            cli_module.get_db = original  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


class TestAPIListFacts:
    """Tests for GET /api/v1/facts."""

    def test_list_empty(self, client: TestClient):
        resp = client.get("/api/v1/facts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_list_with_facts(self, client: TestClient, seeded_db: dict[str, Any]):
        resp = client.get("/api/v1/facts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["vendor"] == "Test Store"

    def test_filter_by_vendor(self, client: TestClient, seeded_db: dict[str, Any]):
        resp = client.get("/api/v1/facts?vendor=Test")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

        resp = client.get("/api/v1/facts?vendor=Nonexistent")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


class TestAPIInspectFact:
    """Tests for GET /api/v1/facts/{id}/inspect."""

    def test_inspect(self, client: TestClient, seeded_db: dict[str, Any]):
        resp = client.get(f"/api/v1/facts/{seeded_db['fact_id']}/inspect")
        assert resp.status_code == 200
        data = resp.json()
        assert data["fact"]["vendor"] == "Test Store"
        assert len(data["bundles"]) == 1
        assert len(data["items"]) == 1

    def test_inspect_not_found(self, client: TestClient, seeded_db: dict[str, Any]):
        resp = client.get("/api/v1/facts/nonexistent/inspect")
        assert resp.status_code == 404


class TestAPIListClouds:
    """Tests for GET /api/v1/facts/clouds."""

    def test_list_clouds(self, client: TestClient, seeded_db: dict[str, Any]):
        resp = client.get("/api/v1/facts/clouds")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1

    def test_filter_by_status(self, client: TestClient, seeded_db: dict[str, Any]):
        resp = client.get("/api/v1/facts/clouds?status=collapsed")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

        resp = client.get("/api/v1/facts/clouds?status=forming")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


class TestAPIUnassigned:
    """Tests for GET /api/v1/facts/unassigned."""

    def test_no_unassigned(self, client: TestClient, seeded_db: dict[str, Any]):
        resp = client.get("/api/v1/facts/unassigned")
        assert resp.status_code == 200
        assert resp.json() == []


class TestAPIMoveBundle:
    """Tests for POST /api/v1/facts/move-bundle."""

    def test_move_to_new_cloud(self, client: TestClient, seeded_db: dict[str, Any]):
        resp = client.post(
            "/api/v1/facts/move-bundle",
            json={"bundle_id": seeded_db["bundle_id"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"]
        assert data["target_cloud_id"] is not None

    def test_move_nonexistent(self, client: TestClient, seeded_db: dict[str, Any]):
        resp = client.post(
            "/api/v1/facts/move-bundle",
            json={"bundle_id": "nonexistent"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert not data["success"]


class TestAPISetCloud:
    """Tests for POST /api/v1/facts/set-cloud."""

    def test_detach_bundle(self, client: TestClient, seeded_db: dict[str, Any]):
        resp = client.post(
            "/api/v1/facts/set-cloud",
            json={"bundle_id": seeded_db["bundle_id"], "cloud_id": None},
        )
        assert resp.status_code == 200
        assert resp.json()["success"]

    def test_set_nonexistent_bundle(
        self, client: TestClient, seeded_db: dict[str, Any]
    ):
        resp = client.post(
            "/api/v1/facts/set-cloud",
            json={"bundle_id": "nonexistent"},
        )
        assert resp.status_code == 404


class TestAPIRecollapse:
    """Tests for POST /api/v1/facts/clouds/{id}/recollapse."""

    def test_recollapse(self, client: TestClient, seeded_db: dict[str, Any]):
        resp = client.post(f"/api/v1/facts/clouds/{seeded_db['cloud_id']}/recollapse")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"]
        assert data["fact_id"] is not None


class TestAPIDispute:
    """Tests for POST /api/v1/facts/clouds/{id}/dispute."""

    def test_dispute(self, client: TestClient, seeded_db: dict[str, Any]):
        resp = client.post(f"/api/v1/facts/clouds/{seeded_db['cloud_id']}/dispute")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"]
        assert data["status"] == "disputed"
