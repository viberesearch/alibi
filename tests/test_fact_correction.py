"""Tests for fact inspection and correction system.

Tests the ability to:
1. Inspect facts — drill down into cloud/bundle/atom composition
2. Move bundles between clouds — fix wrong cloud formation
3. Re-collapse clouds after corrections
4. Handle edge cases (empty clouds, disputed status)
"""

from datetime import date
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from alibi.clouds.collapse import try_collapse
from alibi.clouds.correction import (
    CorrectionResult,
    mark_disputed,
    move_bundle,
    recollapse_cloud,
)
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
# Helpers to build test data
# ---------------------------------------------------------------------------


def _make_document(db: DatabaseManager, file_path: str = "test.jpg") -> str:
    """Create and store a test document, return its ID."""
    doc_id = str(uuid4())
    doc = Document(
        id=doc_id,
        file_path=file_path,
        file_hash=str(uuid4())[:16],
        raw_extraction={"vendor": "Test Store"},
    )
    v2_store.store_document(db, doc)
    return doc_id


def _make_atoms(
    db: DatabaseManager,
    document_id: str,
    vendor: str = "Test Store",
    amount: float = 25.50,
    item_names: list[str] | None = None,
) -> list[Atom]:
    """Create and store atoms for a document."""
    atoms = [
        Atom(
            id=str(uuid4()),
            document_id=document_id,
            atom_type=AtomType.VENDOR,
            data={"name": vendor},
        ),
        Atom(
            id=str(uuid4()),
            document_id=document_id,
            atom_type=AtomType.AMOUNT,
            data={"value": amount, "currency": "EUR", "semantic_type": "total"},
        ),
        Atom(
            id=str(uuid4()),
            document_id=document_id,
            atom_type=AtomType.DATETIME,
            data={"value": "2025-03-15T10:30:00"},
        ),
    ]
    for name in item_names or []:
        atoms.append(
            Atom(
                id=str(uuid4()),
                document_id=document_id,
                atom_type=AtomType.ITEM,
                data={
                    "name": name,
                    "quantity": 1,
                    "unit": "pcs",
                    "total_price": 5.0,
                },
            )
        )
    v2_store.store_atoms(db, atoms)
    return atoms


def _make_bundle(
    db: DatabaseManager,
    document_id: str,
    atoms: list[Atom],
    bundle_type: BundleType = BundleType.BASKET,
) -> str:
    """Create and store a bundle with atom links, return bundle ID."""
    bundle_id = str(uuid4())
    bundle = Bundle(id=bundle_id, document_id=document_id, bundle_type=bundle_type)

    role_map = {
        AtomType.VENDOR: BundleAtomRole.VENDOR_INFO,
        AtomType.AMOUNT: BundleAtomRole.TOTAL,
        AtomType.DATETIME: BundleAtomRole.EVENT_TIME,
        AtomType.ITEM: BundleAtomRole.BASKET_ITEM,
        AtomType.PAYMENT: BundleAtomRole.PAYMENT_INFO,
    }

    bundle_atoms = [
        BundleAtom(
            bundle_id=bundle_id,
            atom_id=a.id,
            role=role_map.get(a.atom_type, BundleAtomRole.BASKET_ITEM),
        )
        for a in atoms
    ]
    v2_store.store_bundle(db, bundle, bundle_atoms)
    return bundle_id


def _make_cloud_with_bundle(
    db: DatabaseManager,
    bundle_id: str,
    status: CloudStatus = CloudStatus.COLLAPSED,
) -> str:
    """Create a cloud containing one bundle, return cloud ID."""
    cloud_id = str(uuid4())
    cloud = Cloud(id=cloud_id, status=status)
    link = CloudBundle(
        cloud_id=cloud_id,
        bundle_id=bundle_id,
        match_type=CloudMatchType.EXACT_AMOUNT,
        match_confidence=Decimal("1.0"),
    )
    v2_store.store_cloud(db, cloud, link)
    return cloud_id


def _make_fact(
    db: DatabaseManager,
    cloud_id: str,
    vendor: str = "Test Store",
    amount: float = 25.50,
    items: list[tuple[str, str]] | None = None,
) -> str:
    """Create a fact for a cloud, return fact ID."""
    fact_id = str(uuid4())
    fact = Fact(
        id=fact_id,
        cloud_id=cloud_id,
        fact_type=FactType.PURCHASE,
        vendor=vendor,
        total_amount=Decimal(str(amount)),
        currency="EUR",
        event_date=date(2025, 3, 15),
        status=FactStatus.CONFIRMED,
    )
    fact_items = []
    for name, atom_id in items or []:
        fact_items.append(
            FactItem(
                id=str(uuid4()),
                fact_id=fact_id,
                atom_id=atom_id,
                name=name,
                quantity=Decimal("1"),
                unit=UnitType.PIECE,
                total_price=Decimal("5.0"),
                tax_type=TaxType.VAT,
            )
        )
    v2_store.store_fact(db, fact, fact_items)
    return fact_id


def _setup_two_cloud_scenario(
    db: DatabaseManager,
) -> dict[str, Any]:
    """Set up a scenario with two clouds, each with one bundle and fact.

    Cloud A: "Store A", receipt with Milk and Bread
    Cloud B: "Store B", card slip (payment record)

    Returns dict with all IDs for assertions.
    """
    # Cloud A: receipt
    doc_a = _make_document(db, "receipt_a.jpg")
    atoms_a = _make_atoms(
        db, doc_a, vendor="Store A", amount=10.0, item_names=["Milk", "Bread"]
    )
    bundle_a = _make_bundle(db, doc_a, atoms_a, BundleType.BASKET)
    cloud_a = _make_cloud_with_bundle(db, bundle_a)
    item_atoms_a = [a for a in atoms_a if a.atom_type == AtomType.ITEM]
    fact_a = _make_fact(
        db,
        cloud_a,
        vendor="Store A",
        amount=10.0,
        items=[(a.data["name"], a.id) for a in item_atoms_a],
    )

    # Cloud B: card slip
    doc_b = _make_document(db, "slip_b.jpg")
    atoms_b = _make_atoms(db, doc_b, vendor="Store B", amount=15.0)
    bundle_b = _make_bundle(db, doc_b, atoms_b, BundleType.PAYMENT_RECORD)
    cloud_b = _make_cloud_with_bundle(db, bundle_b)
    fact_b = _make_fact(db, cloud_b, vendor="Store B", amount=15.0)

    return {
        "doc_a": doc_a,
        "atoms_a": atoms_a,
        "bundle_a": bundle_a,
        "cloud_a": cloud_a,
        "fact_a": fact_a,
        "doc_b": doc_b,
        "atoms_b": atoms_b,
        "bundle_b": bundle_b,
        "cloud_b": cloud_b,
        "fact_b": fact_b,
    }


# ---------------------------------------------------------------------------
# Test: Fact inspection
# ---------------------------------------------------------------------------


class TestInspectFact:
    """Tests for inspect_fact() deep drill-down."""

    def test_inspect_returns_full_tree(self, db: DatabaseManager):
        """Inspect fact shows cloud, bundles, atoms, items, document."""
        doc_id = _make_document(db, "receipt.jpg")
        atoms = _make_atoms(db, doc_id, item_names=["Milk", "Bread"])
        bundle_id = _make_bundle(db, doc_id, atoms)
        cloud_id = _make_cloud_with_bundle(db, bundle_id)
        item_atoms = [a for a in atoms if a.atom_type == AtomType.ITEM]
        fact_id = _make_fact(
            db,
            cloud_id,
            items=[(a.data["name"], a.id) for a in item_atoms],
        )

        result = v2_store.inspect_fact(db, fact_id)

        assert result is not None
        assert result["fact"]["id"] == fact_id
        assert result["fact"]["vendor"] == "Test Store"
        assert result["cloud"]["id"] == cloud_id
        assert len(result["bundles"]) == 1
        assert result["bundles"][0]["id"] == bundle_id
        assert result["bundles"][0]["document"]["file_path"] == "receipt.jpg"
        assert len(result["bundles"][0]["atoms"]) == len(atoms)
        assert len(result["items"]) == 2

    def test_inspect_nonexistent_fact(self, db: DatabaseManager):
        """Inspect returns None for nonexistent fact."""
        assert v2_store.inspect_fact(db, "nonexistent") is None

    def test_inspect_multi_bundle_fact(self, db: DatabaseManager):
        """Inspect fact with multiple bundles from different documents."""
        doc_a = _make_document(db, "receipt.jpg")
        atoms_a = _make_atoms(db, doc_a, item_names=["Milk"])
        bundle_a = _make_bundle(db, doc_a, atoms_a)

        doc_b = _make_document(db, "card_slip.jpg")
        atoms_b = _make_atoms(db, doc_b)
        bundle_b = _make_bundle(db, doc_b, atoms_b, BundleType.PAYMENT_RECORD)

        # Create cloud with both bundles
        cloud_id = _make_cloud_with_bundle(db, bundle_a)
        v2_store.add_cloud_bundle(
            db,
            CloudBundle(
                cloud_id=cloud_id,
                bundle_id=bundle_b,
                match_type=CloudMatchType.EXACT_AMOUNT,
                match_confidence=Decimal("0.9"),
            ),
        )
        fact_id = _make_fact(db, cloud_id)

        result = v2_store.inspect_fact(db, fact_id)
        assert result is not None
        assert len(result["bundles"]) == 2

        file_paths = {b["document"]["file_path"] for b in result["bundles"]}
        assert file_paths == {"receipt.jpg", "card_slip.jpg"}

    def test_inspect_shows_atom_details(self, db: DatabaseManager):
        """Atoms include type, role, data, confidence."""
        doc_id = _make_document(db)
        atoms = _make_atoms(db, doc_id)
        bundle_id = _make_bundle(db, doc_id, atoms)
        cloud_id = _make_cloud_with_bundle(db, bundle_id)
        fact_id = _make_fact(db, cloud_id)

        result = v2_store.inspect_fact(db, fact_id)
        assert result is not None
        atom_data = result["bundles"][0]["atoms"]

        types = {a["atom_type"] for a in atom_data}
        assert "vendor" in types
        assert "amount" in types
        assert "datetime" in types

        vendor_atom = next(a for a in atom_data if a["atom_type"] == "vendor")
        assert vendor_atom["data"]["name"] == "Test Store"
        assert vendor_atom["role"] == "vendor_info"


# ---------------------------------------------------------------------------
# Test: List clouds
# ---------------------------------------------------------------------------


class TestListClouds:
    """Tests for list_clouds() summary view."""

    def test_list_all_clouds(self, db: DatabaseManager):
        """List returns all clouds with summary."""
        doc_id = _make_document(db)
        atoms = _make_atoms(db, doc_id)
        bundle_id = _make_bundle(db, doc_id, atoms)
        cloud_id = _make_cloud_with_bundle(db, bundle_id)
        fact_id = _make_fact(db, cloud_id)

        clouds = v2_store.list_clouds(db)
        assert len(clouds) == 1
        assert clouds[0]["id"] == cloud_id
        assert clouds[0]["bundle_count"] == 1
        assert clouds[0]["fact_id"] == fact_id

    def test_filter_by_status(self, db: DatabaseManager):
        """Filter clouds by status."""
        doc1 = _make_document(db)
        atoms1 = _make_atoms(db, doc1)
        b1 = _make_bundle(db, doc1, atoms1)
        c1 = _make_cloud_with_bundle(db, b1, CloudStatus.COLLAPSED)

        doc2 = _make_document(db)
        atoms2 = _make_atoms(db, doc2)
        b2 = _make_bundle(db, doc2, atoms2)
        c2 = _make_cloud_with_bundle(db, b2, CloudStatus.FORMING)

        collapsed = v2_store.list_clouds(db, status="collapsed")
        assert len(collapsed) == 1
        assert collapsed[0]["id"] == c1

        forming = v2_store.list_clouds(db, status="forming")
        assert len(forming) == 1
        assert forming[0]["id"] == c2


# ---------------------------------------------------------------------------
# Test: Move bundle to existing cloud
# ---------------------------------------------------------------------------


class TestMoveBundleToExistingCloud:
    """Tests for moving a bundle from one cloud to another."""

    def test_move_bundle_reassigns_cloud(self, db: DatabaseManager):
        """Moving a bundle changes its cloud_bundles link."""
        data = _setup_two_cloud_scenario(db)

        result = move_bundle(db, data["bundle_b"], target_cloud_id=data["cloud_a"])

        assert result.success
        assert result.source_cloud_id == data["cloud_b"]
        assert result.target_cloud_id == data["cloud_a"]

        # Bundle B is now in Cloud A
        new_cloud = v2_store.get_cloud_for_bundle(db, data["bundle_b"])
        assert new_cloud == data["cloud_a"]

        # Cloud A now has 2 bundles
        bundles_in_a = v2_store.get_bundles_in_cloud(db, data["cloud_a"])
        assert len(bundles_in_a) == 2

    def test_move_bundle_deletes_source_and_target_facts(self, db: DatabaseManager):
        """Both source and target facts are deleted before re-collapse."""
        data = _setup_two_cloud_scenario(db)

        # Both facts exist before
        assert v2_store.get_fact_by_id(db, data["fact_a"]) is not None
        assert v2_store.get_fact_by_id(db, data["fact_b"]) is not None

        result = move_bundle(db, data["bundle_b"], target_cloud_id=data["cloud_a"])
        assert result.success

        # Original facts are gone (deleted before re-collapse)
        assert v2_store.get_fact_by_id(db, data["fact_a"]) is None
        assert v2_store.get_fact_by_id(db, data["fact_b"]) is None

    def test_move_same_vendor_recollapses(self, db: DatabaseManager):
        """Moving a same-vendor bundle produces a new collapsed fact."""
        # Cloud A: receipt from Store A
        doc_a = _make_document(db, "receipt_a.jpg")
        atoms_a = _make_atoms(
            db, doc_a, vendor="Store A", amount=10.0, item_names=["Milk"]
        )
        bundle_a = _make_bundle(db, doc_a, atoms_a, BundleType.BASKET)
        cloud_a = _make_cloud_with_bundle(db, bundle_a)
        _make_fact(db, cloud_a, vendor="Store A", amount=10.0)

        # Cloud B: card slip from Store A (same vendor, same amount)
        doc_b = _make_document(db, "slip_a.jpg")
        atoms_b = _make_atoms(db, doc_b, vendor="Store A", amount=10.0)
        bundle_b = _make_bundle(db, doc_b, atoms_b, BundleType.PAYMENT_RECORD)
        cloud_b = _make_cloud_with_bundle(db, bundle_b)
        _make_fact(db, cloud_b, vendor="Store A", amount=10.0)

        result = move_bundle(db, bundle_b, target_cloud_id=cloud_a)
        assert result.success
        # Same vendor + same amount → high confidence → re-collapse succeeds
        assert result.target_fact_id is not None

    def test_move_cleans_up_empty_source_cloud(self, db: DatabaseManager):
        """If source cloud becomes empty after move, it's deleted."""
        data = _setup_two_cloud_scenario(db)

        result = move_bundle(db, data["bundle_b"], target_cloud_id=data["cloud_a"])
        assert result.success
        assert result.deleted_clouds >= 1

    def test_move_nonexistent_bundle_fails(self, db: DatabaseManager):
        """Moving a nonexistent bundle returns error."""
        result = move_bundle(db, "nonexistent", target_cloud_id="also-nonexistent")
        assert not result.success
        assert result.error is not None


# ---------------------------------------------------------------------------
# Test: Move bundle to new cloud
# ---------------------------------------------------------------------------


class TestMoveBundleToNewCloud:
    """Tests for moving a bundle to a brand new cloud."""

    def test_move_to_new_cloud_creates_cloud(self, db: DatabaseManager):
        """Moving to new cloud creates a fresh cloud."""
        data = _setup_two_cloud_scenario(db)

        # Add bundle_b to cloud_a (simulating wrong merge)
        v2_store.delete_fact(db, data["fact_a"])
        v2_store.delete_fact(db, data["fact_b"])
        v2_store.move_bundle_to_cloud(db, data["bundle_b"], data["cloud_a"])

        # Now move bundle_b back out to a new cloud
        result = move_bundle(db, data["bundle_b"], target_cloud_id=None)

        assert result.success
        assert result.target_cloud_id is not None
        assert result.target_cloud_id != data["cloud_a"]

        # Bundle B is in the new cloud
        assert (
            v2_store.get_cloud_for_bundle(db, data["bundle_b"])
            == result.target_cloud_id
        )

        # New cloud has a fact
        assert result.target_fact_id is not None

    def test_move_to_new_cloud_recollapses_source(self, db: DatabaseManager):
        """Source cloud is re-collapsed after bundle removal."""
        data = _setup_two_cloud_scenario(db)

        # Merge both bundles into cloud_a
        v2_store.delete_fact(db, data["fact_a"])
        v2_store.delete_fact(db, data["fact_b"])
        v2_store.move_bundle_to_cloud(db, data["bundle_b"], data["cloud_a"])

        # Verify: cloud_a has 2 bundles
        assert len(v2_store.get_bundles_in_cloud(db, data["cloud_a"])) == 2

        # Move bundle_b out
        result = move_bundle(db, data["bundle_b"], target_cloud_id=None)
        assert result.success

        # Source cloud (A) still has bundle_a and got a new fact
        assert result.source_fact_id is not None
        bundles_in_a = v2_store.get_bundles_in_cloud(db, data["cloud_a"])
        assert len(bundles_in_a) == 1
        assert bundles_in_a[0] == data["bundle_a"]


# ---------------------------------------------------------------------------
# Test: Recollapse cloud
# ---------------------------------------------------------------------------


class TestRecollapseCloud:
    """Tests for forced cloud re-collapse."""

    def test_recollapse_single_bundle(self, db: DatabaseManager):
        """Single-bundle cloud always re-collapses into a fact."""
        doc_id = _make_document(db)
        atoms = _make_atoms(db, doc_id, item_names=["Milk"])
        bundle_id = _make_bundle(db, doc_id, atoms)
        cloud_id = _make_cloud_with_bundle(db, bundle_id, CloudStatus.FORMING)

        fact_id = recollapse_cloud(db, cloud_id)
        assert fact_id is not None

        fact = v2_store.get_fact_by_id(db, fact_id)
        assert fact is not None
        assert fact["vendor"] == "Test Store"
        assert float(fact["total_amount"]) == 25.50

    def test_recollapse_replaces_existing_fact(self, db: DatabaseManager):
        """Re-collapse deletes old fact and creates new one."""
        doc_id = _make_document(db)
        atoms = _make_atoms(db, doc_id)
        bundle_id = _make_bundle(db, doc_id, atoms)
        cloud_id = _make_cloud_with_bundle(db, bundle_id)
        old_fact_id = _make_fact(db, cloud_id)

        new_fact_id = recollapse_cloud(db, cloud_id)
        assert new_fact_id is not None
        assert new_fact_id != old_fact_id

        # Old fact gone
        assert v2_store.get_fact_by_id(db, old_fact_id) is None
        # New fact exists
        assert v2_store.get_fact_by_id(db, new_fact_id) is not None

    def test_recollapse_empty_cloud_returns_none(self, db: DatabaseManager):
        """Re-collapsing a cloud with no bundles returns None."""
        cloud_id = str(uuid4())
        # Create empty cloud
        db.execute(
            "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
            (cloud_id, "forming", 0.0),
        )
        db.get_connection().commit()

        assert recollapse_cloud(db, cloud_id) is None


# ---------------------------------------------------------------------------
# Test: Mark disputed
# ---------------------------------------------------------------------------


class TestMarkDisputed:
    """Tests for marking a cloud as disputed."""

    def test_mark_disputed_deletes_fact(self, db: DatabaseManager):
        """Marking disputed removes the fact and sets cloud status."""
        doc_id = _make_document(db)
        atoms = _make_atoms(db, doc_id)
        bundle_id = _make_bundle(db, doc_id, atoms)
        cloud_id = _make_cloud_with_bundle(db, bundle_id)
        fact_id = _make_fact(db, cloud_id)

        assert mark_disputed(db, cloud_id)

        # Fact deleted
        assert v2_store.get_fact_by_id(db, fact_id) is None

        # Cloud status is disputed
        clouds = v2_store.list_clouds(db, status="disputed")
        assert len(clouds) == 1
        assert clouds[0]["id"] == cloud_id

    def test_mark_disputed_no_fact(self, db: DatabaseManager):
        """Marking disputed works even if no fact exists."""
        doc_id = _make_document(db)
        atoms = _make_atoms(db, doc_id)
        bundle_id = _make_bundle(db, doc_id, atoms)
        cloud_id = _make_cloud_with_bundle(db, bundle_id, CloudStatus.FORMING)

        assert mark_disputed(db, cloud_id)

        clouds = v2_store.list_clouds(db, status="disputed")
        assert len(clouds) == 1


# ---------------------------------------------------------------------------
# Test: Delete fact
# ---------------------------------------------------------------------------


class TestDeleteFact:
    """Tests for delete_fact() and cleanup."""

    def test_delete_fact_preserves_cloud(self, db: DatabaseManager):
        """Deleting fact keeps the cloud and bundles intact."""
        doc_id = _make_document(db)
        atoms = _make_atoms(db, doc_id)
        bundle_id = _make_bundle(db, doc_id, atoms)
        cloud_id = _make_cloud_with_bundle(db, bundle_id)
        fact_id = _make_fact(db, cloud_id, items=[("Milk", atoms[0].id)])

        assert v2_store.delete_fact(db, fact_id)

        # Fact and items gone
        assert v2_store.get_fact_by_id(db, fact_id) is None
        assert v2_store.get_fact_items(db, fact_id) == []

        # Cloud still has the bundle
        bundles = v2_store.get_bundles_in_cloud(db, cloud_id)
        assert len(bundles) == 1

        # Cloud reverted to forming
        clouds = v2_store.list_clouds(db, status="forming")
        assert any(c["id"] == cloud_id for c in clouds)

    def test_delete_nonexistent_fact(self, db: DatabaseManager):
        """Deleting nonexistent fact returns False."""
        assert not v2_store.delete_fact(db, "nonexistent")

    def test_delete_empty_clouds(self, db: DatabaseManager):
        """Orphaned clouds with no bundles get cleaned up."""
        # Create a cloud with no bundles
        cloud_id = str(uuid4())
        db.execute(
            "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
            (cloud_id, "forming", 0.0),
        )
        db.get_connection().commit()

        deleted = v2_store.delete_empty_clouds(db)
        assert deleted == 1

        # Cloud is gone
        clouds = v2_store.list_clouds(db)
        assert not any(c["id"] == cloud_id for c in clouds)


# ---------------------------------------------------------------------------
# Test: Low-level store helpers
# ---------------------------------------------------------------------------


class TestStoreHelpers:
    """Tests for move_bundle_to_cloud and move_bundle_to_new_cloud."""

    def test_move_bundle_to_cloud(self, db: DatabaseManager):
        """Low-level move changes cloud_bundles link."""
        data = _setup_two_cloud_scenario(db)

        ok = v2_store.move_bundle_to_cloud(db, data["bundle_b"], data["cloud_a"])
        assert ok

        assert v2_store.get_cloud_for_bundle(db, data["bundle_b"]) == data["cloud_a"]

    def test_move_bundle_to_nonexistent_cloud_fails(self, db: DatabaseManager):
        """Moving to a nonexistent cloud returns False."""
        data = _setup_two_cloud_scenario(db)
        assert not v2_store.move_bundle_to_cloud(db, data["bundle_b"], "nonexistent")

    def test_move_nonexistent_bundle_fails(self, db: DatabaseManager):
        """Moving a nonexistent bundle returns False."""
        data = _setup_two_cloud_scenario(db)
        assert not v2_store.move_bundle_to_cloud(db, "nonexistent", data["cloud_a"])

    def test_move_bundle_to_new_cloud(self, db: DatabaseManager):
        """Low-level move to new cloud creates a forming cloud."""
        data = _setup_two_cloud_scenario(db)

        new_cloud_id = v2_store.move_bundle_to_new_cloud(db, data["bundle_b"])
        assert new_cloud_id is not None
        assert new_cloud_id != data["cloud_b"]

        # Bundle is in the new cloud
        assert v2_store.get_cloud_for_bundle(db, data["bundle_b"]) == new_cloud_id

        # New cloud exists
        forming = v2_store.list_clouds(db, status="forming")
        assert any(c["id"] == new_cloud_id for c in forming)

    def test_move_nonexistent_bundle_to_new_cloud_fails(self, db: DatabaseManager):
        """Moving nonexistent bundle to new cloud returns None."""
        assert v2_store.move_bundle_to_new_cloud(db, "nonexistent") is None

    def test_manual_match_type_stored(self, db: DatabaseManager):
        """Manual moves store MANUAL match_type in cloud_bundles."""
        data = _setup_two_cloud_scenario(db)

        v2_store.move_bundle_to_cloud(db, data["bundle_b"], data["cloud_a"])

        row = db.fetchone(
            "SELECT match_type FROM cloud_bundles WHERE bundle_id = ?",
            (data["bundle_b"],),
        )
        assert row is not None
        assert row["match_type"] == "manual"

    def test_bundle_cloud_id_set_on_store_cloud(self, db: DatabaseManager):
        """store_cloud() sets bundles.cloud_id on the bundle."""
        doc_id = _make_document(db)
        atoms = _make_atoms(db, doc_id)
        bundle_id = _make_bundle(db, doc_id, atoms)
        cloud_id = _make_cloud_with_bundle(db, bundle_id)

        # Check the authoritative field directly
        row = db.fetchone(
            "SELECT cloud_id FROM bundles WHERE id = ?",
            (bundle_id,),
        )
        assert row is not None
        assert row["cloud_id"] == cloud_id

    def test_bundle_cloud_id_updated_on_move(self, db: DatabaseManager):
        """move_bundle_to_cloud() updates bundles.cloud_id."""
        data = _setup_two_cloud_scenario(db)

        v2_store.move_bundle_to_cloud(db, data["bundle_b"], data["cloud_a"])

        row = db.fetchone(
            "SELECT cloud_id FROM bundles WHERE id = ?",
            (data["bundle_b"],),
        )
        assert row is not None
        assert row["cloud_id"] == data["cloud_a"]


# ---------------------------------------------------------------------------
# Test: set_bundle_cloud (direct field edit)
# ---------------------------------------------------------------------------


class TestSetBundleCloud:
    """Tests for set_bundle_cloud() — the user-facing reassignment field."""

    def test_set_to_existing_cloud(self, db: DatabaseManager):
        """User sets bundle.cloud_id to another cloud."""
        data = _setup_two_cloud_scenario(db)

        ok = v2_store.set_bundle_cloud(db, data["bundle_b"], data["cloud_a"])
        assert ok

        # Authoritative field updated
        assert v2_store.get_cloud_for_bundle(db, data["bundle_b"]) == data["cloud_a"]

        # Junction table also updated
        row = db.fetchone(
            "SELECT match_type FROM cloud_bundles WHERE bundle_id = ?",
            (data["bundle_b"],),
        )
        assert row is not None
        assert row["match_type"] == "manual"

    def test_set_to_null_detaches(self, db: DatabaseManager):
        """User clears bundle.cloud_id to detach it."""
        data = _setup_two_cloud_scenario(db)

        ok = v2_store.set_bundle_cloud(db, data["bundle_b"], None)
        assert ok

        # No cloud assigned
        assert v2_store.get_cloud_for_bundle(db, data["bundle_b"]) is None

        # Junction table entry removed
        row = db.fetchone(
            "SELECT cloud_id FROM cloud_bundles WHERE bundle_id = ?",
            (data["bundle_b"],),
        )
        assert row is None

    def test_set_to_null_appears_in_unassigned(self, db: DatabaseManager):
        """Detached bundles appear in get_unassigned_bundles()."""
        data = _setup_two_cloud_scenario(db)

        v2_store.set_bundle_cloud(db, data["bundle_b"], None)

        unassigned = v2_store.get_unassigned_bundles(db)
        assert len(unassigned) == 1
        assert unassigned[0]["id"] == data["bundle_b"]

    def test_set_nonexistent_bundle_fails(self, db: DatabaseManager):
        """Setting cloud_id on nonexistent bundle returns False."""
        assert not v2_store.set_bundle_cloud(db, "nonexistent", None)

    def test_set_to_nonexistent_cloud_fails(self, db: DatabaseManager):
        """Setting to a nonexistent cloud returns False."""
        data = _setup_two_cloud_scenario(db)
        assert not v2_store.set_bundle_cloud(db, data["bundle_b"], "nonexistent")

    def test_no_unassigned_initially(self, db: DatabaseManager):
        """No unassigned bundles when everything is properly assigned."""
        _setup_two_cloud_scenario(db)
        assert v2_store.get_unassigned_bundles(db) == []
