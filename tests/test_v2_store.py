"""Tests for v2 atom-cloud-fact persistence layer."""

import json
import os
import tempfile
from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.config import Config, reset_config
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
from alibi.db import v2_store


@pytest.fixture
def db():
    """Create a fresh in-memory-like temp database."""
    reset_config()
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    config = Config(db_path=db_path)
    manager = DatabaseManager(config)
    if not manager.is_initialized():
        manager.initialize()
    yield manager
    manager.close()
    os.unlink(db_path)


@pytest.fixture
def sample_document():
    """A sample v2 Document."""
    return Document(
        id=str(uuid4()),
        file_path="/test/receipt.jpg",
        file_hash="abc123hash",
        perceptual_hash="phash001",
        raw_extraction={"vendor": "TEST STORE", "total": "42.50"},
    )


@pytest.fixture
def sample_atoms(sample_document):
    """Sample atoms for a receipt."""
    doc_id = sample_document.id
    return [
        Atom(
            id=str(uuid4()),
            document_id=doc_id,
            atom_type=AtomType.VENDOR,
            data={"name": "TEST STORE", "address": "123 Main St"},
        ),
        Atom(
            id=str(uuid4()),
            document_id=doc_id,
            atom_type=AtomType.AMOUNT,
            data={"value": "42.50", "currency": "EUR", "semantic_type": "total"},
        ),
        Atom(
            id=str(uuid4()),
            document_id=doc_id,
            atom_type=AtomType.DATETIME,
            data={"value": "2026-01-15", "semantic_type": "transaction_time"},
        ),
        Atom(
            id=str(uuid4()),
            document_id=doc_id,
            atom_type=AtomType.ITEM,
            data={
                "name": "Milk 1L",
                "quantity": "2",
                "unit": "pcs",
                "unit_price": "1.50",
                "total_price": "3.00",
                "currency": "EUR",
            },
        ),
        Atom(
            id=str(uuid4()),
            document_id=doc_id,
            atom_type=AtomType.ITEM,
            data={
                "name": "Bread",
                "quantity": "1",
                "unit": "pcs",
                "unit_price": "2.50",
                "total_price": "2.50",
                "currency": "EUR",
            },
        ),
    ]


@pytest.fixture
def sample_bundle(sample_document, sample_atoms):
    """Sample bundle for a receipt."""
    bundle = Bundle(
        id=str(uuid4()),
        document_id=sample_document.id,
        bundle_type=BundleType.BASKET,
    )
    bundle_atoms = [
        BundleAtom(
            bundle_id=bundle.id,
            atom_id=sample_atoms[0].id,
            role=BundleAtomRole.VENDOR_INFO,
        ),
        BundleAtom(
            bundle_id=bundle.id,
            atom_id=sample_atoms[1].id,
            role=BundleAtomRole.TOTAL,
        ),
        BundleAtom(
            bundle_id=bundle.id,
            atom_id=sample_atoms[2].id,
            role=BundleAtomRole.EVENT_TIME,
        ),
        BundleAtom(
            bundle_id=bundle.id,
            atom_id=sample_atoms[3].id,
            role=BundleAtomRole.BASKET_ITEM,
        ),
        BundleAtom(
            bundle_id=bundle.id,
            atom_id=sample_atoms[4].id,
            role=BundleAtomRole.BASKET_ITEM,
        ),
    ]
    return bundle, bundle_atoms


class TestStoreDocument:
    def test_store_and_retrieve(self, db, sample_document):
        v2_store.store_document(db, sample_document)
        found = v2_store.get_document_by_hash(db, "abc123hash")
        assert found is not None
        assert found["id"] == sample_document.id
        assert found["file_path"] == "/test/receipt.jpg"

    def test_store_document_no_perceptual_hash(self, db):
        doc = Document(
            id=str(uuid4()),
            file_path="/test/doc.pdf",
            file_hash="pdfhash123",
            raw_extraction=None,
        )
        v2_store.store_document(db, doc)
        found = v2_store.get_document_by_hash(db, "pdfhash123")
        assert found is not None
        assert found["perceptual_hash"] is None

    def test_duplicate_document_ignored(self, db, sample_document):
        v2_store.store_document(db, sample_document)
        # Second insert with same ID should be ignored (INSERT OR IGNORE)
        v2_store.store_document(db, sample_document)
        found = v2_store.get_document_by_hash(db, "abc123hash")
        assert found is not None

    def test_not_found_returns_none(self, db):
        assert v2_store.get_document_by_hash(db, "nonexistent") is None


class TestStoreAtoms:
    def test_store_atoms(self, db, sample_document, sample_atoms):
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)

        rows = db.fetchall(
            "SELECT * FROM atoms WHERE document_id = ?",
            (sample_document.id,),
        )
        assert len(rows) == 5

    def test_atom_types_stored(self, db, sample_document, sample_atoms):
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)

        types = {
            r["atom_type"]
            for r in db.fetchall(
                "SELECT atom_type FROM atoms WHERE document_id = ?",
                (sample_document.id,),
            )
        }
        assert types == {"vendor", "amount", "datetime", "item"}

    def test_atom_data_json(self, db, sample_document, sample_atoms):
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)

        row = db.fetchone(
            "SELECT data FROM atoms WHERE atom_type = 'vendor' AND document_id = ?",
            (sample_document.id,),
        )
        data = json.loads(row["data"])
        assert data["name"] == "TEST STORE"

    def test_empty_atoms_noop(self, db):
        v2_store.store_atoms(db, [])  # Should not raise


class TestStoreBundle:
    def test_store_bundle(self, db, sample_document, sample_atoms, sample_bundle):
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)
        bundle, bundle_atoms = sample_bundle
        v2_store.store_bundle(db, bundle, bundle_atoms)

        row = db.fetchone("SELECT * FROM bundles WHERE id = ?", (bundle.id,))
        assert row is not None
        assert row["bundle_type"] == "basket"

    def test_bundle_atom_links(self, db, sample_document, sample_atoms, sample_bundle):
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)
        bundle, bundle_atoms = sample_bundle
        v2_store.store_bundle(db, bundle, bundle_atoms)

        rows = db.fetchall(
            "SELECT * FROM bundle_atoms WHERE bundle_id = ?", (bundle.id,)
        )
        assert len(rows) == 5
        roles = {r["role"] for r in rows}
        assert "vendor_info" in roles
        assert "basket_item" in roles


class TestStoreCloud:
    def test_store_new_cloud(self, db, sample_document, sample_atoms, sample_bundle):
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)
        bundle, bundle_atoms = sample_bundle
        v2_store.store_bundle(db, bundle, bundle_atoms)

        cloud = Cloud(id=str(uuid4()), status=CloudStatus.FORMING)
        cloud_bundle = CloudBundle(
            cloud_id=cloud.id,
            bundle_id=bundle.id,
            match_type=CloudMatchType.EXACT_AMOUNT,
            match_confidence=Decimal("1.0"),
        )
        v2_store.store_cloud(db, cloud, cloud_bundle)

        row = db.fetchone("SELECT * FROM clouds WHERE id = ?", (cloud.id,))
        assert row is not None
        assert row["status"] == "forming"

    def test_add_bundle_to_cloud(
        self, db, sample_document, sample_atoms, sample_bundle
    ):
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)
        bundle, bundle_atoms = sample_bundle
        v2_store.store_bundle(db, bundle, bundle_atoms)

        cloud = Cloud(id=str(uuid4()))
        cb1 = CloudBundle(
            cloud_id=cloud.id,
            bundle_id=bundle.id,
            match_type=CloudMatchType.EXACT_AMOUNT,
            match_confidence=Decimal("1.0"),
        )
        v2_store.store_cloud(db, cloud, cb1)

        # Add second bundle
        bundle2 = Bundle(
            id=str(uuid4()),
            document_id=sample_document.id,
            bundle_type=BundleType.PAYMENT_RECORD,
        )
        v2_store.store_bundle(db, bundle2, [])
        cb2 = CloudBundle(
            cloud_id=cloud.id,
            bundle_id=bundle2.id,
            match_type=CloudMatchType.VENDOR_DATE,
            match_confidence=Decimal("0.7"),
        )
        v2_store.add_cloud_bundle(db, cb2)

        rows = db.fetchall(
            "SELECT * FROM cloud_bundles WHERE cloud_id = ?", (cloud.id,)
        )
        assert len(rows) == 2

    def test_get_cloud_for_bundle(
        self, db, sample_document, sample_atoms, sample_bundle
    ):
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)
        bundle, bundle_atoms = sample_bundle
        v2_store.store_bundle(db, bundle, bundle_atoms)

        cloud = Cloud(id=str(uuid4()))
        cb = CloudBundle(
            cloud_id=cloud.id,
            bundle_id=bundle.id,
            match_type=CloudMatchType.EXACT_AMOUNT,
            match_confidence=Decimal("1.0"),
        )
        v2_store.store_cloud(db, cloud, cb)

        found = v2_store.get_cloud_for_bundle(db, bundle.id)
        assert found == cloud.id

    def test_get_cloud_for_bundle_not_found(self, db):
        assert v2_store.get_cloud_for_bundle(db, "nonexistent") is None


class TestUpdateCloudStatus:
    def test_update_status(self, db, sample_document, sample_atoms, sample_bundle):
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)
        bundle, bundle_atoms = sample_bundle
        v2_store.store_bundle(db, bundle, bundle_atoms)

        cloud = Cloud(id=str(uuid4()))
        cb = CloudBundle(
            cloud_id=cloud.id,
            bundle_id=bundle.id,
            match_type=CloudMatchType.EXACT_AMOUNT,
            match_confidence=Decimal("1.0"),
        )
        v2_store.store_cloud(db, cloud, cb)

        v2_store.update_cloud_status(
            db, cloud.id, CloudStatus.COLLAPSED, Decimal("1.0")
        )

        row = db.fetchone("SELECT * FROM clouds WHERE id = ?", (cloud.id,))
        assert row["status"] == "collapsed"
        assert row["confidence"] == 1.0


class TestStoreFact:
    def test_store_fact_with_items(
        self, db, sample_document, sample_atoms, sample_bundle
    ):
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)
        bundle, bundle_atoms = sample_bundle
        v2_store.store_bundle(db, bundle, bundle_atoms)

        cloud = Cloud(id=str(uuid4()))
        cb = CloudBundle(
            cloud_id=cloud.id,
            bundle_id=bundle.id,
            match_type=CloudMatchType.EXACT_AMOUNT,
            match_confidence=Decimal("1.0"),
        )
        v2_store.store_cloud(db, cloud, cb)

        fact = Fact(
            id=str(uuid4()),
            cloud_id=cloud.id,
            fact_type=FactType.PURCHASE,
            vendor="TEST STORE",
            total_amount=Decimal("42.50"),
            currency="EUR",
            event_date=date(2026, 1, 15),
            payments=[{"method": "card", "card_last4": "1234", "amount": "42.50"}],
            status=FactStatus.CONFIRMED,
        )

        items = [
            FactItem(
                id=str(uuid4()),
                fact_id=fact.id,
                atom_id=sample_atoms[3].id,
                name="Milk 1L",
                quantity=Decimal("2"),
                unit=UnitType.PIECE,
                unit_price=Decimal("1.50"),
                total_price=Decimal("3.00"),
            ),
            FactItem(
                id=str(uuid4()),
                fact_id=fact.id,
                atom_id=sample_atoms[4].id,
                name="Bread",
                quantity=Decimal("1"),
                unit=UnitType.PIECE,
                unit_price=Decimal("2.50"),
                total_price=Decimal("2.50"),
            ),
        ]

        v2_store.store_fact(db, fact, items)

        # Fact stored
        fact_row = v2_store.get_fact_for_cloud(db, cloud.id)
        assert fact_row is not None
        assert fact_row["vendor"] == "TEST STORE"
        assert float(fact_row["total_amount"]) == 42.50

        # Items stored
        item_rows = v2_store.get_fact_items(db, fact.id)
        assert len(item_rows) == 2
        names = {r["name"] for r in item_rows}
        assert names == {"Bread", "Milk 1L"}

        # Cloud updated to collapsed
        cloud_row = db.fetchone("SELECT * FROM clouds WHERE id = ?", (cloud.id,))
        assert cloud_row["status"] == "collapsed"

    def test_get_fact_for_cloud_not_found(self, db):
        assert v2_store.get_fact_for_cloud(db, "nonexistent") is None


class TestGetBundleSummaries:
    def test_returns_bundles_with_atoms(
        self, db, sample_document, sample_atoms, sample_bundle
    ):
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)
        bundle, bundle_atoms = sample_bundle
        v2_store.store_bundle(db, bundle, bundle_atoms)

        # Create cloud for bundle
        cloud = Cloud(id=str(uuid4()))
        cb = CloudBundle(
            cloud_id=cloud.id,
            bundle_id=bundle.id,
            match_type=CloudMatchType.EXACT_AMOUNT,
            match_confidence=Decimal("1.0"),
        )
        v2_store.store_cloud(db, cloud, cb)

        summaries = v2_store.get_bundle_summaries(db)
        assert len(summaries) == 1
        s = summaries[0]
        assert s["bundle_id"] == bundle.id
        assert s["bundle_type"] == "basket"
        assert s["cloud_id"] == cloud.id
        assert len(s["atoms"]) == 5

    def test_empty_db(self, db):
        assert v2_store.get_bundle_summaries(db) == []


class TestGetCloudBundleData:
    def test_returns_bundle_data_for_collapse(
        self, db, sample_document, sample_atoms, sample_bundle
    ):
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)
        bundle, bundle_atoms = sample_bundle
        v2_store.store_bundle(db, bundle, bundle_atoms)

        cloud = Cloud(id=str(uuid4()))
        cb = CloudBundle(
            cloud_id=cloud.id,
            bundle_id=bundle.id,
            match_type=CloudMatchType.EXACT_AMOUNT,
            match_confidence=Decimal("1.0"),
        )
        v2_store.store_cloud(db, cloud, cb)

        data = v2_store.get_cloud_bundle_data(db, cloud.id)
        assert len(data) == 1
        bd = data[0]
        assert bd["bundle_id"] == bundle.id
        assert bd["bundle_type"] == "basket"
        assert len(bd["atoms"]) == 5

        # Check atom data is properly deserialized
        vendor_atom = next(a for a in bd["atoms"] if a["atom_type"] == "vendor")
        assert vendor_atom["data"]["name"] == "TEST STORE"

    def test_empty_cloud(self, db):
        assert v2_store.get_cloud_bundle_data(db, "nonexistent") == []


class TestCleanupDocument:
    """Tests for cleanup_document() compensating cleanup."""

    def test_cleanup_full_pipeline_data(
        self, db, sample_document, sample_atoms, sample_bundle
    ):
        """Cleanup removes document, atoms, bundles, cloud, fact."""
        # Store full pipeline data
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)
        bundle, bundle_atoms = sample_bundle
        v2_store.store_bundle(db, bundle, bundle_atoms)

        cloud = Cloud(id=str(uuid4()))
        cb = CloudBundle(
            cloud_id=cloud.id,
            bundle_id=bundle.id,
            match_type=CloudMatchType.EXACT_AMOUNT,
            match_confidence=Decimal("1.0"),
        )
        v2_store.store_cloud(db, cloud, cb)

        fact = Fact(
            id=str(uuid4()),
            cloud_id=cloud.id,
            fact_type=FactType.PURCHASE,
            vendor="TEST STORE",
            total_amount=Decimal("42.50"),
            currency="EUR",
            event_date=date(2026, 1, 15),
        )
        item = FactItem(
            id=str(uuid4()),
            fact_id=fact.id,
            atom_id=sample_atoms[3].id,
            name="Milk 1L",
            quantity=Decimal("2"),
            unit=UnitType.PIECE,
            unit_price=Decimal("1.50"),
            total_price=Decimal("3.00"),
        )
        v2_store.store_fact(db, fact, [item])

        # Cleanup should remove everything
        result = v2_store.cleanup_document(db, sample_document.id)
        assert result["cleaned"] is True

        # Verify all data is gone
        assert v2_store.get_document_by_hash(db, "abc123hash") is None
        assert (
            db.fetchone(
                "SELECT COUNT(*) as cnt FROM atoms WHERE document_id = ?",
                (sample_document.id,),
            )["cnt"]
            == 0
        )
        assert (
            db.fetchone(
                "SELECT COUNT(*) as cnt FROM bundles WHERE document_id = ?",
                (sample_document.id,),
            )["cnt"]
            == 0
        )
        assert v2_store.get_fact_by_id(db, fact.id) is None

    def test_cleanup_nonexistent_document(self, db):
        """Cleanup returns dict with cleaned=False for nonexistent document."""
        result = v2_store.cleanup_document(db, "nonexistent")
        assert result["cleaned"] is False

    def test_cleanup_document_only(self, db, sample_document):
        """Cleanup works when only a document exists (no atoms/bundles)."""
        v2_store.store_document(db, sample_document)
        result = v2_store.cleanup_document(db, sample_document.id)
        assert result["cleaned"] is True
        assert v2_store.get_document_by_hash(db, "abc123hash") is None


class TestOrphanedCleanup:
    """Tests for orphaned data cleanup functions."""

    def test_cleanup_orphaned_atoms(self, db, sample_document, sample_atoms):
        """Orphaned atoms (no bundle link) are cleaned up."""
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)
        # Atoms exist but no bundle links them
        count = v2_store.cleanup_orphaned_atoms(db)
        assert count == len(sample_atoms)

        remaining = db.fetchone(
            "SELECT COUNT(*) as cnt FROM atoms WHERE document_id = ?",
            (sample_document.id,),
        )
        assert remaining["cnt"] == 0

    def test_cleanup_orphaned_atoms_skips_linked(
        self, db, sample_document, sample_atoms, sample_bundle
    ):
        """Linked atoms are NOT cleaned up."""
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)
        bundle, bundle_atoms = sample_bundle
        v2_store.store_bundle(db, bundle, bundle_atoms)

        count = v2_store.cleanup_orphaned_atoms(db)
        assert count == 0

    def test_cleanup_orphaned_bundles(self, db, sample_document, sample_atoms):
        """Bundles with NULL cloud_id are cleaned up."""
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)
        bundle = Bundle(
            id=str(uuid4()),
            document_id=sample_document.id,
            bundle_type=BundleType.BASKET,
            cloud_id=None,
        )
        ba = BundleAtom(
            bundle_id=bundle.id,
            atom_id=sample_atoms[0].id,
            role=BundleAtomRole.VENDOR_INFO,
        )
        v2_store.store_bundle(db, bundle, [ba])

        count = v2_store.cleanup_orphaned_bundles(db)
        assert count == 1

    def test_cleanup_orphaned_bundles_skips_assigned(
        self, db, sample_document, sample_atoms, sample_bundle
    ):
        """Bundles assigned to clouds are NOT cleaned up."""
        v2_store.store_document(db, sample_document)
        v2_store.store_atoms(db, sample_atoms)
        bundle, bundle_atoms = sample_bundle
        v2_store.store_bundle(db, bundle, bundle_atoms)

        cloud = Cloud(id=str(uuid4()))
        cb = CloudBundle(
            cloud_id=cloud.id,
            bundle_id=bundle.id,
            match_type=CloudMatchType.EXACT_AMOUNT,
            match_confidence=Decimal("1.0"),
        )
        v2_store.store_cloud(db, cloud, cb)

        count = v2_store.cleanup_orphaned_bundles(db)
        assert count == 0

    def test_run_maintenance(self, db):
        """run_maintenance returns cleanup counts."""
        result = v2_store.run_maintenance(db)
        assert result == {
            "orphaned_atoms": 0,
            "orphaned_bundles": 0,
            "empty_clouds": 0,
        }
