"""Tests for alibi.services.query — the query service facade."""

from __future__ import annotations

import os
from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.db import v2_store
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
from alibi.services import query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(db, file_path: str = "/test/receipt.jpg") -> Document:
    doc = Document(
        id=str(uuid4()),
        file_path=file_path,
        file_hash=str(uuid4()),
    )
    v2_store.store_document(db, doc)
    return doc


def _make_vendor_atom(db, doc: Document, name: str = "Test Store") -> Atom:
    atom = Atom(
        id=str(uuid4()),
        document_id=doc.id,
        atom_type=AtomType.VENDOR,
        data={"name": name},
    )
    v2_store.store_atoms(db, [atom])
    return atom


def _make_item_atom(db, doc: Document, name: str = "Milk 1L") -> Atom:
    atom = Atom(
        id=str(uuid4()),
        document_id=doc.id,
        atom_type=AtomType.ITEM,
        data={
            "name": name,
            "quantity": "1",
            "unit": "pcs",
            "unit_price": "1.50",
            "total_price": "1.50",
            "currency": "EUR",
        },
    )
    v2_store.store_atoms(db, [atom])
    return atom


def _make_bundle(db, doc: Document, atoms: list[Atom]) -> Bundle:
    bundle = Bundle(
        id=str(uuid4()),
        document_id=doc.id,
        bundle_type=BundleType.BASKET,
    )
    bundle_atoms = [
        BundleAtom(
            bundle_id=bundle.id,
            atom_id=a.id,
            role=BundleAtomRole.VENDOR_INFO,
        )
        for a in atoms
    ]
    v2_store.store_bundle(db, bundle, bundle_atoms)
    return bundle


def _make_cloud_and_collapse(
    db,
    bundle: Bundle,
    vendor: str = "Test Store",
    total_amount: Decimal = Decimal("42.50"),
    event_date: date | None = None,
    fact_type: FactType = FactType.PURCHASE,
    items: list[FactItem] | None = None,
) -> Fact:
    """Create a cloud, attach the bundle, and store a collapsed fact."""
    cloud = Cloud(id=str(uuid4()), status=CloudStatus.FORMING)
    cloud_bundle = CloudBundle(
        cloud_id=cloud.id,
        bundle_id=bundle.id,
        match_type=CloudMatchType.MANUAL,
        match_confidence=Decimal("1.0"),
    )
    v2_store.store_cloud(db, cloud, cloud_bundle)

    fact = Fact(
        id=str(uuid4()),
        cloud_id=cloud.id,
        fact_type=fact_type,
        vendor=vendor,
        vendor_key=None,
        total_amount=total_amount,
        currency="EUR",
        event_date=event_date or date(2026, 1, 15),
        status=FactStatus.CONFIRMED,
    )
    v2_store.store_fact(db, fact, items or [])
    return fact


def _make_fact_item(fact: Fact, atom: Atom, name: str = "Milk 1L") -> FactItem:
    return FactItem(
        id=str(uuid4()),
        fact_id=fact.id,
        atom_id=atom.id,
        name=name,
        name_normalized=name.lower(),
        quantity=Decimal("1"),
        unit=UnitType.PIECE,
        unit_price=Decimal("1.50"),
        total_price=Decimal("1.50"),
        tax_type=TaxType.NONE,
    )


# ---------------------------------------------------------------------------
# get_fact
# ---------------------------------------------------------------------------


class TestGetFact:
    def test_returns_none_for_missing_fact(self, db):
        result = query.get_fact(db, str(uuid4()))
        assert result is None

    def test_returns_fact_with_items(self, db):
        doc = _make_doc(db)
        vendor_atom = _make_vendor_atom(db, doc)
        item_atom = _make_item_atom(db, doc, name="Bread")
        bundle = _make_bundle(db, doc, [vendor_atom, item_atom])

        fact = _make_cloud_and_collapse(db, bundle, vendor="BakerShop")
        item = _make_fact_item(fact, item_atom, name="Bread")
        # Store the item directly since _make_cloud_and_collapse accepted []
        # Re-insert a fact item by calling store_fact on a fresh fact — instead
        # use the v2_store helper for items directly.
        with db.transaction() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO fact_items "
                "(id, fact_id, atom_id, name, name_normalized, "
                "quantity, unit, unit_price, total_price, "
                "brand, category, comparable_unit_price, comparable_unit, "
                "tax_rate, tax_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.id,
                    item.fact_id,
                    item.atom_id,
                    item.name,
                    item.name_normalized,
                    float(item.quantity),
                    item.unit.value,
                    float(item.unit_price) if item.unit_price else None,
                    float(item.total_price) if item.total_price else None,
                    item.brand,
                    item.category,
                    None,  # comparable_unit_price
                    None,  # comparable_unit
                    None,  # tax_rate
                    item.tax_type.value,
                ),
            )

        result = query.get_fact(db, fact.id)
        assert result is not None
        assert result["id"] == fact.id
        assert result["vendor"] == "BakerShop"
        assert "items" in result
        assert len(result["items"]) == 1
        assert result["items"][0]["name"] == "Bread"

    def test_fact_with_no_items_returns_empty_items_list(self, db):
        doc = _make_doc(db)
        vendor_atom = _make_vendor_atom(db, doc)
        bundle = _make_bundle(db, doc, [vendor_atom])
        fact = _make_cloud_and_collapse(db, bundle)

        result = query.get_fact(db, fact.id)
        assert result is not None
        assert result["items"] == []


# ---------------------------------------------------------------------------
# inspect_fact
# ---------------------------------------------------------------------------


class TestInspectFact:
    def test_returns_none_for_missing_fact(self, db):
        result = query.inspect_fact(db, str(uuid4()))
        assert result is None

    def test_returns_nested_structure(self, db):
        doc = _make_doc(db)
        vendor_atom = _make_vendor_atom(db, doc)
        bundle = _make_bundle(db, doc, [vendor_atom])
        fact = _make_cloud_and_collapse(db, bundle, vendor="Megamart")

        result = query.inspect_fact(db, fact.id)
        assert result is not None
        assert "fact" in result
        assert "cloud" in result
        assert "bundles" in result
        assert "items" in result
        assert result["fact"]["vendor"] == "Megamart"
        assert len(result["bundles"]) == 1


# ---------------------------------------------------------------------------
# list_facts
# ---------------------------------------------------------------------------


class TestListFacts:
    def test_returns_empty_on_fresh_db(self, db):
        result = query.list_facts(db)
        assert result["total"] == 0
        assert result["facts"] == []

    def test_pagination_structure(self, db):
        result = query.list_facts(db, offset=0, limit=10)
        assert result["offset"] == 0
        assert result["limit"] == 10

    def test_offset_and_limit(self, db):
        # Insert 5 facts
        for i in range(5):
            doc = _make_doc(db, f"/test/receipt_{i}.jpg")
            atom = _make_vendor_atom(db, doc, name=f"Vendor {i}")
            bundle = _make_bundle(db, doc, [atom])
            _make_cloud_and_collapse(
                db,
                bundle,
                vendor=f"Vendor {i}",
                event_date=date(2026, 1, i + 1),
            )

        all_facts = query.list_facts(db)
        assert all_facts["total"] == 5

        page1 = query.list_facts(db, offset=0, limit=2)
        assert len(page1["facts"]) == 2
        assert page1["total"] == 5

        page2 = query.list_facts(db, offset=2, limit=2)
        assert len(page2["facts"]) == 2

        page3 = query.list_facts(db, offset=4, limit=2)
        assert len(page3["facts"]) == 1

    def test_filter_by_vendor(self, db):
        for vendor in ["Alpha Corp", "Beta Ltd", "Gamma Inc"]:
            doc = _make_doc(db, f"/test/{vendor}.jpg")
            atom = _make_vendor_atom(db, doc, name=vendor)
            bundle = _make_bundle(db, doc, [atom])
            _make_cloud_and_collapse(db, bundle, vendor=vendor)

        result = query.list_facts(db, filters={"vendor": "alpha"})
        assert result["total"] == 1
        assert result["facts"][0]["vendor"] == "Alpha Corp"

    def test_filter_by_date_range(self, db):
        dates = [date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1)]
        for d in dates:
            doc = _make_doc(db, f"/test/{d}.jpg")
            atom = _make_vendor_atom(db, doc)
            bundle = _make_bundle(db, doc, [atom])
            _make_cloud_and_collapse(db, bundle, event_date=d)

        result = query.list_facts(
            db, filters={"date_from": date(2026, 1, 15), "date_to": date(2026, 2, 15)}
        )
        assert result["total"] == 1
        # SQLite returns event_date as a date object (row_factory strips the string)
        event_date = result["facts"][0]["event_date"]
        assert str(event_date) == "2026-02-01"

    def test_filter_by_min_max_amount(self, db):
        for amount in [Decimal("10.00"), Decimal("50.00"), Decimal("100.00")]:
            doc = _make_doc(db, f"/test/{amount}.jpg")
            atom = _make_vendor_atom(db, doc)
            bundle = _make_bundle(db, doc, [atom])
            _make_cloud_and_collapse(db, bundle, total_amount=amount)

        result = query.list_facts(db, filters={"min_amount": 20.0, "max_amount": 75.0})
        assert result["total"] == 1
        assert float(result["facts"][0]["total_amount"]) == pytest.approx(50.0)

    def test_filter_by_fact_type(self, db):
        for ft in [FactType.PURCHASE, FactType.REFUND, FactType.PURCHASE]:
            doc = _make_doc(db, f"/test/{uuid4()}.jpg")
            atom = _make_vendor_atom(db, doc)
            bundle = _make_bundle(db, doc, [atom])
            _make_cloud_and_collapse(db, bundle, fact_type=ft)

        result = query.list_facts(db, filters={"fact_type": "purchase"})
        assert result["total"] == 2

        result = query.list_facts(db, filters={"fact_type": "refund"})
        assert result["total"] == 1

    def test_total_is_independent_of_pagination(self, db):
        for i in range(7):
            doc = _make_doc(db, f"/test/r{i}.jpg")
            atom = _make_vendor_atom(db, doc)
            bundle = _make_bundle(db, doc, [atom])
            _make_cloud_and_collapse(db, bundle, event_date=date(2026, 1, i + 1))

        page = query.list_facts(db, offset=0, limit=3)
        assert page["total"] == 7
        assert len(page["facts"]) == 3


# ---------------------------------------------------------------------------
# search_facts
# ---------------------------------------------------------------------------


class TestSearchFacts:
    def test_search_by_vendor_name(self, db):
        for vendor in ["Supermarket Alpha", "BookShop Beta", "Pharmacy Gamma"]:
            doc = _make_doc(db, f"/test/{vendor}.jpg")
            atom = _make_vendor_atom(db, doc, name=vendor)
            bundle = _make_bundle(db, doc, [atom])
            _make_cloud_and_collapse(db, bundle, vendor=vendor)

        result = query.search_facts(db, "bookshop")
        assert result["total"] == 1
        assert result["facts"][0]["vendor"] == "BookShop Beta"

    def test_search_by_item_name(self, db):
        # Fact 1: vendor "Generic Store", item "Organic Oat Milk"
        doc1 = _make_doc(db, "/test/store1.jpg")
        v_atom1 = _make_vendor_atom(db, doc1, name="Generic Store")
        i_atom1 = _make_item_atom(db, doc1, name="Organic Oat Milk")
        bundle1 = _make_bundle(db, doc1, [v_atom1])
        fact1 = _make_cloud_and_collapse(db, bundle1, vendor="Generic Store")

        # Store the fact item for fact1
        item1 = FactItem(
            id=str(uuid4()),
            fact_id=fact1.id,
            atom_id=i_atom1.id,
            name="Organic Oat Milk",
            name_normalized="organic oat milk",
            quantity=Decimal("1"),
            unit=UnitType.PIECE,
            tax_type=TaxType.NONE,
        )
        with db.transaction() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO fact_items "
                "(id, fact_id, atom_id, name, name_normalized, "
                "quantity, unit, unit_price, total_price, "
                "brand, category, comparable_unit_price, comparable_unit, "
                "tax_rate, tax_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item1.id,
                    item1.fact_id,
                    item1.atom_id,
                    item1.name,
                    item1.name_normalized,
                    float(item1.quantity),
                    item1.unit.value,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    item1.tax_type.value,
                ),
            )

        # Fact 2: vendor "Other Vendor", item "Bread"
        doc2 = _make_doc(db, "/test/store2.jpg")
        v_atom2 = _make_vendor_atom(db, doc2, name="Other Vendor")
        i_atom2 = _make_item_atom(db, doc2, name="Bread")
        bundle2 = _make_bundle(db, doc2, [v_atom2])
        fact2 = _make_cloud_and_collapse(db, bundle2, vendor="Other Vendor")

        item2 = FactItem(
            id=str(uuid4()),
            fact_id=fact2.id,
            atom_id=i_atom2.id,
            name="Bread",
            name_normalized="bread",
            quantity=Decimal("1"),
            unit=UnitType.PIECE,
            tax_type=TaxType.NONE,
        )
        with db.transaction() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO fact_items "
                "(id, fact_id, atom_id, name, name_normalized, "
                "quantity, unit, unit_price, total_price, "
                "brand, category, comparable_unit_price, comparable_unit, "
                "tax_rate, tax_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item2.id,
                    item2.fact_id,
                    item2.atom_id,
                    item2.name,
                    item2.name_normalized,
                    float(item2.quantity),
                    item2.unit.value,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    item2.tax_type.value,
                ),
            )

        # Search for "oat milk" — should find fact1 via item, NOT fact2
        result = query.search_facts(db, "oat milk")
        assert result["total"] == 1
        assert result["facts"][0]["id"] == fact1.id

    def test_search_across_vendor_and_items(self, db):
        # Fact with vendor matching
        doc1 = _make_doc(db, "/test/store1.jpg")
        v_atom1 = _make_vendor_atom(db, doc1, name="Omega Supermarket")
        bundle1 = _make_bundle(db, doc1, [v_atom1])
        _make_cloud_and_collapse(db, bundle1, vendor="Omega Supermarket")

        # Fact with item matching but different vendor
        doc2 = _make_doc(db, "/test/store2.jpg")
        v_atom2 = _make_vendor_atom(db, doc2, name="Delta Store")
        i_atom2 = _make_item_atom(db, doc2, name="Omega 3 Capsules")
        bundle2 = _make_bundle(db, doc2, [v_atom2])
        fact2 = _make_cloud_and_collapse(db, bundle2, vendor="Delta Store")

        with db.transaction() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO fact_items "
                "(id, fact_id, atom_id, name, name_normalized, "
                "quantity, unit, unit_price, total_price, "
                "brand, category, comparable_unit_price, comparable_unit, "
                "tax_rate, tax_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    fact2.id,
                    i_atom2.id,
                    "Omega 3 Capsules",
                    "omega 3 capsules",
                    1.0,
                    "pcs",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "none",
                ),
            )

        # "omega" matches both vendor "Omega Supermarket" and item "Omega 3 Capsules"
        result = query.search_facts(db, "omega")
        assert result["total"] == 2

    def test_search_no_match_returns_empty(self, db):
        doc = _make_doc(db)
        atom = _make_vendor_atom(db, doc, name="Totally Unique Store")
        bundle = _make_bundle(db, doc, [atom])
        _make_cloud_and_collapse(db, bundle, vendor="Totally Unique Store")

        result = query.search_facts(db, "xyzzy_does_not_exist")
        assert result["total"] == 0
        assert result["facts"] == []

    def test_search_pagination_structure(self, db):
        result = query.search_facts(db, "anything", offset=5, limit=10)
        assert result["offset"] == 5
        assert result["limit"] == 10


# ---------------------------------------------------------------------------
# list_unassigned
# ---------------------------------------------------------------------------


class TestListUnassigned:
    def test_empty_on_fresh_db(self, db):
        result = query.list_unassigned(db)
        assert result == []

    def test_returns_detached_bundles(self, db):
        doc = _make_doc(db)
        atom = _make_vendor_atom(db, doc)
        bundle = _make_bundle(db, doc, [atom])
        # Bundle has no cloud_id at this point (cloud_id=NULL after store_bundle)

        result = query.list_unassigned(db)
        assert len(result) == 1
        assert result[0]["id"] == bundle.id

    def test_assigned_bundles_not_included(self, db):
        doc = _make_doc(db)
        atom = _make_vendor_atom(db, doc)
        bundle = _make_bundle(db, doc, [atom])
        _make_cloud_and_collapse(db, bundle)

        result = query.list_unassigned(db)
        assert result == []


# ---------------------------------------------------------------------------
# get_document
# ---------------------------------------------------------------------------


class TestGetDocument:
    def test_returns_none_for_missing_document(self, db):
        result = query.get_document(db, str(uuid4()))
        assert result is None

    def test_returns_document_metadata(self, db):
        doc = _make_doc(db, "/some/file.jpg")
        result = query.get_document(db, doc.id)
        assert result is not None
        assert result["id"] == doc.id
        assert result["file_path"] == "/some/file.jpg"


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------


class TestListDocuments:
    def test_empty_on_fresh_db(self, db):
        result = query.list_documents(db)
        assert result["total"] == 0
        assert result["documents"] == []

    def test_pagination_structure(self, db):
        result = query.list_documents(db, offset=0, limit=25)
        assert result["offset"] == 0
        assert result["limit"] == 25

    def test_offset_and_limit(self, db):
        for i in range(6):
            _make_doc(db, f"/test/file_{i}.jpg")

        all_docs = query.list_documents(db)
        assert all_docs["total"] == 6

        page1 = query.list_documents(db, offset=0, limit=4)
        assert len(page1["documents"]) == 4
        assert page1["total"] == 6

        page2 = query.list_documents(db, offset=4, limit=4)
        assert len(page2["documents"]) == 2

    def test_total_is_independent_of_pagination(self, db):
        for i in range(5):
            _make_doc(db, f"/test/doc_{i}.jpg")

        page = query.list_documents(db, offset=0, limit=2)
        assert page["total"] == 5
        assert len(page["documents"]) == 2


# ---------------------------------------------------------------------------
# list_fact_items_with_fact
# ---------------------------------------------------------------------------


class TestListFactItemsWithFact:
    """Tests for list_fact_items_with_fact service function."""

    @staticmethod
    def _setup_fact_with_items(
        db, vendor="Test Store", event_date=None, items_spec=None
    ):
        """Helper: create a fact with items. items_spec is list of (name, category)."""
        items_spec = items_spec or [("Milk", None)]
        event_date = event_date or date(2026, 1, 15)
        doc = _make_doc(db, f"/test/{vendor.replace(' ', '_')}.jpg")
        vendor_atom = _make_vendor_atom(db, doc, vendor)
        item_atoms = [_make_item_atom(db, doc, name) for name, _ in items_spec]
        bundle = _make_bundle(db, doc, [vendor_atom] + item_atoms)
        fact = _make_cloud_and_collapse(
            db,
            bundle,
            vendor=vendor,
            event_date=event_date,
        )
        # Now create and store items with correct fact_id
        fact_items = []
        for (name, category), atom in zip(items_spec, item_atoms):
            fi = _make_fact_item(fact, atom, name)
            if category:
                fi.category = category
            fact_items.append(fi)
        v2_store.store_fact(db, fact, fact_items)
        return fact

    def test_returns_items_with_parent_fact_fields(self, db):
        self._setup_fact_with_items(
            db,
            vendor="Grocery Store",
            event_date=date(2026, 1, 10),
            items_spec=[("Bread", "bakery")],
        )

        rows = query.list_fact_items_with_fact(db)
        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == "Bread"
        assert row["vendor"] == "Grocery Store"
        assert str(row["event_date"]) == "2026-01-10"
        assert row["currency"] == "EUR"
        assert row["category"] == "bakery"

    def test_filter_by_category(self, db):
        self._setup_fact_with_items(
            db,
            items_spec=[("Milk", "dairy"), ("Bread", "bakery")],
        )

        rows = query.list_fact_items_with_fact(db, filters={"category": "dairy"})
        assert len(rows) == 1
        assert rows[0]["name"] == "Milk"

    def test_filter_by_date_range(self, db):
        self._setup_fact_with_items(
            db,
            vendor="Jan Shop",
            event_date=date(2026, 1, 15),
            items_spec=[("Item Jan", None)],
        )
        self._setup_fact_with_items(
            db,
            vendor="Feb Shop",
            event_date=date(2026, 2, 15),
            items_spec=[("Item Feb", None)],
        )

        rows = query.list_fact_items_with_fact(
            db, filters={"date_from": "2026-02-01", "date_to": "2026-02-28"}
        )
        assert len(rows) == 1
        assert rows[0]["name"] == "Item Feb"

    def test_empty_when_no_items(self, db):
        rows = query.list_fact_items_with_fact(db)
        assert rows == []

    def test_ordered_by_date_descending(self, db):
        for day, name in [(5, "Early"), (20, "Late"), (10, "Mid")]:
            self._setup_fact_with_items(
                db,
                vendor=f"Shop {name}",
                event_date=date(2026, 1, day),
                items_spec=[(name, None)],
            )

        rows = query.list_fact_items_with_fact(db)
        names = [r["name"] for r in rows]
        assert names == ["Late", "Mid", "Early"]
