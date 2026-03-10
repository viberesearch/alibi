"""Tests for v2 fact-based Obsidian note generation."""

import json
import os
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any
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
from alibi.obsidian.notes import (
    generate_fact_note,
    format_currency,
    get_note_filename,
    NoteExporter,
    _format_vendor_details_from_atom,
    _format_payment_rows,
    _format_fact_items_table,
)


@pytest.fixture
def db():
    """Create a fresh temp database."""
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


def _populate_full_fact(db) -> tuple[Fact, list[FactItem], Document]:
    """Create a complete fact with document, atoms, bundle, cloud, items."""
    doc_id = str(uuid4())
    doc = Document(
        id=doc_id,
        file_path="/test/receipt_2026-01-21.jpg",
        file_hash="hash_receipt_001",
        perceptual_hash="phash001",
        raw_extraction={"vendor": "PAPAS HYPERMARKET", "total": "85.69"},
    )
    v2_store.store_document(db, doc)

    vendor_atom = Atom(
        id=str(uuid4()),
        document_id=doc_id,
        atom_type=AtomType.VENDOR,
        data={
            "name": "PAPAS HYPERMARKET",
            "address": "Panayioti Tsangari 23",
            "phone": "+357-99-123456",
            "vat_number": "10355430K",
        },
    )
    amount_atom = Atom(
        id=str(uuid4()),
        document_id=doc_id,
        atom_type=AtomType.AMOUNT,
        data={"value": "85.69", "currency": "EUR", "semantic_type": "total"},
    )
    payment_atom = Atom(
        id=str(uuid4()),
        document_id=doc_id,
        atom_type=AtomType.PAYMENT,
        data={"method": "card", "card_last4": "7201", "amount": "85.69"},
    )
    date_atom = Atom(
        id=str(uuid4()),
        document_id=doc_id,
        atom_type=AtomType.DATETIME,
        data={"value": "2026-01-21", "semantic_type": "transaction_time"},
    )
    item1_atom = Atom(
        id=str(uuid4()),
        document_id=doc_id,
        atom_type=AtomType.ITEM,
        data={
            "name": "Red Bull 250ml",
            "quantity": "3",
            "unit": "pcs",
            "unit_price": "1.99",
            "total_price": "5.97",
            "brand": "Red Bull",
            "category": "beverages",
        },
    )
    item2_atom = Atom(
        id=str(uuid4()),
        document_id=doc_id,
        atom_type=AtomType.ITEM,
        data={
            "name": "Organic Bananas 1kg",
            "quantity": "1",
            "unit": "kg",
            "unit_price": "2.49",
            "total_price": "2.49",
            "category": "produce",
            "comparable_unit_price": "2.49",
            "comparable_unit": "kg",
        },
    )
    atoms = [vendor_atom, amount_atom, payment_atom, date_atom, item1_atom, item2_atom]
    v2_store.store_atoms(db, atoms)

    bundle = Bundle(
        id=str(uuid4()),
        document_id=doc_id,
        bundle_type=BundleType.BASKET,
    )
    bundle_atoms = [
        BundleAtom(bundle_id=bundle.id, atom_id=a.id, role=BundleAtomRole.BASKET_ITEM)
        for a in atoms
    ]
    v2_store.store_bundle(db, bundle, bundle_atoms)

    cloud = Cloud(
        id=str(uuid4()),
        status=CloudStatus.COLLAPSED,
        confidence=Decimal("1.0"),
    )
    cloud_bundle = CloudBundle(
        cloud_id=cloud.id,
        bundle_id=bundle.id,
        match_type=CloudMatchType.EXACT_AMOUNT,
        match_confidence=Decimal("1.0"),
    )
    v2_store.store_cloud(db, cloud, cloud_bundle)

    fact = Fact(
        id=str(uuid4()),
        cloud_id=cloud.id,
        fact_type=FactType.PURCHASE,
        vendor="PAPAS HYPERMARKET",
        total_amount=Decimal("85.69"),
        currency="EUR",
        event_date=date(2026, 1, 21),
        payments=[{"method": "card", "card_last4": "7201", "amount": "85.69"}],
        status=FactStatus.CONFIRMED,
    )
    items = [
        FactItem(
            id=str(uuid4()),
            fact_id=fact.id,
            atom_id=item1_atom.id,
            name="Red Bull 250ml",
            quantity=Decimal("3"),
            unit=UnitType.PIECE,
            unit_price=Decimal("1.99"),
            total_price=Decimal("5.97"),
            brand="Red Bull",
            category="beverages",
        ),
        FactItem(
            id=str(uuid4()),
            fact_id=fact.id,
            atom_id=item2_atom.id,
            name="Organic Bananas 1kg",
            quantity=Decimal("1"),
            unit=UnitType.KILOGRAM,
            unit_price=Decimal("2.49"),
            total_price=Decimal("2.49"),
            category="produce",
            comparable_unit_price=Decimal("2.49"),
            comparable_unit=UnitType.KILOGRAM,
        ),
    ]
    v2_store.store_fact(db, fact, items)

    return fact, items, doc


# ---------------------------------------------------------------------------
# v2_store query function tests
# ---------------------------------------------------------------------------


class TestGetFactById:
    def test_found(self, db):
        fact, _, _ = _populate_full_fact(db)
        result = v2_store.get_fact_by_id(db, fact.id)
        assert result is not None
        assert result["id"] == fact.id
        assert result["vendor"] == "PAPAS HYPERMARKET"

    def test_not_found(self, db):
        assert v2_store.get_fact_by_id(db, "nonexistent") is None


class TestListFacts:
    def test_returns_all(self, db):
        _populate_full_fact(db)
        results = v2_store.list_facts(db)
        assert len(results) == 1
        assert results[0]["vendor"] == "PAPAS HYPERMARKET"

    def test_filter_by_date_from(self, db):
        _populate_full_fact(db)
        results = v2_store.list_facts(db, date_from=date(2026, 2, 1))
        assert len(results) == 0

    def test_filter_by_date_to(self, db):
        _populate_full_fact(db)
        results = v2_store.list_facts(db, date_to=date(2026, 1, 31))
        assert len(results) == 1

    def test_filter_by_vendor(self, db):
        _populate_full_fact(db)
        results = v2_store.list_facts(db, vendor="papas")
        assert len(results) == 1

    def test_filter_by_vendor_no_match(self, db):
        _populate_full_fact(db)
        results = v2_store.list_facts(db, vendor="walmart")
        assert len(results) == 0

    def test_filter_by_fact_type(self, db):
        _populate_full_fact(db)
        results = v2_store.list_facts(db, fact_type="purchase")
        assert len(results) == 1
        results = v2_store.list_facts(db, fact_type="refund")
        assert len(results) == 0


class TestGetFactDocuments:
    def test_returns_source_documents(self, db):
        fact, _, doc = _populate_full_fact(db)
        docs = v2_store.get_fact_documents(db, fact.id)
        assert len(docs) == 1
        assert docs[0]["file_path"] == "/test/receipt_2026-01-21.jpg"

    def test_no_documents(self, db):
        docs = v2_store.get_fact_documents(db, "nonexistent")
        assert docs == []


class TestGetFactVendorAtom:
    def test_returns_vendor_data(self, db):
        fact, _, _ = _populate_full_fact(db)
        vendor = v2_store.get_fact_vendor_atom(db, fact.id)
        assert vendor is not None
        assert vendor["name"] == "PAPAS HYPERMARKET"
        assert vendor["address"] == "Panayioti Tsangari 23"
        assert vendor["phone"] == "+357-99-123456"
        assert vendor["vat_number"] == "10355430K"

    def test_not_found(self, db):
        assert v2_store.get_fact_vendor_atom(db, "nonexistent") is None


class TestGetFactsGroupedByVendor:
    def test_groups_by_vendor(self, db):
        _populate_full_fact(db)
        grouped = v2_store.get_facts_grouped_by_vendor(db)
        assert "PAPAS HYPERMARKET" in grouped
        assert len(grouped["PAPAS HYPERMARKET"]) == 1

    def test_empty(self, db):
        grouped = v2_store.get_facts_grouped_by_vendor(db)
        assert grouped == {}


class TestUpdateFactType:
    def test_updates_type(self, db):
        fact, _, _ = _populate_full_fact(db)
        v2_store.update_fact_type(db, fact.id, "subscription_payment")
        updated = v2_store.get_fact_by_id(db, fact.id)
        assert updated is not None
        assert updated["fact_type"] == "subscription_payment"


# ---------------------------------------------------------------------------
# Fact note generation tests
# ---------------------------------------------------------------------------


class TestFormatVendorDetailsFromAtom:
    def test_with_full_data(self):
        data = {
            "name": "TEST",
            "address": "123 Main St",
            "phone": "+1-555-0123",
            "vat_number": "VAT123",
            "tax_id": "TIC456",
        }
        result = _format_vendor_details_from_atom(data)
        assert "**Address**: 123 Main St" in result
        assert "**Phone**: +1-555-0123" in result
        assert "**VAT Number**: VAT123" in result
        assert "**Tax ID**: TIC456" in result

    def test_with_none(self):
        result = _format_vendor_details_from_atom(None)
        assert result == "_No vendor details_"

    def test_with_empty_dict(self):
        result = _format_vendor_details_from_atom({})
        assert result == "_No vendor details_"


class TestFormatPaymentRows:
    def test_single_payment(self):
        payments = [{"method": "card", "card_last4": "7201", "amount": "85.69"}]
        result = _format_payment_rows(payments)
        assert "**Payment**" in result
        assert "Card" in result
        assert "*7201" in result

    def test_multiple_payments(self):
        payments = [
            {"method": "card", "amount": "500"},
            {"method": "transfer", "amount": "500"},
        ]
        result = _format_payment_rows(payments)
        assert "**Payment 1**" in result
        assert "**Payment 2**" in result

    def test_no_payments(self):
        result = _format_payment_rows(None)
        assert result == ""


class TestFormatFactItemsTable:
    def test_with_items(self):
        items = [
            {
                "name": "Red Bull 250ml",
                "brand": "Red Bull",
                "quantity": 3,
                "unit": "pcs",
                "unit_price": 1.99,
                "total_price": 5.97,
                "category": "beverages",
            }
        ]
        result = _format_fact_items_table(items)
        assert "Red Bull 250ml" in result
        assert "1.99" in result
        assert "5.97" in result

    def test_empty_items(self):
        result = _format_fact_items_table([])
        assert result == "_No line items_"


class TestGenerateFactNote:
    def test_basic_note_structure(self):
        fact = {
            "id": "fact-001",
            "fact_type": "purchase",
            "vendor": "PAPAS HYPERMARKET",
            "total_amount": 85.69,
            "currency": "EUR",
            "event_date": date(2026, 1, 21),
            "status": "confirmed",
            "payments": json.dumps(
                [{"method": "card", "card_last4": "7201", "amount": "85.69"}]
            ),
        }
        result = generate_fact_note(fact)

        # Check YAML frontmatter
        assert "---" in result
        assert "type: fact" in result
        assert 'id: "fact-001"' in result
        assert 'vendor: "PAPAS HYPERMARKET"' in result
        assert "amount: 85.69" in result

        # Check markdown body
        assert "# PAPAS HYPERMARKET" in result
        assert "2026-01-21" in result

    def test_with_items(self):
        fact = {
            "id": "fact-002",
            "fact_type": "purchase",
            "vendor": "STORE",
            "total_amount": 10.0,
            "currency": "EUR",
            "event_date": "2026-01-15",
            "status": "confirmed",
            "payments": None,
        }
        items = [
            {
                "name": "Milk 1L",
                "quantity": 2,
                "unit": "pcs",
                "unit_price": 1.50,
                "total_price": 3.00,
                "brand": None,
                "category": "dairy",
                "comparable_unit_price": 1.50,
                "comparable_unit": "l",
            }
        ]
        result = generate_fact_note(fact, items=items)
        assert "Milk 1L" in result
        assert "dairy" in result

    def test_with_documents(self):
        fact = {
            "id": "fact-003",
            "fact_type": "purchase",
            "vendor": "SHOP",
            "total_amount": 5.0,
            "currency": "EUR",
            "event_date": "2026-02-01",
            "status": "confirmed",
            "payments": None,
        }
        documents = [
            {"id": "doc-001", "file_path": "/inbox/receipt.jpg"},
            {"id": "doc-002", "file_path": "/inbox/card_slip.jpg"},
        ]
        result = generate_fact_note(fact, documents=documents)
        assert "[[receipt.jpg]]" in result
        assert "[[card_slip.jpg]]" in result

    def test_with_vendor_atom(self):
        fact = {
            "id": "fact-004",
            "fact_type": "purchase",
            "vendor": "ACME",
            "total_amount": 100.0,
            "currency": "EUR",
            "event_date": "2026-01-10",
            "status": "confirmed",
            "payments": None,
        }
        vendor_atom = {
            "name": "ACME Corp",
            "address": "456 Oak Ave",
            "website": "https://acme.com",
        }
        result = generate_fact_note(fact, vendor_atom=vendor_atom)
        assert "**Address**: 456 Oak Ave" in result
        assert "**Website**: https://acme.com" in result

    def test_with_tags(self):
        fact = {
            "id": "fact-005",
            "fact_type": "purchase",
            "vendor": "TAGGED",
            "total_amount": 1.0,
            "currency": "EUR",
            "event_date": "2026-01-01",
            "status": "confirmed",
            "payments": None,
        }
        result = generate_fact_note(fact, tags=["groceries", "weekly"])
        assert '"groceries"' in result
        assert '"weekly"' in result

    def test_none_amount(self):
        fact = {
            "id": "fact-006",
            "fact_type": "purchase",
            "vendor": "UNKNOWN",
            "total_amount": None,
            "currency": "EUR",
            "event_date": None,
            "status": "partial",
            "payments": None,
        }
        result = generate_fact_note(fact)
        assert "amount: 0" in result
        assert "N/A" in result  # format_currency returns N/A for None


class TestNoteExporterFact:
    def test_export_fact(self, db):
        fact, _, _ = _populate_full_fact(db)

        with tempfile.TemporaryDirectory() as vault:
            exporter = NoteExporter(db, vault_path=Path(vault))
            fact_dict = v2_store.get_fact_by_id(db, fact.id)
            assert fact_dict is not None
            path = exporter.export_fact(fact_dict)

            assert path.exists()
            content = path.read_text()
            assert "PAPAS HYPERMARKET" in content
            assert "Red Bull 250ml" in content
            assert "receipt_2026-01-21.jpg" in content
            assert "Panayioti Tsangari 23" in content

    def test_export_all_facts(self, db):
        _populate_full_fact(db)

        with tempfile.TemporaryDirectory() as vault:
            exporter = NoteExporter(db, vault_path=Path(vault))
            paths = exporter.export_all_facts()
            assert len(paths) == 1

    def test_export_all_facts_since_filter(self, db):
        _populate_full_fact(db)

        with tempfile.TemporaryDirectory() as vault:
            exporter = NoteExporter(db, vault_path=Path(vault))
            paths = exporter.export_all_facts(since=date(2026, 2, 1))
            assert len(paths) == 0

    def test_no_overwrite_by_default(self, db):
        fact, _, _ = _populate_full_fact(db)

        with tempfile.TemporaryDirectory() as vault:
            exporter = NoteExporter(db, vault_path=Path(vault))
            fact_dict = v2_store.get_fact_by_id(db, fact.id)
            assert fact_dict is not None
            path1 = exporter.export_fact(fact_dict)
            path2 = exporter.export_fact(fact_dict)
            assert path1 != path2
            assert path1.exists()
            assert path2.exists()
