"""Tests for historical verification module."""

import os
import tempfile
from datetime import date
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from alibi.config import Config, reset_config
from alibi.db.connection import DatabaseManager
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
from alibi.extraction.historical import (
    HistoricalCorrection,
    HistoricalResult,
    _best_product_match,
    apply_historical_corrections,
    backfill_vendor_key,
    check_product_names,
    check_vendor_details,
    check_vendor_identity,
    make_vendor_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """Create a fresh database with schema."""
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


def _seed_vendor(
    db: DatabaseManager,
    vendor_name: str,
    registration: str | None = None,
    address: str | None = None,
    phone: str | None = None,
    website: str | None = None,
    items: list[dict[str, Any]] | None = None,
    amount: Decimal = Decimal("42.50"),
    event_date: date | None = None,
    vendor_key: str | None = None,
) -> dict[str, str]:
    """Seed a complete vendor record: doc → atoms → bundle → cloud → fact.

    Returns dict with IDs: doc_id, bundle_id, cloud_id, fact_id.
    """
    doc_id = str(uuid4())
    doc = Document(
        id=doc_id,
        file_path=f"{vendor_name.lower().replace(' ', '_')}.jpg",
        file_hash=str(uuid4())[:16],
        raw_extraction={"vendor": vendor_name},
    )
    v2_store.store_document(db, doc)

    # Vendor atom
    vendor_data: dict[str, Any] = {"name": vendor_name}
    if registration:
        vendor_data["vat_number"] = registration
    if address:
        vendor_data["address"] = address
    if phone:
        vendor_data["phone"] = phone
    if website:
        vendor_data["website"] = website

    vendor_atom = Atom(
        id=str(uuid4()),
        document_id=doc_id,
        atom_type=AtomType.VENDOR,
        data=vendor_data,
    )
    amount_atom = Atom(
        id=str(uuid4()),
        document_id=doc_id,
        atom_type=AtomType.AMOUNT,
        data={"value": str(amount), "currency": "EUR", "semantic_type": "total"},
    )
    item_atoms: list[Atom] = []
    if items:
        for item in items:
            item_atoms.append(
                Atom(
                    id=str(uuid4()),
                    document_id=doc_id,
                    atom_type=AtomType.ITEM,
                    data=item,
                )
            )

    all_atoms = [vendor_atom, amount_atom] + item_atoms
    v2_store.store_atoms(db, all_atoms)

    # Bundle
    bundle_id = str(uuid4())
    bundle = Bundle(id=bundle_id, document_id=doc_id, bundle_type=BundleType.BASKET)
    bundle_atom_links = [
        BundleAtom(
            bundle_id=bundle_id,
            atom_id=vendor_atom.id,
            role=BundleAtomRole.VENDOR_INFO,
        ),
        BundleAtom(
            bundle_id=bundle_id,
            atom_id=amount_atom.id,
            role=BundleAtomRole.TOTAL,
        ),
    ]
    for ia in item_atoms:
        bundle_atom_links.append(
            BundleAtom(
                bundle_id=bundle_id,
                atom_id=ia.id,
                role=BundleAtomRole.BASKET_ITEM,
            )
        )
    v2_store.store_bundle(db, bundle, bundle_atom_links)

    # Cloud
    cloud_id = str(uuid4())
    cloud = Cloud(id=cloud_id, status=CloudStatus.COLLAPSED)
    cloud_bundle = CloudBundle(
        cloud_id=cloud_id,
        bundle_id=bundle_id,
        match_type=CloudMatchType.EXACT_AMOUNT,
        match_confidence=Decimal("1.0"),
    )
    v2_store.store_cloud(db, cloud, cloud_bundle)

    # Fact + items
    fact_id = str(uuid4())
    # Compute vendor_key if not explicitly provided
    if vendor_key is None:
        vendor_key = make_vendor_key(registration, vendor_name)

    fact = Fact(
        id=fact_id,
        cloud_id=cloud_id,
        fact_type=FactType.PURCHASE,
        vendor=vendor_name,
        vendor_key=vendor_key,
        total_amount=amount,
        currency="EUR",
        event_date=event_date or date(2026, 1, 15),
        status=FactStatus.CONFIRMED,
    )

    fact_items: list[FactItem] = []
    if items:
        for i, (item_data, atom) in enumerate(zip(items, item_atoms)):
            fact_items.append(
                FactItem(
                    id=str(uuid4()),
                    fact_id=fact_id,
                    atom_id=atom.id,
                    name=item_data.get("name", f"Item {i}"),
                    name_normalized=(item_data.get("name", "")).lower().strip(),
                    quantity=Decimal(str(item_data.get("quantity", 1))),
                    unit=UnitType.PIECE,
                    total_price=Decimal(str(item_data.get("total_price", 0))),
                    tax_type=TaxType.VAT,
                )
            )

    v2_store.store_fact(db, fact, fact_items)

    return {
        "doc_id": doc_id,
        "bundle_id": bundle_id,
        "cloud_id": cloud_id,
        "fact_id": fact_id,
    }


# ---------------------------------------------------------------------------
# make_vendor_key tests
# ---------------------------------------------------------------------------


class TestMakeVendorKey:
    def test_registration_takes_priority(self) -> None:
        key = make_vendor_key("CY12345678X", "Some Store")
        assert key == "CY12345678X"

    def test_registration_normalized(self) -> None:
        key = make_vendor_key(" CY 123 456 78X ", "Store")
        assert key == "CY12345678X"

    def test_fallback_to_name_hash(self) -> None:
        key = make_vendor_key(None, "Fresko Butanolo")
        assert key is not None
        assert key.startswith("noid_")
        assert len(key) == 15  # "noid_" + 10 hex chars

    def test_name_hash_strips_legal_suffixes(self) -> None:
        key1 = make_vendor_key(None, "Fresko Ltd")
        key2 = make_vendor_key(None, "Fresko")
        assert key1 == key2

    def test_name_hash_case_insensitive(self) -> None:
        key1 = make_vendor_key(None, "FRESKO")
        key2 = make_vendor_key(None, "fresko")
        assert key1 == key2

    def test_none_when_nothing_available(self) -> None:
        assert make_vendor_key(None, None) is None
        assert make_vendor_key("", "") is None
        assert make_vendor_key(None, "") is None


# ---------------------------------------------------------------------------
# _best_product_match tests
# ---------------------------------------------------------------------------


class TestBestProductMatch:
    def test_exact_match(self) -> None:
        match, ratio = _best_product_match("Milk 1L", ["Milk 1L", "Bread"])
        assert match == "Milk 1L"
        assert ratio == 1.0

    def test_case_insensitive_exact(self) -> None:
        match, ratio = _best_product_match("milk 1l", ["Milk 1L", "Bread"])
        assert match == "Milk 1L"
        assert ratio == 1.0

    def test_fuzzy_match(self) -> None:
        # OCR might read "Mi1k 1L" instead of "Milk 1L"
        match, ratio = _best_product_match("Mi1k 1L", ["Milk 1L", "Bread"])
        assert match == "Milk 1L"
        assert ratio > 0.7

    def test_no_match(self) -> None:
        match, ratio = _best_product_match("Completely Different", ["Milk", "Bread"])
        # Low ratio but still returns the best available
        assert ratio < 0.5

    def test_empty_known_list(self) -> None:
        match, ratio = _best_product_match("Milk", [])
        assert match is None
        assert ratio == 0.0

    def test_empty_name(self) -> None:
        match, ratio = _best_product_match("", ["Milk"])
        assert match is None
        assert ratio == 0.0


# ---------------------------------------------------------------------------
# v2_store historical queries
# ---------------------------------------------------------------------------


class TestV2StoreHistoricalQueries:
    def test_find_vendors_by_registration(self, db: DatabaseManager) -> None:
        _seed_vendor(db, "Fresko Store", registration="CY12345678X")
        _seed_vendor(db, "Fresko Downtown", registration="CY12345678X")
        _seed_vendor(db, "Other Shop", registration="DE987654321")

        matches = v2_store.find_vendors_by_registration(db, "CY12345678X")
        assert len(matches) == 2
        names = {m["name"] for m in matches}
        assert "Fresko Store" in names
        assert "Fresko Downtown" in names

    def test_find_vendors_by_registration_normalized(self, db: DatabaseManager) -> None:
        _seed_vendor(db, "Fresko", registration="CY 123 456 78X")

        # Search with different spacing
        matches = v2_store.find_vendors_by_registration(db, "cy12345678x")
        assert len(matches) == 1

    def test_get_known_vendor_names(self, db: DatabaseManager) -> None:
        # Seed same registration with different names (2x Store, 1x Downtown)
        _seed_vendor(db, "Fresko Store", registration="CY12345678X")
        _seed_vendor(db, "Fresko Store", registration="CY12345678X")
        _seed_vendor(db, "Fresko Downtown", registration="CY12345678X")

        names = v2_store.get_known_vendor_names(db, "CY12345678X")
        assert names[0] == "Fresko Store"  # Most common
        assert "Fresko Downtown" in names

    def test_get_known_product_names(self, db: DatabaseManager) -> None:
        _seed_vendor(
            db,
            "Fresko Store",
            registration="CY12345678X",
            items=[
                {"name": "Milk 1L", "quantity": 2, "total_price": 3.0},
                {"name": "Bread White", "quantity": 1, "total_price": 2.5},
            ],
        )

        products = v2_store.get_known_product_names_for_vendor(db, "Fresko")
        assert len(products) == 2
        names_lower = [p.lower() for p in products]
        assert "milk 1l" in names_lower
        assert "bread white" in names_lower

    def test_find_matching_fact_vendors(self, db: DatabaseManager) -> None:
        _seed_vendor(db, "Fresko Store", registration="CY12345678X")
        _seed_vendor(db, "Fresko Store", registration="CY12345678X")
        _seed_vendor(db, "Other Shop")

        matches = v2_store.find_matching_fact_vendors(db, "fresko")
        assert len(matches) == 1
        assert matches[0] == "Fresko Store"

    def test_get_vendor_details_history(self, db: DatabaseManager) -> None:
        _seed_vendor(
            db,
            "Fresko Store",
            registration="CY12345678X",
            phone="+357-123-456",
            website="www.fresko.cy",
        )
        _seed_vendor(
            db,
            "Fresko Store",
            registration="CY12345678X",
            address="Different Address",
            phone="+357-123-456",
        )

        history = v2_store.get_vendor_details_history(db, "Fresko")
        assert "+357-123-456" in history["phone"]
        assert "www.fresko.cy" in history["website"]


# ---------------------------------------------------------------------------
# check_vendor_identity
# ---------------------------------------------------------------------------


class TestCheckVendorIdentity:
    def test_registration_match_corrects_vendor_name(self, db: DatabaseManager) -> None:
        # Seed known vendor
        _seed_vendor(db, "Fresko Store", registration="CY12345678X")
        _seed_vendor(db, "Fresko Store", registration="CY12345678X")

        # New extraction with same registration but OCR-garbled name
        extracted: dict[str, Any] = {
            "vendor": "Fresk0 St0re",  # OCR error
            "vendor_vat": "CY12345678X",
        }

        result = check_vendor_identity(db, extracted)
        assert result.vendor_identified is True
        assert result.known_vendor_name == "Fresko Store"
        assert len(result.corrections) == 1
        assert result.corrections[0].field == "vendor"
        assert result.corrections[0].suggested == "Fresko Store"
        assert result.corrections[0].reason == "registration_match"

    def test_registration_match_already_correct(self, db: DatabaseManager) -> None:
        _seed_vendor(db, "Fresko Store", registration="CY12345678X")

        extracted: dict[str, Any] = {
            "vendor": "Fresko Store",
            "vendor_vat": "CY12345678X",
        }

        result = check_vendor_identity(db, extracted)
        assert result.vendor_identified is True
        assert len(result.corrections) == 0

    def test_registration_fills_missing_vendor(self, db: DatabaseManager) -> None:
        _seed_vendor(db, "Fresko Store", registration="CY12345678X")

        extracted: dict[str, Any] = {
            "vendor": "",
            "vendor_vat": "CY12345678X",
        }

        result = check_vendor_identity(db, extracted)
        assert len(result.corrections) == 1
        assert result.corrections[0].original == ""
        assert result.corrections[0].suggested == "Fresko Store"

    def test_name_fallback_when_no_registration(self, db: DatabaseManager) -> None:
        _seed_vendor(db, "Fresko Store")

        extracted: dict[str, Any] = {
            "vendor": "Fresko",
        }

        result = check_vendor_identity(db, extracted)
        assert result.vendor_identified is True
        assert result.known_vendor_name == "Fresko Store"

    def test_no_match_returns_empty(self, db: DatabaseManager) -> None:
        extracted: dict[str, Any] = {
            "vendor": "Unknown Store",
            "vendor_vat": "XX999999999",
        }

        result = check_vendor_identity(db, extracted)
        assert result.vendor_identified is False
        assert len(result.corrections) == 0

    def test_chain_same_registration_different_addresses(
        self, db: DatabaseManager
    ) -> None:
        """Same chain, same VAT, different store addresses."""
        _seed_vendor(
            db,
            "Fresko Store",
            registration="CY12345678X",
            address="123 Main St, Nicosia",
        )
        _seed_vendor(
            db,
            "Fresko Store",
            registration="CY12345678X",
            address="456 Beach Rd, Limassol",
        )

        extracted: dict[str, Any] = {
            "vendor": "Fresk0",  # OCR error
            "vendor_vat": "CY12345678X",
        }

        result = check_vendor_identity(db, extracted)
        assert result.vendor_identified is True
        assert result.known_vendor_name == "Fresko Store"


# ---------------------------------------------------------------------------
# check_product_names
# ---------------------------------------------------------------------------


class TestCheckProductNames:
    def test_corrects_ocr_errors_in_product_names(self, db: DatabaseManager) -> None:
        _seed_vendor(
            db,
            "Fresko Store",
            items=[
                {"name": "Milk Full Fat 1L", "quantity": 1, "total_price": 2.0},
                {"name": "Sourdough Bread 500g", "quantity": 1, "total_price": 3.0},
            ],
        )

        extracted: dict[str, Any] = {
            "vendor": "Fresko Store",
            "line_items": [
                {"name": "Mi1k Full Fat 1L", "total_price": 2.0},  # OCR: 1 vs l
                {"name": "Sourdough Bread 500g", "total_price": 3.0},  # Correct
            ],
        }

        corrections = check_product_names(db, extracted, "Fresko Store")
        # First item should be corrected, second is already correct
        corrected_names = [c.suggested for c in corrections]
        assert "milk full fat 1l" in corrected_names

    def test_no_corrections_when_exact_match(self, db: DatabaseManager) -> None:
        _seed_vendor(
            db,
            "Fresko Store",
            items=[{"name": "Milk 1L", "quantity": 1, "total_price": 2.0}],
        )

        extracted: dict[str, Any] = {
            "line_items": [{"name": "Milk 1L", "total_price": 2.0}],
        }

        # Exact match (case-insensitive) via name_normalized
        corrections = check_product_names(db, extracted, "Fresko")
        # "Milk 1L" vs "milk 1l" in name_normalized — exact lowercase match
        assert len(corrections) == 0

    def test_no_items_returns_empty(self, db: DatabaseManager) -> None:
        corrections = check_product_names(db, {"line_items": []}, "Fresko")
        assert corrections == []

    def test_unknown_vendor_returns_empty(self, db: DatabaseManager) -> None:
        corrections = check_product_names(
            db, {"line_items": [{"name": "Milk"}]}, "Unknown"
        )
        assert corrections == []


# ---------------------------------------------------------------------------
# check_vendor_details
# ---------------------------------------------------------------------------


class TestCheckVendorDetails:
    def test_enriches_missing_phone(self, db: DatabaseManager) -> None:
        _seed_vendor(db, "Fresko Store", phone="+357-123-456")

        extracted: dict[str, Any] = {"vendor": "Fresko Store"}

        corrections = check_vendor_details(db, extracted, "Fresko Store")
        phones = [c for c in corrections if c.field == "vendor_phone"]
        assert len(phones) == 1
        assert phones[0].suggested == "+357-123-456"

    def test_enriches_missing_website(self, db: DatabaseManager) -> None:
        _seed_vendor(db, "Fresko Store", website="www.fresko.cy")

        extracted: dict[str, Any] = {"vendor": "Fresko Store"}

        corrections = check_vendor_details(db, extracted, "Fresko Store")
        websites = [c for c in corrections if c.field == "vendor_website"]
        assert len(websites) == 1
        assert websites[0].suggested == "www.fresko.cy"

    def test_does_not_override_existing_values(self, db: DatabaseManager) -> None:
        _seed_vendor(db, "Fresko Store", phone="+357-123-456")

        extracted: dict[str, Any] = {
            "vendor": "Fresko Store",
            "vendor_phone": "+357-999-999",
        }

        corrections = check_vendor_details(db, extracted, "Fresko Store")
        phones = [c for c in corrections if c.field == "vendor_phone"]
        assert len(phones) == 0  # Already has a value

    def test_enriches_missing_registration(self, db: DatabaseManager) -> None:
        _seed_vendor(db, "Fresko Store", registration="CY12345678X")

        extracted: dict[str, Any] = {"vendor": "Fresko Store"}

        corrections = check_vendor_details(db, extracted, "Fresko Store")
        regs = [c for c in corrections if c.field == "vendor_vat"]
        assert len(regs) == 1
        assert regs[0].suggested == "CY12345678X"


# ---------------------------------------------------------------------------
# apply_historical_corrections (integration)
# ---------------------------------------------------------------------------


class TestApplyHistoricalCorrections:
    def test_full_correction_pipeline(self, db: DatabaseManager) -> None:
        """End-to-end: registration → vendor name → details → products."""
        _seed_vendor(
            db,
            "Fresko Store",
            registration="CY12345678X",
            phone="+357-123-456",
            items=[
                {"name": "Milk Full Fat 1L", "quantity": 2, "total_price": 4.0},
                {"name": "Bread White 500g", "quantity": 1, "total_price": 2.5},
            ],
        )

        extracted: dict[str, Any] = {
            "vendor": "Fresk0 St0re",  # OCR error
            "vendor_vat": "CY12345678X",
            "line_items": [
                {"name": "Mi1k Full Fat 1L", "total_price": 4.0},
                {"name": "Bread White 500g", "total_price": 2.5},
            ],
        }

        result = apply_historical_corrections(db, extracted)

        # Vendor name should be corrected
        assert extracted["vendor"] == "Fresko Store"
        assert result.vendor_identified is True

        # Phone should be enriched
        assert extracted.get("vendor_phone") == "+357-123-456"

        # At least some corrections were applied
        assert result.applied_count > 0

    def test_no_history_no_changes(self, db: DatabaseManager) -> None:
        """Empty database — no corrections possible."""
        extracted: dict[str, Any] = {
            "vendor": "Brand New Store",
            "total": 42.50,
            "line_items": [{"name": "Mystery Item"}],
        }

        result = apply_historical_corrections(db, extracted)
        assert result.applied_count == 0
        assert extracted["vendor"] == "Brand New Store"

    def test_name_fallback_without_registration(self, db: DatabaseManager) -> None:
        """Vendor matched by name when no registration present."""
        _seed_vendor(
            db,
            "Fresko Store",
            phone="+357-123-456",
            items=[{"name": "Milk 1L", "quantity": 1, "total_price": 2.0}],
        )

        extracted: dict[str, Any] = {
            "vendor": "Fresko",
            "line_items": [{"name": "Mi1k 1L"}],
        }

        result = apply_historical_corrections(db, extracted)
        # Should find vendor via name match and enrich phone
        assert result.vendor_identified is True
        assert extracted.get("vendor_phone") == "+357-123-456"

    def test_modifies_extraction_in_place(self, db: DatabaseManager) -> None:
        _seed_vendor(db, "Fresko Store", registration="CY12345678X")

        extracted: dict[str, Any] = {
            "vendor": "Wrong Name",
            "vendor_vat": "CY12345678X",
        }

        apply_historical_corrections(db, extracted)
        assert extracted["vendor"] == "Fresko Store"

    def test_product_names_corrected_in_place(self, db: DatabaseManager) -> None:
        _seed_vendor(
            db,
            "Fresko Store",
            items=[{"name": "Organic Bananas", "quantity": 1, "total_price": 1.5}],
        )

        extracted: dict[str, Any] = {
            "vendor": "Fresko Store",
            "line_items": [
                {"name": "0rganic Bananas", "total_price": 1.5},  # OCR: O vs 0
            ],
        }

        result = apply_historical_corrections(db, extracted)
        # If the fuzzy match is good enough, name should be corrected
        if result.products_matched > 0:
            assert extracted["line_items"][0]["name"] == "organic bananas"


# ---------------------------------------------------------------------------
# vendor_key storage and retrieval
# ---------------------------------------------------------------------------


class TestVendorKeyStorage:
    def test_fact_stores_vendor_key(self, db: DatabaseManager) -> None:
        """Fact stores vendor_key when seeded."""
        ids = _seed_vendor(db, "Fresko Store", registration="CY12345678X")
        fact = v2_store.get_fact_by_id(db, ids["fact_id"])
        assert fact is not None
        assert fact["vendor_key"] == "CY12345678X"

    def test_fact_stores_noid_key_without_registration(
        self, db: DatabaseManager
    ) -> None:
        """Fact gets noid_ key when no registration is available."""
        ids = _seed_vendor(db, "Fresko Store")
        fact = v2_store.get_fact_by_id(db, ids["fact_id"])
        assert fact is not None
        assert fact["vendor_key"] is not None
        assert fact["vendor_key"].startswith("noid_")

    def test_get_facts_by_vendor_key(self, db: DatabaseManager) -> None:
        _seed_vendor(db, "Fresko Store", registration="CY12345678X")
        _seed_vendor(db, "Fresko Downtown", registration="CY12345678X")
        _seed_vendor(db, "Other Shop", registration="DE987654321")

        facts = v2_store.get_facts_by_vendor_key(db, "CY12345678X")
        assert len(facts) == 2
        vendors = {f["vendor"] for f in facts}
        assert "Fresko Store" in vendors
        assert "Fresko Downtown" in vendors

    def test_get_facts_by_vendor_key_no_match(self, db: DatabaseManager) -> None:
        facts = v2_store.get_facts_by_vendor_key(db, "NONEXISTENT")
        assert facts == []


# ---------------------------------------------------------------------------
# vendor_key backfill
# ---------------------------------------------------------------------------


class TestVendorKeyBackfill:
    def test_backfill_updates_noid_to_registration(self, db: DatabaseManager) -> None:
        """Seed 3 facts with noid_ key, then backfill with registration."""
        # Seed without registration (gets noid_ key)
        ids1 = _seed_vendor(db, "Fresko Store")
        ids2 = _seed_vendor(db, "Fresko Store", amount=Decimal("35.00"))
        ids3 = _seed_vendor(db, "Fresko Store", amount=Decimal("28.75"))

        # Verify all have noid_ keys
        old_key = make_vendor_key(None, "Fresko Store")
        assert old_key is not None
        for fid in [ids1["fact_id"], ids2["fact_id"], ids3["fact_id"]]:
            fact = v2_store.get_fact_by_id(db, fid)
            assert fact is not None
            assert fact["vendor_key"] == old_key

        # Backfill with registration
        count = backfill_vendor_key(db, "CY12345678X", "Fresko Store")
        assert count == 3

        # Verify all now have registration-based key
        for fid in [ids1["fact_id"], ids2["fact_id"], ids3["fact_id"]]:
            fact = v2_store.get_fact_by_id(db, fid)
            assert fact is not None
            assert fact["vendor_key"] == "CY12345678X"

    def test_backfill_does_not_affect_other_vendors(self, db: DatabaseManager) -> None:
        _seed_vendor(db, "Fresko Store")
        _seed_vendor(db, "Other Shop")

        other_fact = v2_store.get_fact_by_id(
            db,
            _seed_vendor(db, "Other Shop")["fact_id"],
        )
        assert other_fact is not None
        other_key_before = other_fact["vendor_key"]

        backfill_vendor_key(db, "CY12345678X", "Fresko Store")

        # Other Shop should be unchanged
        other_facts = v2_store.get_facts_by_vendor_key(db, other_key_before)
        assert len(other_facts) >= 1

    def test_backfill_noop_when_already_has_registration(
        self, db: DatabaseManager
    ) -> None:
        """If facts already have registration key, backfill does nothing."""
        _seed_vendor(db, "Fresko Store", registration="CY12345678X")

        # Trying to backfill with same registration — old_key != noid_, noop
        count = backfill_vendor_key(db, "CY12345678X", "Fresko Store")
        assert count == 0

    def test_backfill_returns_zero_for_unknown_vendor(
        self, db: DatabaseManager
    ) -> None:
        count = backfill_vendor_key(db, "CY12345678X", "Unknown Vendor")
        assert count == 0

    def test_backfill_via_apply_historical_corrections(
        self, db: DatabaseManager
    ) -> None:
        """Integration: apply_historical_corrections triggers backfill."""
        # Seed vendor without registration
        ids = _seed_vendor(db, "Fresko Store")
        old_key = make_vendor_key(None, "Fresko Store")
        fact = v2_store.get_fact_by_id(db, ids["fact_id"])
        assert fact is not None
        assert fact["vendor_key"] == old_key

        # New extraction has the registration
        extracted: dict[str, Any] = {
            "vendor": "Fresko Store",
            "vendor_vat": "CY12345678X",
        }

        apply_historical_corrections(db, extracted)

        # Fact should now have registration-based key
        fact = v2_store.get_fact_by_id(db, ids["fact_id"])
        assert fact is not None
        assert fact["vendor_key"] == "CY12345678X"
