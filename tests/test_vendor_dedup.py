"""Tests for vendor deduplication report."""

from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from alibi.analytics.vendors import vendor_deduplication_report
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
    FactType,
)
from alibi.db import v2_store


def _insert_fact(
    db,
    vendor: str,
    vendor_key: str | None,
    amount: str = "10.00",
    event_date: date | None = None,
):
    """Quick fact insertion for testing."""
    doc = Document(
        id=str(uuid4()),
        file_path=f"/test/{vendor}.jpg",
        file_hash=str(uuid4()),
        raw_extraction={"vendor": vendor},
    )
    v2_store.store_document(db, doc)

    atom = Atom(
        id=str(uuid4()),
        document_id=doc.id,
        atom_type=AtomType.AMOUNT,
        data={"value": amount},
    )
    v2_store.store_atoms(db, [atom])

    bundle = Bundle(
        id=str(uuid4()),
        document_id=doc.id,
        bundle_type=BundleType.BASKET,
    )
    ba = BundleAtom(bundle_id=bundle.id, atom_id=atom.id, role=BundleAtomRole.TOTAL)
    v2_store.store_bundle(db, bundle, [ba])

    cloud = Cloud(id=str(uuid4()), status=CloudStatus.COLLAPSED)
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
        vendor=vendor,
        vendor_key=vendor_key,
        total_amount=Decimal(amount),
        currency="EUR",
        event_date=event_date or date(2026, 1, 1),
    )
    v2_store.store_fact(db, fact, [])


class TestVendorDeduplicationReport:
    def test_detects_aliases(self, db):
        """Same vendor_key with different display names creates alias."""
        _insert_fact(db, "FRESKO", "VAT123", "100.00")
        _insert_fact(db, "Fresko Butanolo Ltd", "VAT123", "50.00")

        report = vendor_deduplication_report(db)
        assert len(report.aliases) == 1
        alias = report.aliases[0]
        assert alias.vendor_key == "VAT123"
        assert len(alias.names) == 2
        assert alias.fact_count == 2
        assert alias.total_amount == Decimal("150.00")

    def test_no_aliases_when_same_name(self, db):
        """Same vendor_key and same display name = no alias entry."""
        _insert_fact(db, "Shop A", "VAT456", "50.00")
        _insert_fact(db, "Shop A", "VAT456", "75.00")

        report = vendor_deduplication_report(db)
        assert len(report.aliases) == 0

    def test_unkeyed_vendors(self, db):
        """Vendors without vendor_key appear in unkeyed list."""
        _insert_fact(db, "Unknown Shop", None)

        report = vendor_deduplication_report(db)
        assert "Unknown Shop" in report.unkeyed_vendors

    def test_empty_database(self, db):
        report = vendor_deduplication_report(db)
        assert report.total_vendors == 0
        assert report.aliases == []
        assert report.unkeyed_vendors == []

    def test_most_frequent_name_first(self, db):
        """Names sorted by frequency descending."""
        _insert_fact(db, "FreSko", "VAT123")
        _insert_fact(db, "FreSko", "VAT123")
        _insert_fact(db, "FRESKO BUTANOLO", "VAT123")

        report = vendor_deduplication_report(db)
        alias = report.aliases[0]
        assert alias.names[0] == "FreSko"  # Most frequent
        assert alias.names[1] == "FRESKO BUTANOLO"

    def test_total_vendors_count(self, db):
        _insert_fact(db, "Shop A", "VAT1")
        _insert_fact(db, "Shop B", "VAT2")
        _insert_fact(db, "Shop C", None)

        report = vendor_deduplication_report(db)
        assert report.total_vendors == 3

    def test_dedup_different_vendor_keys_same_name(self, db):
        """Same normalized name but different keys = separate entries, no alias."""
        _insert_fact(db, "Shop A", "VAT1")
        _insert_fact(db, "Shop A", "VAT2")

        report = vendor_deduplication_report(db)
        # Each key has only 1 name → no aliases
        assert len(report.aliases) == 0

    def test_dedup_same_vendor_key_multiple_names(self, db):
        """One key with 3+ name variants."""
        _insert_fact(db, "FRESKO", "VAT123", "10.00")
        _insert_fact(db, "Fresko Butanolo", "VAT123", "20.00")
        _insert_fact(db, "fresko butanolo ltd", "VAT123", "30.00")

        report = vendor_deduplication_report(db)
        assert len(report.aliases) == 1
        alias = report.aliases[0]
        assert len(alias.names) == 3
        assert alias.fact_count == 3
        assert alias.total_amount == Decimal("60.00")

    def test_dedup_no_duplicates(self, db):
        """Clean data with unique names per key returns empty aliases."""
        _insert_fact(db, "Shop A", "VAT1")
        _insert_fact(db, "Shop B", "VAT2")
        _insert_fact(db, "Shop C", "VAT3")

        report = vendor_deduplication_report(db)
        assert report.aliases == []
        assert report.total_vendors == 3

    def test_dedup_with_name_hash_keys(self, db):
        """noid_ prefix keys (name-based hashes) handled correctly."""
        _insert_fact(db, "Local Shop", "noid_abc123")
        _insert_fact(db, "LOCAL SHOP", "noid_abc123")

        report = vendor_deduplication_report(db)
        assert len(report.aliases) == 1
        assert report.aliases[0].vendor_key == "noid_abc123"

    def test_dedup_sorted_by_group_size(self, db):
        """Aliases sorted by fact_count descending (largest groups first)."""
        # Group 1: 2 facts
        _insert_fact(db, "Shop A", "VAT1")
        _insert_fact(db, "SHOP A LTD", "VAT1")

        # Group 2: 4 facts
        _insert_fact(db, "BigStore", "VAT2")
        _insert_fact(db, "BIG STORE", "VAT2")
        _insert_fact(db, "Big Store Ltd", "VAT2")
        _insert_fact(db, "BIGSTORE", "VAT2")

        report = vendor_deduplication_report(db)
        assert len(report.aliases) == 2
        assert report.aliases[0].fact_count > report.aliases[1].fact_count
