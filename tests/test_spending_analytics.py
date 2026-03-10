"""Tests for v2 spending analytics module."""

from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from alibi.analytics.spending import (
    ItemFrequency,
    MonthlySpend,
    SeasonalPattern,
    VendorSpend,
    item_frequency,
    seasonal_patterns,
    spending_by_month,
    spending_by_vendor,
)
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
    UnitType,
)
from alibi.db import v2_store


def _create_fact(
    db,
    vendor: str,
    amount: str,
    event_date: date,
    vendor_key: str | None = None,
    items: list[tuple[str, str]] | None = None,
) -> Fact:
    """Helper to create a full fact with document/atoms/bundle/cloud chain."""
    doc_id = str(uuid4())
    doc = Document(
        id=doc_id,
        file_path=f"/test/{vendor.lower()}.jpg",
        file_hash=str(uuid4()),
        raw_extraction={"vendor": vendor, "total": amount},
    )
    v2_store.store_document(db, doc)

    atoms = [
        Atom(
            id=str(uuid4()),
            document_id=doc_id,
            atom_type=AtomType.VENDOR,
            data={"name": vendor},
        ),
        Atom(
            id=str(uuid4()),
            document_id=doc_id,
            atom_type=AtomType.AMOUNT,
            data={"value": amount, "currency": "EUR"},
        ),
    ]
    v2_store.store_atoms(db, atoms)

    bundle = Bundle(
        id=str(uuid4()),
        document_id=doc_id,
        bundle_type=BundleType.BASKET,
    )
    bundle_atoms = [
        BundleAtom(
            bundle_id=bundle.id,
            atom_id=atoms[0].id,
            role=BundleAtomRole.VENDOR_INFO,
        ),
        BundleAtom(
            bundle_id=bundle.id,
            atom_id=atoms[1].id,
            role=BundleAtomRole.TOTAL,
        ),
    ]
    v2_store.store_bundle(db, bundle, bundle_atoms)

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
        event_date=event_date,
    )

    fact_items = []
    if items:
        for item_name, item_price in items:
            item_atom = Atom(
                id=str(uuid4()),
                document_id=doc_id,
                atom_type=AtomType.ITEM,
                data={"name": item_name, "total_price": item_price},
            )
            v2_store.store_atoms(db, [item_atom])
            fact_items.append(
                FactItem(
                    id=str(uuid4()),
                    fact_id=fact.id,
                    atom_id=item_atom.id,
                    name=item_name,
                    name_normalized=item_name.lower(),
                    quantity=Decimal("1"),
                    unit=UnitType.PIECE,
                    total_price=Decimal(item_price),
                )
            )

    v2_store.store_fact(db, fact, fact_items)
    return fact


class TestSpendingByVendor:
    def test_basic_vendor_ranking(self, db):
        _create_fact(db, "Shop A", "100.00", date(2026, 1, 1))
        _create_fact(db, "Shop A", "50.00", date(2026, 1, 15))
        _create_fact(db, "Shop B", "200.00", date(2026, 1, 10))

        result = spending_by_vendor(db)
        assert len(result) == 2
        assert result[0].vendor == "Shop B"
        assert result[0].total == Decimal("200.00")
        assert result[1].vendor == "Shop A"
        assert result[1].total == Decimal("150.00")
        assert result[1].count == 2

    def test_vendor_share_percentage(self, db):
        _create_fact(db, "Shop A", "75.00", date(2026, 1, 1))
        _create_fact(db, "Shop B", "25.00", date(2026, 1, 2))

        result = spending_by_vendor(db)
        assert result[0].share_pct == 75.0
        assert result[1].share_pct == 25.0

    def test_date_filtering(self, db):
        _create_fact(db, "Shop A", "100.00", date(2026, 1, 1))
        _create_fact(db, "Shop A", "50.00", date(2026, 3, 1))

        result = spending_by_vendor(
            db, date_from=date(2026, 2, 1), date_to=date(2026, 4, 1)
        )
        assert len(result) == 1
        assert result[0].total == Decimal("50.00")

    def test_empty_database(self, db):
        result = spending_by_vendor(db)
        assert result == []

    def test_groups_by_vendor_key(self, db):
        _create_fact(db, "FRESKO", "100.00", date(2026, 1, 1), vendor_key="VAT123")
        _create_fact(
            db, "Fresko Butanolo", "50.00", date(2026, 1, 5), vendor_key="VAT123"
        )

        result = spending_by_vendor(db)
        assert len(result) == 1
        assert result[0].total == Decimal("150.00")
        assert result[0].count == 2


class TestSpendingByMonth:
    def test_monthly_totals(self, db):
        _create_fact(db, "Shop A", "100.00", date(2026, 1, 5))
        _create_fact(db, "Shop B", "50.00", date(2026, 1, 20))
        _create_fact(db, "Shop A", "75.00", date(2026, 2, 10))

        result = spending_by_month(db)
        assert len(result) == 2
        assert result[0].month == "2026-01"
        assert result[0].total == Decimal("150.00")
        assert result[0].count == 2
        assert result[1].month == "2026-02"
        assert result[1].total == Decimal("75.00")

    def test_empty_database(self, db):
        assert spending_by_month(db) == []


class TestItemFrequency:
    def test_counts_items(self, db):
        _create_fact(
            db,
            "Shop A",
            "10.00",
            date(2026, 1, 1),
            items=[("Milk", "3.00"), ("Bread", "2.00")],
        )
        _create_fact(
            db,
            "Shop B",
            "5.00",
            date(2026, 1, 5),
            items=[("Milk", "3.50")],
        )

        result = item_frequency(db)
        assert len(result) == 2
        milk = next(i for i in result if i.name_normalized == "milk")
        assert milk.count == 2
        assert milk.total_spent == Decimal("6.50")
        assert len(milk.vendors) == 2

    def test_empty_database(self, db):
        assert item_frequency(db) == []


class TestSeasonalPatterns:
    def test_detects_seasonal_variation(self, db):
        # Create spending in Jan and Dec across 1 year
        _create_fact(db, "Shop", "100.00", date(2026, 1, 15))
        _create_fact(db, "Shop", "200.00", date(2026, 6, 15))
        _create_fact(db, "Shop", "300.00", date(2026, 12, 15))

        result = seasonal_patterns(db, min_years=1)
        assert len(result) == 12

        jan = result[0]
        assert jan.month_number == 1
        assert jan.month_name == "January"
        assert jan.avg_spend == Decimal("100.00")

    def test_empty_database(self, db):
        assert seasonal_patterns(db) == []

    def test_min_years_filter(self, db):
        _create_fact(db, "Shop", "100.00", date(2026, 1, 15))
        # Only 1 year of data, require 2
        result = seasonal_patterns(db, min_years=2)
        assert result == []
