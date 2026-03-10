"""Tests for v2 subscription/recurring payment detection."""

from datetime import date, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

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
)
from alibi.db import v2_store
from alibi.analytics.subscriptions import (
    detect_subscriptions,
    mark_subscriptions,
    get_upcoming_subscriptions,
    SubscriptionPattern,
    _classify_period,
    _cluster_by_amount,
)


def _create_fact(
    db,
    vendor: str,
    amount: Decimal,
    event_date: date,
    fact_type: str = "purchase",
) -> str:
    """Create a minimal fact for subscription testing. Returns fact ID."""
    doc_id = str(uuid4())
    doc = Document(
        id=doc_id,
        file_path=f"/test/{vendor}_{event_date}.jpg",
        file_hash=str(uuid4()),
    )
    v2_store.store_document(db, doc)

    bundle = Bundle(
        id=str(uuid4()),
        document_id=doc_id,
        bundle_type=BundleType.BASKET,
    )
    v2_store.store_bundle(db, bundle, [])

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
        fact_type=FactType(fact_type),
        vendor=vendor,
        total_amount=amount,
        currency="EUR",
        event_date=event_date,
        status=FactStatus.CONFIRMED,
    )
    v2_store.store_fact(db, fact, [])
    return fact.id


def _create_monthly_subscription(
    db, vendor="Netflix", amount=Decimal("14.99"), months=6
) -> list[str]:
    """Create N months of subscription facts."""
    fact_ids = []
    base_date = date(2025, 7, 15)
    for i in range(months):
        event_date = date(
            base_date.year + (base_date.month + i - 1) // 12,
            (base_date.month + i - 1) % 12 + 1,
            base_date.day,
        )
        fid = _create_fact(db, vendor, amount, event_date)
        fact_ids.append(fid)
    return fact_ids


# ---------------------------------------------------------------------------
# Period classification tests
# ---------------------------------------------------------------------------


class TestClassifyPeriod:
    def test_weekly(self):
        period, conf = _classify_period(7)
        assert period == "weekly"
        assert conf > 0.5

    def test_monthly(self):
        period, conf = _classify_period(30)
        assert period == "monthly"
        assert conf > 0.5

    def test_quarterly(self):
        period, conf = _classify_period(91)
        assert period == "quarterly"
        assert conf > 0.5

    def test_annual(self):
        period, conf = _classify_period(365)
        assert period == "annual"
        assert conf > 0.5

    def test_biweekly(self):
        period, conf = _classify_period(14)
        assert period == "biweekly"

    def test_irregular(self):
        period, conf = _classify_period(45)
        assert period == "irregular"
        assert conf == 0.0

    def test_near_monthly(self):
        period, conf = _classify_period(31)
        assert period == "monthly"
        assert conf > 0.0


# ---------------------------------------------------------------------------
# Amount clustering tests
# ---------------------------------------------------------------------------


class TestClusterByAmount:
    def test_same_amounts(self):
        entries = [
            ("V", Decimal("10.00"), date(2025, 1, 1), "f1"),
            ("V", Decimal("10.00"), date(2025, 2, 1), "f2"),
            ("V", Decimal("10.00"), date(2025, 3, 1), "f3"),
        ]
        clusters = _cluster_by_amount(entries, 0.10)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_different_amounts_separate_clusters(self):
        entries = [
            ("V", Decimal("10.00"), date(2025, 1, 1), "f1"),
            ("V", Decimal("10.00"), date(2025, 2, 1), "f2"),
            ("V", Decimal("50.00"), date(2025, 3, 1), "f3"),
            ("V", Decimal("50.00"), date(2025, 4, 1), "f4"),
        ]
        clusters = _cluster_by_amount(entries, 0.10)
        assert len(clusters) == 2

    def test_within_tolerance(self):
        entries = [
            ("V", Decimal("10.00"), date(2025, 1, 1), "f1"),
            ("V", Decimal("10.50"), date(2025, 2, 1), "f2"),
            ("V", Decimal("10.80"), date(2025, 3, 1), "f3"),
        ]
        clusters = _cluster_by_amount(entries, 0.10)
        assert len(clusters) == 1

    def test_empty(self):
        assert _cluster_by_amount([], 0.10) == []


# ---------------------------------------------------------------------------
# Full detection tests
# ---------------------------------------------------------------------------


class TestDetectSubscriptions:
    def test_monthly_subscription(self, db):
        _create_monthly_subscription(db, "Netflix", Decimal("14.99"), months=6)
        patterns = detect_subscriptions(db, min_occurrences=3)
        assert len(patterns) == 1
        assert patterns[0].vendor == "Netflix"
        assert patterns[0].period_type == "monthly"
        assert Decimal(str(patterns[0].avg_amount)) == Decimal("14.99")
        assert patterns[0].occurrences == 6

    def test_weekly_subscription(self, db):
        base = date(2025, 7, 1)
        for i in range(5):
            _create_fact(db, "Gym", Decimal("5.00"), base + timedelta(days=7 * i))
        patterns = detect_subscriptions(db, min_occurrences=3)
        assert len(patterns) >= 1
        gym = [p for p in patterns if "gym" in p.vendor_normalized]
        assert len(gym) == 1
        assert gym[0].period_type == "weekly"

    def test_no_pattern_too_few(self, db):
        _create_fact(db, "OneOff", Decimal("99.00"), date(2025, 1, 1))
        _create_fact(db, "OneOff", Decimal("99.00"), date(2025, 2, 1))
        patterns = detect_subscriptions(db, min_occurrences=3)
        assert len(patterns) == 0

    def test_irregular_not_detected(self, db):
        # Random intervals shouldn't be detected as subscriptions
        _create_fact(db, "Random", Decimal("20.00"), date(2025, 1, 5))
        _create_fact(db, "Random", Decimal("20.00"), date(2025, 2, 28))
        _create_fact(db, "Random", Decimal("20.00"), date(2025, 3, 3))
        _create_fact(db, "Random", Decimal("20.00"), date(2025, 5, 20))
        patterns = detect_subscriptions(db, min_occurrences=3, min_confidence=0.6)
        # Should either not detect or have low confidence
        high_conf = [p for p in patterns if p.confidence >= 0.6]
        assert len(high_conf) == 0

    def test_multiple_vendors(self, db):
        _create_monthly_subscription(db, "Netflix", Decimal("14.99"), months=4)
        _create_monthly_subscription(db, "Spotify", Decimal("9.99"), months=4)
        patterns = detect_subscriptions(db, min_occurrences=3)
        vendors = {p.vendor for p in patterns}
        assert "Netflix" in vendors
        assert "Spotify" in vendors

    def test_different_amounts_same_vendor(self, db):
        # Two price tiers from same vendor
        base = date(2025, 7, 1)
        for i in range(4):
            _create_fact(db, "Service", Decimal("10.00"), base + timedelta(days=30 * i))
        for i in range(4):
            _create_fact(db, "Service", Decimal("50.00"), base + timedelta(days=30 * i))
        patterns = detect_subscriptions(db, min_occurrences=3)
        service_patterns = [p for p in patterns if "service" in p.vendor_normalized]
        assert len(service_patterns) == 2

    def test_empty_db(self, db):
        patterns = detect_subscriptions(db)
        assert patterns == []

    def test_next_expected_date(self, db):
        _create_monthly_subscription(db, "Netflix", Decimal("14.99"), months=4)
        patterns = detect_subscriptions(db, min_occurrences=3)
        assert len(patterns) == 1
        # Next expected should be roughly a month after the last date
        assert patterns[0].next_expected > patterns[0].last_date


# ---------------------------------------------------------------------------
# Mark subscriptions tests
# ---------------------------------------------------------------------------


class TestMarkSubscriptions:
    def test_marks_facts(self, db):
        fact_ids = _create_monthly_subscription(
            db, "Netflix", Decimal("14.99"), months=4
        )
        patterns = detect_subscriptions(db, min_occurrences=3)
        assert len(patterns) == 1

        count = mark_subscriptions(db, patterns[0])
        assert count == 4

        for fid in fact_ids:
            fact = v2_store.get_fact_by_id(db, fid)
            assert fact is not None
            assert fact["fact_type"] == "subscription_payment"


# ---------------------------------------------------------------------------
# Upcoming subscriptions tests
# ---------------------------------------------------------------------------


class TestGetUpcomingSubscriptions:
    def test_upcoming(self):
        today = date.today()
        pattern = SubscriptionPattern(
            vendor="Netflix",
            vendor_normalized="netflix",
            avg_amount=Decimal("14.99"),
            period_type="monthly",
            frequency_days=30,
            confidence=0.9,
            last_date=today - timedelta(days=25),
            next_expected=today + timedelta(days=5),
            occurrences=6,
            amount_variance=0.0,
            fact_ids=[],
        )
        upcoming = get_upcoming_subscriptions([pattern], days_ahead=30)
        assert len(upcoming) >= 1
        assert upcoming[0][0].vendor == "Netflix"

    def test_no_upcoming(self):
        today = date.today()
        pattern = SubscriptionPattern(
            vendor="Annual",
            vendor_normalized="annual",
            avg_amount=Decimal("100.00"),
            period_type="annual",
            frequency_days=365,
            confidence=0.9,
            last_date=today - timedelta(days=10),
            next_expected=today + timedelta(days=355),
            occurrences=3,
            amount_variance=0.0,
            fact_ids=[],
        )
        upcoming = get_upcoming_subscriptions([pattern], days_ahead=30)
        assert len(upcoming) == 0

    def test_past_next_expected_rolls_forward(self):
        today = date.today()
        pattern = SubscriptionPattern(
            vendor="Weekly",
            vendor_normalized="weekly",
            avg_amount=Decimal("5.00"),
            period_type="weekly",
            frequency_days=7,
            confidence=0.9,
            last_date=today - timedelta(days=20),
            next_expected=today - timedelta(days=13),
            occurrences=4,
            amount_variance=0.0,
            fact_ids=[],
        )
        upcoming = get_upcoming_subscriptions([pattern], days_ahead=30)
        # Should have rolled forward past today
        assert all(u[1] >= today for u in upcoming)
