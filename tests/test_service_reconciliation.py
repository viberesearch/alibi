"""Tests for the item <-> payment reconciliation service."""

import json

from alibi.services.reconciliation import (
    EMPTY,
    ITEMS_ONLY,
    MATCHED,
    PAYMENT_ONLY,
    _build_filters,
    _has_payment,
    _mismatch,
    _sum_payments,
    reconcile,
)


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows
        self.last_sql = None
        self.last_params = None

    def fetchall(self, sql, params):
        self.last_sql = sql
        self.last_params = params
        return self._rows


def _row(**kw):
    base = {
        "id": "x",
        "vendor": "Shop",
        "total_amount": 10.0,
        "currency": "EUR",
        "country": "CY",
        "event_date": "2026-01-01",
        "event_time": None,
        "payments": None,
        "n_items": 0,
        "bundle_types": None,
        "items_sum": 0.0,
    }
    base.update(kw)
    return base


class TestHasPayment:
    def test_payment_record_bundle(self):
        assert _has_payment(None, "basket,payment_record")

    def test_payment_atoms_json(self):
        assert _has_payment(json.dumps([{"method": "card"}]), None)

    def test_none(self):
        assert not _has_payment(None, "basket")
        assert not _has_payment("[]", None)


class TestReconcile:
    def test_classifies_each_coverage(self):
        rows = [
            _row(id="1", n_items=3, bundle_types="basket", items_sum=10.0),
            _row(id="2", n_items=0, bundle_types="payment_record"),
            _row(
                id="3",
                n_items=2,
                bundle_types="basket,payment_record",
                payments=json.dumps([{"method": "card"}]),
                items_sum=10.0,
            ),
            _row(id="4", n_items=0, bundle_types=None),
        ]
        res = reconcile(_FakeDB(rows))
        assert res["summary"] == {
            MATCHED: 1,
            ITEMS_ONLY: 1,
            PAYMENT_ONLY: 1,
            EMPTY: 1,
            "total": 4,
        }
        cov = {t["id"]: t["coverage"] for t in res["transactions"]}
        assert cov == {
            "1": ITEMS_ONLY,
            "2": PAYMENT_ONLY,
            "3": MATCHED,
            "4": EMPTY,
        }

    def test_amount_mismatch_flagged(self):
        # items sum 7.00 but receipt total 10.00 -> mismatch
        rows = [_row(id="1", n_items=2, total_amount=10.0, items_sum=7.0)]
        res = reconcile(_FakeDB(rows))
        assert res["transactions"][0]["amount_mismatch"] is True

    def test_amount_within_tolerance_ok(self):
        rows = [_row(id="1", n_items=2, total_amount=10.0, items_sum=9.99)]
        res = reconcile(_FakeDB(rows))
        assert res["transactions"][0]["amount_mismatch"] is False


class TestSumPayments:
    def test_sums_normalised_amounts(self):
        js = json.dumps([{"amount": "7.50"}, {"amount": "2.50"}])
        assert _sum_payments(js) == 10.0

    def test_none_when_no_amount_key(self):
        assert _sum_payments(json.dumps([{"method": "card"}])) is None

    def test_none_for_empty_or_bad(self):
        assert _sum_payments(None) is None
        assert _sum_payments("[]") is None
        assert _sum_payments("not json") is None

    def test_skips_unparseable_amounts(self):
        js = json.dumps([{"amount": "5.00"}, {"amount": "n/a"}])
        assert _sum_payments(js) == 5.0


class TestPaymentReconciliation:
    def test_payment_amount_surfaced(self):
        rows = [
            _row(
                id="1",
                n_items=2,
                bundle_types="basket,payment_record",
                payments=json.dumps([{"amount": "10.00"}]),
                items_sum=10.0,
                total_amount=10.0,
            )
        ]
        t = reconcile(_FakeDB(rows))["transactions"][0]
        assert t["payment_amount"] == 10.0
        assert t["payment_mismatch"] is False

    def test_payment_mismatch_flagged(self):
        # paid 10.00 but receipt total says 8.00 -> payment mismatch
        rows = [
            _row(
                id="1",
                n_items=2,
                payments=json.dumps([{"amount": "10.00"}]),
                items_sum=8.0,
                total_amount=8.0,
            )
        ]
        t = reconcile(_FakeDB(rows))["transactions"][0]
        assert t["payment_mismatch"] is True

    def test_no_payment_amount_no_mismatch(self):
        rows = [_row(id="1", n_items=2, items_sum=10.0, total_amount=10.0)]
        t = reconcile(_FakeDB(rows))["transactions"][0]
        assert t["payment_amount"] is None
        assert t["payment_mismatch"] is False


class TestCoverageFilter:
    def _mixed_rows(self):
        return [
            _row(id="1", n_items=3, bundle_types="basket", items_sum=10.0),
            _row(id="2", n_items=0, bundle_types="payment_record"),
            _row(
                id="3",
                n_items=2,
                bundle_types="basket,payment_record",
                payments=json.dumps([{"method": "card"}]),
                items_sum=10.0,
            ),
        ]

    def test_filters_to_one_class(self):
        res = reconcile(_FakeDB(self._mixed_rows()), {"coverage": PAYMENT_ONLY})
        assert [t["id"] for t in res["transactions"]] == ["2"]

    def test_summary_scoped_to_filtered_set(self):
        res = reconcile(_FakeDB(self._mixed_rows()), {"coverage": PAYMENT_ONLY})
        assert res["summary"] == {
            MATCHED: 0,
            ITEMS_ONLY: 0,
            PAYMENT_ONLY: 1,
            EMPTY: 0,
            "total": 1,
        }


class TestBuildFilters:
    def test_empty_filters(self):
        where, params = _build_filters({})
        assert where == "1=1"
        assert params == []

    def test_vendor_is_substring(self):
        where, params = _build_filters({"vendor": "Lidl"})
        assert "LOWER(f.vendor) LIKE ?" in where
        assert params == ["%lidl%"]

    def test_exact_and_date_axes(self):
        where, params = _build_filters(
            {"currency": "EUR", "country": "CY", "date_from": "2026-01-01"}
        )
        assert "f.currency = ?" in where
        assert "f.country = ?" in where
        assert "f.event_date >= ?" in where
        assert params == ["EUR", "CY", "2026-01-01"]

    def test_datetime_bounds(self):
        where, params = _build_filters({"datetime_from": "2026-01-01 12:00:00"})
        assert "event_time" in where
        assert params == ["2026-01-01 12:00:00"]

    def test_coverage_not_a_sql_filter(self):
        # coverage is applied post-classification, never in the WHERE clause.
        where, params = _build_filters({"coverage": PAYMENT_ONLY})
        assert where == "1=1"
        assert params == []


def test_mismatch_tolerance():
    assert _mismatch(10.0, 7.0) is True
    assert _mismatch(10.0, 9.99) is False
    assert _mismatch(None, 10.0) is False
    assert _mismatch(10.0, None) is False
