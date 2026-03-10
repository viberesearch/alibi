"""Tests for alibi.services.analytics facade."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from alibi.analytics.anomalies import SpendingAnomaly
from alibi.analytics.spending import MonthlySpend, VendorSpend
from alibi.analytics.subscriptions import SubscriptionPattern
from alibi.analytics.vendors import VendorDeduplicationReport
from alibi.services import analytics as svc


# ---------------------------------------------------------------------------
# Helpers: lightweight stub return values
# ---------------------------------------------------------------------------


def _make_vendor_spend() -> VendorSpend:
    return VendorSpend(
        vendor="ACME",
        vendor_key="CY10057000Y",
        total=Decimal("100.00"),
        count=3,
        avg_amount=Decimal("33.33"),
        first_date=date(2025, 1, 1),
        last_date=date(2025, 3, 1),
        share_pct=50.0,
    )


def _make_monthly_spend() -> MonthlySpend:
    return MonthlySpend(
        month="2025-01",
        total=Decimal("200.00"),
        count=5,
        avg_amount=Decimal("40.00"),
    )


def _make_subscription() -> SubscriptionPattern:
    return SubscriptionPattern(
        vendor="Netflix",
        vendor_normalized="netflix",
        avg_amount=Decimal("14.99"),
        period_type="monthly",
        frequency_days=30,
        confidence=0.95,
        last_date=date(2025, 3, 1),
        next_expected=date(2025, 4, 1),
        occurrences=12,
        amount_variance=0.0,
        fact_ids=["f1", "f2"],
    )


def _make_anomaly() -> SpendingAnomaly:
    return SpendingAnomaly(
        fact_id="f99",
        vendor="Unknown",
        amount=Decimal("999.00"),
        date=date(2025, 2, 15),
        anomaly_type="high_amount",
        severity=0.9,
        explanation="Amount is 3.5 std devs above average",
    )


def _make_vendor_report() -> VendorDeduplicationReport:
    return VendorDeduplicationReport(
        aliases=[],
        total_vendors=10,
        vendors_with_aliases=0,
        unkeyed_vendors=["Unnamed Vendor"],
    )


# ---------------------------------------------------------------------------
# spending_summary
# ---------------------------------------------------------------------------


class TestSpendingSummary:
    def test_month_delegates_to_spending_by_month(self) -> None:
        mock_db = MagicMock()
        expected = [_make_monthly_spend()]
        with patch(
            "alibi.services.analytics._spending.spending_by_month",
            return_value=expected,
        ) as mock_fn:
            result = svc.spending_summary(mock_db, period="month")

        assert result is expected
        mock_fn.assert_called_once_with(mock_db, date_from=None, date_to=None)

    def test_vendor_delegates_to_spending_by_vendor(self) -> None:
        mock_db = MagicMock()
        expected = [_make_vendor_spend()]
        with patch(
            "alibi.services.analytics._spending.spending_by_vendor",
            return_value=expected,
        ) as mock_fn:
            result = svc.spending_summary(mock_db, period="vendor")

        assert result is expected
        mock_fn.assert_called_once_with(mock_db, date_from=None, date_to=None, limit=50)

    def test_month_passes_date_filters(self) -> None:
        mock_db = MagicMock()
        d_from = date(2025, 1, 1)
        d_to = date(2025, 6, 30)
        with patch(
            "alibi.services.analytics._spending.spending_by_month",
            return_value=[],
        ) as mock_fn:
            svc.spending_summary(
                mock_db,
                period="month",
                filters={"date_from": d_from, "date_to": d_to},
            )

        mock_fn.assert_called_once_with(mock_db, date_from=d_from, date_to=d_to)

    def test_vendor_passes_limit_filter(self) -> None:
        mock_db = MagicMock()
        with patch(
            "alibi.services.analytics._spending.spending_by_vendor",
            return_value=[],
        ) as mock_fn:
            svc.spending_summary(mock_db, period="vendor", filters={"limit": 10})

        mock_fn.assert_called_once_with(mock_db, date_from=None, date_to=None, limit=10)

    def test_unknown_period_raises_value_error(self) -> None:
        mock_db = MagicMock()
        with pytest.raises(ValueError, match="Unknown period"):
            svc.spending_summary(mock_db, period="week")

    def test_default_period_is_month(self) -> None:
        mock_db = MagicMock()
        with patch(
            "alibi.services.analytics._spending.spending_by_month",
            return_value=[],
        ) as mock_fn:
            svc.spending_summary(mock_db)

        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# spending_by_vendor
# ---------------------------------------------------------------------------


class TestSpendingByVendor:
    def test_delegates_to_spending_module(self) -> None:
        mock_db = MagicMock()
        expected = [_make_vendor_spend()]
        with patch(
            "alibi.services.analytics._spending.spending_by_vendor",
            return_value=expected,
        ) as mock_fn:
            result = svc.spending_by_vendor(mock_db)

        assert result is expected
        mock_fn.assert_called_once_with(mock_db, date_from=None, date_to=None, limit=50)

    def test_passes_filters(self) -> None:
        mock_db = MagicMock()
        d_from = date(2025, 1, 1)
        d_to = date(2025, 12, 31)
        with patch(
            "alibi.services.analytics._spending.spending_by_vendor",
            return_value=[],
        ) as mock_fn:
            svc.spending_by_vendor(
                mock_db,
                filters={"date_from": d_from, "date_to": d_to, "limit": 5},
            )

        mock_fn.assert_called_once_with(
            mock_db, date_from=d_from, date_to=d_to, limit=5
        )

    def test_none_filters_treated_as_empty(self) -> None:
        mock_db = MagicMock()
        with patch(
            "alibi.services.analytics._spending.spending_by_vendor",
            return_value=[],
        ) as mock_fn:
            svc.spending_by_vendor(mock_db, filters=None)

        mock_fn.assert_called_once_with(mock_db, date_from=None, date_to=None, limit=50)


# ---------------------------------------------------------------------------
# spending_by_month
# ---------------------------------------------------------------------------


class TestSpendingByMonth:
    def test_delegates_to_spending_module(self) -> None:
        mock_db = MagicMock()
        expected = [_make_monthly_spend()]
        with patch(
            "alibi.services.analytics._spending.spending_by_month",
            return_value=expected,
        ) as mock_fn:
            result = svc.spending_by_month(mock_db)

        assert result is expected
        mock_fn.assert_called_once_with(mock_db, date_from=None, date_to=None)

    def test_passes_date_filters(self) -> None:
        mock_db = MagicMock()
        d_from = date(2025, 3, 1)
        d_to = date(2025, 9, 30)
        with patch(
            "alibi.services.analytics._spending.spending_by_month",
            return_value=[],
        ) as mock_fn:
            svc.spending_by_month(
                mock_db,
                filters={"date_from": d_from, "date_to": d_to},
            )

        mock_fn.assert_called_once_with(mock_db, date_from=d_from, date_to=d_to)

    def test_none_filters_treated_as_empty(self) -> None:
        mock_db = MagicMock()
        with patch(
            "alibi.services.analytics._spending.spending_by_month",
            return_value=[],
        ) as mock_fn:
            svc.spending_by_month(mock_db, filters=None)

        mock_fn.assert_called_once_with(mock_db, date_from=None, date_to=None)


# ---------------------------------------------------------------------------
# detect_subscriptions
# ---------------------------------------------------------------------------


class TestDetectSubscriptions:
    def test_delegates_to_subscriptions_module(self) -> None:
        mock_db = MagicMock()
        expected = [_make_subscription()]
        with patch(
            "alibi.services.analytics._subscriptions.detect_subscriptions",
            return_value=expected,
        ) as mock_fn:
            result = svc.detect_subscriptions(mock_db)

        assert result is expected
        mock_fn.assert_called_once_with(mock_db, min_occurrences=3, min_confidence=0.5)

    def test_returns_empty_list_when_no_patterns(self) -> None:
        mock_db = MagicMock()
        with patch(
            "alibi.services.analytics._subscriptions.detect_subscriptions",
            return_value=[],
        ):
            result = svc.detect_subscriptions(mock_db)

        assert result == []


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------


class TestDetectAnomalies:
    def test_delegates_to_anomalies_module(self) -> None:
        mock_db = MagicMock()
        expected = [_make_anomaly()]
        with patch(
            "alibi.services.analytics._anomalies.detect_anomalies",
            return_value=expected,
        ) as mock_fn:
            result = svc.detect_anomalies(mock_db)

        assert result is expected
        mock_fn.assert_called_once_with(mock_db, lookback_days=90, std_threshold=2.0)

    def test_passes_lookback_days(self) -> None:
        mock_db = MagicMock()
        with patch(
            "alibi.services.analytics._anomalies.detect_anomalies",
            return_value=[],
        ) as mock_fn:
            svc.detect_anomalies(mock_db, lookback_days=180)

        mock_fn.assert_called_once_with(mock_db, lookback_days=180, std_threshold=2.0)

    def test_passes_std_threshold(self) -> None:
        mock_db = MagicMock()
        with patch(
            "alibi.services.analytics._anomalies.detect_anomalies",
            return_value=[],
        ) as mock_fn:
            svc.detect_anomalies(mock_db, std_threshold=3.0)

        mock_fn.assert_called_once_with(mock_db, lookback_days=90, std_threshold=3.0)

    def test_passes_all_params(self) -> None:
        mock_db = MagicMock()
        with patch(
            "alibi.services.analytics._anomalies.detect_anomalies",
            return_value=[],
        ) as mock_fn:
            svc.detect_anomalies(mock_db, lookback_days=30, std_threshold=1.5)

        mock_fn.assert_called_once_with(mock_db, lookback_days=30, std_threshold=1.5)


# ---------------------------------------------------------------------------
# vendor_report
# ---------------------------------------------------------------------------


class TestVendorReport:
    def test_delegates_to_vendors_module(self) -> None:
        mock_db = MagicMock()
        expected = _make_vendor_report()
        with patch(
            "alibi.services.analytics._vendors.vendor_deduplication_report",
            return_value=expected,
        ) as mock_fn:
            result = svc.vendor_report(mock_db)

        assert result is expected
        mock_fn.assert_called_once_with(mock_db)

    def test_returns_vendor_deduplication_report_type(self) -> None:
        mock_db = MagicMock()
        report = _make_vendor_report()
        with patch(
            "alibi.services.analytics._vendors.vendor_deduplication_report",
            return_value=report,
        ):
            result = svc.vendor_report(mock_db)

        assert isinstance(result, VendorDeduplicationReport)
        assert result.total_vendors == 10
        assert result.unkeyed_vendors == ["Unnamed Vendor"]
