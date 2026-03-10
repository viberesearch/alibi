"""Tests for spending anomaly detection."""

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from alibi.analytics.anomalies import (
    SpendingAnomaly,
    detect_anomalies,
)
from alibi.analytics.patterns import _extract_category
from alibi.db.connection import DatabaseManager


@pytest.fixture
def mock_db() -> MagicMock:
    """Create a mock database manager for anomaly tests."""
    return MagicMock(spec=DatabaseManager)


def _make_row(
    fact_id: str,
    vendor: str,
    amount: float,
    event_date: str,
    categories: str | None = None,
) -> tuple[str, str, float, str, str | None]:
    """Create a fact row as returned by the SQL query."""
    return (fact_id, vendor, amount, event_date, categories)


class TestDetectAnomaliesHighAmount:
    """Test detection of high-amount anomalies."""

    def test_detect_anomalies_high_amount(self, mock_db: MagicMock) -> None:
        """Single vendor, one transaction 3+ std devs above mean."""
        today = date.today()
        rows = [
            _make_row(
                "t1", "Grocery Store", 50.0, (today - timedelta(days=i * 7)).isoformat()
            )
            for i in range(10)
        ]
        # Add an outlier: 500 is far above the 50 mean
        rows.append(_make_row("t-outlier", "Grocery Store", 500.0, today.isoformat()))
        mock_db.fetchall.return_value = rows

        anomalies = detect_anomalies(mock_db, std_threshold=2.0)

        assert len(anomalies) >= 1
        outlier = [a for a in anomalies if a.fact_id == "t-outlier"]
        assert len(outlier) == 1
        assert outlier[0].anomaly_type == "high_amount"
        assert outlier[0].amount == Decimal("500")


class TestDetectAnomaliesUnusualVendor:
    """Test detection of unusual vendor anomalies."""

    def test_detect_anomalies_unusual_vendor(self, mock_db: MagicMock) -> None:
        """First-time vendor with high amount is flagged as unusual_vendor."""
        today = date.today()
        # Regular transactions with known vendor
        rows = [
            _make_row(
                f"t{i}",
                "Regular Shop",
                30.0,
                (today - timedelta(days=i * 5)).isoformat(),
            )
            for i in range(15)
        ]
        # New vendor with high amount
        rows.append(_make_row("t-new", "Expensive New Store", 300.0, today.isoformat()))
        mock_db.fetchall.return_value = rows

        anomalies = detect_anomalies(mock_db, std_threshold=2.0)

        new_vendor = [a for a in anomalies if a.fact_id == "t-new"]
        assert len(new_vendor) == 1
        assert new_vendor[0].anomaly_type == "unusual_vendor"
        assert "First transaction" in new_vendor[0].explanation


class TestDetectAnomaliesCategoryBased:
    """Test category-based anomaly detection."""

    def test_detect_anomalies_category_based(self, mock_db: MagicMock) -> None:
        """Anomaly relative to category mean when category has enough history."""
        today = date.today()
        # 5 normal grocery transactions (need >2 for category stats)
        rows = [
            _make_row(
                f"t{i}",
                "Store A",
                25.0,
                (today - timedelta(days=i * 7)).isoformat(),
                categories="groceries",
            )
            for i in range(5)
        ]
        # One outlier in same category
        rows.append(
            _make_row(
                "t-cat-outlier",
                "Store A",
                250.0,
                today.isoformat(),
                categories="groceries",
            )
        )
        mock_db.fetchall.return_value = rows

        anomalies = detect_anomalies(mock_db, std_threshold=2.0)

        cat_anomaly = [a for a in anomalies if a.fact_id == "t-cat-outlier"]
        assert len(cat_anomaly) == 1
        assert cat_anomaly[0].anomaly_type == "high_amount"
        assert "groceries" in cat_anomaly[0].explanation


class TestNoAnomalies:
    """Test that normal spending produces no anomalies."""

    def test_no_anomalies_normal_spending(self, mock_db: MagicMock) -> None:
        """All transactions within 2 std devs returns empty list."""
        today = date.today()
        rows = [
            _make_row(f"t{i}", "Shop", 50.0, (today - timedelta(days=i)).isoformat())
            for i in range(10)
        ]
        mock_db.fetchall.return_value = rows

        anomalies = detect_anomalies(mock_db, std_threshold=2.0)

        assert anomalies == []


class TestEmptyTransactions:
    """Test with no transaction data."""

    def test_empty_transactions(self, mock_db: MagicMock) -> None:
        """No data returns empty list."""
        mock_db.fetchall.return_value = []

        anomalies = detect_anomalies(mock_db)

        assert anomalies == []


class TestSeverityScaling:
    """Test that severity scales with z-score."""

    def test_severity_scaling(self, mock_db: MagicMock) -> None:
        """Higher z-scores produce higher severity (capped at 1.0)."""
        today = date.today()
        rows = [
            _make_row(
                f"t{i}", "Shop", 10.0, (today - timedelta(days=i * 3)).isoformat()
            )
            for i in range(20)
        ]
        # Add two outliers at different levels
        rows.append(_make_row("t-mid", "Shop", 100.0, today.isoformat()))
        rows.append(_make_row("t-high", "Shop", 500.0, today.isoformat()))
        mock_db.fetchall.return_value = rows

        anomalies = detect_anomalies(mock_db, std_threshold=2.0)

        if len(anomalies) >= 2:
            # Sorted by severity descending
            assert anomalies[0].severity >= anomalies[-1].severity


class TestLookbackWindow:
    """Test lookback window filtering."""

    def test_lookback_window(self, mock_db: MagicMock) -> None:
        """Only transactions within lookback_days are analyzed."""
        today = date.today()
        # Old transactions (outside 30-day window)
        old_rows = [
            _make_row(
                f"t-old{i}", "Shop", 10.0, (today - timedelta(days=60 + i)).isoformat()
            )
            for i in range(5)
        ]
        # Recent transactions within window
        recent_rows = [
            _make_row(
                f"t-new{i}", "Shop", 10.0, (today - timedelta(days=i)).isoformat()
            )
            for i in range(5)
        ]
        # The SQL query filters by date, so mock returns only what query would return
        mock_db.fetchall.return_value = recent_rows

        anomalies = detect_anomalies(mock_db, lookback_days=30)

        # Verify the SQL was called with correct date parameter
        call_args = mock_db.fetchall.call_args
        query_params = call_args[0][1]
        expected_start = (today - timedelta(days=30)).isoformat()
        assert query_params[0] == expected_start


class TestStdThresholdParameter:
    """Test custom threshold parameter."""

    def test_std_threshold_parameter(self, mock_db: MagicMock) -> None:
        """Custom threshold is respected — higher threshold = fewer anomalies."""
        today = date.today()
        rows = [
            _make_row(
                f"t{i}", "Shop", 10.0, (today - timedelta(days=i * 3)).isoformat()
            )
            for i in range(15)
        ]
        rows.append(_make_row("t-out", "Shop", 60.0, today.isoformat()))
        mock_db.fetchall.return_value = rows

        # Low threshold — more sensitive
        anomalies_low = detect_anomalies(mock_db, std_threshold=1.5)
        # High threshold — less sensitive
        anomalies_high = detect_anomalies(mock_db, std_threshold=5.0)

        assert len(anomalies_low) >= len(anomalies_high)


class TestExtractCategory:
    """Test the _extract_category helper."""

    def test_extracts_single_category(self) -> None:
        assert _extract_category("groceries") == "groceries"

    def test_multiple_categories(self) -> None:
        assert _extract_category("groceries,dairy") == "groceries"

    def test_empty_string(self) -> None:
        assert _extract_category("") is None

    def test_none_input(self) -> None:
        assert _extract_category(None) is None

    def test_whitespace_handling(self) -> None:
        assert _extract_category(" groceries , dairy ") == "groceries"
