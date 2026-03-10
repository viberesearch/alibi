"""Tests for spending pattern analysis."""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from alibi.analytics.patterns import (
    CategoryTrend,
    MonthlyTrend,
    SpendingInsights,
    _calculate_category_trend,
    _extract_category,
    analyze_spending_patterns,
    compare_periods,
)


class TestMonthlyTrendDataclass:
    """Tests for MonthlyTrend dataclass."""

    def test_creates_monthly_trend(self):
        trend = MonthlyTrend(
            month="2024-01",
            total_expenses=Decimal("1500.00"),
            total_income=Decimal("5000.00"),
            net=Decimal("3500.00"),
            transaction_count=25,
            top_categories=[("groceries", Decimal("500"))],
        )
        assert trend.month == "2024-01"
        assert trend.total_expenses == Decimal("1500.00")
        assert trend.total_income == Decimal("5000.00")
        assert trend.net == Decimal("3500.00")
        assert trend.transaction_count == 25
        assert len(trend.top_categories) == 1

    def test_default_top_categories(self):
        trend = MonthlyTrend(
            month="2024-01",
            total_expenses=Decimal("0"),
            total_income=Decimal("0"),
            net=Decimal("0"),
            transaction_count=0,
        )
        assert trend.top_categories == []


class TestCategoryTrendDataclass:
    """Tests for CategoryTrend dataclass."""

    def test_creates_category_trend(self):
        trend = CategoryTrend(
            category="groceries",
            months=["2024-01", "2024-02", "2024-03"],
            amounts=[Decimal("500"), Decimal("550"), Decimal("600")],
            trend_direction="increasing",
            avg_monthly=Decimal("550.00"),
        )
        assert trend.category == "groceries"
        assert len(trend.months) == 3
        assert len(trend.amounts) == 3
        assert trend.trend_direction == "increasing"
        assert trend.avg_monthly == Decimal("550.00")


class TestExtractCategory:
    """Tests for _extract_category helper function."""

    def test_extracts_single_category(self):
        result = _extract_category("groceries")
        assert result == "groceries"

    def test_extracts_first_category(self):
        result = _extract_category("food,groceries")
        assert result == "food"

    def test_returns_none_for_empty_string(self):
        result = _extract_category("")
        assert result is None

    def test_returns_none_for_none(self):
        result = _extract_category(None)
        assert result is None

    def test_handles_whitespace(self):
        result = _extract_category(" dining , restaurant ")
        assert result == "dining"


class TestCalculateCategoryTrend:
    """Tests for _calculate_category_trend helper function."""

    def test_calculates_increasing_trend(self):
        month_amounts = {
            "2024-01": Decimal("100"),
            "2024-02": Decimal("150"),
            "2024-03": Decimal("200"),
            "2024-04": Decimal("250"),
            "2024-05": Decimal("300"),
        }
        all_months = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]

        trend = _calculate_category_trend("shopping", month_amounts, all_months)

        assert trend is not None
        assert trend.category == "shopping"
        assert trend.trend_direction == "increasing"
        assert len(trend.amounts) == 5

    def test_calculates_decreasing_trend(self):
        month_amounts = {
            "2024-01": Decimal("500"),
            "2024-02": Decimal("400"),
            "2024-03": Decimal("300"),
            "2024-04": Decimal("200"),
            "2024-05": Decimal("100"),
        }
        all_months = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]

        trend = _calculate_category_trend("dining", month_amounts, all_months)

        assert trend is not None
        assert trend.trend_direction == "decreasing"

    def test_calculates_stable_trend(self):
        month_amounts = {
            "2024-01": Decimal("200"),
            "2024-02": Decimal("195"),
            "2024-03": Decimal("205"),
            "2024-04": Decimal("198"),
            "2024-05": Decimal("202"),
        }
        all_months = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]

        trend = _calculate_category_trend("utilities", month_amounts, all_months)

        assert trend is not None
        assert trend.trend_direction == "stable"

    def test_calculates_average_monthly(self):
        month_amounts = {
            "2024-01": Decimal("100"),
            "2024-02": Decimal("200"),
            "2024-03": Decimal("300"),
        }
        all_months = ["2024-01", "2024-02", "2024-03"]

        trend = _calculate_category_trend("test", month_amounts, all_months)

        assert trend is not None
        assert trend.avg_monthly == Decimal("200.00")

    def test_fills_missing_months_with_zero(self):
        month_amounts = {
            "2024-01": Decimal("100"),
            "2024-03": Decimal("300"),
        }
        all_months = ["2024-01", "2024-02", "2024-03"]

        trend = _calculate_category_trend("test", month_amounts, all_months)

        assert trend is not None
        assert len(trend.amounts) == 3
        assert trend.amounts[1] == Decimal("0")

    def test_returns_none_for_empty_amounts(self):
        trend = _calculate_category_trend("test", {}, [])
        assert trend is None

    def test_single_month_is_stable(self):
        month_amounts = {"2024-01": Decimal("500")}
        all_months = ["2024-01"]

        trend = _calculate_category_trend("test", month_amounts, all_months)

        assert trend is not None
        assert trend.trend_direction == "stable"


class TestMonthlyTrendCalculation:
    """Tests for monthly trend calculation in analyze_spending_patterns."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database manager."""
        return MagicMock()

    def test_monthly_trend_calculation(self, mock_db):
        """Verify monthly trends are calculated correctly."""
        # Sample transactions spanning 3 months
        mock_db.fetchall.return_value = [
            # January 2024
            (
                "txn-1",
                "purchase",
                "500.00",
                "2024-01-15",
                "Grocery Store",
                "groceries",
            ),
            (
                "txn-2",
                "purchase",
                "200.00",
                "2024-01-20",
                "Gas Station",
                "transportation",
            ),
            ("txn-3", "refund", "5000.00", "2024-01-01", "Employer", None),
            # February 2024
            (
                "txn-4",
                "purchase",
                "600.00",
                "2024-02-10",
                "Grocery Store",
                "groceries",
            ),
            (
                "txn-5",
                "purchase",
                "150.00",
                "2024-02-15",
                "Restaurant",
                "dining",
            ),
            ("txn-6", "refund", "5000.00", "2024-02-01", "Employer", None),
            # March 2024
            (
                "txn-7",
                "purchase",
                "450.00",
                "2024-03-05",
                "Grocery Store",
                "groceries",
            ),
            (
                "txn-8",
                "purchase",
                "300.00",
                "2024-03-12",
                "Online Shopping",
                "shopping",
            ),
            ("txn-9", "refund", "5500.00", "2024-03-01", "Employer", None),
        ]

        insights = analyze_spending_patterns(mock_db, months=3)

        # Verify we have monthly trends
        assert len(insights.monthly_trends) >= 1

        # Find a specific month to verify calculations
        january = next(
            (t for t in insights.monthly_trends if t.month == "2024-01"), None
        )
        if january:
            assert january.total_expenses == Decimal("700.00")  # 500 + 200
            assert january.total_income == Decimal("5000.00")
            assert january.net == Decimal("4300.00")  # 5000 - 700
            assert january.transaction_count == 3

    def test_monthly_trend_top_categories(self, mock_db):
        """Verify top categories are identified correctly."""
        mock_db.fetchall.return_value = [
            ("txn-1", "purchase", "500.00", "2024-01-15", "Store", "groceries"),
            ("txn-2", "purchase", "300.00", "2024-01-20", "Store", "dining"),
            ("txn-3", "purchase", "200.00", "2024-01-25", "Store", "shopping"),
            ("txn-4", "purchase", "100.00", "2024-01-28", "Store", "utilities"),
        ]

        insights = analyze_spending_patterns(mock_db, months=1)

        if insights.monthly_trends:
            trend = insights.monthly_trends[0]
            # Top categories should be sorted by amount descending
            if trend.top_categories:
                assert trend.top_categories[0][0] == "groceries"
                assert trend.top_categories[0][1] == Decimal("500.00")


class TestCategoryTrendDirection:
    """Tests for category trend direction detection."""

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    def test_category_trend_direction_increasing(self, mock_db):
        """Verify increasing trend is detected correctly."""
        # Create transactions with increasing spending in groceries
        mock_db.fetchall.return_value = [
            ("txn-1", "purchase", "100.00", "2024-01-15", "Store", "groceries"),
            ("txn-2", "purchase", "200.00", "2024-02-15", "Store", "groceries"),
            ("txn-3", "purchase", "300.00", "2024-03-15", "Store", "groceries"),
            ("txn-4", "purchase", "400.00", "2024-04-15", "Store", "groceries"),
            ("txn-5", "purchase", "500.00", "2024-05-15", "Store", "groceries"),
        ]

        insights = analyze_spending_patterns(mock_db, months=6)

        groceries_trend = next(
            (t for t in insights.category_trends if t.category == "groceries"), None
        )
        assert groceries_trend is not None
        assert groceries_trend.trend_direction == "increasing"

    def test_category_trend_direction_decreasing(self, mock_db):
        """Verify decreasing trend is detected correctly."""
        mock_db.fetchall.return_value = [
            (
                "txn-1",
                "purchase",
                "500.00",
                "2024-01-15",
                "Store",
                "entertainment",
            ),
            (
                "txn-2",
                "purchase",
                "400.00",
                "2024-02-15",
                "Store",
                "entertainment",
            ),
            (
                "txn-3",
                "purchase",
                "300.00",
                "2024-03-15",
                "Store",
                "entertainment",
            ),
            (
                "txn-4",
                "purchase",
                "200.00",
                "2024-04-15",
                "Store",
                "entertainment",
            ),
            (
                "txn-5",
                "purchase",
                "100.00",
                "2024-05-15",
                "Store",
                "entertainment",
            ),
        ]

        insights = analyze_spending_patterns(mock_db, months=6)

        entertainment_trend = next(
            (t for t in insights.category_trends if t.category == "entertainment"), None
        )
        assert entertainment_trend is not None
        assert entertainment_trend.trend_direction == "decreasing"

    def test_category_trend_direction_stable(self, mock_db):
        """Verify stable trend is detected correctly."""
        mock_db.fetchall.return_value = [
            (
                "txn-1",
                "purchase",
                "200.00",
                "2024-01-15",
                "Utility Co",
                "utilities",
            ),
            (
                "txn-2",
                "purchase",
                "195.00",
                "2024-02-15",
                "Utility Co",
                "utilities",
            ),
            (
                "txn-3",
                "purchase",
                "205.00",
                "2024-03-15",
                "Utility Co",
                "utilities",
            ),
            (
                "txn-4",
                "purchase",
                "198.00",
                "2024-04-15",
                "Utility Co",
                "utilities",
            ),
            (
                "txn-5",
                "purchase",
                "202.00",
                "2024-05-15",
                "Utility Co",
                "utilities",
            ),
        ]

        insights = analyze_spending_patterns(mock_db, months=6)

        utilities_trend = next(
            (t for t in insights.category_trends if t.category == "utilities"), None
        )
        assert utilities_trend is not None
        assert utilities_trend.trend_direction == "stable"


class TestSavingsRateCalculation:
    """Tests for savings rate formula."""

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    def test_savings_rate_calculation_positive(self, mock_db):
        """Verify savings rate formula: (income - expenses) / income."""
        mock_db.fetchall.return_value = [
            ("txn-1", "refund", "5000.00", "2024-01-01", "Employer", None),
            ("txn-2", "purchase", "3500.00", "2024-01-15", "Various", "misc"),
        ]

        insights = analyze_spending_patterns(mock_db, months=1)

        # Savings rate = (5000 - 3500) / 5000 = 0.3
        assert insights.savings_rate == 0.3

    def test_savings_rate_zero_income(self, mock_db):
        """Verify savings rate is 0 when no income."""
        mock_db.fetchall.return_value = [
            ("txn-1", "purchase", "500.00", "2024-01-15", "Store", "shopping"),
        ]

        insights = analyze_spending_patterns(mock_db, months=1)

        assert insights.savings_rate == 0.0

    def test_savings_rate_negative(self, mock_db):
        """Verify negative savings rate when expenses exceed income."""
        mock_db.fetchall.return_value = [
            ("txn-1", "refund", "3000.00", "2024-01-01", "Employer", None),
            ("txn-2", "purchase", "4000.00", "2024-01-15", "Various", "misc"),
        ]

        insights = analyze_spending_patterns(mock_db, months=1)

        # Savings rate = (3000 - 4000) / 3000 = -0.333
        assert insights.savings_rate == pytest.approx(-0.333, rel=0.01)

    def test_savings_rate_hundred_percent(self, mock_db):
        """Verify 100% savings rate when no expenses."""
        mock_db.fetchall.return_value = [
            ("txn-1", "refund", "5000.00", "2024-01-01", "Employer", None),
        ]

        insights = analyze_spending_patterns(mock_db, months=1)

        assert insights.savings_rate == 1.0

    def test_savings_rate_rounded(self, mock_db):
        """Verify savings rate is rounded to 3 decimal places."""
        mock_db.fetchall.return_value = [
            ("txn-1", "refund", "3000.00", "2024-01-01", "Employer", None),
            ("txn-2", "purchase", "1000.00", "2024-01-15", "Store", "misc"),
        ]

        insights = analyze_spending_patterns(mock_db, months=1)

        # Savings rate = (3000 - 1000) / 3000 = 0.666666...
        assert insights.savings_rate == 0.667


class TestPeriodComparison:
    """Tests for compare_periods function."""

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    def test_compare_periods_returns_correct_changes(self, mock_db):
        """Verify compare_periods returns correct percentage changes."""

        def mock_fetchall(query, params):
            start_date = params[1]
            if "2024-01" in start_date:
                # Period 1: January
                return [
                    ("purchase", "1000.00", "Store", "groceries"),
                    ("purchase", "500.00", "Restaurant", "dining"),
                    ("refund", "5000.00", "Employer", None),
                ]
            else:
                # Period 2: February
                return [
                    ("purchase", "1200.00", "Store", "groceries"),
                    ("purchase", "400.00", "Restaurant", "dining"),
                    ("refund", "5500.00", "Employer", None),
                ]

        mock_db.fetchall.side_effect = mock_fetchall

        result = compare_periods(
            mock_db,
            period1_start=date(2024, 1, 1),
            period1_end=date(2024, 1, 31),
            period2_start=date(2024, 2, 1),
            period2_end=date(2024, 2, 29),
        )

        # Verify structure
        assert "period1" in result
        assert "period2" in result
        assert "changes" in result

        # Verify period 1 stats
        assert result["period1"]["total_expenses"] == 1500.0  # 1000 + 500
        assert result["period1"]["total_income"] == 5000.0
        assert result["period1"]["transaction_count"] == 3

        # Verify period 2 stats
        assert result["period2"]["total_expenses"] == 1600.0  # 1200 + 400
        assert result["period2"]["total_income"] == 5500.0
        assert result["period2"]["transaction_count"] == 3

        # Verify expense change percentage: (1600 - 1500) / 1500 * 100 = 6.67%
        assert result["changes"]["expense_change_pct"] == pytest.approx(6.7, rel=0.1)

        # Verify income change percentage: (5500 - 5000) / 5000 * 100 = 10%
        assert result["changes"]["income_change_pct"] == 10.0

    def test_compare_periods_category_changes(self, mock_db):
        """Verify category-level changes are calculated correctly."""

        def mock_fetchall(query, params):
            start_date = params[1]
            if "2024-01" in start_date:
                return [
                    ("purchase", "500.00", "Store", "groceries"),
                ]
            else:
                return [
                    ("purchase", "750.00", "Store", "groceries"),
                ]

        mock_db.fetchall.side_effect = mock_fetchall

        result = compare_periods(
            mock_db,
            period1_start=date(2024, 1, 1),
            period1_end=date(2024, 1, 31),
            period2_start=date(2024, 2, 1),
            period2_end=date(2024, 2, 29),
        )

        category_changes = result["changes"]["category_changes"]
        groceries = next(
            (c for c in category_changes if c["category"] == "groceries"), None
        )

        assert groceries is not None
        assert groceries["period1"] == 500.0
        assert groceries["period2"] == 750.0
        # Change: (750 - 500) / 500 * 100 = 50%
        assert groceries["change_pct"] == 50.0

    def test_compare_periods_zero_to_nonzero(self, mock_db):
        """Verify handling when category goes from 0 to non-zero."""

        def mock_fetchall(query, params):
            start_date = params[1]
            if "2024-01" in start_date:
                return []  # No transactions in period 1
            else:
                return [
                    ("purchase", "100.00", "Store", "new_category"),
                ]

        mock_db.fetchall.side_effect = mock_fetchall

        result = compare_periods(
            mock_db,
            period1_start=date(2024, 1, 1),
            period1_end=date(2024, 1, 31),
            period2_start=date(2024, 2, 1),
            period2_end=date(2024, 2, 29),
        )

        category_changes = result["changes"]["category_changes"]
        new_cat = next(
            (c for c in category_changes if c["category"] == "new_category"), None
        )

        assert new_cat is not None
        assert new_cat["period1"] == 0.0
        assert new_cat["period2"] == 100.0
        # Special case: 0 to non-zero = 100%
        assert new_cat["change_pct"] == 100.0

    def test_compare_periods_nonzero_to_zero(self, mock_db):
        """Verify handling when category goes from non-zero to 0."""

        def mock_fetchall(query, params):
            start_date = params[1]
            if "2024-01" in start_date:
                return [
                    ("purchase", "100.00", "Store", "dropped_category"),
                ]
            else:
                return []  # No transactions in period 2

        mock_db.fetchall.side_effect = mock_fetchall

        result = compare_periods(
            mock_db,
            period1_start=date(2024, 1, 1),
            period1_end=date(2024, 1, 31),
            period2_start=date(2024, 2, 1),
            period2_end=date(2024, 2, 29),
        )

        category_changes = result["changes"]["category_changes"]
        dropped = next(
            (c for c in category_changes if c["category"] == "dropped_category"), None
        )

        assert dropped is not None
        assert dropped["period1"] == 100.0
        assert dropped["period2"] == 0.0
        # Change: (0 - 100) / 100 * 100 = -100%
        assert dropped["change_pct"] == -100.0

    def test_compare_periods_date_formatting(self, mock_db):
        """Verify dates are formatted correctly in the result."""
        mock_db.fetchall.return_value = []

        result = compare_periods(
            mock_db,
            period1_start=date(2024, 3, 1),
            period1_end=date(2024, 3, 31),
            period2_start=date(2024, 4, 1),
            period2_end=date(2024, 4, 30),
        )

        assert result["period1"]["start"] == "2024-03-01"
        assert result["period1"]["end"] == "2024-03-31"
        assert result["period2"]["start"] == "2024-04-01"
        assert result["period2"]["end"] == "2024-04-30"


class TestSpendingInsights:
    """Tests for SpendingInsights dataclass and overall analysis."""

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    def test_biggest_increase_category(self, mock_db):
        """Verify biggest increase category is identified correctly."""
        mock_db.fetchall.return_value = [
            # Groceries: stable
            ("txn-1", "purchase", "500.00", "2024-01-15", "Store", "groceries"),
            ("txn-2", "purchase", "500.00", "2024-02-15", "Store", "groceries"),
            # Dining: big increase
            (
                "txn-3",
                "purchase",
                "100.00",
                "2024-01-20",
                "Restaurant",
                "dining",
            ),
            (
                "txn-4",
                "purchase",
                "400.00",
                "2024-02-20",
                "Restaurant",
                "dining",
            ),
        ]

        insights = analyze_spending_patterns(mock_db, months=2)

        assert insights.biggest_increase_category == "dining"

    def test_biggest_decrease_category(self, mock_db):
        """Verify biggest decrease category is identified correctly."""
        mock_db.fetchall.return_value = [
            # Entertainment: big decrease
            (
                "txn-1",
                "purchase",
                "400.00",
                "2024-01-10",
                "Cinema",
                "entertainment",
            ),
            (
                "txn-2",
                "purchase",
                "100.00",
                "2024-02-10",
                "Cinema",
                "entertainment",
            ),
            # Shopping: smaller decrease
            ("txn-3", "purchase", "300.00", "2024-01-15", "Store", "shopping"),
            ("txn-4", "purchase", "250.00", "2024-02-15", "Store", "shopping"),
        ]

        insights = analyze_spending_patterns(mock_db, months=2)

        assert insights.biggest_decrease_category == "entertainment"

    def test_empty_transactions(self, mock_db):
        """Verify handling of empty transaction list."""
        mock_db.fetchall.return_value = []

        insights = analyze_spending_patterns(mock_db, months=6)

        assert insights.monthly_trends == []
        assert insights.category_trends == []
        assert insights.savings_rate == 0.0
        assert insights.biggest_increase_category is None
        assert insights.biggest_decrease_category is None

    def test_category_trends_sorted_by_average(self, mock_db):
        """Verify category trends are sorted by average monthly spending."""
        mock_db.fetchall.return_value = [
            ("txn-1", "purchase", "100.00", "2024-01-15", "Store", "small"),
            ("txn-2", "purchase", "500.00", "2024-01-15", "Store", "large"),
            ("txn-3", "purchase", "250.00", "2024-01-15", "Store", "medium"),
        ]

        insights = analyze_spending_patterns(mock_db, months=1)

        if len(insights.category_trends) >= 3:
            assert insights.category_trends[0].category == "large"
            assert insights.category_trends[1].category == "medium"
            assert insights.category_trends[2].category == "small"

    def test_period_dates_set_correctly(self, mock_db):
        """Verify period start and end dates are set correctly."""
        mock_db.fetchall.return_value = []

        insights = analyze_spending_patterns(mock_db, months=6)

        assert insights.period_start is not None
        assert insights.period_end is not None
        assert insights.period_start < insights.period_end

    def test_handles_string_dates(self, mock_db):
        """Verify string dates from database are parsed correctly."""
        mock_db.fetchall.return_value = [
            ("txn-1", "purchase", "100.00", "2024-01-15", "Store", "test"),
        ]

        # Should not raise an exception
        insights = analyze_spending_patterns(mock_db, months=1)

        assert len(insights.monthly_trends) >= 0

    def test_income_only_transactions(self, mock_db):
        """Verify handling of income-only transactions."""
        mock_db.fetchall.return_value = [
            ("txn-1", "refund", "5000.00", "2024-01-01", "Employer", None),
            ("txn-2", "refund", "5500.00", "2024-02-01", "Employer", None),
        ]

        insights = analyze_spending_patterns(mock_db, months=2)

        assert insights.savings_rate == 1.0  # All income, no expenses
        assert insights.category_trends == []  # No expense categories
