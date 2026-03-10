"""Tests for report generation."""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from alibi.reports.monthly import (
    CategorySummary,
    InsuranceInventory,
    MonthlyReport,
    ReportGenerator,
    VendorSummary,
    WarrantyItem,
    format_report_text,
)


class TestCategorySummary:
    """Tests for CategorySummary dataclass."""

    def test_clamps_percentage(self):
        summary = CategorySummary(
            category="groceries",
            total=Decimal("100"),
            count=5,
            percentage=150.0,
        )
        assert summary.percentage == 100.0

    def test_negative_percentage_clamped(self):
        summary = CategorySummary(
            category="groceries",
            total=Decimal("100"),
            count=5,
            percentage=-10.0,
        )
        assert summary.percentage == 0.0


class TestVendorSummary:
    """Tests for VendorSummary dataclass."""

    def test_calculates_average(self):
        summary = VendorSummary(
            vendor="Amazon",
            total=Decimal("100"),
            count=4,
        )
        assert summary.average == Decimal("25")

    def test_zero_count(self):
        summary = VendorSummary(
            vendor="Amazon",
            total=Decimal("100"),
            count=0,
        )
        assert summary.average == Decimal("0")


class TestMonthlyReport:
    """Tests for MonthlyReport dataclass."""

    def test_calculates_net_change(self):
        report = MonthlyReport(
            year=2024,
            month=1,
            space_id="default",
            total_income=Decimal("5000"),
            total_expenses=Decimal("3000"),
        )
        assert report.net_change == Decimal("2000")

    def test_negative_net_change(self):
        report = MonthlyReport(
            year=2024,
            month=1,
            space_id="default",
            total_income=Decimal("2000"),
            total_expenses=Decimal("3000"),
        )
        assert report.net_change == Decimal("-1000")


class TestWarrantyItem:
    """Tests for WarrantyItem dataclass."""

    def test_calculates_days_remaining(self):
        future_date = date(2099, 12, 31)
        item = WarrantyItem(
            id="item-1",
            name="MacBook Pro",
            category="electronics",
            warranty_expires=future_date,
            warranty_type="AppleCare+",
        )
        assert item.days_remaining is not None
        assert item.days_remaining > 0

    def test_expired_warranty(self):
        past_date = date(2020, 1, 1)
        item = WarrantyItem(
            id="item-1",
            name="Old Device",
            category="electronics",
            warranty_expires=past_date,
            warranty_type="standard",
        )
        assert item.days_remaining is not None
        assert item.days_remaining < 0

    def test_no_warranty(self):
        item = WarrantyItem(
            id="item-1",
            name="No Warranty",
            category="misc",
            warranty_expires=None,
            warranty_type=None,
        )
        assert item.days_remaining is None


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.fetchone.return_value = None
        db.fetchall.return_value = []
        return db

    @pytest.fixture
    def generator(self, mock_db):
        return ReportGenerator(mock_db)

    def test_generate_monthly_report_empty(self, generator, mock_db):
        mock_db.fetchone.return_value = (0, 0, 0)

        report = generator.generate_monthly_report(2024, 1, "default")

        assert report.year == 2024
        assert report.month == 1
        assert report.total_income == Decimal("0")
        assert report.total_expenses == Decimal("0")

    def test_generate_monthly_report_with_data(self, generator, mock_db):
        # Totals query
        mock_db.fetchone.return_value = ("5000", "3000", 50)

        report = generator.generate_monthly_report(
            2024, 1, "default", include_previous=False
        )

        assert report.total_income == Decimal("5000")
        assert report.total_expenses == Decimal("3000")
        assert report.transaction_count == 50

    def test_get_warranty_status_empty(self, generator, mock_db):
        mock_db.fetchall.return_value = []

        items = generator.get_warranty_status("default")

        assert items == []

    def test_get_warranty_status_with_items(self, generator, mock_db):
        mock_db.fetchall.return_value = [
            ("item-1", "MacBook", "electronics", "2025-12-31", "AppleCare+", "active"),
            ("item-2", "iPhone", "electronics", "2024-06-15", "standard", "active"),
        ]

        items = generator.get_warranty_status("default")

        assert len(items) == 2
        assert items[0].name == "MacBook"
        assert items[1].name == "iPhone"

    def test_get_insurance_inventory_empty(self, generator, mock_db):
        mock_db.fetchall.return_value = []

        inventory = generator.get_insurance_inventory("default")

        assert inventory.item_count == 0
        assert inventory.total_value == Decimal("0")

    def test_get_insurance_inventory_with_items(self, generator, mock_db):
        mock_db.fetchall.return_value = [
            (
                "i1",
                "MacBook",
                "electronics",
                "2499",
                "2200",
                "EUR",
                "2025-12-31",
                "AppleCare+",
            ),
            ("i2", "iPhone", "electronics", "999", "800", "EUR", None, None),
        ]

        inventory = generator.get_insurance_inventory("default")

        assert inventory.item_count == 2
        assert inventory.total_value == Decimal("3000")  # 2200 + 800
        assert "electronics" in inventory.by_category


class TestFormatReportText:
    """Tests for format_report_text function."""

    def test_formats_basic_report(self):
        report = MonthlyReport(
            year=2024,
            month=1,
            space_id="default",
            total_income=Decimal("5000"),
            total_expenses=Decimal("3000"),
            transaction_count=50,
            artifact_count=10,
        )

        text = format_report_text(report)

        assert "January 2024" in text
        assert "Total Income: 5,000.00" in text
        assert "Total Expenses: 3,000.00" in text
        assert "Net Change: +2,000.00" in text

    def test_includes_category_breakdown(self):
        report = MonthlyReport(
            year=2024,
            month=6,
            space_id="default",
            total_expenses=Decimal("1000"),
            by_category=[
                CategorySummary("groceries", Decimal("500"), 10, 50.0),
                CategorySummary("dining", Decimal("300"), 5, 30.0),
            ],
        )

        text = format_report_text(report)

        assert "By Category" in text
        assert "groceries" in text
        assert "500.00" in text

    def test_includes_vendor_breakdown(self):
        report = MonthlyReport(
            year=2024,
            month=3,
            space_id="default",
            by_vendor=[
                VendorSummary("Amazon", Decimal("200"), 5),
                VendorSummary("Walmart", Decimal("100"), 3),
            ],
        )

        text = format_report_text(report)

        assert "Top Vendors" in text
        assert "Amazon" in text
        assert "200.00" in text
