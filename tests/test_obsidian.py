"""Tests for Obsidian note generation."""

from datetime import date
from decimal import Decimal

import pytest

from alibi.db.models import (
    Item,
    ItemStatus,
)
from alibi.obsidian.notes import (
    format_currency,
    generate_item_note,
    get_note_filename,
    sanitize_filename,
)


class TestFormatCurrency:
    """Tests for format_currency function."""

    def test_eur(self):
        result = format_currency(Decimal("99.99"), "EUR")
        assert result == "\u20ac99.99"

    def test_usd(self):
        result = format_currency(Decimal("100"), "USD")
        assert result == "$100.00"

    def test_gbp(self):
        result = format_currency(Decimal("50.50"), "GBP")
        assert result == "\u00a350.50"

    def test_unknown_currency(self):
        result = format_currency(Decimal("100"), "JPY")
        assert result == "JPY 100.00"

    def test_none(self):
        result = format_currency(None, "EUR")
        assert result == "N/A"


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_removes_slashes(self):
        result = sanitize_filename("path/to/file")
        assert result == "path-to-file"

    def test_removes_special_chars(self):
        result = sanitize_filename("file:name*?.txt")
        assert result == "file-name.txt"

    def test_strips_whitespace(self):
        result = sanitize_filename("  filename  ")
        assert result == "filename"


class TestGetNoteFilename:
    """Tests for get_note_filename function."""

    def test_with_date(self):
        result = get_note_filename("transaction", date(2024, 1, 15), "Amazon")
        assert result == "2024-01-15_transaction_Amazon"

    def test_without_date(self):
        result = get_note_filename("item", None, "MacBook Pro")
        assert result == "unknown_item_MacBook Pro"

    def test_truncates_long_names(self):
        long_name = "A" * 100
        result = get_note_filename("item", date(2024, 1, 1), long_name)
        assert len(result.split("_")[-1]) <= 50


class TestGenerateItemNote:
    """Tests for generate_item_note function."""

    @pytest.fixture
    def sample_item(self):
        return Item(
            id="item-456",
            space_id="default",
            name='MacBook Pro 14"',
            category="electronics",
            model="M3 Pro",
            serial_number="C02ABC123",
            purchase_date=date(2024, 1, 10),
            purchase_price=Decimal("2499.00"),
            current_value=Decimal("2200.00"),
            status=ItemStatus.ACTIVE,
            warranty_expires=date(2027, 1, 10),
            warranty_type="AppleCare+",
            insurance_covered=True,
        )

    def test_generates_valid_markdown(self, sample_item):
        result = generate_item_note(sample_item)

        assert "---" in result
        assert "type: item" in result
        assert 'name: "MacBook Pro 14' in result
        assert "M3 Pro" in result
        assert "C02ABC123" in result

    def test_includes_warranty_info(self, sample_item):
        result = generate_item_note(sample_item)

        assert "2027-01-10" in result
        assert "AppleCare+" in result
        assert "Insurance Covered**: Yes" in result

    def test_handles_missing_fields(self):
        minimal_item = Item(
            id="item-789",
            space_id="default",
            name="Simple Item",
        )

        result = generate_item_note(minimal_item)

        assert 'name: "Simple Item"' in result
        assert "_No specifications recorded_" in result


class TestNoteExporter:
    """Tests for NoteExporter class."""

    def test_requires_vault_path(self):
        from unittest.mock import MagicMock, patch

        mock_db = MagicMock()

        with patch("alibi.obsidian.notes.get_config") as mock_config:
            mock_config.return_value.vault_path = None

            with pytest.raises(ValueError, match="No vault path configured"):
                from alibi.obsidian.notes import NoteExporter

                NoteExporter(mock_db)
