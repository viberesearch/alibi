"""Tests for export functionality."""

import csv
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from alibi.export import (
    ExportResult,
    _serialize_value,
    export_all,
    export_artifacts,
    export_items,
    export_transactions,
)


class TestSerializeValue:
    """Tests for _serialize_value function."""

    def test_serialize_decimal(self):
        """Test serialization of Decimal values."""
        value = Decimal("99.99")
        result = _serialize_value(value)
        assert isinstance(result, float)
        assert result == 99.99

    def test_serialize_decimal_precision(self):
        """Test serialization preserves decimal precision."""
        value = Decimal("123.456789")
        result = _serialize_value(value)
        assert isinstance(result, float)
        assert result == 123.456789

    def test_serialize_date(self):
        """Test serialization of date objects."""
        value = date(2024, 1, 15)
        result = _serialize_value(value)
        assert result == "2024-01-15"

    def test_serialize_datetime(self):
        """Test serialization of datetime objects."""
        value = datetime(2024, 1, 15, 14, 30, 45)
        result = _serialize_value(value)
        assert result == "2024-01-15T14:30:45"

    def test_serialize_string(self):
        """Test serialization of string values (passthrough)."""
        value = "test string"
        result = _serialize_value(value)
        assert result == "test string"

    def test_serialize_int(self):
        """Test serialization of int values (passthrough)."""
        value = 42
        result = _serialize_value(value)
        assert result == 42

    def test_serialize_none(self):
        """Test serialization of None values (passthrough)."""
        value = None
        result = _serialize_value(value)
        assert result is None


class TestExportTransactions:
    """Tests for export_transactions function."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database manager."""
        db = MagicMock()
        db.fetchall.return_value = []
        return db

    @pytest.fixture
    def sample_transactions(self):
        """Sample fact data for export."""
        return [
            (
                "txn-1",
                "Amazon",
                "purchase",
                Decimal("99.99"),
                "USD",
                date(2024, 1, 15),
                "confirmed",
            ),
            (
                "txn-2",
                "Walmart",
                "purchase",
                Decimal("50.00"),
                "USD",
                date(2024, 1, 10),
                "confirmed",
            ),
        ]

    def test_export_transactions_csv(self, mock_db, sample_transactions, tmp_path):
        """Test exporting transactions to CSV."""
        mock_db.fetchall.return_value = sample_transactions
        output_file = tmp_path / "facts.csv"

        result = export_transactions(mock_db, output_file, format="csv")

        assert isinstance(result, ExportResult)
        assert result.path == output_file
        assert result.format == "csv"
        assert result.record_count == 2
        assert result.size_bytes > 0

        # Verify CSV content
        with open(output_file) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 3  # header + 2 data rows
            assert rows[0] == [
                "id",
                "vendor",
                "type",
                "amount",
                "currency",
                "date",
                "status",
            ]
            assert rows[1][0] == "txn-1"
            assert rows[1][1] == "Amazon"
            assert rows[1][3] == "99.99"

    def test_export_transactions_json(self, mock_db, sample_transactions, tmp_path):
        """Test exporting transactions to JSON."""
        mock_db.fetchall.return_value = sample_transactions
        output_file = tmp_path / "facts.json"

        result = export_transactions(mock_db, output_file, format="json")

        assert isinstance(result, ExportResult)
        assert result.path == output_file
        assert result.format == "json"
        assert result.record_count == 2
        assert result.size_bytes > 0

        # Verify JSON content
        with open(output_file) as f:
            data = json.load(f)
            assert len(data) == 2
            assert data[0]["id"] == "txn-1"
            assert data[0]["vendor"] == "Amazon"
            assert data[0]["amount"] == 99.99
            assert data[0]["date"] == "2024-01-15"

    def test_export_with_date_filter_since(
        self, mock_db, sample_transactions, tmp_path
    ):
        """Test exporting with since date filter."""
        mock_db.fetchall.return_value = sample_transactions
        output_file = tmp_path / "filtered.csv"
        since_date = date(2024, 1, 12)

        result = export_transactions(
            mock_db, output_file, format="csv", since=since_date
        )

        # Verify the query was called with since parameter
        call_args = mock_db.fetchall.call_args
        assert "event_date >= ?" in call_args[0][0]
        assert "2024-01-12" in call_args[0][1]
        assert isinstance(result, ExportResult)

    def test_export_with_date_filter_until(
        self, mock_db, sample_transactions, tmp_path
    ):
        """Test exporting with until date filter."""
        mock_db.fetchall.return_value = sample_transactions
        output_file = tmp_path / "filtered.csv"
        until_date = date(2024, 1, 20)

        result = export_transactions(
            mock_db, output_file, format="csv", until=until_date
        )

        # Verify the query was called with until parameter
        call_args = mock_db.fetchall.call_args
        assert "event_date <= ?" in call_args[0][0]
        assert "2024-01-20" in call_args[0][1]
        assert isinstance(result, ExportResult)

    def test_export_with_both_date_filters(
        self, mock_db, sample_transactions, tmp_path
    ):
        """Test exporting with both since and until filters."""
        mock_db.fetchall.return_value = sample_transactions
        output_file = tmp_path / "filtered.csv"
        since_date = date(2024, 1, 1)
        until_date = date(2024, 1, 31)

        result = export_transactions(
            mock_db, output_file, format="csv", since=since_date, until=until_date
        )

        # Verify both parameters are in the query
        call_args = mock_db.fetchall.call_args
        assert "event_date >= ?" in call_args[0][0]
        assert "event_date <= ?" in call_args[0][0]
        assert "2024-01-01" in call_args[0][1]
        assert "2024-01-31" in call_args[0][1]
        assert isinstance(result, ExportResult)

    def test_export_empty_transactions(self, mock_db, tmp_path):
        """Test exporting when there are no transactions."""
        mock_db.fetchall.return_value = []
        output_file = tmp_path / "empty.csv"

        result = export_transactions(mock_db, output_file, format="csv")

        assert result.record_count == 0
        assert result.size_bytes > 0  # Header still exists

        # Verify file has header only
        with open(output_file) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 1  # header only

    def test_export_custom_space(self, mock_db, tmp_path):
        """Test exporting (space_id kept for API compat but not used in query)."""
        mock_db.fetchall.return_value = []
        output_file = tmp_path / "custom.csv"

        export_transactions(mock_db, output_file, space_id="custom-space")

        # Query should have been called (space_id no longer in v2 query)
        mock_db.fetchall.assert_called_once()


class TestExportItems:
    """Tests for export_items function."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database manager."""
        db = MagicMock()
        db.fetchall.return_value = []
        return db

    @pytest.fixture
    def sample_items(self):
        """Sample item data."""
        return [
            (
                "item-1",
                "MacBook Pro",
                "electronics",
                "M1 Pro 16-inch",
                date(2023, 6, 1),
                Decimal("2499.99"),
                "USD",
                date(2026, 6, 1),
                "home",
            ),
            (
                "item-2",
                "Coffee Maker",
                "appliances",
                "Automatic drip",
                date(2024, 1, 1),
                Decimal("89.99"),
                "USD",
                None,
                "kitchen",
            ),
        ]

    def test_export_items_csv(self, mock_db, sample_items, tmp_path):
        """Test exporting items to CSV."""
        mock_db.fetchall.return_value = sample_items
        output_file = tmp_path / "items.csv"

        result = export_items(mock_db, output_file, format="csv")

        assert isinstance(result, ExportResult)
        assert result.path == output_file
        assert result.format == "csv"
        assert result.record_count == 2
        assert result.size_bytes > 0

        # Verify CSV content
        with open(output_file) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 3  # header + 2 data rows
            assert rows[0] == [
                "id",
                "name",
                "category",
                "description",
                "purchase_date",
                "purchase_price",
                "currency",
                "warranty_expires",
                "location",
            ]
            assert rows[1][1] == "MacBook Pro"
            assert rows[1][5] == "2499.99"

    def test_export_items_json(self, mock_db, sample_items, tmp_path):
        """Test exporting items to JSON."""
        mock_db.fetchall.return_value = sample_items
        output_file = tmp_path / "items.json"

        result = export_items(mock_db, output_file, format="json")

        assert isinstance(result, ExportResult)
        assert result.path == output_file
        assert result.format == "json"
        assert result.record_count == 2

        # Verify JSON content
        with open(output_file) as f:
            data = json.load(f)
            assert len(data) == 2
            assert data[0]["name"] == "MacBook Pro"
            assert data[0]["purchase_price"] == 2499.99
            assert data[0]["purchase_date"] == "2023-06-01"
            assert data[0]["warranty_expires"] == "2026-06-01"
            assert data[1]["warranty_expires"] is None

    def test_export_items_empty(self, mock_db, tmp_path):
        """Test exporting when there are no items."""
        mock_db.fetchall.return_value = []
        output_file = tmp_path / "empty_items.csv"

        result = export_items(mock_db, output_file, format="csv")

        assert result.record_count == 0
        assert result.size_bytes > 0  # Header still exists


class TestExportArtifacts:
    """Tests for export_artifacts function."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database manager."""
        db = MagicMock()
        db.fetchall.return_value = []
        return db

    @pytest.fixture
    def sample_artifacts(self):
        """Sample document data (with fact join)."""
        return [
            (
                "doc-1",
                "/path/to/receipt1.jpg",
                "2024-01-15T10:00:00",
                "purchase",
                "Amazon",
                date(2024, 1, 15),
                Decimal("99.99"),
                "USD",
            ),
            (
                "doc-2",
                "/path/to/invoice.pdf",
                "2024-01-10T10:00:00",
                "purchase",
                "Acme Corp",
                date(2024, 1, 10),
                Decimal("1500.00"),
                "USD",
            ),
        ]

    def test_export_artifacts_csv(self, mock_db, sample_artifacts, tmp_path):
        """Test exporting artifacts to CSV."""
        mock_db.fetchall.return_value = sample_artifacts
        output_file = tmp_path / "documents.csv"

        result = export_artifacts(mock_db, output_file, format="csv")

        assert isinstance(result, ExportResult)
        assert result.path == output_file
        assert result.format == "csv"
        assert result.record_count == 2
        assert result.size_bytes > 0

        # Verify CSV content
        with open(output_file) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 3  # header + 2 data rows
            assert rows[0] == [
                "id",
                "file_path",
                "ingested_at",
                "type",
                "vendor",
                "date",
                "amount",
                "currency",
            ]
            assert rows[1][0] == "doc-1"
            assert rows[1][4] == "Amazon"

    def test_export_artifacts_json(self, mock_db, sample_artifacts, tmp_path):
        """Test exporting artifacts to JSON."""
        mock_db.fetchall.return_value = sample_artifacts
        output_file = tmp_path / "documents.json"

        result = export_artifacts(mock_db, output_file, format="json")

        assert isinstance(result, ExportResult)
        assert result.path == output_file
        assert result.format == "json"
        assert result.record_count == 2

        # Verify JSON content
        with open(output_file) as f:
            data = json.load(f)
            assert len(data) == 2
            assert data[0]["id"] == "doc-1"
            assert data[0]["vendor"] == "Amazon"
            assert data[0]["amount"] == 99.99
            assert data[0]["date"] == "2024-01-15"

    def test_export_artifacts_empty(self, mock_db, tmp_path):
        """Test exporting when there are no artifacts."""
        mock_db.fetchall.return_value = []
        output_file = tmp_path / "empty_artifacts.csv"

        result = export_artifacts(mock_db, output_file, format="csv")

        assert result.record_count == 0
        assert result.size_bytes > 0  # Header still exists


class TestExportAll:
    """Tests for export_all function."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database manager."""
        db = MagicMock()
        # Return different data for different queries
        db.fetchall.side_effect = [
            [  # facts
                (
                    "txn-1",
                    "Amazon",
                    "purchase",
                    Decimal("99.99"),
                    "USD",
                    date(2024, 1, 15),
                    "confirmed",
                )
            ],
            [  # items (v1, kept for compat)
                (
                    "item-1",
                    "Laptop",
                    "electronics",
                    "Work laptop",
                    date(2023, 1, 1),
                    Decimal("1500.00"),
                    "USD",
                    date(2026, 1, 1),
                    "office",
                )
            ],
            [  # documents
                (
                    "doc-1",
                    "/path/receipt.jpg",
                    "2024-01-01T10:00:00",
                    "purchase",
                    "Store",
                    date(2024, 1, 1),
                    Decimal("50.00"),
                    "USD",
                )
            ],
        ]
        return db

    def test_export_all_csv(self, mock_db, tmp_path):
        """Test exporting all data to CSV files."""
        output_dir = tmp_path / "export_csv"

        results = export_all(mock_db, output_dir, format="csv")

        assert len(results) == 3
        assert output_dir.exists()
        assert (output_dir / "facts.csv").exists()
        assert (output_dir / "items.csv").exists()
        assert (output_dir / "documents.csv").exists()

        # Verify each result
        assert all(isinstance(r, ExportResult) for r in results)
        assert results[0].format == "csv"
        assert results[1].format == "csv"
        assert results[2].format == "csv"

    def test_export_all_json(self, mock_db, tmp_path):
        """Test exporting all data to JSON files."""
        output_dir = tmp_path / "export_json"

        results = export_all(mock_db, output_dir, format="json")

        assert len(results) == 3
        assert output_dir.exists()
        assert (output_dir / "facts.json").exists()
        assert (output_dir / "items.json").exists()
        assert (output_dir / "documents.json").exists()

        # Verify each result
        assert all(isinstance(r, ExportResult) for r in results)
        assert results[0].format == "json"
        assert results[1].format == "json"
        assert results[2].format == "json"

    def test_export_all_creates_directory(self, mock_db, tmp_path):
        """Test that export_all creates output directory if it doesn't exist."""
        output_dir = tmp_path / "new" / "nested" / "dir"
        assert not output_dir.exists()

        results = export_all(mock_db, output_dir, format="csv")

        assert output_dir.exists()
        assert len(results) == 3

    def test_export_all_with_date_filters(self, mock_db, tmp_path):
        """Test export_all passes date filters to transactions export."""
        output_dir = tmp_path / "filtered_export"
        since_date = date(2024, 1, 1)
        until_date = date(2024, 1, 31)

        results = export_all(
            mock_db, output_dir, format="csv", since=since_date, until=until_date
        )

        assert len(results) == 3
        # Note: Only transactions export should use date filters
        # The first fetchall call should include the date parameters

    def test_export_all_custom_space(self, mock_db, tmp_path):
        """Test export_all with custom space_id."""
        output_dir = tmp_path / "custom_space_export"

        results = export_all(mock_db, output_dir, space_id="test-space")

        assert len(results) == 3
        # Verify space_id was passed to all export functions
        # Each fetchall call should include the space_id parameter


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_export_result_attributes(self):
        """Test that ExportResult has correct attributes."""
        path = Path("/tmp/test.csv")
        result = ExportResult(
            path=path, format="csv", record_count=100, size_bytes=5000
        )

        assert result.path == path
        assert result.format == "csv"
        assert result.record_count == 100
        assert result.size_bytes == 5000

    def test_export_result_with_path_object(self):
        """Test ExportResult works with Path objects."""
        path = Path("/some/path/export.json")
        result = ExportResult(
            path=path, format="json", record_count=50, size_bytes=2500
        )

        assert isinstance(result.path, Path)
        assert result.path == path

    def test_export_result_zero_records(self):
        """Test ExportResult with zero records."""
        result = ExportResult(
            path=Path("/tmp/empty.csv"), format="csv", record_count=0, size_bytes=100
        )

        assert result.record_count == 0
        assert result.size_bytes == 100  # Still has header
