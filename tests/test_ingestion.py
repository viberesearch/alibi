"""Tests for the ingestion module."""

import csv
import tempfile
from datetime import date
from decimal import Decimal
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alibi.ingestion.csv_parser import (
    CSVFormat,
    GenericCSVParser,
    N26CSVParser,
    ParsedTransaction,
    RevolutCSVParser,
    detect_csv_format,
)
from alibi.ingestion.importer import ImportResult, TransactionImporter
from alibi.ingestion.ofx_parser import OFXParser, is_ofx_file


class TestParsedTransaction:
    """Tests for ParsedTransaction dataclass."""

    def test_create_basic(self):
        """Test creating a basic transaction."""
        txn = ParsedTransaction(
            date=date(2024, 1, 15),
            description="Coffee",
            amount=Decimal("-4.50"),
        )
        assert txn.date == date(2024, 1, 15)
        assert txn.description == "Coffee"
        assert txn.amount == Decimal("-4.50")
        assert txn.currency == "EUR"

    def test_hash_computed(self):
        """Test that transaction hash is computed."""
        txn = ParsedTransaction(
            date=date(2024, 1, 15),
            description="Coffee",
            amount=Decimal("-4.50"),
        )
        assert txn.transaction_hash
        assert len(txn.transaction_hash) == 16

    def test_same_inputs_same_hash(self):
        """Test that same inputs produce same hash."""
        txn1 = ParsedTransaction(
            date=date(2024, 1, 15),
            description="Coffee",
            amount=Decimal("-4.50"),
            vendor="Starbucks",
        )
        txn2 = ParsedTransaction(
            date=date(2024, 1, 15),
            description="Different desc",
            amount=Decimal("-4.50"),
            vendor="Starbucks",
        )
        # Hash uses vendor if present
        assert txn1.transaction_hash == txn2.transaction_hash

    def test_different_amounts_different_hash(self):
        """Test that different amounts produce different hashes."""
        txn1 = ParsedTransaction(
            date=date(2024, 1, 15),
            description="Coffee",
            amount=Decimal("-4.50"),
        )
        txn2 = ParsedTransaction(
            date=date(2024, 1, 15),
            description="Coffee",
            amount=Decimal("-5.50"),
        )
        assert txn1.transaction_hash != txn2.transaction_hash


class TestBaseCSVParser:
    """Tests for base CSV parser functionality."""

    def test_parse_amount_simple(self):
        """Test parsing simple amounts."""
        assert N26CSVParser.parse_amount("100.00") == Decimal("100.00")
        assert N26CSVParser.parse_amount("-50.25") == Decimal("-50.25")

    def test_parse_amount_with_thousands(self):
        """Test parsing amounts with thousands separator."""
        assert N26CSVParser.parse_amount("1,234.56") == Decimal("1234.56")
        assert N26CSVParser.parse_amount("1,234,567.89") == Decimal("1234567.89")

    def test_parse_amount_eu_format(self):
        """Test parsing EU format (comma as decimal)."""
        assert N26CSVParser.parse_amount("100,50") == Decimal("100.50")
        assert N26CSVParser.parse_amount("-25,99") == Decimal("-25.99")

    def test_parse_amount_with_currency(self):
        """Test parsing amounts with currency symbols."""
        assert N26CSVParser.parse_amount("$100.00") == Decimal("100.00")
        assert N26CSVParser.parse_amount("EUR 50.00") == Decimal("50.00")
        assert N26CSVParser.parse_amount("100.00 EUR") == Decimal("100.00")

    def test_parse_amount_empty(self):
        """Test parsing empty amount raises error."""
        with pytest.raises(ValueError):
            N26CSVParser.parse_amount("")
        with pytest.raises(ValueError):
            N26CSVParser.parse_amount("   ")

    def test_parse_amount_invalid(self):
        """Test parsing invalid amount raises error."""
        with pytest.raises(ValueError):
            N26CSVParser.parse_amount("not a number")

    def test_parse_date_iso(self):
        """Test parsing ISO format date."""
        assert N26CSVParser.parse_date("2024-01-15") == date(2024, 1, 15)

    def test_parse_date_eu(self):
        """Test parsing EU format date."""
        assert N26CSVParser.parse_date("15.01.2024") == date(2024, 1, 15)
        assert N26CSVParser.parse_date("15/01/2024") == date(2024, 1, 15)

    def test_parse_date_invalid(self):
        """Test parsing invalid date raises error."""
        with pytest.raises(ValueError):
            N26CSVParser.parse_date("not a date")


class TestN26CSVParser:
    """Tests for N26 CSV parser."""

    N26_HEADER = '"Date","Payee","Account number","Transaction type","Payment reference","Amount (EUR)","Amount (Foreign Currency)","Type Foreign Currency","Exchange Rate"'

    def test_parse_basic_transaction(self):
        """Test parsing a basic N26 transaction."""
        csv_content = f'{self.N26_HEADER}\n"2024-01-15","Coffee Shop","","Payment","Coffee","-4.50","","",""'

        parser = N26CSVParser()
        transactions = parser.parse_file(StringIO(csv_content))

        assert len(transactions) == 1
        txn = transactions[0]
        assert txn.date == date(2024, 1, 15)
        assert txn.vendor == "Coffee Shop"
        assert txn.amount == Decimal("-4.50")
        assert txn.currency == "EUR"
        assert txn.description == "Coffee"

    def test_parse_with_foreign_currency(self):
        """Test parsing N26 transaction with foreign currency."""
        csv_content = f'{self.N26_HEADER}\n"2024-01-15","Store","","Payment","Purchase","-10.00","-11.50","USD","1.15"'

        parser = N26CSVParser()
        transactions = parser.parse_file(StringIO(csv_content))

        assert len(transactions) == 1
        txn = transactions[0]
        assert txn.amount == Decimal("-10.00")
        assert txn.original_currency == "USD"
        assert txn.original_amount == Decimal("-11.50")
        assert txn.exchange_rate == Decimal("1.15")

    def test_parse_multiple_transactions(self):
        """Test parsing multiple N26 transactions."""
        csv_content = f"""{self.N26_HEADER}
"2024-01-15","Shop A","","Payment","Item A","-10.00","","",""
"2024-01-16","Shop B","","Payment","Item B","-20.00","","",""
"2024-01-17","Shop C","","Income","Refund","5.00","","",""
"""
        parser = N26CSVParser()
        transactions = parser.parse_file(StringIO(csv_content))

        assert len(transactions) == 3
        assert transactions[0].vendor == "Shop A"
        assert transactions[1].vendor == "Shop B"
        assert transactions[2].vendor == "Shop C"
        assert transactions[2].amount == Decimal("5.00")

    def test_parse_from_file(self):
        """Test parsing N26 CSV from actual file."""
        csv_content = f'{self.N26_HEADER}\n"2024-01-15","Test","","Payment","Desc","-5.00","","",""'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()

            parser = N26CSVParser()
            transactions = parser.parse(Path(f.name))

        assert len(transactions) == 1
        assert transactions[0].vendor == "Test"


class TestRevolutCSVParser:
    """Tests for Revolut CSV parser."""

    REVOLUT_HEADER = "Type,Product,Started Date,Completed Date,Description,Amount,Fee,Currency,State,Balance"

    def test_parse_basic_transaction(self):
        """Test parsing a basic Revolut transaction."""
        csv_content = f"{self.REVOLUT_HEADER}\nCARD_PAYMENT,Current,2024-01-15 10:30:00,2024-01-15 10:35:00,To Coffee Shop,-4.50,0.00,EUR,COMPLETED,100.00"

        parser = RevolutCSVParser()
        transactions = parser.parse_file(StringIO(csv_content))

        assert len(transactions) == 1
        txn = transactions[0]
        assert txn.date == date(2024, 1, 15)
        assert txn.vendor == "Coffee Shop"
        assert txn.amount == Decimal("-4.50")
        assert txn.currency == "EUR"
        assert txn.balance == Decimal("100.00")

    def test_parse_skips_pending(self):
        """Test that pending transactions are skipped."""
        csv_content = f"""{self.REVOLUT_HEADER}
CARD_PAYMENT,Current,2024-01-15,2024-01-15,To Shop A,-10.00,0.00,EUR,COMPLETED,100.00
CARD_PAYMENT,Current,2024-01-16,2024-01-16,To Shop B,-20.00,0.00,EUR,PENDING,80.00
CARD_PAYMENT,Current,2024-01-17,2024-01-17,To Shop C,-30.00,0.00,EUR,COMPLETED,50.00
"""
        parser = RevolutCSVParser()
        transactions = parser.parse_file(StringIO(csv_content))

        assert len(transactions) == 2
        assert transactions[0].vendor == "Shop A"
        assert transactions[1].vendor == "Shop C"

    def test_parse_vendor_from_description(self):
        """Test vendor extraction from description."""
        csv_content = f"""{self.REVOLUT_HEADER}
TRANSFER,Current,2024-01-15,2024-01-15,To John Doe,-50.00,0.00,EUR,COMPLETED,100.00
TRANSFER,Current,2024-01-16,2024-01-16,From Jane Doe,100.00,0.00,EUR,COMPLETED,200.00
TRANSFER,Current,2024-01-17,2024-01-17,Some Other Description,-25.00,0.00,EUR,COMPLETED,175.00
"""
        parser = RevolutCSVParser()
        transactions = parser.parse_file(StringIO(csv_content))

        assert len(transactions) == 3
        assert transactions[0].vendor == "John Doe"
        assert transactions[1].vendor == "Jane Doe"
        assert transactions[2].vendor == "Some Other Description"


class TestGenericCSVParser:
    """Tests for generic CSV parser."""

    def test_parse_auto_detect_columns(self):
        """Test auto-detecting common column names."""
        csv_content = "date,description,amount\n2024-01-15,Coffee,-4.50"

        parser = GenericCSVParser()
        transactions = parser.parse_file(StringIO(csv_content))

        assert len(transactions) == 1
        assert transactions[0].date == date(2024, 1, 15)
        assert transactions[0].description == "Coffee"
        assert transactions[0].amount == Decimal("-4.50")

    def test_parse_with_currency_column(self):
        """Test parsing with currency column."""
        csv_content = "date,description,amount,currency\n2024-01-15,Coffee,-4.50,USD"

        parser = GenericCSVParser()
        transactions = parser.parse_file(StringIO(csv_content))

        assert len(transactions) == 1
        assert transactions[0].currency == "USD"

    def test_parse_with_explicit_columns(self):
        """Test parsing with explicitly specified columns."""
        csv_content = "txn_date,memo,value\n2024-01-15,Coffee,-4.50"

        parser = GenericCSVParser(
            date_column="txn_date",
            description_column="memo",
            amount_column="value",
        )
        transactions = parser.parse_file(StringIO(csv_content))

        assert len(transactions) == 1
        assert transactions[0].description == "Coffee"

    def test_parse_fails_without_required_columns(self):
        """Test that parsing fails without required columns."""
        csv_content = "name,value\nItem,100"

        parser = GenericCSVParser()
        with pytest.raises(ValueError, match="Cannot detect required columns"):
            parser.parse_file(StringIO(csv_content))

    def test_parse_default_currency(self):
        """Test default currency is used."""
        csv_content = "date,description,amount\n2024-01-15,Coffee,-4.50"

        parser = GenericCSVParser(default_currency="CHF")
        transactions = parser.parse_file(StringIO(csv_content))

        assert transactions[0].currency == "CHF"


class TestDetectCSVFormat:
    """Tests for CSV format detection."""

    def test_detect_n26_format(self):
        """Test detecting N26 format."""
        csv_content = '"Date","Payee","Account number","Transaction type","Payment reference","Amount (EUR)","Amount (Foreign Currency)","Type Foreign Currency","Exchange Rate"\n'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()

            detected = detect_csv_format(Path(f.name))

        assert detected == CSVFormat.N26

    def test_detect_revolut_format(self):
        """Test detecting Revolut format."""
        csv_content = "Type,Product,Started Date,Completed Date,Description,Amount,Fee,Currency,State,Balance\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()

            detected = detect_csv_format(Path(f.name))

        assert detected == CSVFormat.REVOLUT

    def test_detect_generic_format(self):
        """Test detecting generic format."""
        csv_content = "date,description,amount,currency\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()

            detected = detect_csv_format(Path(f.name))

        assert detected == CSVFormat.GENERIC

    def test_detect_unknown_format(self):
        """Test detecting unknown format."""
        csv_content = "foo,bar,baz\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()

            detected = detect_csv_format(Path(f.name))

        assert detected == CSVFormat.UNKNOWN


class TestOFXParser:
    """Tests for OFX parser."""

    def test_ofx_parser_requires_ofxparse(self):
        """Test that OFX parser requires ofxparse library."""
        # This test will pass if ofxparse is installed
        try:
            parser = OFXParser()
            assert parser is not None
        except ImportError:
            pytest.skip("ofxparse not installed")

    def test_is_ofx_file_by_extension(self):
        """Test detecting OFX file by extension."""
        with tempfile.NamedTemporaryFile(suffix=".ofx", delete=False) as f:
            f.write(b"dummy content")
            f.flush()
            assert is_ofx_file(Path(f.name))

        with tempfile.NamedTemporaryFile(suffix=".qfx", delete=False) as f:
            f.write(b"dummy content")
            f.flush()
            assert is_ofx_file(Path(f.name))

    def test_is_ofx_file_by_content(self):
        """Test detecting OFX file by content."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"OFXHEADER:100\n<OFX>")
            f.flush()
            assert is_ofx_file(Path(f.name))

    def test_is_ofx_file_not_ofx(self):
        """Test non-OFX file is not detected."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"date,amount\n2024-01-01,100")
            f.flush()
            assert not is_ofx_file(Path(f.name))


class TestTransactionImporter:
    """Tests for transaction importer."""

    def test_import_result_dataclass(self):
        """Test ImportResult dataclass."""
        result = ImportResult(
            file_path=Path("/test/file.csv"),
            total_rows=10,
            imported=8,
            duplicates=2,
            errors=0,
            success=True,
        )
        assert result.total_rows == 10
        assert result.imported == 8
        assert result.duplicates == 2

    def test_import_csv_file_not_found(self):
        """Test importing non-existent file."""
        db_mock = MagicMock()
        importer = TransactionImporter(db=db_mock)

        result = importer.import_csv("/nonexistent/file.csv")

        assert not result.success
        assert result.error_message is not None
        assert "not found" in result.error_message.lower()

    def test_import_csv_unknown_format(self):
        """Test importing CSV with unknown format."""
        csv_content = "foo,bar,baz\n1,2,3"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()

            db_mock = MagicMock()
            importer = TransactionImporter(db=db_mock)
            result = importer.import_csv(f.name)

        assert not result.success
        assert result.error_message is not None
        assert "format" in result.error_message.lower()

    def test_import_csv_with_explicit_format(self):
        """Test importing CSV with explicit format."""
        csv_content = '"Date","Payee","Account number","Transaction type","Payment reference","Amount (EUR)","Amount (Foreign Currency)","Type Foreign Currency","Exchange Rate"\n"2024-01-15","Test","","Payment","Desc","-5.00","","",""'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()

            db_mock = MagicMock()
            db_mock.fetchone.return_value = None  # No duplicates
            importer = TransactionImporter(db=db_mock)
            result = importer.import_csv(f.name, format_type="n26")

        assert result.success
        assert result.imported == 1

    def test_import_csv_skips_duplicates(self):
        """Test that duplicates are skipped."""
        csv_content = '"Date","Payee","Account number","Transaction type","Payment reference","Amount (EUR)","Amount (Foreign Currency)","Type Foreign Currency","Exchange Rate"\n"2024-01-15","Test","","Payment","Desc","-5.00","","",""'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()

            db_mock = MagicMock()
            db_mock.fetchone.return_value = (1,)  # Duplicate found
            importer = TransactionImporter(db=db_mock)
            result = importer.import_csv(f.name, format_type="n26")

        assert result.success
        assert result.imported == 0
        assert result.duplicates == 1

    def test_import_ofx_file_not_found(self):
        """Test importing non-existent OFX file."""
        db_mock = MagicMock()
        importer = TransactionImporter(db=db_mock)

        result = importer.import_ofx("/nonexistent/file.ofx")

        assert not result.success
        assert result.error_message is not None
        assert "not found" in result.error_message.lower()

    def test_import_ofx_not_ofx_format(self):
        """Test importing non-OFX file as OFX."""
        csv_content = "date,amount\n2024-01-01,100"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()

            db_mock = MagicMock()
            importer = TransactionImporter(db=db_mock)
            result = importer.import_ofx(f.name)

        assert not result.success
        assert result.error_message is not None
        assert "OFX" in result.error_message


class TestImporterIntegration:
    """Integration tests for importer with mock database."""

    def test_full_import_flow_n26(self):
        """Test full import flow with N26 CSV."""
        csv_content = '"Date","Payee","Account number","Transaction type","Payment reference","Amount (EUR)","Amount (Foreign Currency)","Type Foreign Currency","Exchange Rate"\n"2024-01-15","Coffee Shop","","Payment","Morning coffee","-4.50","","",""\n"2024-01-16","Grocery Store","","Payment","Weekly shopping","-25.00","","",""'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()

            db_mock = MagicMock()
            db_mock.fetchone.return_value = None  # No duplicates
            cursor_mock = MagicMock()
            db_mock.transaction.return_value.__enter__ = MagicMock(
                return_value=cursor_mock
            )
            db_mock.transaction.return_value.__exit__ = MagicMock(return_value=False)

            importer = TransactionImporter(
                db=db_mock,
                space_id="test-space",
                default_currency="EUR",
            )
            result = importer.import_csv(f.name)

        assert result.success
        assert result.total_rows == 2
        assert result.imported == 2
        assert result.duplicates == 0
        assert result.errors == 0

        # Verify database was called (2 txns * 2 inserts each = 4)
        assert cursor_mock.execute.call_count == 4

    def test_import_with_account_name(self):
        """Test import with account name override."""
        csv_content = '"Date","Payee","Account number","Transaction type","Payment reference","Amount (EUR)","Amount (Foreign Currency)","Type Foreign Currency","Exchange Rate"\n"2024-01-15","Test","","Payment","Desc","-5.00","","",""'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()

            db_mock = MagicMock()
            db_mock.fetchone.return_value = None
            cursor_mock = MagicMock()
            db_mock.transaction.return_value.__enter__ = MagicMock(
                return_value=cursor_mock
            )
            db_mock.transaction.return_value.__exit__ = MagicMock(return_value=False)

            importer = TransactionImporter(db=db_mock)
            result = importer.import_csv(f.name, account_name="My N26 Account")

        assert result.success
        # Verify fact was inserted with vendor "Test"
        fact_insert = cursor_mock.execute.call_args_list[
            1
        ]  # second call is facts INSERT
        assert "INSERT INTO facts" in fact_insert[0][0]

    def test_import_canonicalizes_vendor(self, tmp_path):
        """Test that imported vendor names are canonicalized via aliases."""
        from alibi.matching.duplicates import (
            init_vendor_mappings,
            reset_vendor_mappings,
        )

        # Create a vendor aliases YAML
        aliases_file = tmp_path / "vendor_aliases.yaml"
        aliases_file.write_text("fresko butanolo: FRESKO\n")

        try:
            csv_content = (
                '"Date","Payee","Account number","Transaction type",'
                '"Payment reference","Amount (EUR)",'
                '"Amount (Foreign Currency)","Type Foreign Currency",'
                '"Exchange Rate"\n'
                '"2024-01-15","FreSko BUTANOLO LTD","","Payment",'
                '"Groceries","-12.50","","",""\n'
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                f.write(csv_content)
                f.flush()

                db_mock = MagicMock()
                db_mock.fetchone.return_value = None
                cursor_mock = MagicMock()
                db_mock.transaction.return_value.__enter__ = MagicMock(
                    return_value=cursor_mock
                )
                db_mock.transaction.return_value.__exit__ = MagicMock(
                    return_value=False
                )

                importer = TransactionImporter(db=db_mock)
                # Load test aliases AFTER importer init (which loads defaults)
                init_vendor_mappings(aliases_file)
                result = importer.import_csv(f.name, format_type="n26")

            assert result.success
            assert result.imported == 1

            # Extract the vendor from the facts INSERT call
            fact_insert = cursor_mock.execute.call_args_list[1]
            insert_args = fact_insert[0][1]
            vendor_in_db = insert_args[3]  # vendor is 4th param
            assert vendor_in_db == "FRESKO"
        finally:
            reset_vendor_mappings()

    def test_import_vendor_none_handled(self):
        """Test that None vendor doesn't crash canonicalization."""
        from alibi.ingestion.csv_parser import ParsedTransaction

        db_mock = MagicMock()
        cursor_mock = MagicMock()
        db_mock.transaction.return_value.__enter__ = MagicMock(return_value=cursor_mock)
        db_mock.transaction.return_value.__exit__ = MagicMock(return_value=False)

        importer = TransactionImporter(db=db_mock)

        parsed = ParsedTransaction(
            date=date(2024, 1, 15),
            amount=Decimal("-10.00"),
            currency="EUR",
            vendor=None,
            description="Unknown merchant",
        )

        # Should not crash with None vendor
        importer._insert_as_fact(parsed, None)
        # canonicalize_vendor(None) returns None — verify it was inserted
        fact_insert = cursor_mock.execute.call_args_list[1]
        insert_args = fact_insert[0][1]
        assert insert_args[3] is None  # vendor is 4th param


class TestDisambiguateDate:
    """Tests for date disambiguation using contextual signals."""

    def test_future_date_rejected(self) -> None:
        from alibi.normalizers.dates import disambiguate_date

        # "03/05/2026" parsed as May 3 (European DD/MM default)
        # but file created in March -> swap to March 5
        parsed = date(2026, 5, 3)  # May 3
        ref = date(2026, 3, 5)  # March 5 (today)
        result = disambiguate_date(parsed, "03/05/2026", reference_date=ref)
        assert result == date(2026, 3, 5)

    def test_past_date_kept(self) -> None:
        from alibi.normalizers.dates import disambiguate_date

        # Both interpretations are in the past -- pick closer to reference
        parsed = date(2026, 2, 3)  # Feb 3
        ref = date(2026, 3, 5)
        result = disambiguate_date(parsed, "03/02/2026", reference_date=ref)
        # Feb 3 is 30 days away, March 2 is 3 days away -> March 2
        assert result == date(2026, 3, 2)

    def test_unambiguous_date_unchanged(self) -> None:
        from alibi.normalizers.dates import disambiguate_date

        # Day > 12, unambiguous
        parsed = date(2026, 3, 15)  # March 15
        result = disambiguate_date(
            parsed, "15/03/2026", reference_date=date(2026, 3, 5)
        )
        assert result == date(2026, 3, 15)

    def test_same_day_month_unchanged(self) -> None:
        from alibi.normalizers.dates import disambiguate_date

        parsed = date(2026, 5, 5)  # May 5
        result = disambiguate_date(
            parsed, "05/05/2026", reference_date=date(2026, 3, 5)
        )
        assert result == date(2026, 5, 5)  # can't disambiguate

    def test_file_date_proximity(self) -> None:
        from alibi.normalizers.dates import disambiguate_date

        # File created March 4 -> prefer March 5 over May 3
        parsed = date(2026, 5, 3)
        result = disambiguate_date(
            parsed,
            "03/05/2026",
            file_date=date(2026, 3, 4),
            reference_date=date(2026, 6, 1),  # far away ref
        )
        assert result == date(2026, 3, 5)

    def test_both_past_file_date_wins(self) -> None:
        from alibi.normalizers.dates import disambiguate_date

        # Both dates in past, file date is closer to one interpretation
        parsed = date(2026, 1, 3)  # Jan 3
        result = disambiguate_date(
            parsed,
            "03/01/2026",
            file_date=date(2026, 3, 2),
            reference_date=date(2026, 4, 1),
        )
        # Jan 3 is 58 days from file, March 1 is 1 day -> March 1
        assert result == date(2026, 3, 1)
