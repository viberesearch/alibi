"""Transaction importer for Alibi.

Provides high-level import functionality for CSV and OFX files.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from alibi.config import get_config
from alibi.matching.duplicates import canonicalize_vendor, init_vendor_mappings
from alibi.ingestion.csv_parser import (
    CSVFormat,
    GenericCSVParser,
    N26CSVParser,
    ParsedTransaction,
    RevolutCSVParser,
    detect_csv_format,
)
from alibi.ingestion.ofx_parser import OFXParser, is_ofx_file

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager


@dataclass
class ImportResult:
    """Result of an import operation."""

    file_path: Path
    total_rows: int
    imported: int
    duplicates: int
    errors: int
    success: bool
    error_message: str | None = None


class TransactionImporter:
    """Imports transactions from CSV and OFX files into the database."""

    def __init__(
        self,
        db: "DatabaseManager",
        space_id: str = "default",
        default_currency: str = "EUR",
    ):
        """Initialize importer.

        Args:
            db: Database manager instance.
            space_id: Space ID for imported transactions.
            default_currency: Default currency for transactions.
        """
        self.db = db
        self.space_id = space_id
        self.default_currency = default_currency

        # Load vendor alias overrides for canonicalization
        config = get_config()
        init_vendor_mappings(config.get_vendor_aliases_path())

    def import_csv(
        self,
        file_path: Path | str,
        format_type: CSVFormat | str | None = None,
        account_name: str | None = None,
    ) -> ImportResult:
        """Import transactions from a CSV file.

        Args:
            file_path: Path to CSV file.
            format_type: CSV format (n26, revolut, generic) or None to auto-detect.
            account_name: Optional account name for transactions.

        Returns:
            Import result with statistics.
        """
        path = Path(file_path)

        if not path.exists():
            return ImportResult(
                file_path=path,
                total_rows=0,
                imported=0,
                duplicates=0,
                errors=0,
                success=False,
                error_message=f"File not found: {path}",
            )

        # Detect or convert format
        if format_type is None:
            detected = detect_csv_format(path)
            if detected == CSVFormat.UNKNOWN:
                return ImportResult(
                    file_path=path,
                    total_rows=0,
                    imported=0,
                    duplicates=0,
                    errors=0,
                    success=False,
                    error_message="Cannot detect CSV format. Use --format option.",
                )
            format_type = detected
        elif isinstance(format_type, str):
            try:
                format_type = CSVFormat(format_type.lower())
            except ValueError:
                return ImportResult(
                    file_path=path,
                    total_rows=0,
                    imported=0,
                    duplicates=0,
                    errors=0,
                    success=False,
                    error_message=f"Unknown format: {format_type}",
                )

        # Create parser
        parser: N26CSVParser | RevolutCSVParser | GenericCSVParser
        if format_type == CSVFormat.N26:
            parser = N26CSVParser()
        elif format_type == CSVFormat.REVOLUT:
            parser = RevolutCSVParser()
        else:
            parser = GenericCSVParser(default_currency=self.default_currency)

        # Parse transactions
        try:
            parsed = parser.parse(path)
        except Exception as e:
            return ImportResult(
                file_path=path,
                total_rows=0,
                imported=0,
                duplicates=0,
                errors=0,
                success=False,
                error_message=f"Parse error: {e}",
            )

        # Import transactions
        return self._import_transactions(path, parsed, account_name)

    def import_ofx(
        self,
        file_path: Path | str,
        account_name: str | None = None,
    ) -> ImportResult:
        """Import transactions from an OFX file.

        Args:
            file_path: Path to OFX file.
            account_name: Optional account name override.

        Returns:
            Import result with statistics.
        """
        path = Path(file_path)

        if not path.exists():
            return ImportResult(
                file_path=path,
                total_rows=0,
                imported=0,
                duplicates=0,
                errors=0,
                success=False,
                error_message=f"File not found: {path}",
            )

        if not is_ofx_file(path):
            return ImportResult(
                file_path=path,
                total_rows=0,
                imported=0,
                duplicates=0,
                errors=0,
                success=False,
                error_message="File does not appear to be OFX format",
            )

        try:
            parser = OFXParser(default_currency=self.default_currency)
            parsed = parser.parse(path)
        except ImportError as e:
            return ImportResult(
                file_path=path,
                total_rows=0,
                imported=0,
                duplicates=0,
                errors=0,
                success=False,
                error_message=str(e),
            )
        except Exception as e:
            return ImportResult(
                file_path=path,
                total_rows=0,
                imported=0,
                duplicates=0,
                errors=0,
                success=False,
                error_message=f"Parse error: {e}",
            )

        return self._import_transactions(path, parsed, account_name)

    def _import_transactions(
        self,
        file_path: Path,
        parsed: list[ParsedTransaction],
        account_name: str | None,
    ) -> ImportResult:
        """Import parsed transactions into database.

        Args:
            file_path: Source file path.
            parsed: List of parsed transactions.
            account_name: Optional account name.

        Returns:
            Import result with statistics.
        """
        if not parsed:
            return ImportResult(
                file_path=file_path,
                total_rows=0,
                imported=0,
                duplicates=0,
                errors=0,
                success=True,
                error_message="No transactions found in file",
            )

        imported = 0
        duplicates = 0
        errors = 0

        for txn in parsed:
            try:
                # Check for duplicate using hash
                if self._is_duplicate(txn.transaction_hash):
                    duplicates += 1
                    continue

                # Insert as v2 fact
                self._insert_as_fact(txn, account_name)
                imported += 1

            except Exception:
                errors += 1

        return ImportResult(
            file_path=file_path,
            total_rows=len(parsed),
            imported=imported,
            duplicates=duplicates,
            errors=errors,
            success=True,
        )

    def _is_duplicate(self, transaction_hash: str) -> bool:
        """Check if fact already exists.

        Args:
            transaction_hash: Hash to check.

        Returns:
            True if duplicate exists.
        """
        result = self.db.fetchone(
            "SELECT 1 FROM facts WHERE id LIKE ? LIMIT 1",
            (f"%-{transaction_hash}%",),
        )
        return result is not None

    def _insert_as_fact(
        self,
        parsed: ParsedTransaction,
        account_name: str | None,
    ) -> None:
        """Insert a parsed transaction as a v2 fact.

        Args:
            parsed: Parsed transaction.
            account_name: Optional account name.
        """
        # Generate ID including hash for dedup
        fact_id = f"{uuid.uuid4().hex[:16]}-{parsed.transaction_hash}"
        cloud_id = str(uuid.uuid4())

        # Determine fact_type from amount sign
        fact_type = "refund" if parsed.amount >= 0 else "purchase"

        vendor = canonicalize_vendor(parsed.vendor)
        event_date = parsed.date.isoformat() if parsed.date else None

        with self.db.transaction() as cursor:
            cursor.execute(
                "INSERT INTO clouds (id, status) VALUES (?, 'collapsed')",
                (cloud_id,),
            )
            cursor.execute(
                """
                INSERT INTO facts (id, cloud_id, fact_type, vendor,
                                   total_amount, currency, event_date, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'needs_review')
                """,
                (
                    fact_id,
                    cloud_id,
                    fact_type,
                    vendor,
                    str(abs(parsed.amount)),
                    parsed.currency,
                    event_date,
                ),
            )
