"""CSV parsers for bank transaction imports.

Supports multiple bank formats: N26, Revolut, and generic CSV.
"""

from __future__ import annotations

import csv
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from collections.abc import Sequence
from typing import IO


class CSVFormat(str, Enum):
    """Supported CSV formats."""

    N26 = "n26"
    REVOLUT = "revolut"
    GENERIC = "generic"
    UNKNOWN = "unknown"


@dataclass
class ParsedTransaction:
    """A transaction parsed from CSV, ready for import.

    This is an intermediate representation before conversion to Transaction model.
    """

    date: date
    description: str
    amount: Decimal
    currency: str = "EUR"
    vendor: str | None = None
    payment_reference: str | None = None
    account_reference: str | None = None
    original_currency: str | None = None
    original_amount: Decimal | None = None
    exchange_rate: Decimal | None = None
    balance: Decimal | None = None
    transaction_hash: str = field(default="")

    def __post_init__(self) -> None:
        """Compute transaction hash for deduplication."""
        if not self.transaction_hash:
            self.transaction_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute unique hash based on date, amount, and vendor/description."""
        hash_input = (
            f"{self.date.isoformat()}|{self.amount}|{self.vendor or self.description}"
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


class BaseCSVParser(ABC):
    """Base class for CSV parsers."""

    @abstractmethod
    def parse(self, file_path: Path | str) -> list[ParsedTransaction]:
        """Parse CSV file and return transactions.

        Args:
            file_path: Path to CSV file.

        Returns:
            List of parsed transactions.
        """
        ...

    @abstractmethod
    def parse_file(self, file_obj: IO[str]) -> list[ParsedTransaction]:
        """Parse CSV from file object.

        Args:
            file_obj: File-like object with CSV content.

        Returns:
            List of parsed transactions.
        """
        ...

    @staticmethod
    def parse_amount(value: str) -> Decimal:
        """Parse amount string to Decimal.

        Handles various formats: "1,234.56", "-100.00", "100,50" (EU format).

        Args:
            value: Amount string.

        Returns:
            Decimal amount.

        Raises:
            ValueError: If amount cannot be parsed.
        """
        if not value or not value.strip():
            raise ValueError("Empty amount")

        clean = value.strip()

        # Remove currency symbols
        for symbol in ["EUR", "USD", "GBP", "CHF", "$", "E", "Fr."]:
            clean = clean.replace(symbol, "")

        clean = clean.strip()

        # Handle EU format (comma as decimal separator)
        if "," in clean and "." in clean:
            # 1,234.56 format - US style
            clean = clean.replace(",", "")
        elif "," in clean:
            # Might be EU format (100,50) or thousands (1,234)
            parts = clean.split(",")
            if len(parts) == 2 and len(parts[1]) == 2:
                # EU decimal format: 100,50
                clean = clean.replace(",", ".")
            else:
                # Thousands separator: 1,234
                clean = clean.replace(",", "")

        try:
            return Decimal(clean)
        except InvalidOperation as e:
            raise ValueError(f"Cannot parse amount: {value}") from e

    @staticmethod
    def parse_date(value: str, formats: list[str] | None = None) -> date:
        """Parse date string to date object.

        Args:
            value: Date string.
            formats: List of date formats to try.

        Returns:
            Parsed date.

        Raises:
            ValueError: If date cannot be parsed.
        """
        if formats is None:
            formats = [
                "%Y-%m-%d %H:%M:%S",  # Datetime with seconds
                "%Y-%m-%d %H:%M",  # Datetime without seconds
                "%Y-%m-%d",
                "%d.%m.%Y",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%Y/%m/%d",
                "%d-%m-%Y",
            ]

        value = value.strip()

        for fmt in formats:
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue

        raise ValueError(f"Cannot parse date: {value}")


class N26CSVParser(BaseCSVParser):
    """Parser for N26 bank CSV exports.

    N26 format columns:
    - Date
    - Payee
    - Account number
    - Transaction type
    - Payment reference
    - Amount (EUR)
    - Amount (Foreign Currency)
    - Type Foreign Currency
    - Exchange Rate
    """

    EXPECTED_HEADERS = [
        "Date",
        "Payee",
        "Account number",
        "Transaction type",
        "Payment reference",
        "Amount (EUR)",
    ]

    def parse(self, file_path: Path | str) -> list[ParsedTransaction]:
        """Parse N26 CSV file."""
        path = Path(file_path)
        with path.open(encoding="utf-8") as f:
            return self.parse_file(f)

    def parse_file(self, file_obj: IO[str]) -> list[ParsedTransaction]:
        """Parse N26 CSV from file object."""
        reader = csv.DictReader(file_obj)
        transactions = []

        for row in reader:
            try:
                txn = self._parse_row(row)
                transactions.append(txn)
            except (ValueError, KeyError) as e:
                # Log and skip invalid rows
                continue

        return transactions

    def _parse_row(self, row: dict[str, str]) -> ParsedTransaction:
        """Parse a single N26 CSV row."""
        txn_date = self.parse_date(row["Date"])
        amount = self.parse_amount(row["Amount (EUR)"])
        payee = row.get("Payee", "").strip() or None
        payment_ref = row.get("Payment reference", "").strip() or None
        account_num = row.get("Account number", "").strip() or None

        # Handle foreign currency
        original_currency = row.get("Type Foreign Currency", "").strip() or None
        original_amount = None
        exchange_rate = None

        if original_currency:
            foreign_amt = row.get("Amount (Foreign Currency)", "").strip()
            if foreign_amt:
                try:
                    original_amount = self.parse_amount(foreign_amt)
                except ValueError:
                    pass

            rate = row.get("Exchange Rate", "").strip()
            if rate:
                try:
                    exchange_rate = Decimal(rate)
                except InvalidOperation:
                    pass

        # Use payee as vendor, payment reference as description
        description = payment_ref or row.get("Transaction type", "") or ""

        return ParsedTransaction(
            date=txn_date,
            description=description,
            amount=amount,
            currency="EUR",
            vendor=payee,
            payment_reference=payment_ref,
            account_reference=account_num,
            original_currency=original_currency,
            original_amount=original_amount,
            exchange_rate=exchange_rate,
        )


class RevolutCSVParser(BaseCSVParser):
    """Parser for Revolut CSV exports.

    Revolut format columns:
    - Type
    - Product
    - Started Date
    - Completed Date
    - Description
    - Amount
    - Fee
    - Currency
    - State
    - Balance
    """

    EXPECTED_HEADERS = [
        "Type",
        "Product",
        "Started Date",
        "Completed Date",
        "Description",
        "Amount",
        "Currency",
    ]

    def parse(self, file_path: Path | str) -> list[ParsedTransaction]:
        """Parse Revolut CSV file."""
        path = Path(file_path)
        with path.open(encoding="utf-8") as f:
            return self.parse_file(f)

    def parse_file(self, file_obj: IO[str]) -> list[ParsedTransaction]:
        """Parse Revolut CSV from file object."""
        reader = csv.DictReader(file_obj)
        transactions = []

        for row in reader:
            # Skip incomplete or pending transactions
            state = row.get("State", "").strip().upper()
            if state not in ("COMPLETED", ""):
                continue

            try:
                txn = self._parse_row(row)
                transactions.append(txn)
            except (ValueError, KeyError):
                continue

        return transactions

    def _parse_row(self, row: dict[str, str]) -> ParsedTransaction:
        """Parse a single Revolut CSV row."""
        # Prefer Completed Date, fall back to Started Date
        date_str = row.get("Completed Date") or row.get("Started Date", "")
        txn_date = self.parse_date(date_str.strip())

        amount = self.parse_amount(row["Amount"])
        currency = row.get("Currency", "EUR").strip() or "EUR"
        description = row.get("Description", "").strip()
        product = row.get("Product", "").strip()

        # Parse balance if available
        balance = None
        balance_str = row.get("Balance", "").strip()
        if balance_str:
            try:
                balance = self.parse_amount(balance_str)
            except ValueError:
                pass

        # Extract vendor from description (often format: "To Vendor Name")
        vendor = None
        desc_lower = description.lower()
        if desc_lower.startswith("to "):
            vendor = description[3:].strip()
        elif desc_lower.startswith("from "):
            vendor = description[5:].strip()
        else:
            vendor = description

        return ParsedTransaction(
            date=txn_date,
            description=description,
            amount=amount,
            currency=currency,
            vendor=vendor,
            account_reference=product if product else None,
            balance=balance,
        )


class GenericCSVParser(BaseCSVParser):
    """Parser for generic CSV files with flexible column mapping.

    Attempts to detect common column names:
    - Date: date, transaction_date, txn_date, posted_date
    - Description: description, desc, memo, narrative, details
    - Amount: amount, value, sum, total
    - Currency: currency, ccy
    """

    DATE_COLUMNS = ["date", "transaction_date", "txn_date", "posted_date", "value_date"]
    DESC_COLUMNS = ["description", "desc", "memo", "narrative", "details", "payee"]
    AMOUNT_COLUMNS = ["amount", "value", "sum", "total", "credit", "debit"]
    CURRENCY_COLUMNS = ["currency", "ccy", "curr"]

    def __init__(
        self,
        date_column: str | None = None,
        description_column: str | None = None,
        amount_column: str | None = None,
        currency_column: str | None = None,
        default_currency: str = "EUR",
    ):
        """Initialize with optional explicit column mappings.

        Args:
            date_column: Column name for date.
            description_column: Column name for description.
            amount_column: Column name for amount.
            currency_column: Column name for currency.
            default_currency: Default currency if not specified.
        """
        self.date_column = date_column
        self.description_column = description_column
        self.amount_column = amount_column
        self.currency_column = currency_column
        self.default_currency = default_currency

    def parse(self, file_path: Path | str) -> list[ParsedTransaction]:
        """Parse generic CSV file."""
        path = Path(file_path)
        with path.open(encoding="utf-8") as f:
            return self.parse_file(f)

    def parse_file(self, file_obj: IO[str]) -> list[ParsedTransaction]:
        """Parse generic CSV from file object."""
        reader = csv.DictReader(file_obj)
        fieldnames = reader.fieldnames or []

        # Detect column mappings
        date_col = self._find_column(fieldnames, self.date_column, self.DATE_COLUMNS)
        desc_col = self._find_column(
            fieldnames, self.description_column, self.DESC_COLUMNS
        )
        amount_col = self._find_column(
            fieldnames, self.amount_column, self.AMOUNT_COLUMNS
        )
        currency_col = self._find_column(
            fieldnames, self.currency_column, self.CURRENCY_COLUMNS
        )

        if not date_col or not amount_col:
            raise ValueError(
                f"Cannot detect required columns. Found: {fieldnames}. "
                f"Need date ({date_col}) and amount ({amount_col})."
            )

        transactions = []

        for row in reader:
            try:
                txn_date = self.parse_date(row[date_col])
                amount = self.parse_amount(row[amount_col])

                description = ""
                if desc_col and row.get(desc_col):
                    description = row[desc_col].strip()

                currency = self.default_currency
                if currency_col and row.get(currency_col):
                    currency = row[currency_col].strip()

                txn = ParsedTransaction(
                    date=txn_date,
                    description=description,
                    amount=amount,
                    currency=currency,
                    vendor=description if description else None,
                )
                transactions.append(txn)
            except (ValueError, KeyError):
                continue

        return transactions

    def _find_column(
        self,
        fieldnames: Sequence[str] | list[str],
        explicit: str | None,
        candidates: list[str],
    ) -> str | None:
        """Find matching column name.

        Args:
            fieldnames: Available column names.
            explicit: Explicitly specified column (takes priority).
            candidates: List of candidate column names to try.

        Returns:
            Matched column name or None.
        """
        if explicit and explicit in fieldnames:
            return explicit

        # Normalize fieldnames for comparison
        normalized = {fn.lower().strip(): fn for fn in fieldnames}

        for candidate in candidates:
            if candidate.lower() in normalized:
                return normalized[candidate.lower()]

        return None


def detect_csv_format(file_path: Path | str) -> CSVFormat:
    """Detect CSV format based on headers.

    Args:
        file_path: Path to CSV file.

    Returns:
        Detected CSV format.
    """
    path = Path(file_path)

    try:
        with path.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
    except (OSError, csv.Error):
        return CSVFormat.UNKNOWN

    headers_lower = [h.lower().strip() for h in headers]

    # Check for N26 format
    n26_markers = ["payee", "payment reference", "amount (eur)"]
    if all(marker in " ".join(headers_lower) for marker in n26_markers):
        return CSVFormat.N26

    # Check for Revolut format
    revolut_markers = ["started date", "completed date", "state"]
    if all(marker in " ".join(headers_lower) for marker in revolut_markers):
        return CSVFormat.REVOLUT

    # Check if it has basic required columns for generic
    has_date = any(col in headers_lower for col in GenericCSVParser.DATE_COLUMNS)
    has_amount = any(col in headers_lower for col in GenericCSVParser.AMOUNT_COLUMNS)

    if has_date and has_amount:
        return CSVFormat.GENERIC

    return CSVFormat.UNKNOWN
