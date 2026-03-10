"""OFX/QIF parser for bank transaction imports.

Uses ofxparse library to parse OFX (Open Financial Exchange) files.
"""

from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import IO, Any

from alibi.ingestion.csv_parser import ParsedTransaction

try:
    import ofxparse  # type: ignore[import-untyped]
except ImportError:
    ofxparse = None  # type: ignore[assignment]


class OFXParser:
    """Parser for OFX bank export files.

    OFX (Open Financial Exchange) is a standard format used by many banks
    for exporting transaction data.
    """

    def __init__(self, default_currency: str = "EUR"):
        """Initialize OFX parser.

        Args:
            default_currency: Default currency if not specified in file.
        """
        self.default_currency = default_currency

        if ofxparse is None:
            raise ImportError(
                "ofxparse is required for OFX parsing. " "Install with: uv add ofxparse"
            )

    def parse(self, file_path: Path | str) -> list[ParsedTransaction]:
        """Parse OFX file and return transactions.

        Args:
            file_path: Path to OFX file.

        Returns:
            List of parsed transactions.
        """
        path = Path(file_path)
        with path.open("rb") as f:
            return self.parse_file(f)

    def parse_file(self, file_obj: IO[bytes]) -> list[ParsedTransaction]:
        """Parse OFX from file object.

        Args:
            file_obj: Binary file-like object with OFX content.

        Returns:
            List of parsed transactions.
        """
        try:
            ofx = ofxparse.OfxParser.parse(file_obj)
        except Exception as e:
            raise ValueError(f"Failed to parse OFX file: {e}") from e

        transactions = []

        for account in ofx.accounts:
            account_id = getattr(account, "account_id", None)
            currency = getattr(account.statement, "currency", self.default_currency)

            for txn in account.statement.transactions:
                try:
                    parsed = self._parse_transaction(txn, account_id, currency)
                    transactions.append(parsed)
                except (ValueError, AttributeError):
                    continue

        return transactions

    def _parse_transaction(
        self,
        txn: Any,
        account_id: str | None,
        currency: str,
    ) -> ParsedTransaction:
        """Parse a single OFX transaction.

        Args:
            txn: OFX transaction object.
            account_id: Account identifier.
            currency: Currency code.

        Returns:
            Parsed transaction.
        """
        # Extract date
        txn_date: date
        if hasattr(txn, "date") and txn.date:
            if hasattr(txn.date, "date"):
                txn_date = txn.date.date()
            else:
                txn_date = txn.date
        else:
            raise ValueError("Transaction has no date")

        # Extract amount
        amount = Decimal(str(getattr(txn, "amount", 0)))

        # Extract description/memo
        payee = getattr(txn, "payee", None) or ""
        memo = getattr(txn, "memo", None) or ""

        # Combine payee and memo for description
        if payee and memo:
            description = f"{payee} - {memo}"
        else:
            description = payee or memo or "Unknown"

        # Use payee as vendor if available
        vendor = payee.strip() if payee else None

        # Get transaction ID for reference
        txn_id = getattr(txn, "id", None)

        # Compute hash
        hash_input = f"{txn_date.isoformat()}|{amount}|{vendor or description}"
        txn_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        return ParsedTransaction(
            date=txn_date,
            description=description.strip(),
            amount=amount,
            currency=currency,
            vendor=vendor,
            payment_reference=txn_id,
            account_reference=account_id,
            transaction_hash=txn_hash,
        )


def is_ofx_file(file_path: Path | str) -> bool:
    """Check if file appears to be an OFX file.

    Args:
        file_path: Path to check.

    Returns:
        True if file looks like OFX.
    """
    path = Path(file_path)

    # Check extension
    if path.suffix.lower() in (".ofx", ".qfx"):
        return True

    # Check content signature
    try:
        with path.open("rb") as f:
            header = f.read(100).decode("utf-8", errors="ignore").upper()
            return "OFXHEADER" in header or "<OFX>" in header
    except (OSError, UnicodeDecodeError):
        return False
