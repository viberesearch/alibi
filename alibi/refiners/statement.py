"""Statement record refiner."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from alibi.db.models import RecordType
from alibi.normalizers.vendors import normalize_vendor
from alibi.refiners.base import BaseRefiner, _normalize_amount, _normalize_currency


class StatementRefiner(BaseRefiner):
    """Refine bank statement records from raw extraction.

    Statement records can contain multiple transactions:
    - Account information
    - Statement period
    - Opening/closing balance
    - Individual transaction entries
    """

    def _refine_specific(
        self, raw: dict[str, Any], artifact_id: str | None
    ) -> dict[str, Any]:
        """Apply statement-specific refinement logic.

        Args:
            raw: Partially normalized data dict
            artifact_id: Source artifact ID for provenance

        Returns:
            Enriched dict with statement fields
        """
        enriched = raw.copy()

        # Set record type
        enriched["record_type"] = RecordType.STATEMENT

        # Extract account information
        if "account_number" in enriched:
            enriched["account_number"] = self._normalize_account_number(
                enriched.get("account_number")
            )
        elif "account" in enriched:
            enriched["account_number"] = self._normalize_account_number(
                enriched.get("account")
            )

        # Extract account holder/issuer
        if "issuer" in enriched:
            enriched["issuer"] = self._normalize_issuer(enriched.get("issuer"))
        elif "bank" in enriched:
            enriched["issuer"] = self._normalize_issuer(enriched.get("bank"))
        elif "account_holder" in enriched:
            enriched["issuer"] = self._normalize_issuer(enriched.get("account_holder"))

        # Extract balances
        if "opening_balance" in enriched:
            enriched["opening_balance"] = _normalize_amount(
                enriched.get("opening_balance")
            )

        if "closing_balance" in enriched:
            enriched["closing_balance"] = _normalize_amount(
                enriched.get("closing_balance")
            )
        elif "balance" in enriched:
            enriched["closing_balance"] = _normalize_amount(enriched.get("balance"))

        # Normalize currency
        if "currency" in enriched:
            enriched["currency"] = _normalize_currency(enriched.get("currency"))
        else:
            enriched["currency"] = "EUR"

        # Parse statement period
        if "period_start" in enriched or "period_end" in enriched:
            # Already normalized by base class
            pass
        elif "statement_period" in enriched:
            # Parse period string like "2024-01-01 to 2024-01-31"
            period_str = str(enriched.get("statement_period"))
            parts = period_str.split(" to ")
            if len(parts) == 2:
                enriched["period_start"] = parts[0].strip()
                enriched["period_end"] = parts[1].strip()

        # Parse individual transactions from statement
        raw_transactions = enriched.get("transactions", [])
        if raw_transactions:
            enriched["transactions"] = self._parse_transactions(
                raw_transactions, artifact_id, enriched.get("currency", "EUR")
            )

        return enriched

    def _parse_transactions(
        self,
        raw_transactions: list[dict[str, Any]],
        artifact_id: str | None,
        currency: str,
    ) -> list[dict[str, Any]]:
        """Parse individual transactions from statement.

        Args:
            raw_transactions: List of raw transaction dicts
            artifact_id: Source artifact ID
            currency: Currency code for the statement

        Returns:
            List of normalized transaction dicts
        """
        transactions = []

        for raw_tx in raw_transactions:
            tx = self._parse_single_transaction(raw_tx, artifact_id, currency)
            if tx:
                transactions.append(tx)

        return transactions

    def _parse_single_transaction(
        self, raw_tx: dict[str, Any], artifact_id: str | None, currency: str
    ) -> dict[str, Any] | None:
        """Parse a single transaction from statement.

        Args:
            raw_tx: Raw transaction dict
            artifact_id: Source artifact ID
            currency: Currency code

        Returns:
            Normalized transaction dict or None if invalid
        """
        # Extract date (required)
        date = raw_tx.get("date") or raw_tx.get("transaction_date")
        if not date:
            return None

        # Extract amount (required)
        amount = _normalize_amount(raw_tx.get("amount"))
        if amount is None:
            return None

        tx: dict[str, Any] = {
            "id": str(uuid4()),
            "artifact_id": artifact_id,
            "transaction_date": date,
            "amount": amount,
            "currency": currency,
        }

        # Extract description
        if "description" in raw_tx:
            tx["description"] = str(raw_tx.get("description")).strip()

        # Extract vendor/merchant
        if "vendor" in raw_tx:
            tx["vendor"] = str(raw_tx.get("vendor")).strip()
        elif "merchant" in raw_tx:
            tx["vendor"] = str(raw_tx.get("merchant")).strip()

        # Extract transaction type (debit/credit)
        if "type" in raw_tx:
            tx_type = str(raw_tx.get("type")).lower()
            if "debit" in tx_type or "expense" in tx_type:
                tx["type"] = "expense"
            elif "credit" in tx_type or "income" in tx_type:
                tx["type"] = "income"

        # Extract reference number
        if "reference" in raw_tx:
            tx["reference"] = str(raw_tx.get("reference")).strip()

        return tx

    def _normalize_account_number(self, account_number: Any) -> str | None:
        """Normalize account number.

        Masks middle digits for privacy while keeping first 2 and last 4.

        Args:
            account_number: Raw account number

        Returns:
            Normalized account number (partially masked)
        """
        if not account_number:
            return None

        # Convert to string and remove spaces/dashes
        number_str = str(account_number).strip().replace(" ", "").replace("-", "")

        # Mask middle digits (keep first 2 and last 4)
        if len(number_str) > 6:
            masked = number_str[:2] + "*" * (len(number_str) - 6) + number_str[-4:]
            return masked

        return number_str

    def _normalize_issuer(self, issuer: Any) -> str | None:
        """Normalize issuer/bank name via alibi.normalizers.vendors."""
        if not issuer:
            return None
        result = normalize_vendor(str(issuer))
        return result if result else None
