"""Payment record refiner."""

from __future__ import annotations

from typing import Any

from alibi.db.models import RecordType
from alibi.normalizers.vendors import normalize_vendor
from alibi.refiners.base import BaseRefiner, _normalize_amount, _normalize_currency


class PaymentRefiner(BaseRefiner):
    """Refine payment records from raw extraction.

    Payment records represent monetary transactions:
    - Vendor/merchant name
    - Amount and currency
    - Payment method and card details
    - Transaction date
    """

    def _refine_specific(
        self, raw: dict[str, Any], artifact_id: str | None
    ) -> dict[str, Any]:
        """Apply payment-specific refinement logic.

        Args:
            raw: Partially normalized data dict
            artifact_id: Source artifact ID for provenance

        Returns:
            Enriched dict with payment fields
        """
        enriched = raw.copy()

        # Set record type
        enriched["record_type"] = RecordType.PAYMENT

        # Normalize vendor name
        if "vendor" in enriched:
            enriched["vendor"] = self._normalize_vendor(enriched.get("vendor"))

        # Ensure amount is normalized
        if "amount" in enriched and enriched["amount"] is None:
            enriched["amount"] = _normalize_amount(enriched.get("amount"))

        # Normalize currency if present
        if "currency" in enriched:
            enriched["currency"] = _normalize_currency(enriched.get("currency"))
        else:
            enriched["currency"] = "EUR"

        # Extract payment method details
        if "payment_method" in enriched:
            enriched["payment_method"] = self._normalize_payment_method(
                enriched.get("payment_method")
            )

        # Extract card last 4 digits if present
        if "card_last4" in enriched:
            enriched["card_last4"] = self._extract_card_last4(
                enriched.get("card_last4")
            )

        # Map transaction_date to date field if needed
        if "transaction_date" in enriched and "date" not in enriched:
            enriched["date"] = enriched["transaction_date"]

        return enriched

    def _normalize_vendor(self, vendor: Any) -> str | None:
        """Normalize vendor name via alibi.normalizers.vendors."""
        if not vendor:
            return None
        result = normalize_vendor(str(vendor))
        return result if result else None

    def _normalize_payment_method(self, payment_method: Any) -> str | None:
        """Normalize payment method name.

        Args:
            payment_method: Raw payment method value

        Returns:
            Normalized payment method
        """
        if not payment_method:
            return None

        method_str = str(payment_method).strip().lower()

        # Map common variants to standard names
        method_map = {
            "card": "card",
            "credit card": "card",
            "debit card": "card",
            "cash": "cash",
            "bank transfer": "bank_transfer",
            "transfer": "bank_transfer",
            "paypal": "paypal",
            "apple pay": "apple_pay",
            "google pay": "google_pay",
        }

        return method_map.get(method_str, method_str)

    def _extract_card_last4(self, card_info: Any) -> str | None:
        """Extract last 4 digits from card info.

        Args:
            card_info: Raw card info (may include full number or last 4)

        Returns:
            Last 4 digits as string, or None
        """
        if not card_info:
            return None

        card_str = str(card_info).strip().replace(" ", "")

        # If it's exactly 4 digits, return as-is
        if len(card_str) == 4 and card_str.isdigit():
            return card_str

        # If it's longer, extract last 4
        if len(card_str) > 4:
            last4 = card_str[-4:]
            if last4.isdigit():
                return last4

        return None
