"""Invoice record refiner."""

from __future__ import annotations

from typing import Any

from alibi.db.models import RecordType
from alibi.normalizers.vendors import normalize_vendor
from alibi.refiners.base import BaseRefiner, _normalize_amount, _normalize_currency
from alibi.refiners.purchase import PurchaseRefiner


class InvoiceRefiner(BaseRefiner):
    """Refine invoice records from raw extraction.

    Invoice records are similar to payments but with additional fields:
    - Invoice number
    - Due date
    - Issuer/vendor information
    """

    def _refine_specific(
        self, raw: dict[str, Any], artifact_id: str | None
    ) -> dict[str, Any]:
        """Apply invoice-specific refinement logic.

        Args:
            raw: Partially normalized data dict
            artifact_id: Source artifact ID for provenance

        Returns:
            Enriched dict with invoice fields
        """
        enriched = raw.copy()

        # Set record type
        enriched["record_type"] = RecordType.INVOICE

        # Normalize issuer/vendor
        if "issuer" in enriched:
            enriched["issuer"] = self._normalize_issuer(enriched.get("issuer"))
        elif "vendor" in enriched:
            enriched["issuer"] = self._normalize_issuer(enriched.get("vendor"))

        # Map issuer_* fields to vendor_* for unified storage
        _issuer_map = {
            "issuer_address": "vendor_address",
            "issuer_phone": "vendor_phone",
            "issuer_website": "vendor_website",
            "issuer_vat": "vendor_vat",
            "issuer_tax_id": "vendor_tax_id",
            "issuer_id": "vendor_vat",
            # Backward compat: old field name from LLM output
            "issuer_registration": "vendor_vat",
        }
        for src, dst in _issuer_map.items():
            if src in enriched and dst not in enriched:
                enriched[dst] = enriched[src]

        # Also set vendor from issuer for unified pipeline
        if "issuer" in enriched and "vendor" not in enriched:
            enriched["vendor"] = enriched["issuer"]

        # Extract invoice number
        if "invoice_number" in enriched:
            enriched["invoice_number"] = self._normalize_invoice_number(
                enriched.get("invoice_number")
            )
        elif "invoice_id" in enriched:
            enriched["invoice_number"] = self._normalize_invoice_number(
                enriched.get("invoice_id")
            )

        # Ensure amount is normalized
        if "amount" in enriched and enriched["amount"] is None:
            enriched["amount"] = _normalize_amount(enriched.get("amount"))

        # Normalize currency
        if "currency" in enriched:
            enriched["currency"] = _normalize_currency(enriched.get("currency"))
        else:
            enriched["currency"] = "EUR"

        # Extract due date (already normalized by base class)
        # Just ensure it's present
        if "due_date" not in enriched and "payment_due" in enriched:
            enriched["due_date"] = enriched.get("payment_due")

        # Extract issue date (use normalized value from base class)
        if "issue_date" in enriched and "date" not in enriched:
            # issue_date was already normalized by base class
            enriched["date"] = enriched["issue_date"]
        elif "invoice_date" in enriched and "date" not in enriched:
            # invoice_date was already normalized by base class
            enriched["date"] = enriched["invoice_date"]

        # Parse and normalize line items (reuse PurchaseRefiner logic)
        raw_items = enriched.get("line_items", []) or enriched.get("items", [])
        if raw_items:
            purchase_refiner = PurchaseRefiner()
            enriched["line_items"] = purchase_refiner._parse_line_items(
                raw_items,
                artifact_id,
                enriched.get("currency", "EUR"),
                enriched.get("language"),
            )

        return enriched

    def _normalize_issuer(self, issuer: Any) -> str | None:
        """Normalize issuer/vendor name via alibi.normalizers.vendors."""
        if not issuer:
            return None
        result = normalize_vendor(str(issuer))
        return result if result else None

    def _normalize_invoice_number(self, invoice_number: Any) -> str | None:
        """Normalize invoice number.

        Args:
            invoice_number: Raw invoice number

        Returns:
            Normalized invoice number (uppercase, trimmed)
        """
        if not invoice_number:
            return None

        # Convert to string, clean, and uppercase
        number_str = str(invoice_number).strip().upper()

        # Remove common prefixes
        prefixes = ["INV-", "INVOICE-", "INV#", "#"]
        for prefix in prefixes:
            if number_str.startswith(prefix):
                number_str = number_str[len(prefix) :].strip()

        return number_str if number_str else None
