"""Contract record refiner."""

from __future__ import annotations

from typing import Any

from alibi.db.models import RecordType
from alibi.normalizers.vendors import normalize_vendor
from alibi.refiners.base import BaseRefiner


class ContractRefiner(BaseRefiner):
    """Refine contract records from raw extraction.

    Contract records track agreements and service contracts:
    - Contract dates (start, end)
    - Payment terms and renewal conditions
    - Parties involved
    - Contract value
    """

    def _refine_specific(
        self, raw: dict[str, Any], artifact_id: str | None
    ) -> dict[str, Any]:
        """Apply contract-specific refinement logic.

        Args:
            raw: Partially normalized data dict
            artifact_id: Source artifact ID for provenance

        Returns:
            Enriched dict with contract fields
        """
        enriched = raw.copy()

        # Set record type
        enriched["record_type"] = RecordType.CONTRACT

        # Normalize payment terms
        if "payment_terms" in enriched:
            enriched["payment_terms"] = self._normalize_payment_terms(
                enriched.get("payment_terms")
            )

        # Extract vendor/counterparty
        if "vendor" in enriched:
            enriched["vendor"] = self._normalize_vendor(enriched.get("vendor"))
        elif "issuer" in enriched:
            enriched["vendor"] = self._normalize_vendor(enriched.get("issuer"))

        # Normalize customer name
        if "customer" in enriched:
            customer = enriched.get("customer")
            if customer:
                enriched["customer"] = str(customer).strip()

        # Map start_date to the main date field if not present
        if "start_date" in enriched and "date" not in enriched:
            enriched["date"] = enriched.get("start_date")

        # Normalize renewal type
        if "renewal" in enriched:
            enriched["renewal"] = self._normalize_renewal(enriched.get("renewal"))

        return enriched

    def _normalize_payment_terms(self, terms: Any) -> str | None:
        """Normalize payment terms.

        Args:
            terms: Raw payment terms value

        Returns:
            Normalized payment terms
        """
        if not terms:
            return None

        terms_str = str(terms).strip().lower()

        # Map common variants
        terms_map = {
            "monthly": "monthly",
            "annual": "annual",
            "yearly": "annual",
            "quarterly": "quarterly",
            "one-time": "one-time",
            "one time": "one-time",
            "upfront": "one-time",
            "weekly": "weekly",
            "biweekly": "biweekly",
        }

        for key, value in terms_map.items():
            if key in terms_str:
                return value

        return str(terms).strip()

    def _normalize_renewal(self, renewal: Any) -> str | None:
        """Normalize renewal type.

        Args:
            renewal: Raw renewal value

        Returns:
            Normalized renewal type
        """
        if not renewal:
            return None

        renewal_str = str(renewal).strip().lower()

        renewal_map = {
            "auto": "auto",
            "automatic": "auto",
            "manual": "manual",
            "none": "none",
            "no": "none",
        }

        for key, value in renewal_map.items():
            if key in renewal_str:
                return value

        return str(renewal).strip()

    def _normalize_vendor(self, vendor: Any) -> str | None:
        """Normalize vendor name via alibi.normalizers.vendors."""
        if not vendor:
            return None
        result = normalize_vendor(str(vendor))
        return result if result else None
