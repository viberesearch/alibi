"""Insurance record refiner."""

from __future__ import annotations

from typing import Any

from alibi.db.models import RecordType
from alibi.normalizers.vendors import normalize_vendor
from alibi.refiners.base import BaseRefiner, _normalize_amount, _normalize_currency


class InsuranceRefiner(BaseRefiner):
    """Refine insurance records from raw extraction.

    Insurance records track insurance policies:
    - Policy number
    - Coverage details
    - Premium amount
    - Renewal/expiration dates
    """

    def _refine_specific(
        self, raw: dict[str, Any], artifact_id: str | None
    ) -> dict[str, Any]:
        """Apply insurance-specific refinement logic.

        Args:
            raw: Partially normalized data dict
            artifact_id: Source artifact ID for provenance

        Returns:
            Enriched dict with insurance fields
        """
        enriched = raw.copy()

        # Set record type
        enriched["record_type"] = RecordType.INSURANCE

        # Extract policy number
        if "policy_number" in enriched:
            enriched["policy_number"] = self._normalize_policy_number(
                enriched.get("policy_number")
            )
        elif "policy_id" in enriched:
            enriched["policy_number"] = self._normalize_policy_number(
                enriched.get("policy_id")
            )

        # Extract coverage information
        if "coverage" in enriched:
            enriched["coverage"] = str(enriched.get("coverage")).strip()
        elif "coverage_type" in enriched:
            enriched["coverage"] = str(enriched.get("coverage_type")).strip()

        # Extract premium amount
        if "premium" in enriched:
            enriched["premium"] = _normalize_amount(enriched.get("premium"))
        elif "premium_amount" in enriched:
            enriched["premium"] = _normalize_amount(enriched.get("premium_amount"))

        # Normalize currency
        if "currency" in enriched:
            enriched["currency"] = _normalize_currency(enriched.get("currency"))
        else:
            enriched["currency"] = "EUR"

        # Handle renewal date
        if "renewal_date" not in enriched and "expiration_date" in enriched:
            enriched["renewal_date"] = enriched.get("expiration_date")

        # Extract issuer/provider
        if "issuer" in enriched:
            enriched["issuer"] = self._normalize_issuer(enriched.get("issuer"))
        elif "provider" in enriched:
            enriched["issuer"] = self._normalize_issuer(enriched.get("provider"))
        elif "vendor" in enriched:
            enriched["issuer"] = self._normalize_issuer(enriched.get("vendor"))

        # Map issue_date to date field if not present
        if "issue_date" in enriched and "date" not in enriched:
            enriched["date"] = enriched.get("issue_date")
        elif "policy_date" in enriched and "date" not in enriched:
            enriched["date"] = enriched.get("policy_date")

        return enriched

    def _normalize_policy_number(self, policy_number: Any) -> str | None:
        """Normalize policy number.

        Args:
            policy_number: Raw policy number

        Returns:
            Normalized policy number (uppercase, trimmed)
        """
        if not policy_number:
            return None

        # Convert to string, clean, and uppercase
        number_str = str(policy_number).strip().upper()

        # Remove common prefixes
        prefixes = ["POL-", "POLICY-", "POL#", "#"]
        for prefix in prefixes:
            if number_str.startswith(prefix):
                number_str = number_str[len(prefix) :].strip()

        return number_str if number_str else None

    def _normalize_issuer(self, issuer: Any) -> str | None:
        """Normalize issuer/provider name via alibi.normalizers.vendors."""
        if not issuer:
            return None
        result = normalize_vendor(str(issuer))
        return result if result else None
