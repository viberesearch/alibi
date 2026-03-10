"""Base refiner interface and shared helpers.

BaseRefiner defines the refine(raw, artifact_id) -> dict contract.
Helper functions delegate to alibi.normalizers for field normalization.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from alibi.normalizers.currency import (
    normalize_currency as _norm_currency,
    parse_amount_with_currency,
)
from alibi.normalizers.dates import parse_date


def _normalize_amount(value: Any) -> Decimal | None:
    """Normalize monetary amount to Decimal.

    Delegates string parsing to alibi.normalizers.currency.

    Args:
        value: Raw amount value (str, int, float, Decimal)

    Returns:
        Normalized Decimal or None if invalid
    """
    if value is None:
        return None

    if isinstance(value, Decimal):
        return value

    if isinstance(value, (int, float)):
        return Decimal(str(value))

    if isinstance(value, str):
        amount, _ = parse_amount_with_currency(value)
        return amount

    return None


def _normalize_date(value: Any) -> date | None:
    """Normalize date value to date object.

    Delegates to alibi.normalizers.dates.parse_date.

    Args:
        value: Raw date value (str, date, datetime)

    Returns:
        Normalized date or None if invalid
    """
    if isinstance(value, datetime):
        return value.date()

    return parse_date(value)


def _extract_currency_from_amount(amount_str: str) -> str | None:
    """Extract currency symbol/code from amount string.

    Args:
        amount_str: Amount string that may contain currency symbol

    Returns:
        Currency code if found, None otherwise
    """
    _, currency = parse_amount_with_currency(amount_str)
    return currency if currency != "EUR" else None


def _normalize_currency(value: Any) -> str:
    """Normalize currency code to ISO 4217.

    Delegates to alibi.normalizers.currency.normalize_currency.

    Args:
        value: Raw currency value (symbol or code)

    Returns:
        Normalized currency code (default: EUR)
    """
    return _norm_currency(str(value) if value else "")


def _parse_quantity_unit(quantity_str: str) -> tuple[Decimal, str | None]:
    """Parse quantity string into amount and unit.

    Examples:
        "2.5 kg" -> (Decimal("2.5"), "kg")
        "500ml" -> (Decimal("500"), "ml")
        "3" -> (Decimal("3"), None)

    Args:
        quantity_str: Raw quantity string

    Returns:
        Tuple of (amount, unit_raw)
    """
    if not quantity_str:
        return Decimal("1"), None

    # Match number followed by optional unit
    match = re.match(r"^([\d.,]+)\s*([a-zA-Z]+)?$", str(quantity_str).strip())
    if match:
        amount_str, unit = match.groups()
        amount_clean = amount_str.replace(",", ".")
        try:
            amount = Decimal(amount_clean)
            return amount, unit
        except Exception:
            pass

    return Decimal("1"), None


class BaseRefiner(ABC):
    """Abstract base class for per-type record refiners.

    Each refiner transforms raw extracted data into a structured,
    normalized record dict ready for model creation.

    The refine() method:
    1. Normalizes common fields (amounts, dates, currency)
    2. Calls _refine_specific() for type-specific logic
    3. Builds provenance record
    4. Returns enriched dict (not model instance)

    Subclasses override _refine_specific() for type-specific logic.
    """

    @abstractmethod
    def _refine_specific(
        self, raw: dict[str, Any], artifact_id: str | None
    ) -> dict[str, Any]:
        """Apply type-specific refinement logic.

        Called after base normalization. Should return enriched dict
        with type-specific fields populated.

        Args:
            raw: Partially normalized data dict
            artifact_id: Source artifact ID for provenance

        Returns:
            Enriched dict with type-specific fields
        """
        ...

    def refine(
        self, raw: dict[str, Any], artifact_id: str | None = None
    ) -> dict[str, Any]:
        """Refine raw extraction output into structured record.

        Args:
            raw: Raw extracted data dict from LLM/OCR
            artifact_id: Source artifact ID for provenance

        Returns:
            Enriched dict with normalized fields, ready for model creation.
        """
        # Start with a copy to avoid mutating input
        data: dict[str, Any] = raw.copy()

        # Extract currency from amount string if not explicitly provided
        if "amount" in data and "currency" not in data:
            amount_raw = data.get("amount")
            if isinstance(amount_raw, str):
                extracted_currency = _extract_currency_from_amount(amount_raw)
                if extracted_currency:
                    data["currency"] = extracted_currency

        # Base normalization: amounts, dates, currency
        if "amount" in data:
            data["amount"] = self._normalize_amounts(data)

        # Always normalize dates (the function checks for each field)
        data = self._normalize_dates(data)

        if "currency" in data:
            data["currency"] = _normalize_currency(data.get("currency"))

        # Type-specific refinement
        enriched = self._refine_specific(data, artifact_id)

        # Build provenance
        enriched["provenance"] = self._build_provenance(
            artifact_id=artifact_id,
            source_type="ai_refinement",
            confidence=raw.get("confidence"),
        )

        # Generate ID if not present
        if "id" not in enriched:
            enriched["id"] = str(uuid4())

        return enriched

    def _normalize_amounts(self, data: dict[str, Any]) -> Decimal | None:
        """Normalize monetary amounts in the record.

        Handles common amount fields: amount, total, subtotal, etc.

        Args:
            data: Record data dict

        Returns:
            Normalized amount as Decimal
        """
        # Try common field names
        for field in ["amount", "total", "total_price", "total_amount"]:
            if field in data:
                normalized = _normalize_amount(data[field])
                if normalized is not None:
                    return normalized

        return None

    def _normalize_dates(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize date fields in the record.

        Handles common date fields: date, transaction_date, document_date, etc.

        Args:
            data: Record data dict

        Returns:
            Data dict with normalized date fields
        """
        result = data.copy()

        date_fields = [
            "date",
            "transaction_date",
            "document_date",
            "due_date",
            "issue_date",
            "invoice_date",
            "warranty_expires",
            "renewal_date",
            "period_start",
            "period_end",
        ]

        for field in date_fields:
            if field in result:
                normalized = _normalize_date(result[field])
                if normalized:
                    result[field] = normalized

        return result

    def _build_provenance(
        self,
        artifact_id: str | None,
        source_type: str,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Build a provenance record dict.

        Args:
            artifact_id: Source artifact ID
            source_type: Type of source (ocr, ai_refinement, manual, etc.)
            confidence: Optional confidence score

        Returns:
            Provenance dict ready for ProvenanceRecord model
        """
        return {
            "id": str(uuid4()),
            "source_type": source_type,
            "source_id": artifact_id,
            "confidence": Decimal(str(confidence)) if confidence is not None else None,
            "processor": "alibi:refiner:v2",
            "created_at": datetime.now(),
        }
