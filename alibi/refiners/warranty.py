"""Warranty record refiner."""

from __future__ import annotations

from typing import Any

from alibi.db.models import RecordType
from alibi.normalizers.vendors import normalize_vendor
from alibi.refiners.base import BaseRefiner


class WarrantyRefiner(BaseRefiner):
    """Refine warranty records from raw extraction.

    Warranty records track product warranties:
    - Warranty expiration date
    - Warranty type (manufacturer, extended, etc.)
    - Product information
    - Link to item if identifiable
    """

    def _refine_specific(
        self, raw: dict[str, Any], artifact_id: str | None
    ) -> dict[str, Any]:
        """Apply warranty-specific refinement logic.

        Args:
            raw: Partially normalized data dict
            artifact_id: Source artifact ID for provenance

        Returns:
            Enriched dict with warranty fields
        """
        enriched = raw.copy()

        # Set record type
        enriched["record_type"] = RecordType.WARRANTY

        # Normalize warranty type
        if "warranty_type" in enriched:
            enriched["warranty_type"] = self._normalize_warranty_type(
                enriched.get("warranty_type")
            )

        # Extract product information
        if "product" in enriched:
            enriched["product"] = self._normalize_product_name(enriched.get("product"))
        elif "product_name" in enriched:
            enriched["product"] = self._normalize_product_name(
                enriched.get("product_name")
            )

        # Extract model/serial
        if "model" in enriched:
            enriched["model"] = str(enriched.get("model")).strip()

        if "serial_number" in enriched:
            enriched["serial_number"] = str(enriched.get("serial_number")).strip()

        # Map warranty_expires to the main date field if not present
        if "warranty_expires" in enriched and "date" not in enriched:
            enriched["date"] = enriched.get("warranty_expires")

        # Extract vendor/issuer
        if "vendor" in enriched:
            enriched["vendor"] = self._normalize_vendor(enriched.get("vendor"))
        elif "issuer" in enriched:
            enriched["vendor"] = self._normalize_vendor(enriched.get("issuer"))

        return enriched

    def _normalize_warranty_type(self, warranty_type: Any) -> str | None:
        """Normalize warranty type.

        Args:
            warranty_type: Raw warranty type value

        Returns:
            Normalized warranty type
        """
        if not warranty_type:
            return None

        type_str = str(warranty_type).strip().lower()

        # Map common variants
        type_map = {
            "manufacturer": "manufacturer",
            "extended": "extended",
            "lifetime": "lifetime",
            "limited": "limited",
            "full": "full",
            "store": "store",
            "seller": "seller",
        }

        for key, value in type_map.items():
            if key in type_str:
                return value

        return type_str

    def _normalize_product_name(self, product: Any) -> str | None:
        """Normalize product name.

        Args:
            product: Raw product name

        Returns:
            Normalized product name
        """
        if not product:
            return None

        return str(product).strip()

    def _normalize_vendor(self, vendor: Any) -> str | None:
        """Normalize vendor name via alibi.normalizers.vendors."""
        if not vendor:
            return None
        result = normalize_vendor(str(vendor))
        return result if result else None
