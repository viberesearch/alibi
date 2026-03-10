"""Purchase record refiner with line item parsing."""

from __future__ import annotations

import logging
import re
from decimal import Decimal
from typing import Any
from uuid import uuid4

from alibi.db.models import RecordType, TaxType, UnitType
from alibi.normalizers.units import normalize_unit
from alibi.normalizers.vendors import normalize_vendor
from alibi.refiners.base import (
    BaseRefiner,
    _normalize_amount,
    _normalize_currency,
    _parse_quantity_unit,
)

logger = logging.getLogger(__name__)

# Patterns to extract unit/volume from product names
# Matches things like "500g", "250ml", "1lt", "1.5L", "500 g", "1 kg"
_NAME_UNIT_PATTERN = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(kg|g|gr|ml|l|lt|ltr|cl|oz|lb|lbs)\b",
    re.IGNORECASE,
)

# Trailing unit pattern: "Avocado kg", "Chicken breast kg"
_TRAILING_UNIT_PATTERN = re.compile(
    r"\s+(kg|g|gr|ml|l|lt|ltr|cl|oz|lb|lbs)\s*$",
    re.IGNORECASE,
)

# Bare trailing number: "Frozen Blueberries 500", "Basmati Rice 1000"
# Common grocery packaging weights in grams.
_BARE_NUMBER_PATTERN = re.compile(r"\s(\d{2,4})\s*$")
_COMMON_GRAM_VALUES = frozenset(
    {50, 75, 100, 125, 150, 175, 200, 250, 300, 330, 350, 400, 450, 500, 750, 800, 1000}
)

# Weight/volume units where weighed-item heuristics apply
_WEIGHT_VOLUME_UNITS = frozenset(
    {
        UnitType.GRAM,
        UnitType.KILOGRAM,
        UnitType.POUND,
        UnitType.OUNCE,
        UnitType.MILLILITER,
        UnitType.LITER,
        UnitType.GALLON,
    }
)

# Units where a conversion factor of 1000 is a known LLM error pattern
# (e.g. kg with unit_quantity=1000 means "1 kg = 1000 g" conversion, not content)
_CONVERSION_FACTOR_SUSPECTS: dict[UnitType, frozenset[Decimal]] = {
    UnitType.KILOGRAM: frozenset({Decimal("1000")}),
    UnitType.LITER: frozenset({Decimal("1000")}),
}

# Continuous units where fractional quantity is always a measurement, not a count
_CONTINUOUS_UNITS = frozenset({UnitType.KILOGRAM, UnitType.LITER, UnitType.POUND})

# VAT Analysis table pattern: "103 5.00%" or "100  19.00%"
_VAT_ANALYSIS_PATTERN = re.compile(
    r"\b(\d{2,3})\s+(\d+(?:\.\d+)?)\s*%",
)

# No real-world VAT/sales tax rate exceeds 30% (Hungary at 27% is the highest).
# Values above this threshold are likely VAT category codes, not rates.
_MAX_REASONABLE_TAX_RATE = Decimal("50")


class PurchaseRefiner(BaseRefiner):
    """Refine purchase records with line item extraction.

    Purchase records are the most complex:
    - Parse line items array from raw extraction
    - For each item: normalize name, detect unit, parse tax
    - Build LineItem-compatible dicts
    - Cross-validate: sum of line items ≈ receipt total
    - Create provenance chain: artifact -> purchase -> line items
    """

    def _refine_specific(
        self, raw: dict[str, Any], artifact_id: str | None
    ) -> dict[str, Any]:
        """Apply purchase-specific refinement logic.

        Args:
            raw: Partially normalized data dict
            artifact_id: Source artifact ID for provenance

        Returns:
            Enriched dict with purchase fields and line items
        """
        enriched = raw.copy()

        # Set record type
        enriched["record_type"] = RecordType.PURCHASE

        # Normalize vendor
        if "vendor" in enriched:
            enriched["vendor"] = self._normalize_vendor(enriched.get("vendor"))

        # Map vendor_vat from vendor_id if not already set
        if "vendor_id" in enriched and "vendor_vat" not in enriched:
            enriched["vendor_vat"] = enriched.get("vendor_id")

        # Parse VAT analysis table from raw_text (maps category codes to rates)
        raw_text = enriched.get("raw_text", "") or ""
        vat_mapping = self._parse_vat_analysis(raw_text)

        # Parse and normalize line items
        raw_items = enriched.get("line_items", []) or enriched.get("items", [])
        doc_language = enriched.get("language")
        if raw_items:
            enriched["line_items"] = self._parse_line_items(
                raw_items,
                artifact_id,
                enriched.get("currency", "EUR"),
                doc_language,
                vat_mapping,
            )

            # Cross-validate total
            if "amount" in enriched and enriched["amount"]:
                self._validate_total(enriched["amount"], enriched["line_items"])

        # Normalize receipt-level fields
        if "subtotal" in enriched:
            enriched["subtotal"] = _normalize_amount(enriched.get("subtotal"))

        if "tax_amount" in enriched:
            enriched["tax_amount"] = _normalize_amount(enriched.get("tax_amount"))

        if "discount_amount" in enriched:
            enriched["discount_amount"] = _normalize_amount(
                enriched.get("discount_amount")
            )

        # Ensure currency
        if "currency" not in enriched or not enriched["currency"]:
            enriched["currency"] = "EUR"

        return enriched

    def _parse_line_items(
        self,
        raw_items: list[dict[str, Any]],
        artifact_id: str | None,
        currency: str,
        doc_language: str | None = None,
        vat_mapping: dict[int, Decimal] | None = None,
    ) -> list[dict[str, Any]]:
        """Parse and normalize line items from raw extraction.

        Args:
            raw_items: List of raw line item dicts
            artifact_id: Source artifact ID for provenance
            currency: Currency code for the purchase
            doc_language: Document-level language code (fallback for items)
            vat_mapping: VAT category code to rate mapping from receipt footer

        Returns:
            List of normalized LineItem-compatible dicts
        """
        line_items = []

        for idx, raw_item in enumerate(raw_items):
            # Inject document-level language if item doesn't have its own
            if doc_language and "original_language" not in raw_item:
                raw_item.setdefault("language", doc_language)
            item = self._parse_single_line_item(
                raw_item, artifact_id, currency, idx, vat_mapping
            )
            if item:
                line_items.append(item)

        return line_items

    def _parse_single_line_item(
        self,
        raw_item: dict[str, Any],
        artifact_id: str | None,
        currency: str,
        index: int,
        vat_mapping: dict[int, Decimal] | None = None,
    ) -> dict[str, Any] | None:
        """Parse a single line item.

        Args:
            raw_item: Raw line item dict
            artifact_id: Source artifact ID
            currency: Currency code
            index: Item index in the list
            vat_mapping: VAT category code to rate mapping

        Returns:
            Normalized LineItem dict or None if invalid
        """
        # Extract name (required)
        name = (
            raw_item.get("name") or raw_item.get("description") or raw_item.get("item")
        )
        if not name:
            return None

        name = str(name).strip()

        item: dict[str, Any] = {
            "id": str(uuid4()),
            "artifact_id": artifact_id,
            "name": name,
            "currency": currency,
        }

        # -- Quantity and unit --
        # 1. Use explicit unit_raw from LLM if available (v2 prompt)
        unit_raw_from_llm = raw_item.get("unit_raw")

        # 2. Parse quantity
        quantity_raw = raw_item.get("quantity", "1")
        if isinstance(quantity_raw, str):
            quantity, unit_from_qty = _parse_quantity_unit(quantity_raw)
            item["quantity"] = quantity
        else:
            item["quantity"] = (
                Decimal(str(quantity_raw)) if quantity_raw else Decimal("1")
            )
            unit_from_qty = None

        # 3. Determine unit from best available source
        llm_unit_type = None
        if unit_raw_from_llm:
            llm_unit_type = normalize_unit(str(unit_raw_from_llm))

        # Also check LLM-provided unit_quantity (v2 prompt)
        llm_unit_quantity = raw_item.get("unit_quantity")

        if llm_unit_type and llm_unit_type not in (
            UnitType.PIECE,
            UnitType.OTHER,
        ):
            # LLM provided a specific (non-piece) unit — trust it
            item["unit_raw"] = str(unit_raw_from_llm)
            item["unit"] = llm_unit_type
            # Clean trailing unit from name and extract unit_quantity
            extracted = self._extract_unit_from_name(item["name"])
            if extracted:
                item["name"] = extracted[2]
                if extracted[3] is not None:
                    item["unit_quantity"] = extracted[3]
        elif unit_from_qty:
            # Unit was embedded in quantity string (e.g. "2.5kg")
            item["unit_raw"] = unit_from_qty
            item["unit"] = normalize_unit(unit_from_qty)
        else:
            # Try to extract unit from product name (e.g. "Red Bull 250ml")
            extracted = self._extract_unit_from_name(name)
            if extracted:
                item["unit_raw"] = extracted[0]
                item["unit"] = extracted[1]
                item["name"] = extracted[2]  # Clean name
                if extracted[3] is not None:
                    item["unit_quantity"] = extracted[3]
            elif llm_unit_type:
                # LLM returned piece/other — use it as-is
                item["unit_raw"] = str(unit_raw_from_llm)
                item["unit"] = llm_unit_type
            else:
                item["unit"] = UnitType.PIECE

        # Use LLM-provided unit_quantity if not already set from name
        if "unit_quantity" not in item and llm_unit_quantity:
            try:
                item["unit_quantity"] = Decimal(str(llm_unit_quantity))
            except Exception:
                logger.debug(
                    "Failed to parse unit_quantity %r", llm_unit_quantity, exc_info=True
                )

        # -- Name normalization (v2 fields) --
        name_en = raw_item.get("name_en")
        if name_en:
            item["name_normalized"] = str(name_en).strip()

        original_language = raw_item.get("original_language") or raw_item.get(
            "language"
        )
        if original_language:
            item["original_language"] = str(original_language).strip()

        # -- Prices --
        if "unit_price" in raw_item:
            item["unit_price"] = _normalize_amount(raw_item.get("unit_price"))

        if "total_price" in raw_item or "price" in raw_item or "total" in raw_item:
            total_raw = (
                raw_item.get("total_price")
                or raw_item.get("price")
                or raw_item.get("total")
            )
            item["total_price"] = _normalize_amount(total_raw)

        # Calculate missing price field
        if "unit_price" in item and item["unit_price"] and "total_price" not in item:
            item["total_price"] = item["unit_price"] * item["quantity"]

        if "total_price" in item and item["total_price"] and "unit_price" not in item:
            if item["quantity"] and item["quantity"] != 0:
                item["unit_price"] = item["total_price"] / item["quantity"]

        # -- Tax --
        tax_info = self._parse_tax(raw_item, vat_mapping)
        item.update(tax_info)

        # -- Discount --
        if "discount_amount" in raw_item:
            item["discount_amount"] = _normalize_amount(raw_item.get("discount_amount"))

        if "discount_percentage" in raw_item:
            discount_pct = raw_item.get("discount_percentage")
            if discount_pct:
                item["discount_percentage"] = Decimal(str(discount_pct))

        # -- Fix weighed item quantity/unit_quantity swap --
        self._fix_weighed_item_quantities(item)

        # -- Comparable unit price (for cross-shop comparison) --
        self._calculate_comparable_price(item)

        # -- Metadata fields --
        if "category" in raw_item:
            item["category"] = raw_item.get("category")

        if "subcategory" in raw_item:
            item["subcategory"] = raw_item.get("subcategory")

        if "brand" in raw_item:
            item["brand"] = raw_item.get("brand")

        if "barcode" in raw_item:
            item["barcode"] = raw_item.get("barcode")

        return item

    @staticmethod
    def _extract_unit_from_name(
        name: str,
    ) -> tuple[str, UnitType, str, Decimal | None] | None:
        """Extract unit/volume from product name when LLM didn't separate it.

        Handles patterns like:
            "Avocado kg" -> ("kg", KILOGRAM, "Avocado", None)
            "Red Bull White 250ml" -> ("ml", MILLILITER, "Red Bull White", 250)
            "Barilla Penne Rigate 500g" -> ("g", GRAM, "Barilla Penne Rigate", 500)
            "St. George Extra Virgin Oil 1lt" -> ("lt", LITER, "St. George Extra Virgin Oil", 1)

        Returns:
            Tuple of (unit_raw, UnitType, cleaned_name, unit_quantity) or None
        """
        # First try: "250ml", "500g", "1.5L", "1lt" embedded in name
        match = _NAME_UNIT_PATTERN.search(name)
        if match:
            qty_str = match.group(1).replace(",", ".")
            unit_str = match.group(2)
            unit_type = normalize_unit(unit_str)
            if unit_type != UnitType.OTHER:
                # Remove the matched portion from name
                clean = name[: match.start()].strip().rstrip("-").strip()
                if not clean:
                    clean = name[match.end() :].strip()
                unit_qty = Decimal(qty_str)
                return unit_str, unit_type, clean, unit_qty

        # Second try: trailing unit "Avocado kg", "Chicken breast kg"
        match = _TRAILING_UNIT_PATTERN.search(name)
        if match:
            unit_str = match.group(1)
            unit_type = normalize_unit(unit_str)
            if unit_type != UnitType.OTHER:
                clean = name[: match.start()].strip()
                return unit_str, unit_type, clean, None

        # Third try: bare trailing number for common gram weights
        # "Frozen Blueberries 500" -> ("g", GRAM, "Frozen Blueberries", 500)
        match = _BARE_NUMBER_PATTERN.search(name)
        if match:
            qty = int(match.group(1))
            if qty in _COMMON_GRAM_VALUES:
                clean = name[: match.start()].strip()
                if clean:  # Don't strip if it would leave name empty
                    return "g", UnitType.GRAM, clean, Decimal(str(qty))

        return None

    @staticmethod
    def _fix_weighed_item_quantities(item: dict[str, Any]) -> None:
        """Fix common LLM error: weight in quantity, conversion factor in unit_quantity.

        The LLM sometimes returns:
            quantity=0.76, unit="kg", unit_quantity=1000
        instead of:
            quantity=1, unit="kg", unit_quantity=0.76

        Also normalizes fractional quantities for continuous weight/volume units
        (kg, l, lb) when unit_quantity is missing.
        """
        unit = item.get("unit")
        quantity = item.get("quantity")
        unit_quantity = item.get("unit_quantity")

        if not unit or not quantity or unit not in _WEIGHT_VOLUME_UNITS:
            return

        # Case 1: unit_quantity is a known conversion factor
        suspects = _CONVERSION_FACTOR_SUSPECTS.get(unit)
        if suspects and unit_quantity is not None and unit_quantity in suspects:
            item["unit_quantity"] = quantity
            item["quantity"] = Decimal("1")
            return

        # Case 2: fractional quantity with no unit_quantity for continuous units
        if (
            unit_quantity is None
            and unit in _CONTINUOUS_UNITS
            and quantity > 0
            and quantity < Decimal("100")
            and quantity % 1 != 0
        ):
            item["unit_quantity"] = quantity
            item["quantity"] = Decimal("1")
            return

        # Case 3: weight duplicated in both quantity and unit_quantity
        # LLM puts the same weight in both fields (e.g. qty=0.77, unit_qty=0.77)
        if (
            unit_quantity is not None
            and unit in _CONTINUOUS_UNITS
            and quantity == unit_quantity
            and quantity > 0
            and quantity < Decimal("100")
            and quantity % 1 != 0
        ):
            item["quantity"] = Decimal("1")

    @staticmethod
    def _calculate_comparable_price(item: dict[str, Any]) -> None:
        """Calculate normalized price per standard unit for comparison.

        Normalizes weights to EUR/kg, volumes to EUR/l, pieces to EUR/pcs.
        Mutates item dict in-place, adding comparable_unit_price and
        comparable_unit.

        Args:
            item: Parsed line item dict (must have quantity, unit, total_price)
        """
        total_price = item.get("total_price")
        quantity = item.get("quantity")
        unit = item.get("unit")
        unit_quantity = item.get("unit_quantity")

        if not total_price or not quantity or quantity == 0 or not unit:
            return

        # Total content in item's unit
        if unit_quantity and unit_quantity > 0:
            total_content = quantity * unit_quantity
        else:
            total_content = quantity

        if total_content == 0:
            return

        # Conversion factors to standard units
        weight_units = {
            UnitType.GRAM: (Decimal("1000"), "kg"),
            UnitType.KILOGRAM: (Decimal("1"), "kg"),
            UnitType.POUND: (Decimal("2.20462"), "kg"),
            UnitType.OUNCE: (Decimal("35.274"), "kg"),
        }
        volume_units = {
            UnitType.MILLILITER: (Decimal("1000"), "l"),
            UnitType.LITER: (Decimal("1"), "l"),
            UnitType.GALLON: (Decimal("0.264172"), "l"),
        }

        if unit in weight_units:
            factor, std_unit = weight_units[unit]
            # price_per_std = total_price / (total_content / factor)
            raw_price = (total_price * factor) / total_content
            item["comparable_unit_price"] = raw_price.quantize(Decimal("0.01"))
            item["comparable_unit"] = std_unit
        elif unit in volume_units:
            factor, std_unit = volume_units[unit]
            raw_price = (total_price * factor) / total_content
            item["comparable_unit_price"] = raw_price.quantize(Decimal("0.01"))
            item["comparable_unit"] = std_unit
        else:
            # Pieces, packs, kWh, etc. — price per unit as-is
            raw_price = total_price / total_content
            item["comparable_unit_price"] = raw_price.quantize(Decimal("0.01"))
            item["comparable_unit"] = (
                unit.value if hasattr(unit, "value") else str(unit)
            )

    @staticmethod
    def _parse_vat_analysis(raw_text: str | None) -> dict[int, Decimal]:
        """Parse VAT Analysis table from receipt text.

        Many receipts include a footer table like:
            Vat  Rate    Net    Tax    Gross
            103  5.00%   23.80  1.19   24.99
            100  19.00%  0.92   0.17   1.09

        This maps category codes to their actual percentage rates.
        Returns empty dict if no table found or on any error — VAT
        resolution is best-effort and never blocks processing.

        Args:
            raw_text: Full receipt text (may be None)

        Returns:
            Dict mapping category code (int) to rate (Decimal).
            Empty dict if no VAT analysis found or on error.
        """
        if not raw_text:
            return {}
        try:
            mapping: dict[int, Decimal] = {}
            for match in _VAT_ANALYSIS_PATTERN.finditer(raw_text):
                code = int(match.group(1))
                rate = Decimal(match.group(2))
                mapping[code] = rate
            return mapping
        except Exception:
            return {}

    def _parse_tax(
        self,
        raw_item: dict[str, Any],
        vat_mapping: dict[int, Decimal] | None = None,
    ) -> dict[str, Any]:
        """Parse tax information from line item.

        Tax details are optional — the core value is line item cost breakdown.
        This method never raises; any parsing failure is silently ignored so
        that the item is still stored with its prices intact.

        Handles tax info from any receipt layout:
        - Inline per-item tax (tax_rate field on the line item)
        - Footer VAT Analysis table (resolved via vat_mapping)
        - Letter-based tax categories (e.g. "A", "B") — ignored gracefully

        Also handles a common LLM extraction error: receipts with VAT
        category codes (e.g. 103, 100, 106) that the LLM returns as
        tax_rate instead of the actual percentage.

        Args:
            raw_item: Raw line item dict
            vat_mapping: VAT category code to rate mapping from receipt footer

        Returns:
            Dict with tax_type and optionally tax_rate, tax_amount.
            Always returns at least {"tax_type": TaxType.NONE}.
        """
        tax_info: dict[str, Any] = {
            "tax_type": TaxType.NONE,
        }

        try:
            # Check for tax type
            tax_type_raw = raw_item.get("tax_type")
            if tax_type_raw:
                tax_type_str = str(tax_type_raw).strip().lower()
                if "vat" in tax_type_str:
                    tax_info["tax_type"] = TaxType.VAT
                elif "sales" in tax_type_str:
                    tax_info["tax_type"] = TaxType.SALES_TAX
                elif "gst" in tax_type_str:
                    tax_info["tax_type"] = TaxType.GST
                elif "exempt" in tax_type_str:
                    tax_info["tax_type"] = TaxType.EXEMPT
                elif "included" in tax_type_str:
                    tax_info["tax_type"] = TaxType.INCLUDED

            # Parse tax rate (percentage)
            tax_rate_raw = raw_item.get("tax_rate")
            if tax_rate_raw is not None:
                tax_rate_str = str(tax_rate_raw).strip().replace("%", "")
                # Skip non-numeric values (letter codes like "A", "B", "C")
                if not tax_rate_str or not tax_rate_str[0].isdigit():
                    pass  # Gracefully ignore letter-based tax categories
                else:
                    rate = Decimal(tax_rate_str)
                    if rate > _MAX_REASONABLE_TAX_RATE:
                        # Likely a VAT category code (e.g. 103), not a rate
                        code = int(rate)
                        if vat_mapping and code in vat_mapping:
                            tax_info["tax_rate"] = vat_mapping[code]
                        # else: discard — don't store a category code as rate
                    elif rate > 1:
                        tax_info["tax_rate"] = rate
                    elif rate > 0:
                        tax_info["tax_rate"] = rate * 100
                    else:
                        # Explicit 0 — zero-rated tax
                        tax_info["tax_rate"] = Decimal("0")

            # If we resolved a VAT rate and type was not set, default to VAT
            if "tax_rate" in tax_info and tax_info["tax_type"] == TaxType.NONE:
                tax_info["tax_type"] = TaxType.VAT

            # Parse tax amount
            if "tax_amount" in raw_item:
                tax_info["tax_amount"] = _normalize_amount(raw_item.get("tax_amount"))

        except Exception:
            # Tax parsing is best-effort; never block item storage
            logger.debug("Tax parsing failed for item", exc_info=True)

        return tax_info

    def _map_unit(self, unit_raw: str) -> UnitType:
        """Map raw unit string to UnitType enum.

        Args:
            unit_raw: Raw unit string (e.g., "kg", "ml", "pcs")

        Returns:
            UnitType enum value
        """
        return normalize_unit(unit_raw)

    def _validate_total(
        self, expected_total: Decimal, line_items: list[dict[str, Any]]
    ) -> None:
        """Cross-validate that sum of line items ≈ receipt total.

        Logs a warning if mismatch exceeds tolerance.

        Args:
            expected_total: Total from receipt
            line_items: List of parsed line items
        """
        # Sum line item totals
        calculated_total = Decimal("0")
        for item in line_items:
            if "total_price" in item and item["total_price"]:
                calculated_total += item["total_price"]

        # Check tolerance (allow 1% difference for rounding)
        tolerance = abs(expected_total) * Decimal("0.01")
        diff = abs(expected_total - calculated_total)

        if diff > tolerance:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Line item total mismatch: expected {expected_total}, "
                f"calculated {calculated_total} (diff={diff})"
            )

    def _normalize_vendor(self, vendor: Any) -> str | None:
        """Normalize vendor name via alibi.normalizers.vendors."""
        if not vendor:
            return None
        result = normalize_vendor(str(vendor))
        return result if result else None
