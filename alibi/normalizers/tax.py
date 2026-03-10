"""Tax rate parsing and calculation functions.

Pure functions for handling tax information across multiple regions.
"""

from __future__ import annotations

import re
from decimal import Decimal

from alibi.db.models import TaxType

# Region-specific tax keywords
_VAT_KEYWORDS = {
    "vat",
    "mwst",  # German: Mehrwertsteuer
    "mva",  # Norwegian: Merverdiavgift
    "moms",  # Swedish/Danish: Moms
    "φπα",  # Greek: ΦΠΑ
    "fpa",  # Greek transliteration
    "iva",  # Italian/Spanish: IVA
    "tva",  # French: TVA
    "btw",  # Dutch: BTW
    "dph",  # Czech/Slovak: DPH
    "ндс",  # Russian: НДС (VAT)
}

_SALES_TAX_KEYWORDS = {
    "sales tax",
    "sales_tax",
    "salestax",
}

_GST_KEYWORDS = {
    "gst",
    "goods and services tax",
}

_EXEMPT_KEYWORDS = {
    "exempt",
    "tax-free",
    "tax free",
    "zero-rated",
}


def parse_tax_rate(raw: str) -> Decimal | None:
    """Extract tax rate from a string.

    Examples:
        "incl. 24% VAT" -> Decimal("24")
        "ΦΠΑ 13%" -> Decimal("13")
        "MwSt 19%" -> Decimal("19")
        "20% included" -> Decimal("20")

    Returns:
        Tax rate as Decimal percentage, or None if not found.
    """
    if not raw:
        return None

    s = raw.strip().lower()

    # Pattern: look for percentage number
    # Handles: "24%", "19 %", "13.5%"
    matches = re.findall(r"(\d+\.?\d*)\s*%", s)
    if matches:
        try:
            return Decimal(matches[0])
        except Exception:
            return None

    # Pattern: look for "rate:" or "tax:" followed by number
    match = re.search(r"(?:rate|tax)[:\s]+(\d+\.?\d*)", s)
    if match:
        try:
            return Decimal(match.group(1))
        except Exception:
            return None

    return None


def infer_tax_type(raw: str, country: str | None = None) -> TaxType:
    """Infer tax type from description and optional country context.

    Examples:
        "incl. 24% VAT", "FI" -> TaxType.VAT
        "ΦΠΑ 13%", "GR" -> TaxType.VAT
        "MwSt 19%", "DE" -> TaxType.VAT
        "Sales Tax 8%", "US" -> TaxType.SALES_TAX
        "GST 10%", "AU" -> TaxType.GST
        "Tax-free" -> TaxType.EXEMPT

    Returns:
        TaxType enum value.
    """
    if not raw:
        return TaxType.NONE

    s = raw.strip().lower()

    # Check for exempt keywords
    for keyword in _EXEMPT_KEYWORDS:
        if keyword in s:
            return TaxType.EXEMPT

    # Check for VAT keywords (most common in Europe)
    for keyword in _VAT_KEYWORDS:
        if keyword in s:
            return TaxType.VAT

    # Check for Sales Tax keywords (US/Canada)
    for keyword in _SALES_TAX_KEYWORDS:
        if keyword in s:
            return TaxType.SALES_TAX

    # Check for GST keywords (Australia, India, etc.)
    for keyword in _GST_KEYWORDS:
        if keyword in s:
            return TaxType.GST

    # Country-based inference if no keyword match
    if country:
        country_upper = country.upper()
        # VAT countries (Europe, etc.)
        if country_upper in {
            "AT",
            "BE",
            "BG",
            "CY",
            "CZ",
            "DE",
            "DK",
            "EE",
            "ES",
            "FI",
            "FR",
            "GB",
            "GR",
            "HR",
            "HU",
            "IE",
            "IT",
            "LT",
            "LU",
            "LV",
            "MT",
            "NL",
            "NO",
            "PL",
            "PT",
            "RO",
            "SE",
            "SI",
            "SK",
            "CH",
            "UK",
            "RU",
            "TR",
            "IL",
        }:
            return TaxType.VAT
        # Sales Tax countries
        elif country_upper in {"US", "CA"}:
            return TaxType.SALES_TAX
        # GST countries
        elif country_upper in {"AU", "IN", "NZ", "SG", "MY"}:
            return TaxType.GST

    # Check if "included" or "incl" is mentioned
    if "incl" in s or "included" in s:
        return TaxType.INCLUDED

    return TaxType.NONE


def calculate_tax(amount: Decimal, rate: Decimal) -> Decimal:
    """Calculate tax amount from base amount and rate.

    Example:
        calculate_tax(Decimal("100"), Decimal("20")) -> Decimal("20.00")

    Returns:
        Tax amount (amount * rate / 100).
    """
    return (amount * rate / Decimal("100")).quantize(Decimal("0.01"))
