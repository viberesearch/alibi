"""Currency normalization and parsing functions.

Pure functions for handling currency codes and amounts.
"""

from __future__ import annotations

import re
from decimal import Decimal

# Currency symbol to ISO code mappings
_CURRENCY_SYMBOLS: dict[str, str] = {
    "€": "EUR",
    "$": "USD",
    "£": "GBP",
    "¥": "JPY",
    "₽": "RUB",
    "₪": "ILS",
    "₹": "INR",
    "¢": "USD",  # cents
    "₩": "KRW",
    "₴": "UAH",
    "₺": "TRY",
    "₱": "PHP",
    "฿": "THB",
    "₫": "VND",
    "zł": "PLN",
    "Kč": "CZK",
    "kr": "SEK",  # Can also be NOK or DKK - context-dependent
    "CHF": "CHF",
    "Fr": "CHF",
}

# Common currency codes for validation
_VALID_CURRENCY_CODES = {
    "EUR",
    "USD",
    "GBP",
    "JPY",
    "CHF",
    "SEK",
    "NOK",
    "DKK",
    "PLN",
    "CZK",
    "RUB",
    "ILS",
    "INR",
    "CNY",
    "KRW",
    "UAH",
    "TRY",
    "PHP",
    "THB",
    "VND",
    "AUD",
    "CAD",
    "NZD",
    "SGD",
    "HKD",
    "MYR",
    "IDR",
    "AED",
    "SAR",
    "ZAR",
    "BRL",
    "MXN",
    "ARS",
}


def normalize_currency(code: str) -> str:
    """Map currency symbols to ISO codes.

    Examples:
        "€" -> "EUR"
        "$" -> "USD"
        "£" -> "GBP"
        "EUR" -> "EUR" (passthrough)

    Returns:
        ISO 4217 currency code (uppercase), or original input if not recognized.
    """
    if not code:
        return "EUR"  # Default to EUR

    code_stripped = code.strip()

    # Check if it's already a valid ISO code
    code_upper = code_stripped.upper()
    if code_upper in _VALID_CURRENCY_CODES:
        return code_upper

    # Check symbol mappings
    if code_stripped in _CURRENCY_SYMBOLS:
        return _CURRENCY_SYMBOLS[code_stripped]

    # Return as-is (uppercase) if not recognized
    return code_upper


def parse_amount_with_currency(raw: str) -> tuple[Decimal | None, str]:
    """Parse amount and currency from a string.

    Examples:
        "$42.50" -> (Decimal("42.50"), "USD")
        "€1,234.56" -> (Decimal("1234.56"), "EUR")
        "1500 RUB" -> (Decimal("1500"), "RUB")
        "£3.2M" -> (Decimal("3200000"), "GBP")
        "42.50 EUR" -> (Decimal("42.50"), "EUR")

    Returns:
        Tuple of (amount, currency_code). Default currency is "EUR".
    """
    if not raw:
        return None, "EUR"

    s = raw.strip()
    detected_currency = "EUR"

    # Check for currency symbols at the start
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if s.startswith(symbol):
            detected_currency = code
            s = s[len(symbol) :].strip()
            break

    # Check for currency codes (as separate tokens)
    # Pattern: match currency codes at start or end
    for code in _VALID_CURRENCY_CODES:
        # At start: "EUR 42.50"
        if s.upper().startswith(code + " ") or s.upper().startswith(code + "\t"):
            detected_currency = code
            s = s[len(code) :].strip()
            break
        # At end: "42.50 EUR"
        if s.upper().endswith(" " + code) or s.upper().endswith("\t" + code):
            detected_currency = code
            s = s[: -len(code)].strip()
            break

    # Parse the numeric part
    amount = _parse_amount(s)

    return amount, detected_currency


def _parse_amount(value_str: str) -> Decimal | None:
    """Parse numeric amount from string, handling various formats.

    Handles European (49,99) and US (1,234.56) number formats.
    Internal helper for parse_amount_with_currency.
    """
    if not value_str:
        return None

    s = value_str.strip()

    # Handle multipliers (K, M, B)
    multiplier = Decimal("1")
    s_upper = s.upper()
    if s_upper.endswith("K"):
        multiplier = Decimal("1000")
        s = s[:-1].strip()
    elif s_upper.endswith("M"):
        multiplier = Decimal("1000000")
        s = s[:-1].strip()
    elif s_upper.endswith("B"):
        multiplier = Decimal("1000000000")
        s = s[:-1].strip()

    # Remove spaces
    s = s.replace(" ", "")

    # Detect European format: comma as decimal separator (e.g., "49,99")
    # vs US format: comma as thousands separator (e.g., "1,234.56")
    if "," in s and "." not in s:
        # Likely European: replace comma with period
        s = s.replace(",", ".")
    else:
        # US format or no comma: remove commas (thousands separators)
        s = s.replace(",", "")

    try:
        return Decimal(s) * multiplier
    except Exception:
        return None
