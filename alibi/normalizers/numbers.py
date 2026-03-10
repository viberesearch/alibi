"""Number parsing and normalization functions.

Pure functions for parsing numbers, currencies, and percentages from various string formats.
"""

from __future__ import annotations

import re
from typing import Any

_MULTIPLIERS = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
_CURRENCY_SYMBOLS = {
    "€": "EUR",
    "$": "USD",
    "£": "GBP",
    "¥": "JPY",
    "₽": "RUB",
    "₪": "ILS",
}
_CURRENCY_CODES = {
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
}


def parse_number(value: Any) -> float | None:
    """Parse a numeric value from various string formats.

    Handles:
        - Plain numbers: 42, 3.14, -7.5
        - Comma-separated: "4,200,000", "1,234.56"
        - European format: "1.234,56" (dot as thousands separator)
        - Multiplier suffixes: "4.2M", "1.5K", "2B", "0.8T"
        - Approximate prefix: "~4M", "~1,200"
        - Currency prefixed: "$4.2M", "€1.5M" (strips symbol, returns number only)
        - Percentage suffix: "45%" -> 45.0 (not 0.45)

    Returns:
        Parsed float, or None if unparsable.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return None

    # Strip approximate markers
    s = s.lstrip("~≈≈")

    # Strip currency symbols
    for sym in _CURRENCY_SYMBOLS:
        s = s.replace(sym, "")

    # Strip currency codes (only if they appear as whole tokens)
    s_upper = s.upper().strip()
    for code in _CURRENCY_CODES:
        if s_upper.startswith(code):
            s_upper = s_upper[len(code) :].strip()
            s = s_upper
        elif s_upper.endswith(code):
            s_upper = s_upper[: -len(code)].strip()
            s = s_upper

    s = s.strip()

    # Strip percentage sign (keep the number as-is, e.g. "45%" -> 45.0)
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1].strip()

    # Extract multiplier suffix
    multiplier = 1.0
    s_check = s.upper()
    for suffix, mult in _MULTIPLIERS.items():
        if s_check.endswith(suffix):
            multiplier = mult
            s = s[: -len(suffix)].strip()
            break

    # Detect European format (dot as thousands separator, comma as decimal)
    # Heuristic: if there's both comma and dot, and comma comes after dot, it's European
    has_comma = "," in s
    has_dot = "." in s
    if has_comma and has_dot:
        comma_pos = s.rfind(",")
        dot_pos = s.rfind(".")
        if comma_pos > dot_pos:
            # European format: "1.234,56"
            s = s.replace(".", "").replace(",", ".")
        else:
            # US format: "1,234.56"
            s = s.replace(",", "")
    elif has_comma:
        # Check if comma is used as decimal (e.g., "3,14" vs "1,234")
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            # Likely European decimal: "3,14"
            s = s.replace(",", ".")
        else:
            # Likely thousands separator: "1,234"
            s = s.replace(",", "")
    elif has_dot:
        # Keep dot as-is (decimal point)
        pass

    # Remove remaining spaces
    s = s.strip()

    try:
        return float(s) * multiplier
    except (ValueError, TypeError):
        return None


def parse_currency(value: str) -> tuple[float | None, str | None]:
    """Extract amount and currency code from a string.

    Examples:
        "$42.50" -> (42.50, "USD")
        "€1,234.56" -> (1234.56, "EUR")
        "1500 RUB" -> (1500.0, "RUB")
        "£3.2M" -> (3200000.0, "GBP")

    Returns:
        Tuple of (amount, currency_code) or (None, None) if unparsable.
    """
    if not value:
        return None, None

    s = str(value).strip()
    if not s:
        return None, None

    # Detect currency symbol
    detected_currency = None
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in s:
            detected_currency = code
            break

    # Detect currency code
    if not detected_currency:
        s_upper = s.upper()
        for code in _CURRENCY_CODES:
            # Check if code appears as a separate token
            if re.search(rf"\b{code}\b", s_upper):
                detected_currency = code
                break

    # Parse the numeric value
    amount = parse_number(s)

    return amount, detected_currency


def parse_percentage(value: str) -> float | None:
    """Parse a percentage string into a float.

    Examples:
        "45%" -> 45.0
        "3.5%" -> 3.5
        "100%" -> 100.0

    Returns:
        Percentage value (not fraction), or None if unparsable.
    """
    if not value:
        return None

    s = str(value).strip()
    if not s:
        return None

    # Strip percentage sign if present
    if s.endswith("%"):
        s = s[:-1].strip()

    # Parse as number
    return parse_number(s)
