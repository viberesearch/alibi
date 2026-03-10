"""Field type inference functions.

Pure functions for inferring semantic field types from names and values.
Adapted from ma-engine patterns.
"""

from __future__ import annotations

import re
from typing import Any

from alibi.db.models import FieldType

# Keyword sets for field type inference
_CURRENCY_KEYWORDS = {
    "price",
    "cost",
    "amount",
    "total",
    "subtotal",
    "revenue",
    "income",
    "expense",
    "fee",
    "charge",
    "payment",
    "balance",
    "salary",
    "wage",
    "debt",
    "credit",
    "refund",
}

_PERCENTAGE_KEYWORDS = {
    "percentage",
    "percent",
    "pct",
    "rate",
    "ratio",
    "share",
    "margin",
    "discount",
    "tax_rate",
    "interest",
    "growth",
    "change",
}

_WEIGHT_KEYWORDS = {
    "weight",
    "mass",
    "kg",
    "gram",
    "pound",
    "lbs",
    "ounce",
    "oz",
}

_VOLUME_KEYWORDS = {
    "volume",
    "capacity",
    "liter",
    "litre",
    "ml",
    "gallon",
    "gal",
}

_COUNT_KEYWORDS = {
    "count",
    "quantity",
    "qty",
    "number",
    "amount",
    "units",
    "pieces",
    "items",
}

_ENERGY_KEYWORDS = {
    "energy",
    "power",
    "kwh",
    "watt",
    "electricity",
}

_DISTANCE_KEYWORDS = {
    "distance",
    "length",
    "height",
    "width",
    "depth",
    "meter",
    "km",
    "mile",
    "foot",
    "ft",
}

_DURATION_KEYWORDS = {
    "duration",
    "time",
    "hours",
    "minutes",
    "seconds",
    "period",
}

_AREA_KEYWORDS = {
    "area",
    "surface",
    "sqm",
    "sqft",
    "square",
}

_DATE_KEYWORDS = {
    "date",
    "day",
    "month",
    "year",
    "timestamp",
    "when",
    "created",
    "updated",
    "expires",
}

_BOOLEAN_KEYWORDS = {
    "is_",
    "has_",
    "can_",
    "should_",
    "active",
    "enabled",
    "visible",
    "verified",
    "confirmed",
    "approved",
    "taxable",
}


def infer_field_type(key: str, value: Any) -> FieldType:
    """Infer semantic field type from key name and value.

    Uses keyword sets and value patterns to guess the appropriate FieldType.
    Adapted from ma-engine pattern.

    Args:
        key: Field name (e.g., "unit_price", "tax_rate", "quantity")
        value: Field value (used for type hints if key is ambiguous)

    Returns:
        FieldType enum value, defaults to TEXT if undetermined.
    """
    if not key:
        # Value-only inference
        return _infer_from_value(value)

    # Normalize key: lowercase, split into tokens
    key_lower = key.lower().strip()
    tokens = set(re.split(r"[_\-\s]+", key_lower))

    # Boolean patterns (check first — prefix matches)
    for kw in _BOOLEAN_KEYWORDS:
        if key_lower.startswith(kw) or key_lower == kw.rstrip("_"):
            return FieldType.BOOLEAN

    # Date patterns
    for kw in _DATE_KEYWORDS:
        if kw in tokens:
            return FieldType.DATE

    # Percentage patterns (check before currency to avoid ambiguity)
    for kw in _PERCENTAGE_KEYWORDS:
        if kw in tokens:
            return FieldType.PERCENTAGE

    # Currency patterns
    for kw in _CURRENCY_KEYWORDS:
        if kw in tokens:
            return FieldType.CURRENCY

    # Weight patterns
    for kw in _WEIGHT_KEYWORDS:
        if kw in tokens:
            return FieldType.WEIGHT

    # Volume patterns
    for kw in _VOLUME_KEYWORDS:
        if kw in tokens:
            return FieldType.VOLUME

    # Energy patterns
    for kw in _ENERGY_KEYWORDS:
        if kw in tokens:
            return FieldType.ENERGY

    # Distance patterns
    for kw in _DISTANCE_KEYWORDS:
        if kw in tokens:
            return FieldType.DISTANCE

    # Duration patterns
    for kw in _DURATION_KEYWORDS:
        if kw in tokens:
            return FieldType.DURATION

    # Area patterns
    for kw in _AREA_KEYWORDS:
        if kw in tokens:
            return FieldType.AREA

    # Count patterns
    for kw in _COUNT_KEYWORDS:
        if kw in tokens:
            return FieldType.COUNT

    # Fall back to value-based inference
    return _infer_from_value(value)


def _infer_from_value(value: Any) -> FieldType:
    """Infer field type from value alone.

    Lowest confidence inference method.
    """
    if value is None:
        return FieldType.TEXT

    if isinstance(value, bool):
        return FieldType.BOOLEAN

    if isinstance(value, (int, float)):
        return FieldType.NUMBER

    if isinstance(value, str):
        v = value.strip()

        # Boolean-like strings
        if v.lower() in ("true", "false", "yes", "no", "1", "0"):
            return FieldType.BOOLEAN

        # Date-like strings (ISO format)
        if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            return FieldType.DATE

        # Percentage strings
        if v.endswith("%"):
            return FieldType.PERCENTAGE

        # Currency symbols
        if any(sym in v for sym in ["€", "$", "£", "¥", "₽"]):
            return FieldType.CURRENCY

        # Numeric strings
        if re.match(r"^-?\d+\.?\d*$", v):
            return FieldType.NUMBER

    return FieldType.TEXT
