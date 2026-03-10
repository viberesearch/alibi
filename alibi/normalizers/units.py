"""Unit normalization and conversion functions.

Pure functions for handling measurement units across multiple languages.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from alibi.db.models import UnitType

logger = logging.getLogger(__name__)

# Multilingual unit mappings
_UNIT_MAPPINGS: dict[str, UnitType] = {
    # Weight
    "g": UnitType.GRAM,
    "gr": UnitType.GRAM,
    "gram": UnitType.GRAM,
    "grams": UnitType.GRAM,
    "γρ": UnitType.GRAM,  # Greek
    "кг": UnitType.KILOGRAM,  # Russian/Cyrillic
    "ka": UnitType.KILOGRAM,  # OCR artifact for kg
    "kg": UnitType.KILOGRAM,
    "kgs": UnitType.KILOGRAM,
    "kilo": UnitType.KILOGRAM,
    "kilogram": UnitType.KILOGRAM,
    "kilograms": UnitType.KILOGRAM,
    "χιλιόγραμμο": UnitType.KILOGRAM,  # Greek
    "lb": UnitType.POUND,
    "lbs": UnitType.POUND,
    "pound": UnitType.POUND,
    "pounds": UnitType.POUND,
    "oz": UnitType.OUNCE,
    "ounce": UnitType.OUNCE,
    "ounces": UnitType.OUNCE,
    # Volume
    "ml": UnitType.MILLILITER,
    "milliliter": UnitType.MILLILITER,
    "milliliters": UnitType.MILLILITER,
    "μλ": UnitType.MILLILITER,  # Greek
    "l": UnitType.LITER,
    "lt": UnitType.LITER,
    "ltr": UnitType.LITER,
    "liter": UnitType.LITER,
    "liters": UnitType.LITER,
    "litre": UnitType.LITER,
    "litres": UnitType.LITER,
    "ℓ": UnitType.LITER,  # Unicode liter symbol (U+2113)
    "λίτρο": UnitType.LITER,  # Greek
    "λίτρα": UnitType.LITER,  # Greek plural
    "gal": UnitType.GALLON,
    "gallon": UnitType.GALLON,
    "gallons": UnitType.GALLON,
    # Count
    "ea": UnitType.PIECE,  # "each"
    "each": UnitType.PIECE,
    "pc": UnitType.PIECE,
    "pcs": UnitType.PIECE,
    "piece": UnitType.PIECE,
    "pieces": UnitType.PIECE,
    "st": UnitType.PIECE,  # German: Stück
    "stk": UnitType.PIECE,  # German: Stück
    "stück": UnitType.PIECE,  # German
    "τεμ": UnitType.PIECE,  # Greek: τεμάχιο
    "τεμάχιο": UnitType.PIECE,  # Greek
    "τεμάχια": UnitType.PIECE,  # Greek plural
    "шт": UnitType.PIECE,  # Russian: штука
    "pack": UnitType.PACK,
    "package": UnitType.PACK,
    "packages": UnitType.PACK,
    "pkg": UnitType.PACK,
    "συσκ": UnitType.PACK,  # Greek: συσκευασία
    # Energy
    "kwh": UnitType.KWH,
    "κβτω": UnitType.KWH,  # Greek
    # Distance
    "m": UnitType.METER,
    "meter": UnitType.METER,
    "meters": UnitType.METER,
    "μ": UnitType.METER,  # Greek
    "μέτρο": UnitType.METER,  # Greek
    # Area
    "sqm": UnitType.SQ_METER,
    "m2": UnitType.SQ_METER,
    "m²": UnitType.SQ_METER,
    "τμ": UnitType.SQ_METER,  # Greek: τετραγωνικό μέτρο
    # Volume (cubic)
    "cbm": UnitType.CUBIC_METER,
    "m3": UnitType.CUBIC_METER,
    "m³": UnitType.CUBIC_METER,
    "κυβ.μ": UnitType.CUBIC_METER,  # Greek
    # Time
    "hr": UnitType.HOUR,
    "hour": UnitType.HOUR,
    "hours": UnitType.HOUR,
    "ώρα": UnitType.HOUR,  # Greek
    "ώρες": UnitType.HOUR,  # Greek plural
    "min": UnitType.MINUTE,
    "minute": UnitType.MINUTE,
    "minutes": UnitType.MINUTE,
    "λεπτό": UnitType.MINUTE,  # Greek
    "λεπτά": UnitType.MINUTE,  # Greek plural
}

# User-defined alias overrides (loaded from YAML at startup)
_user_mappings: dict[str, UnitType] = {}

# Valid UnitType values for YAML validation
_VALID_UNIT_VALUES: dict[str, UnitType] = {ut.value: ut for ut in UnitType}


def load_user_aliases(path: Path) -> dict[str, UnitType]:
    """Read unit alias YAML and return validated mappings.

    Args:
        path: Path to YAML file with alias definitions.

    Returns:
        Dict mapping lowercase raw strings to UnitType values.
        Returns empty dict on missing file, empty file, or parse error.
    """
    if not path.exists():
        return {}

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot read unit aliases from %s: %s", path, exc)
        return {}

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning("Invalid YAML in %s: %s", path, exc)
        return {}

    if not isinstance(data, dict):
        return {}

    result: dict[str, UnitType] = {}
    for key, value in data.items():
        key_str = str(key).strip().lower()
        val_str = str(value).strip()
        if val_str in _VALID_UNIT_VALUES:
            result[key_str] = _VALID_UNIT_VALUES[val_str]
        else:
            logger.warning(
                "Skipping unit alias '%s: %s' — '%s' is not a valid UnitType "
                "(valid: %s)",
                key,
                value,
                value,
                ", ".join(sorted(_VALID_UNIT_VALUES.keys())),
            )
    return result


def init_unit_mappings(path: Path | None = None) -> None:
    """Load user alias overrides into module cache.

    Call once at startup. If path is None, aliases are cleared.
    """
    global _user_mappings
    if path is None:
        _user_mappings = {}
        return
    _user_mappings = load_user_aliases(path)
    if _user_mappings:
        logger.info("Loaded %d user unit alias(es) from %s", len(_user_mappings), path)


def reset_unit_mappings() -> None:
    """Clear user alias cache (for test isolation)."""
    global _user_mappings
    _user_mappings = {}


# Unit conversion factors (to base unit)
_WEIGHT_CONVERSIONS = {
    UnitType.GRAM: 1.0,
    UnitType.KILOGRAM: 1000.0,
    UnitType.POUND: 453.592,
    UnitType.OUNCE: 28.3495,
}

_VOLUME_CONVERSIONS = {
    UnitType.MILLILITER: 1.0,
    UnitType.LITER: 1000.0,
    UnitType.GALLON: 3785.41,
}

_DISTANCE_CONVERSIONS = {
    UnitType.METER: 1.0,
}

_TIME_CONVERSIONS = {
    UnitType.MINUTE: 1.0,
    UnitType.HOUR: 60.0,
}


def normalize_unit(raw: str) -> UnitType:
    """Map multilingual unit strings to UnitType enum.

    Examples:
        "kg" -> UnitType.KILOGRAM
        "кг" -> UnitType.KILOGRAM (Russian)
        "τεμ" -> UnitType.PIECE (Greek)
        "Stk" -> UnitType.PIECE (German)
        "lbs" -> UnitType.POUND
        "λίτρα" -> UnitType.LITER (Greek)

    Returns:
        UnitType enum value, or UnitType.OTHER if unmapped.
    """
    if not raw:
        return UnitType.OTHER

    # Normalize: lowercase, strip whitespace
    normalized = raw.strip().lower()

    # User overrides take priority
    unit_type = _user_mappings.get(normalized)
    if unit_type:
        return unit_type

    # Direct lookup in hardcoded defaults
    unit_type = _UNIT_MAPPINGS.get(normalized)
    if unit_type:
        return unit_type

    # Try without punctuation
    no_punct = re.sub(r"[.,;:]", "", normalized)

    unit_type = _user_mappings.get(no_punct)
    if unit_type:
        return unit_type

    unit_type = _UNIT_MAPPINGS.get(no_punct)
    if unit_type:
        return unit_type

    # X<number> pattern (e.g., "X12" = pack of 12 pieces)
    if re.match(r"^x\d+$", normalized):
        return UnitType.PIECE

    # Single letter that's not a known unit — likely a tax code column, not a unit
    if len(normalized) == 1 and normalized.isalpha():
        return UnitType.PIECE

    return UnitType.OTHER


def convert_unit(value: float, from_unit: UnitType, to_unit: UnitType) -> float | None:
    """Convert between compatible units.

    Examples:
        convert_unit(1000, UnitType.GRAM, UnitType.KILOGRAM) -> 1.0
        convert_unit(1, UnitType.LITER, UnitType.MILLILITER) -> 1000.0
        convert_unit(1, UnitType.POUND, UnitType.KILOGRAM) -> 0.453592

    Returns:
        Converted value, or None if units are incompatible.
    """
    if from_unit == to_unit:
        return value

    # Check if both units are in the same category
    if from_unit in _WEIGHT_CONVERSIONS and to_unit in _WEIGHT_CONVERSIONS:
        # Convert to base (grams), then to target
        base_value = value * _WEIGHT_CONVERSIONS[from_unit]
        return base_value / _WEIGHT_CONVERSIONS[to_unit]

    if from_unit in _VOLUME_CONVERSIONS and to_unit in _VOLUME_CONVERSIONS:
        # Convert to base (milliliters), then to target
        base_value = value * _VOLUME_CONVERSIONS[from_unit]
        return base_value / _VOLUME_CONVERSIONS[to_unit]

    if from_unit in _DISTANCE_CONVERSIONS and to_unit in _DISTANCE_CONVERSIONS:
        # Convert to base (meters), then to target
        base_value = value * _DISTANCE_CONVERSIONS[from_unit]
        return base_value / _DISTANCE_CONVERSIONS[to_unit]

    if from_unit in _TIME_CONVERSIONS and to_unit in _TIME_CONVERSIONS:
        # Convert to base (minutes), then to target
        base_value = value * _TIME_CONVERSIONS[from_unit]
        return base_value / _TIME_CONVERSIONS[to_unit]

    # Incompatible units
    return None


def parse_quantity_unit(raw: str) -> tuple[Decimal, UnitType, str | None]:
    """Parse quantity with unit from a string.

    Examples:
        "500g" -> (Decimal("500"), UnitType.GRAM, "g")
        "1.5L" -> (Decimal("1.5"), UnitType.LITER, "L")
        "12 pieces" -> (Decimal("12"), UnitType.PIECE, "pieces")
        "3 kg" -> (Decimal("3"), UnitType.KILOGRAM, "kg")

    Returns:
        Tuple of (quantity, normalized_unit, original_unit_string).
    """
    if not raw:
        return Decimal("1"), UnitType.PIECE, None

    s = raw.strip()

    # Pattern: number followed by optional space and unit
    # Supports: "500g", "1.5 L", "12 pieces", "3,5kg"
    pattern = r"^([\d.,]+)\s*([a-zA-Zα-ωΑ-Ω]+)$"
    match = re.match(pattern, s)

    if match:
        qty_str = match.group(1)
        unit_str = match.group(2)

        # Parse quantity (handle comma as decimal separator)
        qty_str_normalized = qty_str.replace(",", ".")
        try:
            quantity = Decimal(qty_str_normalized)
        except Exception:
            quantity = Decimal("1")

        # Normalize unit
        unit_type = normalize_unit(unit_str)

        return quantity, unit_type, unit_str

    # If no match, try to parse as just a number
    try:
        quantity = Decimal(s.replace(",", "."))
        return quantity, UnitType.PIECE, None
    except Exception:
        return Decimal("1"), UnitType.PIECE, None
