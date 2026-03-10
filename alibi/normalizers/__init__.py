"""Normalizer functions for alibi.

Pure functions for parsing, normalizing, and converting data across multiple formats
and languages. No I/O, no side effects.
"""

from alibi.normalizers.currency import (
    normalize_currency,
    parse_amount_with_currency,
)
from alibi.normalizers.dates import (
    normalize_date_format,
    parse_date,
)
from alibi.normalizers.fields import (
    infer_field_type,
)
from alibi.normalizers.language import (
    detect_language,
    normalize_text,
)
from alibi.normalizers.numbers import (
    parse_currency,
    parse_number,
    parse_percentage,
)
from alibi.normalizers.tax import (
    calculate_tax,
    infer_tax_type,
    parse_tax_rate,
)
from alibi.normalizers.units import (
    convert_unit,
    init_unit_mappings,
    normalize_unit,
    parse_quantity_unit,
    reset_unit_mappings,
)
from alibi.normalizers.vendors import (
    normalize_vendor,
    normalize_vendor_slug,
)

__all__ = [
    # numbers.py
    "parse_number",
    "parse_currency",
    "parse_percentage",
    # units.py
    "normalize_unit",
    "convert_unit",
    "parse_quantity_unit",
    "init_unit_mappings",
    "reset_unit_mappings",
    # dates.py
    "parse_date",
    "normalize_date_format",
    # tax.py
    "parse_tax_rate",
    "infer_tax_type",
    "calculate_tax",
    # currency.py
    "normalize_currency",
    "parse_amount_with_currency",
    # vendors.py
    "normalize_vendor",
    "normalize_vendor_slug",
    # language.py
    "detect_language",
    "normalize_text",
    # fields.py
    "infer_field_type",
]
