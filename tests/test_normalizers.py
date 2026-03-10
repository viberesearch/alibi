"""Tests for alibi.normalizers module.

Comprehensive tests covering normal cases, edge cases, and multilingual support.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from alibi.db.models import FieldType, TaxType, UnitType
from alibi.normalizers import (
    calculate_tax,
    convert_unit,
    detect_language,
    infer_field_type,
    infer_tax_type,
    init_unit_mappings,
    normalize_currency,
    normalize_date_format,
    normalize_text,
    normalize_unit,
    normalize_vendor,
    parse_amount_with_currency,
    parse_currency,
    parse_date,
    parse_number,
    parse_percentage,
    parse_quantity_unit,
    parse_tax_rate,
    reset_unit_mappings,
)


# ---------------------------------------------------------------------------
# numbers.py tests
# ---------------------------------------------------------------------------


class TestParseNumber:
    """Tests for parse_number function."""

    def test_plain_numbers(self):
        """Test parsing plain numeric values."""
        assert parse_number(42) == 42.0
        assert parse_number(3.14) == 3.14
        assert parse_number(-7.5) == -7.5
        assert parse_number("100") == 100.0

    def test_comma_separated(self):
        """Test parsing comma-separated numbers."""
        assert parse_number("4,200,000") == 4200000.0
        assert parse_number("1,234.56") == 1234.56

    def test_european_format(self):
        """Test parsing European number format (dot as thousands, comma as decimal)."""
        assert parse_number("1.234,56") == 1234.56
        assert parse_number("3,14") == 3.14

    def test_multiplier_suffixes(self):
        """Test parsing numbers with multiplier suffixes."""
        assert parse_number("4.2M") == 4200000.0
        assert parse_number("1.5K") == 1500.0
        assert parse_number("2B") == 2000000000.0
        assert parse_number("0.8T") == 800000000000.0

    def test_approximate_prefix(self):
        """Test parsing numbers with approximate markers."""
        assert parse_number("~4M") == 4000000.0
        assert parse_number("~1,200") == 1200.0

    def test_currency_prefixed(self):
        """Test parsing numbers with currency symbols."""
        assert parse_number("$4.2M") == 4200000.0
        assert parse_number("€1.5M") == 1500000.0
        assert parse_number("£100") == 100.0

    def test_percentage_suffix(self):
        """Test parsing percentages."""
        assert parse_number("45%") == 45.0
        assert parse_number("3.5%") == 3.5

    def test_edge_cases(self):
        """Test edge cases."""
        assert parse_number(None) is None
        assert parse_number("") is None
        assert parse_number("   ") is None
        assert parse_number("invalid") is None


class TestParseCurrency:
    """Tests for parse_currency function."""

    def test_currency_with_symbol(self):
        """Test parsing currency with symbols."""
        assert parse_currency("$42.50") == (42.50, "USD")
        assert parse_currency("€1,234.56") == (1234.56, "EUR")
        assert parse_currency("£100") == (100.0, "GBP")

    def test_currency_with_code(self):
        """Test parsing currency with ISO codes."""
        assert parse_currency("1500 RUB") == (1500.0, "RUB")
        assert parse_currency("EUR 42.50") == (42.50, "EUR")

    def test_currency_with_multiplier(self):
        """Test parsing currency with multipliers."""
        assert parse_currency("£3.2M") == (3200000.0, "GBP")

    def test_edge_cases(self):
        """Test edge cases."""
        assert parse_currency("") == (None, None)
        assert parse_currency(None) == (None, None)  # type: ignore[arg-type]


class TestParsePercentage:
    """Tests for parse_percentage function."""

    def test_valid_percentages(self):
        """Test parsing valid percentages."""
        assert parse_percentage("45%") == 45.0
        assert parse_percentage("3.5%") == 3.5
        assert parse_percentage("100%") == 100.0

    def test_without_percent_sign(self):
        """Test parsing numbers without percent sign."""
        assert parse_percentage("45") == 45.0

    def test_edge_cases(self):
        """Test edge cases."""
        assert parse_percentage("") is None
        assert parse_percentage(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# units.py tests
# ---------------------------------------------------------------------------


class TestNormalizeUnit:
    """Tests for normalize_unit function."""

    def test_weight_units(self):
        """Test normalizing weight units."""
        assert normalize_unit("kg") == UnitType.KILOGRAM
        assert normalize_unit("кг") == UnitType.KILOGRAM  # Russian
        assert normalize_unit("g") == UnitType.GRAM
        assert normalize_unit("lbs") == UnitType.POUND

    def test_volume_units(self):
        """Test normalizing volume units."""
        assert normalize_unit("l") == UnitType.LITER
        assert normalize_unit("λίτρα") == UnitType.LITER  # Greek
        assert normalize_unit("ml") == UnitType.MILLILITER

    def test_count_units(self):
        """Test normalizing count units."""
        assert normalize_unit("pcs") == UnitType.PIECE
        assert normalize_unit("τεμ") == UnitType.PIECE  # Greek
        assert normalize_unit("Stk") == UnitType.PIECE  # German
        assert normalize_unit("шт") == UnitType.PIECE  # Russian

    def test_each_maps_to_piece(self):
        """Test 'ea' and 'each' map to PIECE."""
        assert normalize_unit("ea") == UnitType.PIECE
        assert normalize_unit("each") == UnitType.PIECE
        assert normalize_unit("EA") == UnitType.PIECE

    def test_unicode_liter_symbol(self):
        """Test Unicode liter symbol maps to LITER."""
        assert normalize_unit("ℓ") == UnitType.LITER

    def test_ka_ocr_artifact(self):
        """Test 'ka' OCR artifact maps to KILOGRAM."""
        assert normalize_unit("ka") == UnitType.KILOGRAM

    def test_x_pattern_maps_to_piece(self):
        """Test X<N> pattern (e.g., X12) maps to PIECE."""
        assert normalize_unit("X12") == UnitType.PIECE
        assert normalize_unit("x6") == UnitType.PIECE
        assert normalize_unit("X1") == UnitType.PIECE

    def test_single_letter_tax_code_maps_to_piece(self):
        """Test single-letter unknowns (tax codes) map to PIECE."""
        assert normalize_unit("C") == UnitType.PIECE
        assert normalize_unit("D") == UnitType.PIECE

    def test_edge_cases(self):
        """Test edge cases."""
        assert normalize_unit("") == UnitType.OTHER
        assert normalize_unit("unknown") == UnitType.OTHER


class TestConvertUnit:
    """Tests for convert_unit function."""

    def test_weight_conversions(self):
        """Test weight unit conversions."""
        assert convert_unit(1000, UnitType.GRAM, UnitType.KILOGRAM) == 1.0
        assert convert_unit(1, UnitType.KILOGRAM, UnitType.GRAM) == 1000.0
        result = convert_unit(1, UnitType.POUND, UnitType.KILOGRAM)
        assert result is not None and abs(result - 0.453592) < 0.001

    def test_volume_conversions(self):
        """Test volume unit conversions."""
        assert convert_unit(1, UnitType.LITER, UnitType.MILLILITER) == 1000.0
        assert convert_unit(1000, UnitType.MILLILITER, UnitType.LITER) == 1.0

    def test_same_unit(self):
        """Test conversion with same source and target unit."""
        assert convert_unit(42, UnitType.KILOGRAM, UnitType.KILOGRAM) == 42.0

    def test_incompatible_units(self):
        """Test conversion between incompatible units."""
        assert convert_unit(1, UnitType.KILOGRAM, UnitType.LITER) is None


class TestParseQuantityUnit:
    """Tests for parse_quantity_unit function."""

    def test_quantity_with_unit(self):
        """Test parsing quantity with unit."""
        qty, unit, raw = parse_quantity_unit("500g")
        assert qty == Decimal("500")
        assert unit == UnitType.GRAM
        assert raw == "g"

        qty, unit, raw = parse_quantity_unit("1.5L")
        assert qty == Decimal("1.5")
        assert unit == UnitType.LITER
        assert raw == "L"

    def test_quantity_with_space(self):
        """Test parsing quantity with space before unit."""
        qty, unit, raw = parse_quantity_unit("12 pieces")
        assert qty == Decimal("12")
        assert unit == UnitType.PIECE
        assert raw == "pieces"

    def test_european_format(self):
        """Test parsing quantity with European decimal format."""
        qty, unit, raw = parse_quantity_unit("3,5kg")
        assert qty == Decimal("3.5")
        assert unit == UnitType.KILOGRAM

    def test_edge_cases(self):
        """Test edge cases."""
        qty, unit, raw = parse_quantity_unit("")
        assert qty == Decimal("1")
        assert unit == UnitType.PIECE
        assert raw is None


# ---------------------------------------------------------------------------
# dates.py tests
# ---------------------------------------------------------------------------


class TestParseDate:
    """Tests for parse_date function."""

    def test_iso_format(self):
        """Test parsing ISO date format."""
        assert parse_date("2024-12-31") == date(2024, 12, 31)
        assert parse_date("2024-01-15") == date(2024, 1, 15)

    def test_european_format(self):
        """Test parsing European date format."""
        assert parse_date("31/12/2024") == date(2024, 12, 31)
        assert parse_date("15.01.2024") == date(2024, 1, 15)
        assert parse_date("31-12-2024") == date(2024, 12, 31)

    def test_us_format_unambiguous(self):
        """Test parsing unambiguous US date format."""
        assert parse_date("13/01/2024") == date(2024, 1, 13)  # Day > 12, must be DD/MM

    def test_short_year(self):
        """Test parsing dates with 2-digit year."""
        assert parse_date("31/12/24") == date(2024, 12, 31)
        assert parse_date("15.01.24") == date(2024, 1, 15)

    def test_edge_cases(self):
        """Test edge cases."""
        assert parse_date(None) is None
        assert parse_date("") is None
        assert parse_date("invalid") is None
        assert parse_date("2024-13-01") is None  # Invalid month


class TestNormalizeDateFormat:
    """Tests for normalize_date_format function."""

    def test_iso_format(self):
        """Test formatting to ISO."""
        d = date(2024, 12, 31)
        assert normalize_date_format(d, "iso") == "2024-12-31"

    def test_european_format(self):
        """Test formatting to European format."""
        d = date(2024, 12, 31)
        assert normalize_date_format(d, "eu") == "31/12/2024"

    def test_us_format(self):
        """Test formatting to US format."""
        d = date(2024, 12, 31)
        assert normalize_date_format(d, "us") == "12/31/2024"


# ---------------------------------------------------------------------------
# tax.py tests
# ---------------------------------------------------------------------------


class TestParseTaxRate:
    """Tests for parse_tax_rate function."""

    def test_parse_with_percentage(self):
        """Test parsing tax rates with percentage sign."""
        assert parse_tax_rate("incl. 24% VAT") == Decimal("24")
        assert parse_tax_rate("ΦΠΑ 13%") == Decimal("13")
        assert parse_tax_rate("MwSt 19%") == Decimal("19")

    def test_parse_decimal_rate(self):
        """Test parsing decimal tax rates."""
        assert parse_tax_rate("13.5%") == Decimal("13.5")

    def test_edge_cases(self):
        """Test edge cases."""
        assert parse_tax_rate("") is None
        assert parse_tax_rate("no tax info") is None


class TestInferTaxType:
    """Tests for infer_tax_type function."""

    def test_vat_inference(self):
        """Test inferring VAT from text."""
        assert infer_tax_type("incl. 24% VAT") == TaxType.VAT
        assert infer_tax_type("ΦΠΑ 13%") == TaxType.VAT
        assert infer_tax_type("MwSt 19%") == TaxType.VAT

    def test_sales_tax_inference(self):
        """Test inferring sales tax."""
        assert infer_tax_type("Sales Tax 8%") == TaxType.SALES_TAX

    def test_gst_inference(self):
        """Test inferring GST."""
        assert infer_tax_type("GST 10%") == TaxType.GST

    def test_exempt_inference(self):
        """Test inferring tax-exempt."""
        assert infer_tax_type("Tax-free") == TaxType.EXEMPT

    def test_country_based_inference(self):
        """Test country-based tax type inference."""
        assert infer_tax_type("Tax 20%", "FI") == TaxType.VAT
        assert infer_tax_type("Tax 8%", "US") == TaxType.SALES_TAX
        assert infer_tax_type("Tax 10%", "AU") == TaxType.GST


class TestCalculateTax:
    """Tests for calculate_tax function."""

    def test_calculate_tax(self):
        """Test tax calculation."""
        assert calculate_tax(Decimal("100"), Decimal("20")) == Decimal("20.00")
        assert calculate_tax(Decimal("50"), Decimal("10")) == Decimal("5.00")


# ---------------------------------------------------------------------------
# currency.py tests
# ---------------------------------------------------------------------------


class TestNormalizeCurrency:
    """Tests for normalize_currency function."""

    def test_symbol_to_code(self):
        """Test converting symbols to ISO codes."""
        assert normalize_currency("€") == "EUR"
        assert normalize_currency("$") == "USD"
        assert normalize_currency("£") == "GBP"

    def test_iso_code_passthrough(self):
        """Test ISO codes pass through unchanged."""
        assert normalize_currency("EUR") == "EUR"
        assert normalize_currency("usd") == "USD"

    def test_edge_cases(self):
        """Test edge cases."""
        assert normalize_currency("") == "EUR"  # Default
        assert normalize_currency("UNKNOWN") == "UNKNOWN"


class TestParseAmountWithCurrency:
    """Tests for parse_amount_with_currency function."""

    def test_parse_with_symbol(self):
        """Test parsing with currency symbols."""
        amount, currency = parse_amount_with_currency("$42.50")
        assert amount == Decimal("42.50")
        assert currency == "USD"

    def test_parse_with_code(self):
        """Test parsing with ISO codes."""
        amount, currency = parse_amount_with_currency("42.50 EUR")
        assert amount == Decimal("42.50")
        assert currency == "EUR"

    def test_parse_with_multiplier(self):
        """Test parsing with multipliers."""
        amount, currency = parse_amount_with_currency("£3.2M")
        assert amount == Decimal("3200000")
        assert currency == "GBP"

    def test_edge_cases(self):
        """Test edge cases."""
        amount, currency = parse_amount_with_currency("")
        assert amount is None
        assert currency == "EUR"


# ---------------------------------------------------------------------------
# vendors.py tests
# ---------------------------------------------------------------------------


class TestNormalizeVendor:
    """Tests for normalize_vendor function."""

    def test_strip_legal_suffixes(self):
        """Test stripping legal suffixes."""
        assert normalize_vendor("ACME Corporation, Inc.") == "Acme"
        assert normalize_vendor("Best Shop Ltd") == "Best Shop"

    def test_strip_prefixes(self):
        """Test stripping prefixes."""
        assert normalize_vendor("The Best Shop Ltd") == "Best Shop"

    def test_multilingual(self):
        """Test multilingual vendor names."""
        assert normalize_vendor("Μαγαζί Μου Ε.Ε.") == "Μαγαζί Μου"
        assert normalize_vendor("ООО Компания") == "Компания"

    def test_edge_cases(self):
        """Test edge cases."""
        assert normalize_vendor("") == ""
        assert normalize_vendor("   ") == ""


# ---------------------------------------------------------------------------
# language.py tests
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_latin_scripts(self):
        """Test detecting Latin scripts."""
        assert detect_language("Hello world") == "en"
        assert detect_language("Bonjour le monde") == "en"  # Defaults to 'en' for Latin

    def test_greek(self):
        """Test detecting Greek."""
        assert detect_language("Γεια σου κόσμε") == "el"

    def test_cyrillic(self):
        """Test detecting Cyrillic (Russian)."""
        assert detect_language("Привет мир") == "ru"

    def test_mixed_scripts(self):
        """Test detecting dominant script in mixed text."""
        # Greek text with some Latin
        assert detect_language("Γεια σου world") == "el"

    def test_edge_cases(self):
        """Test edge cases."""
        assert detect_language("") == "en"
        assert detect_language("123") == "en"


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_strip_whitespace(self):
        """Test stripping whitespace."""
        assert normalize_text("  hello  ") == "hello"
        assert normalize_text("hello  world") == "hello world"

    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        # é can be represented as single char or e + combining accent
        assert normalize_text("café") == "café"

    def test_edge_cases(self):
        """Test edge cases."""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""


# ---------------------------------------------------------------------------
# fields.py tests
# ---------------------------------------------------------------------------


class TestInferFieldType:
    """Tests for infer_field_type function."""

    def test_currency_inference(self):
        """Test inferring currency fields."""
        assert infer_field_type("unit_price", 42.50) == FieldType.CURRENCY
        assert infer_field_type("total_amount", 100) == FieldType.CURRENCY
        assert infer_field_type("cost", None) == FieldType.CURRENCY

    def test_percentage_inference(self):
        """Test inferring percentage fields."""
        assert infer_field_type("tax_rate", 19) == FieldType.PERCENTAGE
        assert infer_field_type("discount_pct", "10%") == FieldType.PERCENTAGE

    def test_weight_inference(self):
        """Test inferring weight fields."""
        assert infer_field_type("weight", 500) == FieldType.WEIGHT
        assert infer_field_type("weight_kg", 1.5) == FieldType.WEIGHT

    def test_count_inference(self):
        """Test inferring count fields."""
        assert infer_field_type("quantity", 5) == FieldType.COUNT
        assert infer_field_type("qty", 10) == FieldType.COUNT

    def test_boolean_inference(self):
        """Test inferring boolean fields."""
        assert infer_field_type("is_active", True) == FieldType.BOOLEAN
        assert infer_field_type("has_discount", False) == FieldType.BOOLEAN

    def test_date_inference(self):
        """Test inferring date fields."""
        assert infer_field_type("created_date", "2024-12-31") == FieldType.DATE
        assert infer_field_type("date", None) == FieldType.DATE

    def test_value_only_inference(self):
        """Test inferring from value alone."""
        assert infer_field_type("", True) == FieldType.BOOLEAN
        assert infer_field_type("", 42) == FieldType.NUMBER
        assert infer_field_type("", "hello") == FieldType.TEXT


# ---------------------------------------------------------------------------
# User unit alias YAML tests
# ---------------------------------------------------------------------------


class TestUserUnitAliases:
    """Tests for user-defined unit alias loading from YAML."""

    def teardown_method(self) -> None:
        """Reset unit mappings after each test."""
        reset_unit_mappings()

    def test_load_valid_aliases(self, tmp_path: Path):
        """User YAML aliases are used by normalize_unit()."""
        yaml_file = tmp_path / "aliases.yaml"
        yaml_file.write_text("kilogramm: kg\nlitre: l\n")
        init_unit_mappings(yaml_file)
        assert normalize_unit("kilogramm") == UnitType.KILOGRAM
        assert normalize_unit("litre") == UnitType.LITER

    def test_user_alias_overrides_hardcoded(self, tmp_path: Path):
        """User aliases take priority over hardcoded defaults."""
        yaml_file = tmp_path / "aliases.yaml"
        # Remap "st" (normally PIECE) to PACK via user config
        yaml_file.write_text("st: pack\n")
        init_unit_mappings(yaml_file)
        assert normalize_unit("st") == UnitType.PACK

    def test_missing_yaml_file(self, tmp_path: Path):
        """Missing YAML file falls back gracefully to hardcoded only."""
        init_unit_mappings(tmp_path / "nonexistent.yaml")
        # Hardcoded mappings still work
        assert normalize_unit("kg") == UnitType.KILOGRAM

    def test_empty_yaml_file(self, tmp_path: Path):
        """Empty YAML file falls back to hardcoded only."""
        yaml_file = tmp_path / "aliases.yaml"
        yaml_file.write_text("")
        init_unit_mappings(yaml_file)
        assert normalize_unit("kg") == UnitType.KILOGRAM

    def test_corrupt_yaml_file(self, tmp_path: Path):
        """Corrupt YAML logs warning, falls back to hardcoded."""
        yaml_file = tmp_path / "aliases.yaml"
        yaml_file.write_text(":\n  - [\n  invalid yaml {{{\n")
        init_unit_mappings(yaml_file)
        assert normalize_unit("kg") == UnitType.KILOGRAM

    def test_invalid_unit_value_skipped(self, tmp_path: Path):
        """Invalid UnitType values in YAML are skipped, valid ones loaded."""
        yaml_file = tmp_path / "aliases.yaml"
        yaml_file.write_text("kilogramm: kg\nbogus_unit: not_a_unit\n")
        init_unit_mappings(yaml_file)
        assert normalize_unit("kilogramm") == UnitType.KILOGRAM
        assert normalize_unit("bogus_unit") == UnitType.OTHER

    def test_reset_clears_cache(self, tmp_path: Path):
        """reset_unit_mappings() clears loaded user aliases."""
        yaml_file = tmp_path / "aliases.yaml"
        yaml_file.write_text("kilogramm: kg\n")
        init_unit_mappings(yaml_file)
        assert normalize_unit("kilogramm") == UnitType.KILOGRAM

        reset_unit_mappings()
        assert normalize_unit("kilogramm") == UnitType.OTHER

    def test_case_insensitive_keys(self, tmp_path: Path):
        """YAML keys are lowercased for case-insensitive matching."""
        yaml_file = tmp_path / "aliases.yaml"
        yaml_file.write_text("Kilogramm: kg\n")
        init_unit_mappings(yaml_file)
        assert normalize_unit("KILOGRAMM") == UnitType.KILOGRAM
        assert normalize_unit("kilogramm") == UnitType.KILOGRAM

    def test_init_with_none_clears(self, tmp_path: Path):
        """init_unit_mappings(None) clears any loaded aliases."""
        yaml_file = tmp_path / "aliases.yaml"
        yaml_file.write_text("kilogramm: kg\n")
        init_unit_mappings(yaml_file)
        assert normalize_unit("kilogramm") == UnitType.KILOGRAM

        init_unit_mappings(None)
        assert normalize_unit("kilogramm") == UnitType.OTHER
