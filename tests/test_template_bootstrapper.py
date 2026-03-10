"""Tests for Gemini-assisted template bootstrapping."""

import os
import tempfile
import pytest
from unittest.mock import patch

from alibi.extraction.template_bootstrapper import (
    apply_vendor_details,
    bootstrap_with_gemini,
    build_enhanced_template,
    extract_vendor_details,
    merge_extraction,
    needs_bootstrapping,
    _merge_line_items,
)
from alibi.extraction.templates import (
    VendorTemplate,
    load_vendor_details,
    save_vendor_details,
)


# ---------------------------------------------------------------------------
# needs_bootstrapping
# ---------------------------------------------------------------------------


class TestNeedsBootstrapping:
    def test_no_template_gemini_enabled(self):
        assert needs_bootstrapping(None, gemini_enabled=True) is True

    def test_no_template_gemini_disabled(self):
        assert needs_bootstrapping(None, gemini_enabled=False) is False

    def test_template_not_bootstrapped(self):
        tpl = VendorTemplate(success_count=3, gemini_bootstrapped=False)
        assert needs_bootstrapping(tpl, gemini_enabled=True) is True

    def test_template_already_bootstrapped(self):
        tpl = VendorTemplate(success_count=3, gemini_bootstrapped=True)
        assert needs_bootstrapping(tpl, gemini_enabled=True) is False

    def test_template_already_bootstrapped_gemini_disabled(self):
        tpl = VendorTemplate(success_count=3, gemini_bootstrapped=True)
        assert needs_bootstrapping(tpl, gemini_enabled=False) is False


# ---------------------------------------------------------------------------
# merge_extraction
# ---------------------------------------------------------------------------


class TestMergeExtraction:
    def test_fills_missing_vendor_fields(self):
        parser = {"vendor": "ACME", "total": "10.00", "currency": "EUR"}
        gemini = {
            "vendor": "ACME Corp",
            "vendor_address": "123 Main St",
            "vendor_phone": "+1-555-0123",
            "vendor_vat": "CY12345678X",
            "total": "10.00",
            "currency": "EUR",
        }
        result = merge_extraction(parser, gemini, "receipt")
        assert result["vendor"] == "ACME"  # parser value preserved
        assert result["vendor_address"] == "123 Main St"  # filled
        assert result["vendor_phone"] == "+1-555-0123"  # filled
        assert result["vendor_vat"] == "CY12345678X"  # filled
        assert result["_gemini_bootstrapped"] is True

    def test_does_not_overwrite_parser_values(self):
        parser = {"vendor": "ACME", "vendor_vat": "PARSER_VAT", "total": "10.00"}
        gemini = {
            "vendor": "Different Name",
            "vendor_vat": "GEMINI_VAT",
            "total": "9.99",
        }
        result = merge_extraction(parser, gemini, "receipt")
        assert result["vendor"] == "ACME"
        assert result["vendor_vat"] == "PARSER_VAT"
        assert result["total"] == "10.00"

    def test_empty_gemini_returns_parser(self):
        parser = {"vendor": "ACME", "total": "10.00"}
        result = merge_extraction(parser, {}, "receipt")
        assert result["vendor"] == "ACME"

    def test_none_gemini_returns_parser(self):
        parser = {"vendor": "ACME", "total": "10.00"}
        result = merge_extraction(parser, None, "receipt")
        assert result["vendor"] == "ACME"

    def test_skips_internal_keys(self):
        parser = {"vendor": "ACME"}
        gemini = {
            "vendor": "ACME",
            "_pipeline": "gemini_bootstrap",
            "_parser_confidence": 0.95,
        }
        result = merge_extraction(parser, gemini, "receipt")
        assert (
            "_pipeline" not in result or result.get("_pipeline") != "gemini_bootstrap"
        )

    def test_invoice_fields(self):
        parser = {"issuer": "Widgets Inc", "amount": "500.00"}
        gemini = {
            "issuer": "Widgets Inc",
            "issuer_address": "456 Business Ave",
            "issuer_vat": "DE123456789",
            "issuer_phone": "+49-555-9999",
            "amount": "500.00",
        }
        result = merge_extraction(parser, gemini, "invoice")
        assert result["issuer_address"] == "456 Business Ave"
        assert result["issuer_vat"] == "DE123456789"

    def test_payment_fields(self):
        parser = {"vendor": "Store", "total": "25.00"}
        gemini = {
            "vendor": "Store",
            "vendor_address": "789 Shopping St",
            "vendor_tax_id": "TIC123",
            "total": "25.00",
        }
        result = merge_extraction(parser, gemini, "payment_confirmation")
        assert result["vendor_address"] == "789 Shopping St"
        assert result["vendor_tax_id"] == "TIC123"

    def test_gemini_items_used_when_parser_has_none(self):
        parser = {"vendor": "ACME", "line_items": []}
        gemini = {
            "vendor": "ACME",
            "line_items": [
                {"name": "Widget", "total_price": 5.0},
            ],
        }
        result = merge_extraction(parser, gemini, "receipt")
        assert len(result["line_items"]) == 1
        assert result["line_items"][0]["name"] == "Widget"

    def test_parser_items_preserved_when_both_have_items(self):
        parser = {
            "vendor": "ACME",
            "line_items": [
                {"name": "Widget", "quantity": "2", "total_price": "10.00"},
            ],
        }
        gemini = {
            "vendor": "ACME",
            "line_items": [
                {
                    "name": "Widget",
                    "quantity": 2,
                    "total_price": 10.0,
                    "brand": "Widgetco",
                    "barcode": "5901234123457",
                },
            ],
        }
        result = merge_extraction(parser, gemini, "receipt")
        assert len(result["line_items"]) == 1
        item = result["line_items"][0]
        assert item["name"] == "Widget"  # parser
        assert item["quantity"] == "2"  # parser
        assert item["brand"] == "Widgetco"  # from Gemini
        assert item["barcode"] == "5901234123457"  # from Gemini


# ---------------------------------------------------------------------------
# _merge_line_items
# ---------------------------------------------------------------------------


class TestMergeLineItems:
    def test_enriches_parser_items(self):
        parser_items = [
            {"name": "Milk", "quantity": "1", "total_price": "2.50"},
            {"name": "Bread", "quantity": "1", "total_price": "1.80"},
        ]
        gemini_items = [
            {
                "name": "Milk",
                "brand": "Dairy Fresh",
                "barcode": "1234567890123",
                "category": "Dairy",
            },
            {
                "name": "Bread",
                "brand": "Baker's Best",
                "unit_quantity": 0.5,
                "category": "Bakery",
            },
        ]
        result = _merge_line_items(parser_items, gemini_items)
        assert len(result) == 2
        assert result[0]["brand"] == "Dairy Fresh"
        assert result[0]["barcode"] == "1234567890123"
        assert result[0]["category"] == "Dairy"
        assert result[1]["brand"] == "Baker's Best"
        assert result[1]["unit_quantity"] == 0.5

    def test_does_not_overwrite_existing_fields(self):
        parser_items = [{"name": "Item", "brand": "ParserBrand"}]
        gemini_items = [{"name": "Item", "brand": "GeminiBrand"}]
        result = _merge_line_items(parser_items, gemini_items)
        assert result[0]["brand"] == "ParserBrand"

    def test_more_parser_items_than_gemini(self):
        parser_items = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        gemini_items = [{"name": "A", "brand": "BrandA"}]
        result = _merge_line_items(parser_items, gemini_items)
        assert len(result) == 3
        assert result[0]["brand"] == "BrandA"
        assert "brand" not in result[1]

    def test_empty_gemini_items(self):
        parser_items = [{"name": "A"}]
        result = _merge_line_items(parser_items, [])
        assert len(result) == 1
        assert result[0]["name"] == "A"


# ---------------------------------------------------------------------------
# extract_vendor_details
# ---------------------------------------------------------------------------


class TestExtractVendorDetails:
    def test_receipt_fields(self):
        data = {
            "vendor": "ACME",
            "vendor_address": "123 Main St",
            "vendor_phone": "+1-555-0123",
            "vendor_website": "www.acme.com",
            "vendor_vat": "CY12345678X",
            "vendor_tax_id": "TIC123",
            "vendor_legal_name": "ACME Corp Ltd",
        }
        details = extract_vendor_details(data, "receipt")
        assert details["address"] == "123 Main St"
        assert details["phone"] == "+1-555-0123"
        assert details["website"] == "www.acme.com"
        assert details["vat"] == "CY12345678X"
        assert details["tax_id"] == "TIC123"
        assert details["legal_name"] == "ACME Corp Ltd"

    def test_invoice_uses_issuer_fields(self):
        data = {
            "issuer": "Widgets Inc",
            "issuer_address": "456 Business Ave",
            "issuer_vat": "DE123456789",
        }
        details = extract_vendor_details(data, "invoice")
        assert details["address"] == "456 Business Ave"
        assert details["vat"] == "DE123456789"

    def test_missing_fields_not_included(self):
        data = {"vendor": "ACME", "vendor_vat": "CY123"}
        details = extract_vendor_details(data, "receipt")
        assert "address" not in details
        assert details["vat"] == "CY123"

    def test_empty_strings_not_included(self):
        data = {"vendor": "ACME", "vendor_address": "", "vendor_phone": "  "}
        details = extract_vendor_details(data, "receipt")
        assert "address" not in details
        assert "phone" not in details

    def test_unknown_doc_type_defaults_to_receipt(self):
        data = {"vendor_address": "123 St"}
        details = extract_vendor_details(data, "warranty")
        assert details["address"] == "123 St"


# ---------------------------------------------------------------------------
# apply_vendor_details
# ---------------------------------------------------------------------------


class TestApplyVendorDetails:
    def test_fills_gaps_in_extraction(self):
        extraction = {"vendor": "ACME", "total": "10.00"}
        details = {
            "address": "123 Main St",
            "phone": "+1-555-0123",
            "vat": "CY12345678X",
        }
        result = apply_vendor_details(extraction, details, "receipt")
        assert result["vendor_address"] == "123 Main St"
        assert result["vendor_phone"] == "+1-555-0123"
        assert result["vendor_vat"] == "CY12345678X"
        assert result["vendor"] == "ACME"  # unchanged

    def test_does_not_overwrite_existing(self):
        extraction = {"vendor": "ACME", "vendor_vat": "EXISTING_VAT"}
        details = {"vat": "CACHED_VAT", "address": "123 Main St"}
        result = apply_vendor_details(extraction, details, "receipt")
        assert result["vendor_vat"] == "EXISTING_VAT"
        assert result["vendor_address"] == "123 Main St"

    def test_invoice_maps_to_issuer_fields(self):
        extraction = {"issuer": "Widgets Inc"}
        details = {"address": "456 Ave", "vat": "DE123"}
        result = apply_vendor_details(extraction, details, "invoice")
        assert result["issuer_address"] == "456 Ave"
        assert result["issuer_vat"] == "DE123"

    def test_empty_details_returns_unchanged(self):
        extraction = {"vendor": "ACME"}
        result = apply_vendor_details(extraction, {}, "receipt")
        assert result == {"vendor": "ACME"}

    def test_none_details_returns_unchanged(self):
        extraction = {"vendor": "ACME"}
        # apply_vendor_details expects dict, but should handle empty gracefully
        result = apply_vendor_details(extraction, {}, "receipt")
        assert result == {"vendor": "ACME"}


# ---------------------------------------------------------------------------
# build_enhanced_template
# ---------------------------------------------------------------------------


class TestBuildEnhancedTemplate:
    def test_basic_receipt(self):
        data = {
            "currency": "EUR",
            "language": "el",
            "line_items": [
                {"name": "Milk", "barcode": "1234567890123"},
                {"name": "Bread"},
            ],
        }
        tpl = build_enhanced_template(data, "RECEIPT TEXT", "receipt")
        assert tpl.gemini_bootstrapped is True
        assert tpl.currency == "EUR"
        assert tpl.language == "el"
        assert tpl.has_barcodes is True
        assert tpl.has_unit_quantities is False
        assert tpl.typical_item_count == 2
        assert tpl.success_count == 1

    def test_with_unit_quantities(self):
        data = {
            "line_items": [
                {"name": "Cheese", "unit_quantity": 0.25},
                {"name": "Ham", "unit_quantity": 0.15},
            ],
        }
        tpl = build_enhanced_template(data, "", "receipt")
        assert tpl.has_unit_quantities is True

    def test_no_items(self):
        data = {"vendor": "ACME", "total": "10.00", "line_items": []}
        tpl = build_enhanced_template(data, "", "receipt")
        assert tpl.has_barcodes is None
        assert tpl.typical_item_count == 0

    def test_pos_detection(self):
        ocr = "RECEIPT\nJCC PAYMENT SYSTEMS\nTOTAL 10.00"
        data = {"currency": "EUR", "line_items": []}
        tpl = build_enhanced_template(data, ocr, "receipt")
        assert tpl.pos_provider == "JCC"

    def test_layout_detection(self):
        data = {"_layout_type": "columnar", "line_items": []}
        tpl = build_enhanced_template(data, "", "receipt")
        assert tpl.layout_type == "columnar"


# ---------------------------------------------------------------------------
# bootstrap_with_gemini
# ---------------------------------------------------------------------------


class TestBootstrapWithGemini:
    @patch("alibi.extraction.gemini_structurer.structure_ocr_text_gemini")
    def test_success(self, mock_gemini):
        mock_gemini.return_value = {
            "vendor": "ACME",
            "vendor_address": "123 St",
            "total": "10.00",
        }
        result = bootstrap_with_gemini("OCR TEXT", "receipt")
        assert result is not None
        assert result["vendor"] == "ACME"
        assert result["_pipeline"] == "gemini_bootstrap"
        mock_gemini.assert_called_once_with(
            "OCR TEXT",
            doc_type="receipt",
            api_key=None,
            model=None,
        )

    @patch("alibi.extraction.gemini_structurer.structure_ocr_text_gemini")
    def test_failure_returns_none(self, mock_gemini):
        mock_gemini.side_effect = Exception("API error")
        result = bootstrap_with_gemini("OCR TEXT", "receipt")
        assert result is None

    def test_empty_text_returns_none(self):
        result = bootstrap_with_gemini("", "receipt")
        assert result is None

    def test_whitespace_text_returns_none(self):
        result = bootstrap_with_gemini("   ", "receipt")
        assert result is None

    @patch("alibi.extraction.gemini_structurer.structure_ocr_text_gemini")
    def test_custom_api_key_and_model(self, mock_gemini):
        mock_gemini.return_value = {"vendor": "X"}
        bootstrap_with_gemini("TEXT", "invoice", api_key="key123", model="gemini-pro")
        mock_gemini.assert_called_once_with(
            "TEXT",
            doc_type="invoice",
            api_key="key123",
            model="gemini-pro",
        )


# ---------------------------------------------------------------------------
# VendorTemplate serialization (extended fields)
# ---------------------------------------------------------------------------


class TestVendorTemplateExtended:
    def test_round_trip_with_gemini_fields(self):
        tpl = VendorTemplate(
            layout_type="columnar",
            currency="EUR",
            pos_provider="JCC",
            success_count=5,
            gemini_bootstrapped=True,
            language="el",
            has_barcodes=True,
            has_unit_quantities=False,
            typical_item_count=12,
        )
        d = tpl.to_dict()
        restored = VendorTemplate.from_dict(d)
        assert restored.gemini_bootstrapped is True
        assert restored.language == "el"
        assert restored.has_barcodes is True
        assert restored.has_unit_quantities is False
        assert restored.typical_item_count == 12

    def test_backward_compat_no_gemini_fields(self):
        """Old templates without Gemini fields deserialize cleanly."""
        d = {"layout_type": "standard", "success_count": 3}
        tpl = VendorTemplate.from_dict(d)
        assert tpl.gemini_bootstrapped is False
        assert tpl.language is None
        assert tpl.has_barcodes is None

    def test_to_dict_omits_none_fields(self):
        tpl = VendorTemplate(layout_type="standard", success_count=1)
        d = tpl.to_dict()
        assert "gemini_bootstrapped" not in d
        assert "language" not in d
        assert "has_barcodes" not in d

    def test_to_dict_includes_gemini_when_set(self):
        tpl = VendorTemplate(
            gemini_bootstrapped=True,
            language="en",
            has_barcodes=False,
            has_unit_quantities=True,
            typical_item_count=5,
        )
        d = tpl.to_dict()
        assert d["gemini_bootstrapped"] is True
        assert d["language"] == "en"
        assert d["has_barcodes"] is False
        assert d["has_unit_quantities"] is True
        assert d["typical_item_count"] == 5


# ---------------------------------------------------------------------------
# Vendor details CRUD on identity metadata
# ---------------------------------------------------------------------------


class TestVendorDetailsCRUD:
    @pytest.fixture
    def mock_db(self):
        from alibi.config import Config, reset_config
        from alibi.db.connection import DatabaseManager

        reset_config()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        config = Config(db_path=db_path)
        manager = DatabaseManager(config)
        if not manager.is_initialized():
            manager.initialize()
        yield manager
        manager.close()
        os.unlink(db_path)

    def _create_identity(self, db, name="Test Vendor"):
        from alibi.identities.store import create_identity

        return create_identity(db, entity_type="vendor", canonical_name=name)

    def test_save_and_load_vendor_details(self, mock_db):
        identity_id = self._create_identity(mock_db)
        details = {
            "address": "123 Main St",
            "phone": "+1-555-0123",
            "website": "www.acme.com",
        }
        save_vendor_details(mock_db, identity_id, details)
        loaded = load_vendor_details(mock_db, identity_id)
        assert loaded["address"] == "123 Main St"
        assert loaded["phone"] == "+1-555-0123"
        assert loaded["website"] == "www.acme.com"

    def test_save_preserves_existing_template(self, mock_db):
        from alibi.extraction.templates import save_vendor_template

        identity_id = self._create_identity(mock_db)
        tpl = VendorTemplate(layout_type="columnar", success_count=3)
        save_vendor_template(mock_db, identity_id, tpl)

        details = {"address": "123 St"}
        save_vendor_details(mock_db, identity_id, details)

        # Template should still be there
        from alibi.extraction.templates import load_vendor_template

        loaded_tpl = load_vendor_template(mock_db, identity_id)
        assert loaded_tpl is not None
        assert loaded_tpl.layout_type == "columnar"

        # Details should also be there
        loaded_details = load_vendor_details(mock_db, identity_id)
        assert loaded_details["address"] == "123 St"

    def test_save_does_not_overwrite_existing_details(self, mock_db):
        identity_id = self._create_identity(mock_db)
        save_vendor_details(mock_db, identity_id, {"address": "First", "phone": "111"})
        save_vendor_details(mock_db, identity_id, {"address": "Second", "vat": "CY123"})
        loaded = load_vendor_details(mock_db, identity_id)
        assert loaded["address"] == "First"  # NOT overwritten
        assert loaded["phone"] == "111"
        assert loaded["vat"] == "CY123"  # new field added

    def test_save_filters_unknown_fields(self, mock_db):
        identity_id = self._create_identity(mock_db)
        details = {"address": "123 St", "unknown_field": "should be ignored"}
        save_vendor_details(mock_db, identity_id, details)
        loaded = load_vendor_details(mock_db, identity_id)
        assert "unknown_field" not in loaded
        assert loaded["address"] == "123 St"

    def test_save_skips_empty_values(self, mock_db):
        identity_id = self._create_identity(mock_db)
        details = {"address": "123 St", "phone": "", "website": None}
        save_vendor_details(mock_db, identity_id, details)
        loaded = load_vendor_details(mock_db, identity_id)
        assert "phone" not in loaded
        assert "website" not in loaded

    def test_load_nonexistent_identity(self, mock_db):
        loaded = load_vendor_details(mock_db, "99999")
        assert loaded == {}

    def test_load_identity_without_details(self, mock_db):
        identity_id = self._create_identity(mock_db)
        loaded = load_vendor_details(mock_db, identity_id)
        assert loaded == {}

    def test_save_empty_details_is_noop(self, mock_db):
        identity_id = self._create_identity(mock_db)
        save_vendor_details(mock_db, identity_id, {})
        loaded = load_vendor_details(mock_db, identity_id)
        assert loaded == {}
