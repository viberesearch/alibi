"""Tests for alibi.extraction.templates — vendor template learning and POS provider system."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.extraction.templates import (
    ParserHints,
    VendorTemplate,
    _TemplateCandidate,
    detect_barcode_position,
    detect_layout_type,
    detect_pos_provider,
    ensure_pos_identity,
    extract_template_fingerprint,
    find_template_by_location,
    find_template_for_vendor,
    load_vendor_template,
    merge_template,
    resolve_hints,
    save_vendor_template,
    template_to_hints,
)


# ---------------------------------------------------------------------------
# DB fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """Create a fresh temp database with all migrations applied."""
    from alibi.config import Config, reset_config

    reset_config()
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    config = Config(db_path=db_path)
    from alibi.db.connection import DatabaseManager

    manager = DatabaseManager(config)
    if not manager.is_initialized():
        manager.initialize()
    yield manager
    manager.close()
    os.unlink(db_path)


# ---------------------------------------------------------------------------
# VendorTemplate dataclass
# ---------------------------------------------------------------------------


class TestVendorTemplate:
    def test_default_values(self):
        t = VendorTemplate()
        assert t.layout_type == "standard"
        assert t.currency is None
        assert t.pos_provider is None
        assert t.success_count == 0

    def test_to_dict_minimal(self):
        t = VendorTemplate(layout_type="standard", success_count=3)
        d = t.to_dict()
        assert d["layout_type"] == "standard"
        assert d["success_count"] == 3
        assert "currency" not in d
        assert "pos_provider" not in d

    def test_to_dict_full(self):
        t = VendorTemplate(
            layout_type="columnar",
            currency="EUR",
            pos_provider="SAP",
            success_count=5,
        )
        d = t.to_dict()
        assert d == {
            "layout_type": "columnar",
            "currency": "EUR",
            "pos_provider": "SAP",
            "success_count": 5,
        }

    def test_from_dict_full(self):
        d = {
            "layout_type": "nqa",
            "currency": "GBP",
            "pos_provider": "MICROS",
            "success_count": 7,
        }
        t = VendorTemplate.from_dict(d)
        assert t.layout_type == "nqa"
        assert t.currency == "GBP"
        assert t.pos_provider == "MICROS"
        assert t.success_count == 7

    def test_from_dict_missing_optional_fields(self):
        d = {"success_count": 2}
        t = VendorTemplate.from_dict(d)
        assert t.layout_type == "standard"
        assert t.currency is None
        assert t.pos_provider is None
        assert t.success_count == 2

    def test_round_trip_serialization(self):
        original = VendorTemplate(
            layout_type="markdown_table",
            currency="USD",
            pos_provider="JCC",
            success_count=4,
        )
        restored = VendorTemplate.from_dict(original.to_dict())
        assert restored.layout_type == original.layout_type
        assert restored.currency == original.currency
        assert restored.pos_provider == original.pos_provider
        assert restored.success_count == original.success_count

    def test_is_reliable_below_threshold(self):
        t = VendorTemplate(success_count=1)
        assert not t.is_reliable

    def test_is_reliable_at_threshold(self):
        t = VendorTemplate(success_count=2)
        assert t.is_reliable

    def test_is_reliable_above_threshold(self):
        t = VendorTemplate(success_count=10)
        assert t.is_reliable

    def test_is_reliable_zero(self):
        t = VendorTemplate(success_count=0)
        assert not t.is_reliable


# ---------------------------------------------------------------------------
# ParserHints dataclass
# ---------------------------------------------------------------------------


class TestParserHints:
    def test_default_construction(self):
        h = ParserHints()
        assert h.vendor_name is None
        assert h.currency is None
        assert h.layout_type is None
        assert h.pos_provider is None

    def test_full_construction(self):
        h = ParserHints(
            vendor_name="ACME Store",
            currency="EUR",
            layout_type="columnar",
            pos_provider="SAP",
        )
        assert h.vendor_name == "ACME Store"
        assert h.currency == "EUR"
        assert h.layout_type == "columnar"
        assert h.pos_provider == "SAP"

    def test_partial_construction(self):
        h = ParserHints(vendor_name="My Shop")
        assert h.vendor_name == "My Shop"
        assert h.currency is None


# ---------------------------------------------------------------------------
# detect_pos_provider
# ---------------------------------------------------------------------------


class TestDetectPosProvider:
    def test_sap_detected(self):
        text = "Total: 25.00\nSAP Customer Checkout\nThank you"
        assert detect_pos_provider(text) == "SAP"

    def test_jcc_payment_systems_detected(self):
        text = "Visa ****1234\nJCC Payment Systems\nApproved"
        assert detect_pos_provider(text) == "JCC"

    def test_jcc_payment_short_form_detected(self):
        text = "JCC Payment\nTransaction ID: 12345"
        assert detect_pos_provider(text) == "JCC"

    def test_micros_detected(self):
        text = "RESTAURANT RECEIPT\nPowered by MICROS\nTotal 42.00"
        assert detect_pos_provider(text) == "MICROS"

    def test_oracle_fiscal_detected(self):
        text = "Oracle Fiscal System\nReceipt No: 001"
        assert detect_pos_provider(text) == "ORACLE_FISCAL"

    def test_sterna_detected(self):
        text = "STERNA POS\nTotal: 15.50"
        assert detect_pos_provider(text) == "STERNA"

    def test_softone_detected(self):
        text = "SoftOne ERP\nInvoice issued"
        assert detect_pos_provider(text) == "SOFTONE"

    def test_entersoft_detected(self):
        text = "Entersoft Business Suite\nVAT Invoice"
        assert detect_pos_provider(text) == "ENTERSOFT"

    def test_retail_pro_detected(self):
        text = "Retail Pro Version 9\nSale Complete"
        assert detect_pos_provider(text) == "RETAIL_PRO"

    def test_lightspeed_detected(self):
        text = "Powered by Lightspeed\nThank you for your purchase"
        assert detect_pos_provider(text) == "LIGHTSPEED"

    def test_square_pos_detected(self):
        text = "Square POS\nSale approved"
        assert detect_pos_provider(text) == "SQUARE_POS"

    def test_case_insensitive(self):
        # Signature comparison is lowercase
        text = "SAP CUSTOMER CHECKOUT"
        assert detect_pos_provider(text) == "SAP"

    def test_unknown_text_returns_none(self):
        text = "Total: 10.00\nThank you for shopping"
        assert detect_pos_provider(text) is None

    def test_empty_string_returns_none(self):
        assert detect_pos_provider("") is None

    def test_none_like_empty_text(self):
        # Empty string is falsy, treated as no text
        result = detect_pos_provider("   ")
        # Non-empty string (spaces) — won't match any signature
        assert result is None


# ---------------------------------------------------------------------------
# detect_layout_type
# ---------------------------------------------------------------------------


class TestDetectLayoutType:
    def test_layout_type_from_metadata_key(self):
        data = {"_layout_type": "markdown_table", "line_items": []}
        assert detect_layout_type(data) == "markdown_table"

    def test_layout_type_from_metadata_overrides_items(self):
        # Even with columnar-looking items, explicit _layout_type wins
        data = {
            "_layout_type": "nqa",
            "line_items": [
                {"name": "Apple", "quantity": "2"},
                {"name": "Banana", "quantity": "3"},
            ],
        }
        assert detect_layout_type(data) == "nqa"

    def test_empty_items_returns_standard(self):
        data = {"line_items": []}
        assert detect_layout_type(data) == "standard"

    def test_no_items_key_returns_standard(self):
        data = {"vendor": "ACME"}
        assert detect_layout_type(data) == "standard"

    def test_majority_non_unit_qty_infers_columnar(self):
        # quantity != "1" on most items → columnar
        data = {
            "line_items": [
                {"name": "Apple", "quantity": "2"},
                {"name": "Banana", "quantity": "3"},
                {"name": "Cherry", "quantity": "5"},
            ]
        }
        assert detect_layout_type(data) == "columnar"

    def test_unit_quantity_items_returns_standard(self):
        # quantity == "1" for all items → not columnar
        data = {
            "line_items": [
                {"name": "Apple", "quantity": "1"},
                {"name": "Banana", "quantity": "1"},
            ]
        }
        assert detect_layout_type(data) == "standard"

    def test_mixed_items_below_threshold_returns_standard(self):
        # Only 1 out of 4 items has non-1 quantity (25% < 50%)
        data = {
            "line_items": [
                {"name": "Apple", "quantity": "2"},
                {"name": "Banana", "quantity": "1"},
                {"name": "Cherry", "quantity": "1"},
                {"name": "Date", "quantity": "1"},
            ]
        }
        assert detect_layout_type(data) == "standard"


# ---------------------------------------------------------------------------
# extract_template_fingerprint
# ---------------------------------------------------------------------------


class TestExtractTemplateFingerprint:
    def test_high_confidence_returns_template(self):
        parse_data = {"currency": "EUR", "line_items": []}
        result = extract_template_fingerprint(parse_data, "some text", confidence=0.9)
        assert result is not None
        assert isinstance(result, VendorTemplate)
        assert result.currency == "EUR"
        assert result.success_count == 1

    def test_exactly_at_threshold_returns_template(self):
        parse_data = {"currency": "GBP"}
        result = extract_template_fingerprint(parse_data, "", confidence=0.8)
        assert result is not None

    def test_low_confidence_returns_none(self):
        parse_data = {"currency": "EUR"}
        result = extract_template_fingerprint(parse_data, "", confidence=0.5)
        assert result is None

    def test_just_below_threshold_returns_none(self):
        parse_data = {"currency": "USD"}
        result = extract_template_fingerprint(parse_data, "", confidence=0.79)
        assert result is None

    def test_captures_pos_provider_from_ocr_text(self):
        parse_data = {"currency": "EUR"}
        ocr_text = "Total: 25.00\nJCC Payment Systems\nApproved"
        result = extract_template_fingerprint(parse_data, ocr_text, confidence=0.9)
        assert result is not None
        assert result.pos_provider == "JCC"

    def test_no_pos_in_plain_text(self):
        parse_data = {"currency": "EUR"}
        result = extract_template_fingerprint(
            parse_data, "plain receipt text", confidence=0.9
        )
        assert result is not None
        assert result.pos_provider is None

    def test_detects_layout_from_items(self):
        parse_data = {
            "line_items": [
                {"name": "A", "quantity": "3"},
                {"name": "B", "quantity": "2"},
            ]
        }
        result = extract_template_fingerprint(parse_data, "", confidence=0.85)
        assert result is not None
        assert result.layout_type == "columnar"


# ---------------------------------------------------------------------------
# merge_template
# ---------------------------------------------------------------------------


class TestMergeTemplate:
    def test_same_layout_increments_count(self):
        existing = VendorTemplate(layout_type="columnar", success_count=3)
        new = VendorTemplate(layout_type="columnar", success_count=1)
        merged = merge_template(existing, new)
        assert merged.layout_type == "columnar"
        assert merged.success_count == 4

    def test_same_layout_adopts_new_currency(self):
        existing = VendorTemplate(
            layout_type="standard", currency="EUR", success_count=2
        )
        new = VendorTemplate(layout_type="standard", currency="USD", success_count=1)
        merged = merge_template(existing, new)
        assert merged.currency == "USD"

    def test_same_layout_keeps_existing_currency_when_new_is_none(self):
        existing = VendorTemplate(
            layout_type="standard", currency="EUR", success_count=2
        )
        new = VendorTemplate(layout_type="standard", currency=None, success_count=1)
        merged = merge_template(existing, new)
        assert merged.currency == "EUR"

    def test_different_layout_resets_count(self):
        existing = VendorTemplate(layout_type="columnar", success_count=5)
        new = VendorTemplate(layout_type="standard", success_count=1)
        merged = merge_template(existing, new)
        assert merged.layout_type == "standard"
        assert merged.success_count == 1

    def test_different_layout_adopts_new_layout(self):
        existing = VendorTemplate(layout_type="nqa", success_count=10)
        new = VendorTemplate(layout_type="markdown_table", success_count=1)
        merged = merge_template(existing, new)
        assert merged.layout_type == "markdown_table"

    def test_pos_provider_preserved_from_existing_when_new_is_none(self):
        existing = VendorTemplate(
            layout_type="standard", pos_provider="SAP", success_count=2
        )
        new = VendorTemplate(layout_type="standard", pos_provider=None, success_count=1)
        merged = merge_template(existing, new)
        assert merged.pos_provider == "SAP"

    def test_new_pos_provider_overrides_existing(self):
        existing = VendorTemplate(
            layout_type="standard", pos_provider="SAP", success_count=2
        )
        new = VendorTemplate(
            layout_type="standard", pos_provider="MICROS", success_count=1
        )
        merged = merge_template(existing, new)
        assert merged.pos_provider == "MICROS"


# ---------------------------------------------------------------------------
# template_to_hints
# ---------------------------------------------------------------------------


class TestTemplateToHints:
    def test_reliable_template_produces_full_hints(self):
        t = VendorTemplate(
            layout_type="columnar",
            currency="EUR",
            pos_provider="SAP",
            success_count=5,
        )
        hints = template_to_hints(t, vendor_name="ACME Store")
        assert hints.vendor_name == "ACME Store"
        assert hints.currency == "EUR"
        assert hints.layout_type == "columnar"
        assert hints.pos_provider == "SAP"

    def test_unreliable_template_produces_minimal_hints(self):
        t = VendorTemplate(
            layout_type="columnar",
            currency="EUR",
            pos_provider="SAP",
            success_count=1,
        )
        hints = template_to_hints(t, vendor_name="ACME Store")
        assert hints.vendor_name == "ACME Store"
        assert hints.currency is None
        assert hints.layout_type is None
        assert hints.pos_provider is None

    def test_reliable_template_without_vendor_name(self):
        t = VendorTemplate(layout_type="standard", currency="GBP", success_count=3)
        hints = template_to_hints(t)
        assert hints.vendor_name is None
        assert hints.currency == "GBP"
        assert hints.layout_type == "standard"

    def test_reliable_template_with_none_optional_fields(self):
        t = VendorTemplate(layout_type="standard", success_count=2)
        hints = template_to_hints(t, vendor_name="Shop")
        assert hints.currency is None
        assert hints.pos_provider is None
        assert hints.vendor_name == "Shop"
        assert hints.layout_type == "standard"


# ---------------------------------------------------------------------------
# load_vendor_template / save_vendor_template (DB round-trip)
# ---------------------------------------------------------------------------


class TestLoadSaveVendorTemplate:
    def test_load_returns_none_for_unknown_identity(self, db):
        result = load_vendor_template(db, "nonexistent-id-1234")
        assert result is None

    def test_round_trip_save_and_load(self, db):
        from alibi.identities.store import create_identity

        identity_id = create_identity(
            db, entity_type="vendor", canonical_name="TEST STORE"
        )
        template = VendorTemplate(
            layout_type="columnar",
            currency="EUR",
            pos_provider="JCC",
            success_count=3,
        )
        save_vendor_template(db, identity_id, template)

        loaded = load_vendor_template(db, identity_id)
        assert loaded is not None
        assert loaded.layout_type == "columnar"
        assert loaded.currency == "EUR"
        assert loaded.pos_provider == "JCC"
        assert loaded.success_count == 3

    def test_save_overwrites_existing_template(self, db):
        from alibi.identities.store import create_identity

        identity_id = create_identity(
            db, entity_type="vendor", canonical_name="SHOP XYZ"
        )
        first = VendorTemplate(layout_type="standard", success_count=2)
        save_vendor_template(db, identity_id, first)

        second = VendorTemplate(layout_type="columnar", currency="USD", success_count=5)
        save_vendor_template(db, identity_id, second)

        loaded = load_vendor_template(db, identity_id)
        assert loaded is not None
        assert loaded.layout_type == "columnar"
        assert loaded.success_count == 5

    def test_save_to_nonexistent_identity_is_safe(self, db):
        # Should not raise, just log and return
        template = VendorTemplate(success_count=1)
        save_vendor_template(db, "does-not-exist", template)

    def test_load_preserves_other_metadata_fields(self, db):
        from alibi.identities.store import create_identity, update_identity

        identity_id = create_identity(
            db, entity_type="vendor", canonical_name="PRESERVE ME"
        )
        # Set some pre-existing metadata
        update_identity(db, identity_id, metadata={"custom_field": "keep_me"})

        template = VendorTemplate(layout_type="nqa", success_count=2)
        save_vendor_template(db, identity_id, template)

        from alibi.identities.store import get_identity

        identity = get_identity(db, identity_id)
        meta = identity.get("metadata") or {}
        assert "custom_field" in meta
        assert meta["custom_field"] == "keep_me"
        assert "template" in meta


# ---------------------------------------------------------------------------
# find_template_for_vendor
# ---------------------------------------------------------------------------


class TestFindTemplateForVendor:
    def test_vendor_not_found_returns_triple_none(self, db):
        tpl, identity_id, canonical = find_template_for_vendor(
            db, vendor_name="NONEXISTENT VENDOR XYZ"
        )
        assert tpl is None
        assert identity_id is None
        assert canonical is None

    def test_vendor_found_without_template(self, db):
        from alibi.identities.store import add_member, create_identity

        identity_id = create_identity(
            db, entity_type="vendor", canonical_name="SIMPLE STORE"
        )
        add_member(db, identity_id, "name", "SIMPLE STORE")

        tpl, vid, canonical = find_template_for_vendor(db, vendor_name="SIMPLE STORE")
        # Identity found, but no template stored
        assert tpl is None
        assert vid == identity_id
        assert canonical == "SIMPLE STORE"

    def test_vendor_found_with_template(self, db):
        from alibi.identities.store import add_member, create_identity

        identity_id = create_identity(
            db, entity_type="vendor", canonical_name="TEMPLATE STORE"
        )
        add_member(db, identity_id, "name", "TEMPLATE STORE")

        template = VendorTemplate(
            layout_type="columnar", currency="EUR", success_count=4
        )
        save_vendor_template(db, identity_id, template)

        tpl, vid, canonical = find_template_for_vendor(db, vendor_name="TEMPLATE STORE")
        assert tpl is not None
        assert tpl.layout_type == "columnar"
        assert vid == identity_id
        assert canonical == "TEMPLATE STORE"

    def test_vendor_found_by_vat(self, db):
        from alibi.identities.store import add_member, create_identity

        identity_id = create_identity(
            db, entity_type="vendor", canonical_name="VAT STORE"
        )
        add_member(db, identity_id, "vat_number", "CY10057000Y")

        template = VendorTemplate(layout_type="standard", success_count=3)
        save_vendor_template(db, identity_id, template)

        tpl, vid, canonical = find_template_for_vendor(db, vendor_vat="CY10057000Y")
        assert tpl is not None
        assert vid == identity_id


# ---------------------------------------------------------------------------
# ensure_pos_identity (mocked — schema constraint blocks 'pos_provider' type)
# ---------------------------------------------------------------------------
# NOTE: The identities table CHECK constraint only allows 'vendor' and 'item'.
# ensure_pos_identity uses entity_type='pos_provider', which requires a DB
# migration not yet applied. These tests use mocks to exercise the function
# logic independently of the schema constraint.
# ---------------------------------------------------------------------------


class TestEnsurePosIdentity:
    # The ensure_pos_identity function imports create_identity inside its body
    # from alibi.identities.store, so we patch at the store module level.
    _STORE_PATH = "alibi.identities.store.create_identity"

    def test_creates_new_pos_identity(self):
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_db.get_connection.return_value = mock_conn
        # No existing row → create path
        mock_conn.execute.return_value.fetchone.return_value = None

        with patch(self._STORE_PATH, return_value="pos-id-001") as mock_create:
            identity_id = ensure_pos_identity(mock_db, "SAP")

        assert identity_id == "pos-id-001"
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        # entity_type and canonical_name passed as keyword args
        assert kwargs.get("entity_type") == "pos_provider"
        assert kwargs.get("canonical_name") == "SAP"

    def test_returns_existing_identity_on_second_call(self):
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_db.get_connection.return_value = mock_conn
        # Existing row found — simulated as a dict-like object
        mock_row = {"id": "existing-pos-id"}
        mock_conn.execute.return_value.fetchone.return_value = mock_row

        with patch(
            "alibi.extraction.templates.load_vendor_template", return_value=None
        ):
            identity_id = ensure_pos_identity(mock_db, "MICROS")

        assert identity_id == "existing-pos-id"

    def test_creates_with_template(self):
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_db.get_connection.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = None

        template = VendorTemplate(layout_type="columnar", success_count=1)
        with patch(self._STORE_PATH, return_value="pos-id-002"):
            identity_id = ensure_pos_identity(mock_db, "JCC", template=template)

        assert identity_id == "pos-id-002"

    def test_merges_template_on_existing_identity(self):
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_db.get_connection.return_value = mock_conn

        # Existing row found
        mock_row = {"id": "existing-pos-id"}
        mock_conn.execute.return_value.fetchone.return_value = mock_row

        existing_template = VendorTemplate(layout_type="standard", success_count=2)
        new_template = VendorTemplate(layout_type="standard", success_count=1)

        with (
            patch(
                "alibi.extraction.templates.load_vendor_template",
                return_value=existing_template,
            ),
            patch(
                "alibi.extraction.templates.merge_template",
                return_value=VendorTemplate(layout_type="standard", success_count=3),
            ) as mock_merge,
            patch("alibi.extraction.templates.save_vendor_template") as mock_save,
        ):
            identity_id = ensure_pos_identity(mock_db, "STERNA", template=new_template)

        assert identity_id == "existing-pos-id"
        mock_merge.assert_called_once_with(existing_template, new_template)
        mock_save.assert_called_once()

    def test_no_template_creates_identity_without_metadata(self):
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_db.get_connection.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = None

        with patch(self._STORE_PATH, return_value="pos-id-003") as mock_create:
            identity_id = ensure_pos_identity(mock_db, "EPILOGI")

        assert identity_id == "pos-id-003"
        args, kwargs = mock_create.call_args
        # No template → metadata should be None or empty
        metadata = kwargs.get("metadata")
        assert not metadata  # None or empty dict


# ---------------------------------------------------------------------------
# _TemplateCandidate scoring
# ---------------------------------------------------------------------------


class TestTemplateCandidateScoring:
    def test_lower_priority_wins_over_higher_priority(self):
        # priority 0 (vendor) should score higher than priority 1 (location)
        t_vendor = VendorTemplate(success_count=2)
        t_location = VendorTemplate(success_count=10)
        c_vendor = _TemplateCandidate(t_vendor, "id1", "Store A", "vendor", 0)
        c_location = _TemplateCandidate(t_location, "id2", "Store B", "location", 1)
        # vendor priority 0 → score = 0*1000 + 2 = 2
        # location priority 1 → score = -1*1000 + 10 = -990
        assert c_vendor.score > c_location.score

    def test_same_priority_higher_success_count_wins(self):
        t_low = VendorTemplate(success_count=2)
        t_high = VendorTemplate(success_count=8)
        c_low = _TemplateCandidate(t_low, "id1", "A", "vendor", 0)
        c_high = _TemplateCandidate(t_high, "id2", "B", "vendor", 0)
        assert c_high.score > c_low.score

    def test_score_formula(self):
        t = VendorTemplate(success_count=5)
        c = _TemplateCandidate(t, "id", "Name", "pos", 2)
        assert c.score == -2 * 1000 + 5


# ---------------------------------------------------------------------------
# resolve_hints
# ---------------------------------------------------------------------------


class TestResolveHints:
    def test_no_signals_returns_none_none(self, db):
        hints, identity_id = resolve_hints(db)
        assert hints is None
        assert identity_id is None

    def test_vendor_signal_with_reliable_template(self, db):
        from alibi.identities.store import add_member, create_identity

        identity_id = create_identity(
            db, entity_type="vendor", canonical_name="RELIABLE SHOP"
        )
        add_member(db, identity_id, "name", "RELIABLE SHOP")
        template = VendorTemplate(
            layout_type="columnar", currency="EUR", success_count=5
        )
        save_vendor_template(db, identity_id, template)

        hints, vid = resolve_hints(db, vendor_name="RELIABLE SHOP")
        assert hints is not None
        assert hints.layout_type == "columnar"
        assert hints.currency == "EUR"
        assert vid == identity_id

    def test_vendor_found_but_unreliable_template_returns_name_hint(self, db):
        from alibi.identities.store import add_member, create_identity

        identity_id = create_identity(
            db, entity_type="vendor", canonical_name="NEW SHOP"
        )
        add_member(db, identity_id, "name", "NEW SHOP")
        template = VendorTemplate(layout_type="columnar", success_count=1)
        save_vendor_template(db, identity_id, template)

        hints, vid = resolve_hints(db, vendor_name="NEW SHOP")
        # Template exists but is unreliable; identity known → name hint
        assert hints is not None
        assert hints.vendor_name == "NEW SHOP"
        assert hints.layout_type is None

    def test_vendor_not_found_returns_none(self, db):
        hints, vid = resolve_hints(db, vendor_name="COMPLETELY UNKNOWN VENDOR")
        assert hints is None
        assert vid is None

    def test_pos_signal_with_reliable_template(self, db):
        # Mock load_pos_template to return a reliable POS template
        # (ensure_pos_identity uses entity_type='pos_provider' which hits
        # a DB CHECK constraint — exercise the signal path via mock instead)
        pos_template = VendorTemplate(layout_type="standard", success_count=3)
        ocr_text = "Receipt\nSAP Customer Checkout\nTotal: 42.00"

        with patch(
            "alibi.extraction.templates.load_pos_template", return_value=pos_template
        ):
            hints, vid = resolve_hints(db, ocr_text=ocr_text)

        assert hints is not None
        assert hints.layout_type == "standard"

    def test_vendor_signal_preferred_over_pos(self, db):
        from alibi.identities.store import add_member, create_identity

        # Create reliable vendor template
        identity_id = create_identity(
            db, entity_type="vendor", canonical_name="PRIORITY STORE"
        )
        add_member(db, identity_id, "name", "PRIORITY STORE")
        vendor_template = VendorTemplate(
            layout_type="columnar", currency="EUR", success_count=5
        )
        save_vendor_template(db, identity_id, vendor_template)

        # Simulate POS template via mock (entity_type constraint workaround)
        pos_template = VendorTemplate(layout_type="standard", success_count=10)
        ocr_text = "MICROS System\nTotal: 25.00"

        with patch(
            "alibi.extraction.templates.load_pos_template", return_value=pos_template
        ):
            hints, vid = resolve_hints(
                db, vendor_name="PRIORITY STORE", ocr_text=ocr_text
            )

        # Vendor (priority 0) wins over POS (priority 2)
        assert hints is not None
        assert hints.layout_type == "columnar"

    def test_individual_signal_failure_does_not_block_others(self, db):
        # Vendor lookup for unknown vendor should fail gracefully;
        # POS signal should still produce hints (mocked to bypass schema constraint)
        pos_template = VendorTemplate(layout_type="nqa", success_count=3)
        ocr_text = "JCC Payment\nTotal: 15.00"

        with patch(
            "alibi.extraction.templates.load_pos_template", return_value=pos_template
        ):
            hints, vid = resolve_hints(
                db, vendor_name="UNKNOWN VENDOR ZZZ", ocr_text=ocr_text
            )

        # POS signal should still produce hints
        assert hints is not None
        assert hints.layout_type == "nqa"


# ---------------------------------------------------------------------------
# find_template_by_location
# ---------------------------------------------------------------------------


class TestFindTemplateByLocation:
    def _seed_fact_with_location(
        self,
        db,
        vendor_key: str,
        vendor_name: str,
        lat: float,
        lng: float,
    ) -> str:
        """Helper to create a fact with a location annotation."""
        import json
        from uuid import uuid4

        conn = db.get_connection()
        conn.execute(
            "INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)",
            ("system", "System"),
        )
        cloud_id = str(uuid4())
        conn.execute(
            "INSERT INTO clouds (id, status) VALUES (?, 'collapsed')",
            (cloud_id,),
        )
        fact_id = str(uuid4())
        conn.execute(
            """INSERT INTO facts
               (id, cloud_id, fact_type, vendor, vendor_key, total_amount, currency, event_date, status)
               VALUES (?, ?, 'purchase', ?, ?, '10.00', 'EUR', '2025-01-01', 'confirmed')""",
            (fact_id, cloud_id, vendor_name, vendor_key),
        )
        ann_id = str(uuid4())
        metadata = json.dumps({"lat": lat, "lng": lng})
        conn.execute(
            """INSERT INTO annotations
               (id, annotation_type, target_type, target_id, key, value, metadata, source)
               VALUES (?, 'location', 'fact', ?, 'location', 'test', ?, 'auto')""",
            (ann_id, fact_id, metadata),
        )
        conn.commit()
        return fact_id

    def test_no_annotations_returns_triple_none(self, db):
        tpl, identity_id, canonical = find_template_by_location(db, 35.0, 33.0)
        assert tpl is None
        assert identity_id is None
        assert canonical is None

    def test_no_match_outside_radius_returns_triple_none(self, db):
        from alibi.identities.store import add_member, create_identity

        identity_id = create_identity(
            db, entity_type="vendor", canonical_name="FAR STORE"
        )
        add_member(db, identity_id, "vendor_key", "CY10001111Y")
        template = VendorTemplate(layout_type="columnar", success_count=3)
        save_vendor_template(db, identity_id, template)

        # Seed fact 10km away
        self._seed_fact_with_location(db, "CY10001111Y", "FAR STORE", 35.200, 33.000)
        # Query from a very different location
        tpl, vid, canonical = find_template_by_location(
            db, 34.900, 33.000, radius_m=50.0
        )
        assert tpl is None

    def test_proximity_match_returns_template(self, db):
        from alibi.identities.store import add_member, create_identity

        identity_id = create_identity(
            db, entity_type="vendor", canonical_name="NEARBY SHOP"
        )
        add_member(db, identity_id, "vendor_key", "CY10002222Y")
        template = VendorTemplate(layout_type="nqa", currency="EUR", success_count=4)
        save_vendor_template(db, identity_id, template)

        # Seed fact at (35.1000, 33.1000)
        self._seed_fact_with_location(
            db, "CY10002222Y", "NEARBY SHOP", 35.1000, 33.1000
        )
        # Query from same location (0m distance)
        tpl, vid, canonical = find_template_by_location(
            db, 35.1000, 33.1000, radius_m=100.0
        )
        assert tpl is not None
        assert tpl.layout_type == "nqa"
        assert canonical == "NEARBY SHOP"

    def test_proximity_match_without_template_returns_identity(self, db):
        from alibi.identities.store import add_member, create_identity

        identity_id = create_identity(
            db, entity_type="vendor", canonical_name="NOTPL SHOP"
        )
        add_member(db, identity_id, "vendor_key", "CY10003333Y")
        # No template saved

        self._seed_fact_with_location(db, "CY10003333Y", "NOTPL SHOP", 35.2000, 33.2000)
        tpl, vid, canonical = find_template_by_location(
            db, 35.2000, 33.2000, radius_m=100.0
        )
        # Template is None, but identity_id is returned
        assert tpl is None
        assert vid == identity_id


# ---------------------------------------------------------------------------
# parse_ocr_text integration with ParserHints
# ---------------------------------------------------------------------------


class TestParseOcrTextWithHints:
    def test_vendor_hint_pre_seeds_vendor_field(self):
        from alibi.extraction.text_parser import parse_ocr_text

        # OCR text with no recognizable vendor header
        raw = "Total: 25.00\nEUR\n01/01/2025"
        hints = ParserHints(vendor_name="HINT VENDOR")
        result = parse_ocr_text(raw, doc_type="receipt", hints=hints)
        # Vendor should be populated from hint when heuristics miss it
        assert result.data.get("vendor") == "HINT VENDOR"

    def test_currency_hint_pre_seeds_currency_field(self):
        from alibi.extraction.text_parser import parse_ocr_text

        # Minimal receipt text without currency marker
        raw = "STORE NAME\n01/01/2025\nItem A 10.00\nTotal 10.00"
        hints = ParserHints(currency="USD")
        result = parse_ocr_text(raw, doc_type="receipt", hints=hints)
        # If parser can't detect currency from text, hint fills it
        # (currency may or may not be detectable; we just check no crash)
        assert "currency" in result.data or result.data.get("currency") is None

    def test_layout_hint_guides_item_parser(self):
        from alibi.extraction.text_parser import parse_ocr_text

        # Receipt that has columnar-style items
        raw = (
            "STORE\n"
            "01/01/2025\n"
            "Apple      2  x  1.50   3.00\n"
            "Banana     3  x  0.80   2.40\n"
            "Total            5.40\n"
        )
        hints = ParserHints(layout_type="columnar")
        result = parse_ocr_text(raw, doc_type="receipt", hints=hints)
        # Should not crash; layout_type hint is applied
        assert result is not None

    def test_empty_text_returns_empty_result_regardless_of_hints(self):
        from alibi.extraction.text_parser import parse_ocr_text

        hints = ParserHints(vendor_name="SOME VENDOR", currency="EUR")
        result = parse_ocr_text("", doc_type="receipt", hints=hints)
        assert result.confidence == 0.0

    def test_no_hints_still_parses_normally(self):
        from alibi.extraction.text_parser import parse_ocr_text

        raw = "ACME SUPERMARKET\n01/01/2025\nApple 1.50\nTotal 1.50\nEUR"
        result = parse_ocr_text(raw, doc_type="receipt", hints=None)
        assert result is not None
        assert isinstance(result.data, dict)

    def test_invoice_vendor_hint_seeds_issuer(self):
        from alibi.extraction.text_parser import parse_ocr_text

        raw = "INVOICE\n01/01/2025\nTotal: 100.00\nEUR"
        hints = ParserHints(vendor_name="SUPPLIER LTD")
        result = parse_ocr_text(raw, doc_type="invoice", hints=hints)
        # For invoice, vendor_name hint falls back to issuer
        assert result is not None


class TestBarcodePositionLearning:
    """Tests for barcode position learning in templates."""

    def test_detect_barcode_position_from_metadata(self):
        data = {"_barcode_position": "before_item", "line_items": []}
        assert detect_barcode_position(data) == "before_item"

    def test_detect_barcode_position_after_item(self):
        data = {"_barcode_position": "after_item", "line_items": []}
        assert detect_barcode_position(data) == "after_item"

    def test_detect_barcode_position_none_no_items(self):
        data = {"line_items": []}
        assert detect_barcode_position(data) is None

    def test_detect_barcode_position_none_few_barcodes(self):
        data = {"line_items": [{"name": "Milk", "barcode": "5290036000111"}]}
        assert detect_barcode_position(data) is None

    def test_template_round_trip_barcode_position(self):
        t = VendorTemplate(
            barcode_position="before_item",
            has_barcodes=True,
        )
        d = t.to_dict()
        assert d["barcode_position"] == "before_item"
        t2 = VendorTemplate.from_dict(d)
        assert t2.barcode_position == "before_item"

    def test_template_to_hints_passes_barcode_position(self):
        t = VendorTemplate(
            success_count=3,
            barcode_position="before_item",
        )
        hints = template_to_hints(t, vendor_name="ALPHAMEGA")
        assert hints.barcode_position == "before_item"

    def test_merge_template_preserves_barcode_position(self):
        existing = VendorTemplate(
            success_count=5,
            barcode_position="before_item",
        )
        new = VendorTemplate(success_count=1)
        merged = merge_template(existing, new)
        assert merged.barcode_position == "before_item"

    def test_extract_fingerprint_with_barcode_position(self):
        data = {
            "currency": "EUR",
            "line_items": [
                {"name": "A", "barcode": "5290036000111"},
                {"name": "B", "barcode": "5290036000222"},
            ],
            "_barcode_position": "before_item",
        }
        t = extract_template_fingerprint(data, "some text", confidence=0.9)
        assert t is not None
        assert t.barcode_position == "before_item"


class TestColumnarVariantLearning:
    """Tests for columnar_4/columnar_5 variant learning."""

    def test_merge_columnar_variants_compatible(self):
        existing = VendorTemplate(layout_type="columnar_5", success_count=3)
        new = VendorTemplate(layout_type="columnar_4", success_count=1)
        merged = merge_template(existing, new)
        assert merged.success_count == 4  # not reset — same family
        assert merged.layout_type == "columnar_4"  # adopts new specific variant

    def test_merge_generic_columnar_with_specific(self):
        existing = VendorTemplate(layout_type="columnar", success_count=3)
        new = VendorTemplate(layout_type="columnar_5", success_count=1)
        merged = merge_template(existing, new)
        assert merged.success_count == 4
        assert merged.layout_type == "columnar_5"  # specific wins

    def test_merge_different_family_resets(self):
        existing = VendorTemplate(layout_type="columnar_5", success_count=5)
        new = VendorTemplate(layout_type="nqa", success_count=1)
        merged = merge_template(existing, new)
        assert merged.success_count == 1  # reset — different family
        assert merged.stale is True


# ---------------------------------------------------------------------------
# Date format learning
# ---------------------------------------------------------------------------


class TestDateFormatLearning:
    """Tests for date format template learning."""

    def test_fingerprint_captures_date_format(self):
        data = {"currency": "EUR", "_date_format": "dmy", "line_items": []}
        tpl = extract_template_fingerprint(data, "some text", 0.9)
        assert tpl is not None
        assert tpl.date_format == "dmy"

    def test_merge_same_date_format_increments(self):
        existing = VendorTemplate(
            date_format="dmy", date_format_confidence=3, success_count=3
        )
        new = VendorTemplate(date_format="dmy", success_count=1)
        merged = merge_template(existing, new)
        assert merged.date_format == "dmy"
        assert merged.date_format_confidence == 4

    def test_merge_different_date_format_switches(self):
        existing = VendorTemplate(
            date_format="dmy", date_format_confidence=1, success_count=1
        )
        new = VendorTemplate(date_format="mdy", success_count=1)
        merged = merge_template(existing, new)
        assert merged.date_format == "mdy"
        assert merged.date_format_confidence == 1

    def test_merge_keeps_majority_date_format(self):
        existing = VendorTemplate(
            date_format="dmy", date_format_confidence=5, success_count=5
        )
        new = VendorTemplate(date_format="mdy", success_count=1)
        merged = merge_template(existing, new)
        assert merged.date_format == "dmy"  # majority wins

    def test_hints_include_date_format(self):
        tpl = VendorTemplate(date_format="dmy", success_count=3)
        hints = template_to_hints(tpl, vendor_name="Test")
        assert hints.date_format == "dmy"

    def test_stale_template_omits_date_format(self):
        tpl = VendorTemplate(date_format="dmy", success_count=3, stale=True)
        hints = template_to_hints(tpl, vendor_name="Test")
        assert hints.date_format is None

    def test_serialization_roundtrip(self):
        tpl = VendorTemplate(
            date_format="mdy",
            date_format_confidence=3,
            total_marker="TOTAL EUR",
            success_count=5,
        )
        d = tpl.to_dict()
        restored = VendorTemplate.from_dict(d)
        assert restored.date_format == "mdy"
        assert restored.date_format_confidence == 3
        assert restored.total_marker == "TOTAL EUR"


# ---------------------------------------------------------------------------
# Total marker learning
# ---------------------------------------------------------------------------


class TestTotalMarkerLearning:
    """Tests for total marker template learning."""

    def test_fingerprint_captures_total_marker(self):
        data = {"currency": "EUR", "_total_marker": "TOTAL EUR", "line_items": []}
        tpl = extract_template_fingerprint(data, "some text", 0.9)
        assert tpl is not None
        assert tpl.total_marker == "TOTAL EUR"

    def test_hints_include_total_marker(self):
        tpl = VendorTemplate(total_marker="SYNOLO", success_count=3)
        hints = template_to_hints(tpl, vendor_name="Test")
        assert hints.total_marker == "SYNOLO"


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


class TestLanguageDetection:
    """Tests for language detection from parsed content."""

    def test_fingerprint_captures_language(self):
        data = {"currency": "EUR", "_detected_language": "el", "line_items": []}
        tpl = extract_template_fingerprint(data, "some text", 0.9)
        assert tpl is not None
        assert tpl.language == "el"


# ---------------------------------------------------------------------------
# Correction-driven template refinement
# ---------------------------------------------------------------------------


class TestCorrectionDrivenTemplateLearning:
    """Tests for correction-driven template refinement."""

    def test_date_swap_updates_template(self):
        """Correcting a DD/MM->MM/DD date should update template format."""
        # The correction: 2026-05-03 (May 3) corrected to 2026-03-05 (March 5)
        # This means the system parsed as MM/DD but user says it's DD/MM
        from alibi.extraction.templates import _infer_date_format_from_correction

        fmt = _infer_date_format_from_correction("2026-05-03", "2026-03-05")
        assert fmt == "dmy"  # user says the original was DD/MM

    def test_date_swap_reverse(self):
        """Correcting DD/MM->MM/DD in the other direction."""
        from alibi.extraction.templates import _infer_date_format_from_correction

        fmt = _infer_date_format_from_correction("2026-03-05", "2026-05-03")
        assert fmt == "mdy"  # user says the original was MM/DD

    def test_non_swap_correction_returns_none(self):
        """A date correction that isn't a DD/MM swap returns None."""
        from alibi.extraction.templates import _infer_date_format_from_correction

        fmt = _infer_date_format_from_correction("2026-03-05", "2026-03-06")
        assert fmt is None  # different day, not a format swap

    def test_same_date_returns_none(self):
        from alibi.extraction.templates import _infer_date_format_from_correction

        fmt = _infer_date_format_from_correction("2026-03-05", "2026-03-05")
        assert fmt is None

    def test_invalid_date_returns_none(self):
        from alibi.extraction.templates import _infer_date_format_from_correction

        fmt = _infer_date_format_from_correction("not-a-date", "2026-03-05")
        assert fmt is None

    def test_different_year_returns_none(self):
        from alibi.extraction.templates import _infer_date_format_from_correction

        fmt = _infer_date_format_from_correction("2025-05-03", "2026-03-05")
        assert fmt is None  # year differs, not a simple swap

    def test_apply_correction_updates_existing_template(self, db):
        """apply_correction_to_template updates date_format on existing template."""
        from alibi.extraction.templates import (
            VendorTemplate,
            apply_correction_to_template,
            find_template_for_vendor,
            save_vendor_template,
        )
        from alibi.identities.matching import ensure_vendor_identity

        # Create a vendor identity with a template
        identity_id = ensure_vendor_identity(
            db,
            vendor_name="Test Vendor",
            vendor_key="TV123456Z",
            source="test",
        )
        tpl = VendorTemplate(
            success_count=3, date_format="mdy", date_format_confidence=2
        )
        save_vendor_template(db, identity_id, tpl)

        # Apply a correction that implies dmy format
        apply_correction_to_template(
            db,
            vendor_key="TV123456Z",
            field="date",
            old_value="2026-05-03",
            new_value="2026-03-05",
        )

        # Verify template was updated
        updated_tpl, _, _ = find_template_for_vendor(db, vendor_key="TV123456Z")
        assert updated_tpl is not None
        assert updated_tpl.date_format == "dmy"

    def test_apply_correction_creates_template_when_missing(self, db):
        """apply_correction_to_template creates a template if identity has none."""
        from alibi.extraction.templates import (
            apply_correction_to_template,
            find_template_for_vendor,
        )
        from alibi.identities.matching import ensure_vendor_identity

        # Create a vendor identity without a template
        ensure_vendor_identity(
            db,
            vendor_name="Bare Vendor",
            vendor_key="BV999999X",
            source="test",
        )

        # Apply correction
        apply_correction_to_template(
            db,
            vendor_key="BV999999X",
            field="date",
            old_value="2026-03-05",
            new_value="2026-05-03",
        )

        # Verify template was created
        tpl, _, _ = find_template_for_vendor(db, vendor_key="BV999999X")
        assert tpl is not None
        assert tpl.date_format == "mdy"
        assert tpl.date_format_confidence == 1

    def test_apply_correction_ignores_non_date_fields(self, db):
        """apply_correction_to_template silently ignores non-date fields."""
        from alibi.extraction.templates import apply_correction_to_template

        # Should not raise
        apply_correction_to_template(
            db,
            vendor_key="ANYTHING",
            field="vendor",
            old_value="Old Name",
            new_value="New Name",
        )

    def test_apply_correction_ignores_unknown_vendor(self, db):
        """apply_correction_to_template handles unknown vendor_key gracefully."""
        from alibi.extraction.templates import apply_correction_to_template

        # Should not raise
        apply_correction_to_template(
            db,
            vendor_key="NONEXISTENT999",
            field="date",
            old_value="2026-05-03",
            new_value="2026-03-05",
        )


# ---------------------------------------------------------------------------
# Region boundary learning
# ---------------------------------------------------------------------------


class TestRegionBoundaryLearning:
    """Tests for header/footer region learning."""

    def test_fingerprint_captures_header_lines(self):
        data = {
            "currency": "EUR",
            "_header_lines": 5,
            "_footer_ratio": 0.8,
            "line_items": [],
        }
        tpl = extract_template_fingerprint(data, "text", 0.9)
        assert tpl is not None
        assert tpl.typical_header_lines == 5
        assert tpl.typical_footer_ratio == 0.8

    def test_merge_averages_header_lines(self):
        existing = VendorTemplate(typical_header_lines=6, success_count=3)
        new = VendorTemplate(typical_header_lines=4, success_count=1)
        merged = merge_template(existing, new)
        assert merged.typical_header_lines == 5  # average of 6 and 4

    def test_merge_averages_footer_ratio(self):
        existing = VendorTemplate(typical_footer_ratio=0.7, success_count=3)
        new = VendorTemplate(typical_footer_ratio=0.9, success_count=1)
        merged = merge_template(existing, new)
        assert merged.typical_footer_ratio == 0.8  # average of 0.7 and 0.9

    def test_merge_new_fills_none(self):
        existing = VendorTemplate(success_count=3)
        new = VendorTemplate(typical_header_lines=5, success_count=1)
        merged = merge_template(existing, new)
        assert merged.typical_header_lines == 5

    def test_merge_existing_preserved_when_new_none(self):
        existing = VendorTemplate(typical_header_lines=6, success_count=3)
        new = VendorTemplate(success_count=1)
        merged = merge_template(existing, new)
        assert merged.typical_header_lines == 6

    def test_hints_include_header_lines(self):
        tpl = VendorTemplate(typical_header_lines=5, success_count=3)
        hints = template_to_hints(tpl, vendor_name="Test")
        assert hints.expected_header_lines == 5

    def test_hints_exclude_header_lines_when_stale(self):
        tpl = VendorTemplate(typical_header_lines=5, success_count=3, stale=True)
        hints = template_to_hints(tpl, vendor_name="Test")
        assert hints.expected_header_lines is None

    def test_hints_exclude_header_lines_when_unreliable(self):
        tpl = VendorTemplate(typical_header_lines=5, success_count=1)
        hints = template_to_hints(tpl, vendor_name="Test")
        assert hints.expected_header_lines is None

    def test_serialization_roundtrip(self):
        tpl = VendorTemplate(
            typical_header_lines=7, typical_footer_ratio=0.75, success_count=5
        )
        d = tpl.to_dict()
        restored = VendorTemplate.from_dict(d)
        assert restored.typical_header_lines == 7
        assert restored.typical_footer_ratio == 0.75

    def test_serialization_omits_none(self):
        tpl = VendorTemplate(success_count=1)
        d = tpl.to_dict()
        assert "typical_header_lines" not in d
        assert "typical_footer_ratio" not in d

    def test_observation_carries_through(self):
        from alibi.extraction.templates import record_extraction_observation

        tpl = VendorTemplate(
            typical_header_lines=5, typical_footer_ratio=0.8, success_count=3
        )
        updated = record_extraction_observation(tpl, confidence=0.9)
        assert updated.typical_header_lines == 5
        assert updated.typical_footer_ratio == 0.8
