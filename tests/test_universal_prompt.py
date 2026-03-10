"""Tests for universal extraction prompt, statement V2 prompt, and prompt_mode config."""

from typing import Any
from unittest.mock import patch

import pytest

from alibi.extraction.prompts import (
    STATEMENT_PROMPT_V2,
    UNIVERSAL_PROMPT_V2,
    classify_from_extraction,
    get_prompt_for_type,
)
from alibi.extraction.schemas import (
    STATEMENT_SCHEMA,
    UNIVERSAL_SCHEMA,
    get_schema,
    validate_extraction,
)


# ---------------------------------------------------------------------------
# UNIVERSAL_PROMPT_V2 tests
# ---------------------------------------------------------------------------


class TestUniversalPromptV2:
    """Tests for the universal extraction prompt."""

    def test_prompt_is_string(self) -> None:
        assert isinstance(UNIVERSAL_PROMPT_V2, str)
        assert len(UNIVERSAL_PROMPT_V2) > 100

    def test_prompt_mentions_all_document_types(self) -> None:
        for doc_type in [
            "receipt",
            "invoice",
            "statement",
            "payment_confirmation",
            "warranty",
            "contract",
        ]:
            assert doc_type in UNIVERSAL_PROMPT_V2

    def test_prompt_has_document_type_field(self) -> None:
        assert '"document_type"' in UNIVERSAL_PROMPT_V2

    def test_prompt_has_vendor_fields(self) -> None:
        for field in [
            "vendor",
            "vendor_address",
            "vendor_phone",
            "vendor_website",
            "vendor_vat",
        ]:
            assert f'"{field}"' in UNIVERSAL_PROMPT_V2

    def test_prompt_has_payment_fields(self) -> None:
        for field in [
            "payment_method",
            "card_type",
            "card_last4",
            "authorization_code",
            "terminal_id",
            "iban",
            "bic",
        ]:
            assert f'"{field}"' in UNIVERSAL_PROMPT_V2

    def test_prompt_has_line_items_section(self) -> None:
        assert '"line_items"' in UNIVERSAL_PROMPT_V2

    def test_prompt_has_transactions_section(self) -> None:
        assert '"transactions"' in UNIVERSAL_PROMPT_V2

    def test_prompt_has_statement_fields(self) -> None:
        for field in [
            "institution",
            "account_type",
            "account_last4",
            "opening_balance",
            "closing_balance",
        ]:
            assert f'"{field}"' in UNIVERSAL_PROMPT_V2

    def test_prompt_has_warranty_fields(self) -> None:
        for field in [
            "product_name",
            "product_model",
            "serial_number",
            "warranty_end",
            "warranty_type",
        ]:
            assert f'"{field}"' in UNIVERSAL_PROMPT_V2

    def test_prompt_has_contract_fields(self) -> None:
        for field in ["start_date", "end_date", "renewal", "notice_period"]:
            assert f'"{field}"' in UNIVERSAL_PROMPT_V2

    def test_prompt_has_language_field(self) -> None:
        assert '"language"' in UNIVERSAL_PROMPT_V2

    def test_prompt_has_raw_text_field(self) -> None:
        assert '"raw_text"' in UNIVERSAL_PROMPT_V2


# ---------------------------------------------------------------------------
# STATEMENT_PROMPT_V2 tests
# ---------------------------------------------------------------------------


class TestStatementPromptV2:
    """Tests for the V2 statement prompt."""

    def test_prompt_is_string(self) -> None:
        assert isinstance(STATEMENT_PROMPT_V2, str)
        assert len(STATEMENT_PROMPT_V2) > 100

    def test_prompt_has_language_detection(self) -> None:
        assert "ISO 639-1" in STATEMENT_PROMPT_V2
        assert '"language"' in STATEMENT_PROMPT_V2

    def test_prompt_has_institution_fields(self) -> None:
        for field in [
            "institution",
            "institution_address",
            "institution_phone",
            "institution_website",
            "institution_registration",
        ]:
            assert f'"{field}"' in STATEMENT_PROMPT_V2

    def test_prompt_has_account_fields(self) -> None:
        for field in ["account_type", "account_last4", "account_holder"]:
            assert f'"{field}"' in STATEMENT_PROMPT_V2

    def test_prompt_has_balance_fields(self) -> None:
        for field in ["opening_balance", "closing_balance"]:
            assert f'"{field}"' in STATEMENT_PROMPT_V2

    def test_prompt_has_transaction_fields(self) -> None:
        for field in ["date", "value_date", "description", "vendor", "amount", "type"]:
            assert f'"{field}"' in STATEMENT_PROMPT_V2

    def test_prompt_has_document_type(self) -> None:
        assert '"document_type": "statement"' in STATEMENT_PROMPT_V2

    def test_prompt_has_raw_text(self) -> None:
        assert '"raw_text"' in STATEMENT_PROMPT_V2


# ---------------------------------------------------------------------------
# get_prompt_for_type with mode parameter
# ---------------------------------------------------------------------------


class TestGetPromptForType:
    """Tests for prompt selection with mode parameter."""

    def test_default_mode_is_specialized(self) -> None:
        """Default behavior unchanged — returns type-specific V2 prompt."""
        prompt = get_prompt_for_type("receipt", version=2)
        assert "receipt" in prompt.lower()
        assert prompt != UNIVERSAL_PROMPT_V2

    def test_universal_mode_returns_universal(self) -> None:
        """Universal mode returns UNIVERSAL_PROMPT_V2 regardless of doc_type."""
        for doc_type in [
            "receipt",
            "invoice",
            "statement",
            "warranty",
            "contract",
            "payment_confirmation",
            "other",
        ]:
            prompt = get_prompt_for_type(doc_type, version=2, mode="universal")
            assert prompt is UNIVERSAL_PROMPT_V2

    def test_specialized_mode_returns_type_specific(self) -> None:
        """Specialized mode returns different prompts per type."""
        receipt = get_prompt_for_type("receipt", version=2, mode="specialized")
        invoice = get_prompt_for_type("invoice", version=2, mode="specialized")
        assert receipt != invoice

    def test_statement_v2_now_available(self) -> None:
        """Statement should now return V2 prompt, not V1 fallback."""
        prompt = get_prompt_for_type("statement", version=2, mode="specialized")
        assert prompt is STATEMENT_PROMPT_V2
        assert '"language"' in prompt

    def test_v1_still_works(self) -> None:
        """V1 prompts still work for backward compatibility."""
        prompt = get_prompt_for_type("receipt", version=1)
        assert "document_date" in prompt  # V1 uses document_date, not date


# ---------------------------------------------------------------------------
# Universal schema tests
# ---------------------------------------------------------------------------


class TestUniversalSchema:
    """Tests for the UNIVERSAL_SCHEMA."""

    def test_schema_exists(self) -> None:
        assert UNIVERSAL_SCHEMA is not None
        assert isinstance(UNIVERSAL_SCHEMA, dict)

    def test_schema_accessible_by_name(self) -> None:
        schema = get_schema("universal")
        assert schema is UNIVERSAL_SCHEMA

    def test_document_type_is_required(self) -> None:
        assert "document_type" in UNIVERSAL_SCHEMA["required"]

    def test_has_key_properties(self) -> None:
        props = UNIVERSAL_SCHEMA["properties"]
        assert "vendor" in props
        assert "date" in props
        assert "total" in props
        assert "currency" in props
        assert "line_items" in props
        assert "transactions" in props

    def test_validates_valid_data(self) -> None:
        data: dict[str, Any] = {
            "document_type": "receipt",
            "vendor": "Test Store",
            "total": 42.50,
            "currency": "EUR",
            "line_items": [{"name": "Item 1"}],
            "transactions": [],
        }
        errors = validate_extraction(data, "universal")
        assert errors == []

    def test_validates_missing_document_type(self) -> None:
        data: dict[str, Any] = {
            "vendor": "Test Store",
            "total": 42.50,
        }
        errors = validate_extraction(data, "universal")
        assert any("document_type" in e for e in errors)


class TestStatementSchema:
    """Tests for the STATEMENT_SCHEMA."""

    def test_schema_exists(self) -> None:
        assert STATEMENT_SCHEMA is not None

    def test_schema_accessible_by_name(self) -> None:
        schema = get_schema("statement")
        assert schema is STATEMENT_SCHEMA

    def test_required_fields(self) -> None:
        assert "currency" in STATEMENT_SCHEMA["required"]
        assert "transactions" in STATEMENT_SCHEMA["required"]

    def test_validates_valid_statement(self) -> None:
        data: dict[str, Any] = {
            "institution": "Bank of Test",
            "currency": "EUR",
            "transactions": [
                {"date": "2026-01-15", "description": "Purchase", "amount": 42.50}
            ],
        }
        errors = validate_extraction(data, "statement")
        assert errors == []


# ---------------------------------------------------------------------------
# classify_from_extraction tests
# ---------------------------------------------------------------------------


class TestClassifyFromExtraction:
    """Tests for post-extraction type classification."""

    def test_trusts_known_llm_type(self) -> None:
        for doc_type in [
            "receipt",
            "invoice",
            "statement",
            "payment_confirmation",
            "warranty",
            "contract",
        ]:
            data = {"document_type": doc_type}
            assert classify_from_extraction(data) == doc_type

    def test_handles_case_insensitive(self) -> None:
        assert classify_from_extraction({"document_type": "RECEIPT"}) == "receipt"
        assert classify_from_extraction({"document_type": "Invoice"}) == "invoice"

    def test_handles_whitespace(self) -> None:
        assert classify_from_extraction({"document_type": " receipt "}) == "receipt"

    def test_statement_heuristic(self) -> None:
        """Documents with institution + transactions → statement."""
        data: dict[str, Any] = {
            "document_type": "other",
            "institution": "Bank of Cyprus",
            "transactions": [{"date": "2026-01-15", "amount": 42.50}],
        }
        assert classify_from_extraction(data) == "statement"

    def test_warranty_heuristic(self) -> None:
        """Documents with warranty_end → warranty."""
        data: dict[str, Any] = {
            "document_type": "other",
            "warranty_end": "2028-01-01",
        }
        assert classify_from_extraction(data) == "warranty"

    def test_warranty_heuristic_by_type(self) -> None:
        data: dict[str, Any] = {
            "document_type": "other",
            "warranty_type": "manufacturer",
        }
        assert classify_from_extraction(data) == "warranty"

    def test_contract_heuristic(self) -> None:
        """Documents with start_date + end_date → contract."""
        data: dict[str, Any] = {
            "document_type": "other",
            "start_date": "2026-01-01",
            "end_date": "2027-01-01",
        }
        assert classify_from_extraction(data) == "contract"

    def test_contract_heuristic_with_renewal(self) -> None:
        data: dict[str, Any] = {
            "document_type": "other",
            "start_date": "2026-01-01",
            "renewal": "auto",
        }
        assert classify_from_extraction(data) == "contract"

    def test_invoice_heuristic(self) -> None:
        """Documents with invoice_number → invoice."""
        data: dict[str, Any] = {
            "document_type": "other",
            "invoice_number": "INV-001",
        }
        assert classify_from_extraction(data) == "invoice"

    def test_invoice_heuristic_due_date(self) -> None:
        data: dict[str, Any] = {
            "document_type": "other",
            "due_date": "2026-02-28",
        }
        assert classify_from_extraction(data) == "invoice"

    def test_payment_confirmation_heuristic(self) -> None:
        """Payment details + no real line items → payment_confirmation."""
        data: dict[str, Any] = {
            "document_type": "other",
            "card_last4": "7201",
            "authorization_code": "083646",
            "line_items": [],
        }
        assert classify_from_extraction(data) == "payment_confirmation"

    def test_payment_confirmation_phantom_items(self) -> None:
        """Phantom items (PURCHASE, SALE, etc.) don't count as real items."""
        data: dict[str, Any] = {
            "document_type": "other",
            "terminal_id": "T123",
            "line_items": [{"name": "PURCHASE"}, {"name": "TOTAL"}],
        }
        assert classify_from_extraction(data) == "payment_confirmation"

    def test_payment_with_real_items_is_receipt(self) -> None:
        """Payment details WITH real line items → receipt (not payment_confirmation)."""
        data: dict[str, Any] = {
            "document_type": "other",
            "card_last4": "7201",
            "line_items": [
                {"name": "Red Bull 250ml", "total_price": 1.99},
                {"name": "Bread", "total_price": 2.50},
            ],
        }
        assert classify_from_extraction(data) == "receipt"

    def test_default_is_receipt(self) -> None:
        """Unknown document type with no heuristic match → receipt."""
        data: dict[str, Any] = {
            "document_type": "other",
            "vendor": "Some Store",
            "total": 42.50,
        }
        assert classify_from_extraction(data) == "receipt"

    def test_empty_data(self) -> None:
        assert classify_from_extraction({}) == "receipt"

    def test_missing_document_type(self) -> None:
        data: dict[str, Any] = {"vendor": "Store", "total": 10.0}
        assert classify_from_extraction(data) == "receipt"


# ---------------------------------------------------------------------------
# Config prompt_mode tests
# ---------------------------------------------------------------------------


class TestPromptModeConfig:
    """Tests for the prompt_mode configuration field."""

    def test_default_is_specialized(self) -> None:
        from alibi.config import Config

        config = Config(_env_file=None)
        assert config.prompt_mode == "specialized"

    def test_can_set_universal(self) -> None:
        from alibi.config import Config

        config = Config(prompt_mode="universal", _env_file=None)
        assert config.prompt_mode == "universal"

    def test_env_var_override(self) -> None:
        from alibi.config import Config

        with patch.dict("os.environ", {"ALIBI_PROMPT_MODE": "universal"}):
            config = Config(_env_file=None)
            assert config.prompt_mode == "universal"


# ---------------------------------------------------------------------------
# Pipeline integration with universal mode
# ---------------------------------------------------------------------------


class TestPipelineUniversalMode:
    """Tests for pipeline type classification in universal mode."""

    def test_universal_mode_classifies_receipt(self) -> None:
        """Pipeline uses classify_from_extraction in universal mode."""
        from alibi.db.models import DocumentType
        from alibi.processing.pipeline import STR_TO_ARTIFACT_TYPE

        from alibi.extraction.prompts import classify_from_extraction

        data: dict[str, Any] = {
            "document_type": "invoice",
            "vendor": "ACME Corp",
            "total": 1000.00,
            "invoice_number": "INV-001",
        }
        result = classify_from_extraction(data)
        assert result == "invoice"
        assert STR_TO_ARTIFACT_TYPE[result] == DocumentType.INVOICE

    def test_universal_mode_classifies_statement(self) -> None:
        from alibi.db.models import DocumentType
        from alibi.processing.pipeline import STR_TO_ARTIFACT_TYPE

        data: dict[str, Any] = {
            "document_type": "statement",
            "institution": "Bank of Cyprus",
            "transactions": [{"date": "2026-01-15", "amount": 42.50}],
        }
        result = classify_from_extraction(data)
        assert result == "statement"
        assert STR_TO_ARTIFACT_TYPE[result] == DocumentType.STATEMENT

    def test_vision_receives_prompt_mode(self) -> None:
        """extract_from_image passes prompt_mode through two-stage pipeline."""
        from alibi.config import Config

        config = Config(prompt_mode="universal", _env_file=None)

        with (
            patch("alibi.extraction.vision.get_config", return_value=config),
            patch("alibi.extraction.ocr.get_config", return_value=config),
            patch("alibi.extraction.structurer.get_config", return_value=config),
            patch("alibi.extraction.ocr._call_ollama_ocr") as mock_ocr,
            patch("alibi.extraction.structurer._call_ollama_text") as mock_text,
        ):
            mock_ocr.return_value = {"response": "A" * 50 + " universal mode OCR text"}
            mock_text.return_value = {
                "response": '{"vendor": "test", "date": "2024-01-01", '
                '"total": 5, "currency": "EUR"}'
            }

            from PIL import Image
            import tempfile
            from pathlib import Path

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img = Image.new("RGB", (100, 100))
                img.save(tmp.name)
                tmp_path = Path(tmp.name)

            try:
                from alibi.extraction.vision import extract_from_image

                extract_from_image(tmp_path)
                # Verify the prompt sent to structure model contains
                # universal mode content (document_type field in prompt)
                prompt_arg = mock_text.call_args[0][2]
                assert "document_type" in prompt_arg
            finally:
                tmp_path.unlink(missing_ok=True)
