"""Tests for v2 extraction pipeline: schemas, prompts, and refiner integration."""

import json
import uuid
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alibi.db.models import DocumentType, RecordType, TaxType, UnitType
from alibi.extraction.prompts import (
    INVOICE_PROMPT,
    INVOICE_PROMPT_V2,
    PURCHASE_ATOMIZATION_PROMPT,
    RECEIPT_PROMPT,
    RECEIPT_PROMPT_V2,
    STATEMENT_PROMPT,
    WARRANTY_PROMPT,
    get_prompt_for_type,
    get_purchase_atomization_prompt,
)
from alibi.extraction.schemas import (
    INVOICE_SCHEMA,
    LINE_ITEM_SCHEMA,
    RECEIPT_SCHEMA,
    get_schema,
    validate_extraction,
)
from alibi.processing.pipeline import ARTIFACT_TO_RECORD_TYPE, ProcessingPipeline
from alibi.refiners.registry import get_refiner


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestLineItemSchema:
    """Tests for the LINE_ITEM_SCHEMA definition."""

    def test_schema_is_dict(self) -> None:
        assert isinstance(LINE_ITEM_SCHEMA, dict)

    def test_schema_type_is_object(self) -> None:
        assert LINE_ITEM_SCHEMA["type"] == "object"

    def test_name_is_required(self) -> None:
        assert "name" in LINE_ITEM_SCHEMA["required"]

    def test_has_unit_raw_property(self) -> None:
        assert "unit_raw" in LINE_ITEM_SCHEMA["properties"]

    def test_has_tax_rate_property(self) -> None:
        assert "tax_rate" in LINE_ITEM_SCHEMA["properties"]

    def test_has_tax_type_property(self) -> None:
        assert "tax_type" in LINE_ITEM_SCHEMA["properties"]

    def test_has_discount_property(self) -> None:
        assert "discount" in LINE_ITEM_SCHEMA["properties"]

    def test_has_brand_property(self) -> None:
        assert "brand" in LINE_ITEM_SCHEMA["properties"]

    def test_has_barcode_property(self) -> None:
        assert "barcode" in LINE_ITEM_SCHEMA["properties"]

    def test_has_category_property(self) -> None:
        assert "category" in LINE_ITEM_SCHEMA["properties"]

    def test_has_name_en_property(self) -> None:
        assert "name_en" in LINE_ITEM_SCHEMA["properties"]

    def test_has_quantity_property(self) -> None:
        assert "quantity" in LINE_ITEM_SCHEMA["properties"]

    def test_has_unit_price_property(self) -> None:
        assert "unit_price" in LINE_ITEM_SCHEMA["properties"]

    def test_has_total_price_property(self) -> None:
        assert "total_price" in LINE_ITEM_SCHEMA["properties"]


class TestReceiptSchema:
    """Tests for the RECEIPT_SCHEMA definition."""

    def test_schema_is_dict(self) -> None:
        assert isinstance(RECEIPT_SCHEMA, dict)

    def test_has_required_fields(self) -> None:
        required = RECEIPT_SCHEMA["required"]
        assert "total" in required
        assert "currency" in required
        assert "line_items" in required

    def test_has_vendor_property(self) -> None:
        assert "vendor" in RECEIPT_SCHEMA["properties"]

    def test_has_payment_method_property(self) -> None:
        assert "payment_method" in RECEIPT_SCHEMA["properties"]

    def test_has_language_property(self) -> None:
        assert "language" in RECEIPT_SCHEMA["properties"]

    def test_line_items_references_line_item_schema(self) -> None:
        items_prop = RECEIPT_SCHEMA["properties"]["line_items"]
        assert items_prop["type"] == "array"
        assert items_prop["items"] is LINE_ITEM_SCHEMA


class TestInvoiceSchema:
    """Tests for the INVOICE_SCHEMA definition."""

    def test_schema_is_dict(self) -> None:
        assert isinstance(INVOICE_SCHEMA, dict)

    def test_has_required_fields(self) -> None:
        required = INVOICE_SCHEMA["required"]
        assert "amount" in required
        assert "currency" in required
        assert "line_items" in required

    def test_has_issuer_property(self) -> None:
        assert "issuer" in INVOICE_SCHEMA["properties"]

    def test_has_invoice_number_property(self) -> None:
        assert "invoice_number" in INVOICE_SCHEMA["properties"]

    def test_has_issue_date_property(self) -> None:
        assert "issue_date" in INVOICE_SCHEMA["properties"]

    def test_has_due_date_property(self) -> None:
        assert "due_date" in INVOICE_SCHEMA["properties"]

    def test_has_language_property(self) -> None:
        assert "language" in INVOICE_SCHEMA["properties"]

    def test_line_items_references_line_item_schema(self) -> None:
        items_prop = INVOICE_SCHEMA["properties"]["line_items"]
        assert items_prop["type"] == "array"
        assert items_prop["items"] is LINE_ITEM_SCHEMA


class TestGetSchema:
    """Tests for the get_schema helper."""

    def test_get_receipt_schema(self) -> None:
        assert get_schema("receipt") is RECEIPT_SCHEMA

    def test_get_invoice_schema(self) -> None:
        assert get_schema("invoice") is INVOICE_SCHEMA

    def test_get_warranty_schema(self) -> None:
        from alibi.extraction.schemas import WARRANTY_SCHEMA

        assert get_schema("warranty") is WARRANTY_SCHEMA

    def test_get_contract_schema(self) -> None:
        from alibi.extraction.schemas import CONTRACT_SCHEMA

        assert get_schema("contract") is CONTRACT_SCHEMA

    def test_get_nonsense_returns_none(self) -> None:
        assert get_schema("nonexistent") is None


class TestValidateExtraction:
    """Tests for lightweight schema validation."""

    def test_valid_receipt(self) -> None:
        data: dict[str, Any] = {
            "total": 42.50,
            "currency": "EUR",
            "line_items": [{"name": "Milk", "quantity": 1, "total_price": 1.29}],
        }
        errors = validate_extraction(data, "receipt")
        assert errors == []

    def test_missing_required_field(self) -> None:
        data: dict[str, Any] = {
            "currency": "EUR",
            "line_items": [{"name": "Milk"}],
        }
        errors = validate_extraction(data, "receipt")
        assert any("total" in e for e in errors)

    def test_missing_line_item_name(self) -> None:
        data: dict[str, Any] = {
            "total": 10.0,
            "currency": "EUR",
            "line_items": [{"quantity": 2}],
        }
        errors = validate_extraction(data, "receipt")
        assert any("name" in e for e in errors)

    def test_line_items_not_a_list(self) -> None:
        data: dict[str, Any] = {
            "total": 10.0,
            "currency": "EUR",
            "line_items": "not a list",
        }
        errors = validate_extraction(data, "receipt")
        assert any("list" in e for e in errors)

    def test_line_item_not_a_dict(self) -> None:
        data: dict[str, Any] = {
            "total": 10.0,
            "currency": "EUR",
            "line_items": ["string_item"],
        }
        errors = validate_extraction(data, "receipt")
        assert any("object" in e for e in errors)

    def test_unknown_type_no_errors(self) -> None:
        data: dict[str, Any] = {"random": "data"}
        errors = validate_extraction(data, "unknown_doc_type")
        assert errors == []

    def test_valid_invoice(self) -> None:
        data: dict[str, Any] = {
            "amount": 500.00,
            "currency": "USD",
            "line_items": [{"name": "Consulting", "total_price": 500.00}],
        }
        errors = validate_extraction(data, "invoice")
        assert errors == []

    def test_invoice_missing_amount(self) -> None:
        data: dict[str, Any] = {
            "currency": "USD",
            "line_items": [{"name": "Item"}],
        }
        errors = validate_extraction(data, "invoice")
        assert any("amount" in e for e in errors)


# ---------------------------------------------------------------------------
# V2 prompt tests
# ---------------------------------------------------------------------------


class TestV2Prompts:
    """Tests for v2 prompt variants."""

    def test_receipt_v2_is_string(self) -> None:
        assert isinstance(RECEIPT_PROMPT_V2, str)

    def test_receipt_v2_has_language_instruction(self) -> None:
        assert "ISO 639-1" in RECEIPT_PROMPT_V2

    def test_receipt_v2_has_original_language_instruction(self) -> None:
        assert "ORIGINAL language" in RECEIPT_PROMPT_V2

    def test_receipt_v2_has_name_en(self) -> None:
        assert "name_en" in RECEIPT_PROMPT_V2

    def test_receipt_v2_has_unit_raw(self) -> None:
        assert "unit_raw" in RECEIPT_PROMPT_V2

    def test_receipt_v2_has_tax_rate(self) -> None:
        assert "tax_rate" in RECEIPT_PROMPT_V2

    def test_receipt_v2_has_tax_type(self) -> None:
        assert "tax_type" in RECEIPT_PROMPT_V2

    def test_receipt_v2_has_discount(self) -> None:
        assert "discount" in RECEIPT_PROMPT_V2

    def test_receipt_v2_has_brand(self) -> None:
        assert "brand" in RECEIPT_PROMPT_V2

    def test_receipt_v2_has_barcode(self) -> None:
        assert "barcode" in RECEIPT_PROMPT_V2

    def test_invoice_v2_is_string(self) -> None:
        assert isinstance(INVOICE_PROMPT_V2, str)

    def test_invoice_v2_has_language_instruction(self) -> None:
        assert "ISO 639-1" in INVOICE_PROMPT_V2

    def test_invoice_v2_has_name_en(self) -> None:
        assert "name_en" in INVOICE_PROMPT_V2

    def test_invoice_v2_has_issuer(self) -> None:
        assert "issuer" in INVOICE_PROMPT_V2

    def test_invoice_v2_has_issue_date(self) -> None:
        assert "issue_date" in INVOICE_PROMPT_V2

    def test_invoice_v2_has_due_date(self) -> None:
        assert "due_date" in INVOICE_PROMPT_V2

    def test_atomization_prompt_is_string(self) -> None:
        assert isinstance(PURCHASE_ATOMIZATION_PROMPT, str)

    def test_atomization_prompt_has_placeholder(self) -> None:
        assert "{receipt_json}" in PURCHASE_ATOMIZATION_PROMPT

    def test_atomization_prompt_has_all_fields(self) -> None:
        for field in [
            "name",
            "name_en",
            "unit_raw",
            "unit_price",
            "total_price",
            "tax_rate",
            "tax_type",
            "discount",
            "brand",
            "barcode",
            "category",
        ]:
            assert field in PURCHASE_ATOMIZATION_PROMPT


class TestGetPromptForTypeVersioning:
    """Tests for get_prompt_for_type with version parameter."""

    def test_default_version_is_v1(self) -> None:
        prompt = get_prompt_for_type("receipt")
        assert prompt is RECEIPT_PROMPT

    def test_v1_explicit(self) -> None:
        prompt = get_prompt_for_type("receipt", version=1)
        assert prompt is RECEIPT_PROMPT

    def test_v2_receipt(self) -> None:
        prompt = get_prompt_for_type("receipt", version=2)
        assert prompt is RECEIPT_PROMPT_V2

    def test_v2_invoice(self) -> None:
        prompt = get_prompt_for_type("invoice", version=2)
        assert prompt is INVOICE_PROMPT_V2

    def test_v2_statement_returns_v2(self) -> None:
        from alibi.extraction.prompts import STATEMENT_PROMPT_V2

        prompt = get_prompt_for_type("statement", version=2)
        assert prompt is STATEMENT_PROMPT_V2

    def test_v2_warranty_returns_v2(self) -> None:
        from alibi.extraction.prompts import WARRANTY_PROMPT_V2

        prompt = get_prompt_for_type("warranty", version=2)
        assert prompt is WARRANTY_PROMPT_V2

    def test_v2_contract_returns_v2(self) -> None:
        from alibi.extraction.prompts import CONTRACT_PROMPT_V2

        prompt = get_prompt_for_type("contract", version=2)
        assert prompt is CONTRACT_PROMPT_V2

    def test_v2_unknown_falls_back_to_receipt_v2(self) -> None:
        prompt = get_prompt_for_type("unknown", version=2)
        assert prompt is RECEIPT_PROMPT_V2


class TestGetPurchaseAtomizationPrompt:
    """Tests for the purchase atomization prompt formatter."""

    def test_inserts_receipt_json(self) -> None:
        receipt = json.dumps({"vendor": "Store", "total": 42.50})
        prompt = get_purchase_atomization_prompt(receipt)
        assert "Store" in prompt
        assert "42.5" in prompt

    def test_returns_string(self) -> None:
        prompt = get_purchase_atomization_prompt("{}")
        assert isinstance(prompt, str)


# ---------------------------------------------------------------------------
# Pipeline-refiner integration tests (mocked LLM extraction)
# ---------------------------------------------------------------------------


class TestArtifactToRecordTypeMapping:
    """Tests for the ARTIFACT_TO_RECORD_TYPE mapping."""

    def test_receipt_maps_to_purchase(self) -> None:
        assert ARTIFACT_TO_RECORD_TYPE[DocumentType.RECEIPT] == RecordType.PURCHASE

    def test_invoice_maps_to_invoice(self) -> None:
        assert ARTIFACT_TO_RECORD_TYPE[DocumentType.INVOICE] == RecordType.INVOICE

    def test_statement_maps_to_statement(self) -> None:
        assert ARTIFACT_TO_RECORD_TYPE[DocumentType.STATEMENT] == RecordType.STATEMENT

    def test_warranty_maps_to_warranty(self) -> None:
        assert ARTIFACT_TO_RECORD_TYPE[DocumentType.WARRANTY] == RecordType.WARRANTY

    def test_policy_maps_to_insurance(self) -> None:
        assert ARTIFACT_TO_RECORD_TYPE[DocumentType.POLICY] == RecordType.INSURANCE

    def test_other_not_in_mapping(self) -> None:
        assert DocumentType.OTHER not in ARTIFACT_TO_RECORD_TYPE


class TestPipelineRefinerRouting:
    """Tests for pipeline step 4.5: refiner routing."""

    def _make_pipeline(self) -> ProcessingPipeline:
        """Create a pipeline with mocked DB."""
        mock_db = MagicMock()
        mock_db.is_initialized.return_value = True
        pipeline = ProcessingPipeline(db=mock_db, space_id="test", user_id="tester")
        return pipeline

    def test_refine_extraction_uses_correct_refiner(self) -> None:
        pipeline = self._make_pipeline()
        raw_data: dict[str, Any] = {
            "vendor": "Test Store",
            "total": "42.50",
            "currency": "EUR",
            "line_items": [
                {"name": "Milk", "quantity": 1, "unit_price": 1.29, "total": 1.29}
            ],
        }
        result = pipeline._refine_extraction(raw_data, RecordType.PURCHASE)
        assert result["record_type"] == RecordType.PURCHASE
        assert "provenance" in result
        assert "id" in result

    def test_refine_extraction_normalizes_amounts(self) -> None:
        pipeline = self._make_pipeline()
        raw_data: dict[str, Any] = {
            "vendor": "Store",
            "amount": "100.50",
            "currency": "EUR",
        }
        result = pipeline._refine_extraction(raw_data, RecordType.INVOICE)
        assert result["record_type"] == RecordType.INVOICE
        assert isinstance(result["amount"], Decimal)
        assert result["amount"] == Decimal("100.50")

    def test_refine_extraction_builds_line_items(self) -> None:
        pipeline = self._make_pipeline()
        raw_data: dict[str, Any] = {
            "vendor": "Grocery Store",
            "total": 15.50,
            "currency": "EUR",
            "line_items": [
                {
                    "name": "Milch",
                    "quantity": 2,
                    "unit_price": 1.29,
                    "total": 2.58,
                    "tax_rate": 7,
                    "tax_type": "vat",
                    "brand": "BioMilch",
                    "barcode": "4012345678901",
                },
                {
                    "name": "Brot",
                    "quantity": 1,
                    "unit_price": 3.49,
                    "total": 3.49,
                },
            ],
        }
        result = pipeline._refine_extraction(
            raw_data, RecordType.PURCHASE, artifact_id="art-123"
        )
        line_items = result.get("line_items", [])
        assert len(line_items) == 2

        milk = line_items[0]
        assert milk["name"] == "Milch"
        assert milk["brand"] == "BioMilch"
        assert milk["barcode"] == "4012345678901"
        assert milk["artifact_id"] == "art-123"
        assert milk["tax_type"] == TaxType.VAT

    def test_refine_extraction_with_empty_data(self) -> None:
        pipeline = self._make_pipeline()
        raw_data: dict[str, Any] = {}
        result = pipeline._refine_extraction(raw_data, RecordType.PURCHASE)
        # Should still produce a valid refined dict
        assert "provenance" in result
        assert "id" in result

    def test_refiner_provenance_has_correct_processor(self) -> None:
        pipeline = self._make_pipeline()
        raw_data: dict[str, Any] = {
            "vendor": "Store",
            "amount": 10.0,
            "currency": "EUR",
        }
        result = pipeline._refine_extraction(raw_data, RecordType.PAYMENT)
        prov = result["provenance"]
        assert prov["source_type"] == "ai_refinement"
        assert prov["processor"] == "alibi:refiner:v2"

    def test_processing_result_has_v2_fields(self) -> None:
        """Test that ProcessingResult dataclass includes v2 fields."""
        from alibi.processing.pipeline import ProcessingResult

        result = ProcessingResult(
            success=True,
            file_path=Path("/tmp/test.jpg"),
            document_id="abc-123",
            refined_data={"id": "ref-1", "record_type": RecordType.PURCHASE},
            line_items=[{"name": "Item", "quantity": 1}],
            record_type=RecordType.PURCHASE,
        )
        assert result.refined_data is not None
        assert result.record_type == RecordType.PURCHASE
        assert len(result.line_items) == 1

    def test_processing_result_defaults(self) -> None:
        """Test ProcessingResult default values for v2 fields."""
        from alibi.processing.pipeline import ProcessingResult

        result = ProcessingResult(success=True, file_path=Path("/tmp/test.jpg"))
        assert result.refined_data is None
        assert result.line_items == []
        assert result.record_type is None


class TestRefinerRegistryIntegration:
    """Tests that the refiner registry works with all mapped types."""

    @pytest.mark.parametrize(
        "record_type",
        [
            RecordType.PURCHASE,
            RecordType.INVOICE,
            RecordType.STATEMENT,
            RecordType.WARRANTY,
            RecordType.INSURANCE,
            RecordType.PAYMENT,
        ],
    )
    def test_get_refiner_returns_instance(self, record_type: RecordType) -> None:
        refiner = get_refiner(record_type)
        assert refiner is not None
        assert hasattr(refiner, "refine")

    @pytest.mark.parametrize(
        "record_type",
        [
            RecordType.PURCHASE,
            RecordType.INVOICE,
            RecordType.PAYMENT,
        ],
    )
    def test_refiner_produces_provenance(self, record_type: RecordType) -> None:
        refiner = get_refiner(record_type)
        result = refiner.refine({"amount": "10.00", "currency": "EUR"})
        assert "provenance" in result
        assert result["provenance"]["source_type"] == "ai_refinement"

    def test_default_refiner_for_unmapped_type(self) -> None:
        # RecordType.CONTRACT is not in the registry, should get DefaultRefiner
        refiner = get_refiner(RecordType.CONTRACT)
        result = refiner.refine({"amount": "5.00"})
        assert "provenance" in result
