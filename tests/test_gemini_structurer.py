"""Tests for Gemini OCR text structuring (Stage 3 replacement)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.extraction.gemini_structurer import (
    BatchDocumentExtraction,
    ExtractionBatchResponse,
    GeminiExtractionError,
    InvoiceExtraction,
    LineItemExtraction,
    PaymentExtraction,
    ReceiptExtraction,
    StatementLineExtraction,
    _get_extraction_model,
    _parse_batch_response,
    _parse_response,
    structure_ocr_text_gemini,
    structure_ocr_texts_gemini,
)


# ===========================================================================
# Pydantic model tests
# ===========================================================================


class TestLineItemExtraction:
    def test_minimal_construction(self):
        item = LineItemExtraction()
        assert item.name is None
        assert item.quantity is None
        assert item.unit_raw is None
        assert item.unit_quantity is None
        assert item.unit_price is None
        assert item.total_price is None
        assert item.tax_rate is None
        assert item.tax_type is None
        assert item.discount is None
        assert item.brand is None
        assert item.barcode is None
        assert item.category is None

    def test_full_construction(self):
        item = LineItemExtraction(
            name="Milk 500ml",
            name_en="Milk 500ml",
            quantity=2.0,
            unit_raw="pcs",
            unit_quantity=0.5,
            unit_price=1.20,
            total_price=2.40,
            tax_rate=5.0,
            tax_type="vat",
            discount=0.10,
            brand="Arla",
            barcode="5901234123457",
            category="Dairy",
        )
        assert item.name == "Milk 500ml"
        assert item.quantity == 2.0
        assert item.unit_raw == "pcs"
        assert item.unit_quantity == 0.5
        assert item.unit_price == 1.20
        assert item.total_price == 2.40
        assert item.tax_rate == 5.0
        assert item.tax_type == "vat"
        assert item.discount == 0.10
        assert item.brand == "Arla"
        assert item.barcode == "5901234123457"
        assert item.category == "Dairy"

    def test_exclude_none_serialization(self):
        item = LineItemExtraction(name="Bread", total_price=2.50)
        data = item.model_dump(exclude_none=True)
        assert "name" in data
        assert "total_price" in data
        assert "quantity" not in data
        assert "brand" not in data
        assert "barcode" not in data

    def test_all_none_serialization(self):
        item = LineItemExtraction()
        data = item.model_dump(exclude_none=True)
        assert data == {}


class TestReceiptExtraction:
    def test_minimal_construction(self):
        r = ReceiptExtraction()
        assert r.vendor is None
        assert r.total is None
        assert r.line_items == []

    def test_full_construction(self):
        r = ReceiptExtraction(
            vendor="SuperMarket",
            vendor_address="123 Main St",
            vendor_phone="+357 22 123456",
            vendor_website="www.supermarket.cy",
            vendor_vat="CY10057000Y",
            vendor_tax_id="12057000A",
            date="2026-01-15",
            time="14:30:00",
            document_id="RCT-001",
            subtotal=10.00,
            tax=1.90,
            total=11.90,
            currency="EUR",
            payment_method="card",
            card_type="VISA",
            card_last4="1234",
            language="en",
            line_items=[LineItemExtraction(name="Bread", total_price=2.50)],
        )
        assert r.vendor == "SuperMarket"
        assert r.vendor_vat == "CY10057000Y"
        assert r.date == "2026-01-15"
        assert r.total == 11.90
        assert r.currency == "EUR"
        assert len(r.line_items) == 1
        assert r.line_items[0].name == "Bread"

    def test_exclude_none_serialization(self):
        r = ReceiptExtraction(vendor="TestVendor", total=5.00, currency="EUR")
        data = r.model_dump(exclude_none=True)
        assert data["vendor"] == "TestVendor"
        assert data["total"] == 5.00
        assert data["currency"] == "EUR"
        assert "vendor_address" not in data
        assert "card_last4" not in data

    def test_line_items_default_empty_list(self):
        r = ReceiptExtraction(vendor="Store")
        data = r.model_dump(exclude_none=True)
        assert data["line_items"] == []

    def test_nested_line_items_serialization(self):
        r = ReceiptExtraction(
            vendor="Store",
            line_items=[
                LineItemExtraction(name="Apple", total_price=0.50),
                LineItemExtraction(name="Butter", brand="Lurpak", total_price=2.20),
            ],
        )
        data = r.model_dump(exclude_none=True)
        items = data["line_items"]
        assert len(items) == 2
        assert items[0]["name"] == "Apple"
        assert "brand" not in items[0]
        assert items[1]["brand"] == "Lurpak"


class TestInvoiceExtraction:
    def test_minimal_construction(self):
        inv = InvoiceExtraction()
        assert inv.issuer is None
        assert inv.invoice_number is None
        assert inv.amount is None
        assert inv.line_items == []

    def test_full_construction(self):
        inv = InvoiceExtraction(
            issuer="Acme Corp",
            issuer_address="456 Business Ave",
            issuer_vat="CY20000001Z",
            issuer_tax_id="99000001A",
            customer="Customer Ltd",
            invoice_number="INV-2026-001",
            issue_date="2026-01-10",
            due_date="2026-02-10",
            subtotal=100.00,
            tax=19.00,
            amount=119.00,
            currency="EUR",
            payment_terms="Net 30",
            language="en",
        )
        assert inv.issuer == "Acme Corp"
        assert inv.invoice_number == "INV-2026-001"
        assert inv.amount == 119.00
        assert inv.due_date == "2026-02-10"

    def test_exclude_none_serialization(self):
        inv = InvoiceExtraction(issuer="Corp", amount=500.00)
        data = inv.model_dump(exclude_none=True)
        assert "issuer" in data
        assert "amount" in data
        assert "customer" not in data
        assert "issuer_address" not in data


class TestPaymentExtraction:
    def test_minimal_construction(self):
        p = PaymentExtraction()
        assert p.vendor is None
        assert p.total is None
        assert p.authorization_code is None

    def test_full_construction(self):
        p = PaymentExtraction(
            vendor="Gas Station",
            vendor_vat="CY10370773Q",
            date="2026-01-20",
            time="09:15",
            document_id="TXN-9988",
            total=45.00,
            currency="EUR",
            payment_method="card",
            card_type="Mastercard",
            card_last4="5678",
            authorization_code="AUTH123456",
            language="el",
        )
        assert p.vendor == "Gas Station"
        assert p.total == 45.00
        assert p.authorization_code == "AUTH123456"
        assert p.card_last4 == "5678"

    def test_exclude_none_serialization(self):
        p = PaymentExtraction(vendor="Shop", total=10.00)
        data = p.model_dump(exclude_none=True)
        assert "vendor" in data
        assert "total" in data
        assert "authorization_code" not in data
        assert "vendor_address" not in data

    def test_no_line_items_field(self):
        p = PaymentExtraction()
        assert not hasattr(p, "line_items")


class TestExtractionBatchResponse:
    def test_empty_documents(self):
        resp = ExtractionBatchResponse(documents=[])
        assert resp.documents == []

    def test_multiple_documents(self):
        resp = ExtractionBatchResponse(
            documents=[
                BatchDocumentExtraction(
                    idx=1, document_type="receipt", extraction={"vendor": "Store A"}
                ),
                BatchDocumentExtraction(
                    idx=2, document_type="invoice", extraction={"issuer": "Corp B"}
                ),
            ]
        )
        assert len(resp.documents) == 2
        assert resp.documents[0].idx == 1
        assert resp.documents[0].extraction["vendor"] == "Store A"
        assert resp.documents[1].document_type == "invoice"

    def test_batch_document_extraction_defaults(self):
        doc = BatchDocumentExtraction(idx=3)
        assert doc.idx == 3
        assert doc.document_type is None
        assert doc.extraction == {}


# ===========================================================================
# _get_extraction_model tests
# ===========================================================================


class TestGetExtractionModel:
    def test_receipt_type(self):
        model = _get_extraction_model("receipt")
        assert model is ReceiptExtraction

    def test_invoice_type(self):
        model = _get_extraction_model("invoice")
        assert model is InvoiceExtraction

    def test_payment_confirmation_type(self):
        model = _get_extraction_model("payment_confirmation")
        assert model is PaymentExtraction

    def test_statement_type(self):
        model = _get_extraction_model("statement")
        assert model is StatementLineExtraction

    def test_unknown_type_defaults_to_receipt(self):
        model = _get_extraction_model("unknown_type")
        assert model is ReceiptExtraction

    def test_empty_string_defaults_to_receipt(self):
        model = _get_extraction_model("")
        assert model is ReceiptExtraction

    def test_warranty_defaults_to_receipt(self):
        model = _get_extraction_model("warranty")
        assert model is ReceiptExtraction

    def test_contract_defaults_to_receipt(self):
        model = _get_extraction_model("contract")
        assert model is ReceiptExtraction


# ===========================================================================
# _parse_response tests
# ===========================================================================


class TestParseResponse:
    def test_structured_parsed_response(self):
        extraction = ReceiptExtraction(vendor="TestVendor", total=10.00)
        mock_response = MagicMock()
        mock_response.parsed = extraction

        result = _parse_response(mock_response, ReceiptExtraction)

        assert result["vendor"] == "TestVendor"
        assert result["total"] == 10.00

    def test_parsed_response_excludes_none(self):
        extraction = ReceiptExtraction(vendor="Store", total=5.00)
        mock_response = MagicMock()
        mock_response.parsed = extraction

        result = _parse_response(mock_response, ReceiptExtraction)

        assert "vendor" in result
        assert "total" in result
        assert "vendor_address" not in result

    def test_json_fallback_when_parsed_none(self):
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = '{"vendor": "FallbackStore", "total": 20.00}'

        result = _parse_response(mock_response, ReceiptExtraction)

        assert result["vendor"] == "FallbackStore"
        assert result["total"] == 20.00

    def test_json_fallback_when_parsed_wrong_type(self):
        mock_response = MagicMock()
        mock_response.parsed = "not a pydantic model"
        mock_response.text = '{"vendor": "JsonVendor", "total": 15.00}'

        result = _parse_response(mock_response, ReceiptExtraction)

        assert result["vendor"] == "JsonVendor"

    def test_invalid_json_raises_error(self):
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = "this is not json at all"

        with pytest.raises(GeminiExtractionError, match="Failed to parse"):
            _parse_response(mock_response, ReceiptExtraction)

    def test_empty_text_raises_error(self):
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = ""

        with pytest.raises(GeminiExtractionError):
            _parse_response(mock_response, ReceiptExtraction)

    def test_none_text_raises_error(self):
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = None

        with pytest.raises(GeminiExtractionError):
            _parse_response(mock_response, ReceiptExtraction)

    def test_invoice_model_structured_parsing(self):
        extraction = InvoiceExtraction(issuer="Corp", amount=500.00)
        mock_response = MagicMock()
        mock_response.parsed = extraction

        result = _parse_response(mock_response, InvoiceExtraction)

        assert result["issuer"] == "Corp"
        assert result["amount"] == 500.00

    def test_payment_model_structured_parsing(self):
        extraction = PaymentExtraction(vendor="Gas Station", total=45.00)
        mock_response = MagicMock()
        mock_response.parsed = extraction

        result = _parse_response(mock_response, PaymentExtraction)

        assert result["vendor"] == "Gas Station"
        assert result["total"] == 45.00


# ===========================================================================
# _parse_batch_response tests
# ===========================================================================


class TestParseBatchResponse:
    def test_structured_batch_parsed(self):
        batch = ExtractionBatchResponse(
            documents=[
                BatchDocumentExtraction(
                    idx=1,
                    document_type="receipt",
                    extraction={"vendor": "Store A", "total": 10.00},
                ),
                BatchDocumentExtraction(
                    idx=2,
                    document_type="invoice",
                    extraction={"issuer": "Corp B", "amount": 500.00},
                ),
            ]
        )
        mock_response = MagicMock()
        mock_response.parsed = batch

        documents = [
            {"raw_text": "receipt text", "doc_type": "receipt"},
            {"raw_text": "invoice text", "doc_type": "invoice"},
        ]
        results = _parse_batch_response(mock_response, documents)

        assert len(results) == 2
        assert results[0]["vendor"] == "Store A"
        assert results[0]["_pipeline"] == "gemini_batch_extraction"
        assert results[1]["issuer"] == "Corp B"
        assert results[1]["_pipeline"] == "gemini_batch_extraction"

    def test_batch_idx_is_one_based(self):
        batch = ExtractionBatchResponse(
            documents=[
                BatchDocumentExtraction(idx=1, extraction={"vendor": "First"}),
                BatchDocumentExtraction(idx=2, extraction={"vendor": "Second"}),
            ]
        )
        mock_response = MagicMock()
        mock_response.parsed = batch

        documents = [
            {"raw_text": "text1", "doc_type": "receipt"},
            {"raw_text": "text2", "doc_type": "receipt"},
        ]
        results = _parse_batch_response(mock_response, documents)

        assert results[0]["vendor"] == "First"
        assert results[1]["vendor"] == "Second"

    def test_missing_documents_get_error_placeholder(self):
        batch = ExtractionBatchResponse(
            documents=[
                BatchDocumentExtraction(idx=1, extraction={"vendor": "Only One"})
            ]
        )
        mock_response = MagicMock()
        mock_response.parsed = batch

        documents = [
            {"raw_text": "text1", "doc_type": "receipt"},
            {"raw_text": "text2", "doc_type": "receipt"},
        ]
        results = _parse_batch_response(mock_response, documents)

        assert results[0]["vendor"] == "Only One"
        assert results[1]["_error"] == "not_returned"

    def test_json_fallback_when_parsed_none(self):
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = (
            '{"documents": ['
            '{"idx": 1, "extraction": {"vendor": "FromJson", "total": 5.00}}'
            "]}"
        )

        documents = [{"raw_text": "text", "doc_type": "receipt"}]
        results = _parse_batch_response(mock_response, documents)

        assert results[0]["vendor"] == "FromJson"
        assert results[0]["_pipeline"] == "gemini_batch_extraction"

    def test_json_fallback_idx_mapping(self):
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = (
            '{"documents": ['
            '{"idx": 2, "extraction": {"vendor": "Second"}},'
            '{"idx": 1, "extraction": {"vendor": "First"}}'
            "]}"
        )

        documents = [
            {"raw_text": "text1", "doc_type": "receipt"},
            {"raw_text": "text2", "doc_type": "receipt"},
        ]
        results = _parse_batch_response(mock_response, documents)

        assert results[0]["vendor"] == "First"
        assert results[1]["vendor"] == "Second"

    def test_invalid_json_returns_error_placeholders(self):
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = "not valid json"

        documents = [
            {"raw_text": "text1", "doc_type": "receipt"},
            {"raw_text": "text2", "doc_type": "receipt"},
        ]
        results = _parse_batch_response(mock_response, documents)

        assert len(results) == 2
        assert all(r.get("_error") == "not_returned" for r in results)

    def test_empty_documents_list(self):
        mock_response = MagicMock()
        mock_response.parsed = ExtractionBatchResponse(documents=[])

        results = _parse_batch_response(mock_response, [])
        assert results == []

    def test_out_of_range_idx_ignored(self):
        batch = ExtractionBatchResponse(
            documents=[
                BatchDocumentExtraction(idx=99, extraction={"vendor": "OOB"}),
                BatchDocumentExtraction(idx=1, extraction={"vendor": "Valid"}),
            ]
        )
        mock_response = MagicMock()
        mock_response.parsed = batch

        documents = [{"raw_text": "text", "doc_type": "receipt"}]
        results = _parse_batch_response(mock_response, documents)

        assert results[0]["vendor"] == "Valid"


# ===========================================================================
# structure_ocr_text_gemini tests
# ===========================================================================


class TestStructureOcrTextGemini:
    @patch("google.genai.Client")
    def test_successful_structured_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ReceiptExtraction(
            vendor="TestMart",
            total=15.50,
            currency="EUR",
            date="2026-01-15",
        )
        mock_client.models.generate_content.return_value = mock_response

        result = structure_ocr_text_gemini(
            "TestMart\nTotal 15.50 EUR",
            doc_type="receipt",
            api_key="test-key",
        )

        assert result["vendor"] == "TestMart"
        assert result["total"] == 15.50
        assert result["currency"] == "EUR"
        assert result["_pipeline"] == "gemini_extraction"

    @patch("google.genai.Client")
    def test_pipeline_tag_always_present(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ReceiptExtraction(vendor="Store")
        mock_client.models.generate_content.return_value = mock_response

        result = structure_ocr_text_gemini("text", api_key="test-key")

        assert "_pipeline" in result
        assert result["_pipeline"] == "gemini_extraction"

    @patch("google.genai.Client")
    def test_json_fallback_when_parsed_none(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = '{"vendor": "JsonStore", "total": 9.99, "currency": "EUR"}'
        mock_client.models.generate_content.return_value = mock_response

        result = structure_ocr_text_gemini(
            "some ocr text", doc_type="receipt", api_key="test-key"
        )

        assert result["vendor"] == "JsonStore"
        assert result["total"] == 9.99

    @patch("google.genai.Client")
    def test_invoice_doc_type_uses_invoice_model(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = InvoiceExtraction(issuer="Invoicer Corp", amount=250.00)
        mock_client.models.generate_content.return_value = mock_response

        result = structure_ocr_text_gemini(
            "invoice text", doc_type="invoice", api_key="test-key"
        )

        assert result["issuer"] == "Invoicer Corp"
        assert result["amount"] == 250.00

    @patch("google.genai.Client")
    def test_payment_doc_type_uses_payment_model(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = PaymentExtraction(vendor="Gas Co", total=50.00)
        mock_client.models.generate_content.return_value = mock_response

        result = structure_ocr_text_gemini(
            "payment text", doc_type="payment_confirmation", api_key="test-key"
        )

        assert result["vendor"] == "Gas Co"
        assert result["total"] == 50.00

    def test_missing_api_key_raises_error(self):
        with patch(
            "alibi.extraction.gemini_structurer._get_api_key", return_value=None
        ):
            with pytest.raises(GeminiExtractionError, match="ALIBI_GEMINI_API_KEY"):
                structure_ocr_text_gemini("text", doc_type="receipt")

    @patch("google.genai.Client")
    def test_api_exception_raises_gemini_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("API down")

        with pytest.raises(GeminiExtractionError, match="Gemini extraction failed"):
            structure_ocr_text_gemini("text", api_key="test-key")

    @patch("google.genai.Client")
    def test_uses_specified_model(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ReceiptExtraction()
        mock_client.models.generate_content.return_value = mock_response

        structure_ocr_text_gemini("text", api_key="test-key", model="gemini-1.5-pro")

        call_args = mock_client.models.generate_content.call_args
        model_arg = call_args.kwargs.get("model") or call_args.args[0]
        assert model_arg == "gemini-1.5-pro"

    @patch("google.genai.Client")
    def test_uses_default_model_from_config(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ReceiptExtraction()
        mock_client.models.generate_content.return_value = mock_response

        with patch(
            "alibi.extraction.gemini_structurer._get_model",
            return_value="gemini-2.5-flash",
        ):
            structure_ocr_text_gemini("text", api_key="test-key")

        call_args = mock_client.models.generate_content.call_args
        model_arg = call_args.kwargs.get("model") or call_args.args[0]
        assert model_arg == "gemini-2.5-flash"

    @patch("google.genai.Client")
    def test_prompt_contains_ocr_text(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ReceiptExtraction()
        mock_client.models.generate_content.return_value = mock_response

        structure_ocr_text_gemini(
            "UNIQUE OCR CONTENT 12345", doc_type="receipt", api_key="test-key"
        )

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args.args[1]
        assert "UNIQUE OCR CONTENT 12345" in contents

    @patch("google.genai.Client")
    def test_prompt_contains_doc_type(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = InvoiceExtraction()
        mock_client.models.generate_content.return_value = mock_response

        structure_ocr_text_gemini("text", doc_type="invoice", api_key="test-key")

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args.args[1]
        assert "invoice" in contents

    @patch("google.genai.Client")
    def test_api_key_passed_to_client(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ReceiptExtraction()
        mock_client.models.generate_content.return_value = mock_response

        structure_ocr_text_gemini("text", api_key="my-secret-key")

        mock_client_cls.assert_called_once_with(api_key="my-secret-key")

    @patch("google.genai.Client")
    def test_falls_back_to_config_api_key(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ReceiptExtraction()
        mock_client.models.generate_content.return_value = mock_response

        with patch(
            "alibi.extraction.gemini_structurer._get_api_key",
            return_value="config-key",
        ):
            structure_ocr_text_gemini("text")

        mock_client_cls.assert_called_once_with(api_key="config-key")

    @patch("google.genai.Client")
    def test_line_items_in_structured_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ReceiptExtraction(
            vendor="Store",
            total=5.00,
            line_items=[
                LineItemExtraction(name="Apple", total_price=2.00),
                LineItemExtraction(name="Butter", brand="Lurpak", total_price=3.00),
            ],
        )
        mock_client.models.generate_content.return_value = mock_response

        result = structure_ocr_text_gemini("text", api_key="test-key")

        assert len(result["line_items"]) == 2
        assert result["line_items"][0]["name"] == "Apple"
        assert result["line_items"][1]["brand"] == "Lurpak"


# ===========================================================================
# structure_ocr_texts_gemini (batch) tests
# ===========================================================================


class TestStructureOcrTextsGemini:
    def test_empty_input_returns_empty_list(self):
        result = structure_ocr_texts_gemini([], api_key="test-key")
        assert result == []

    def test_missing_api_key_raises_error(self):
        with patch(
            "alibi.extraction.gemini_structurer._get_api_key", return_value=None
        ):
            with pytest.raises(GeminiExtractionError, match="ALIBI_GEMINI_API_KEY"):
                structure_ocr_texts_gemini(
                    [{"raw_text": "text", "doc_type": "receipt"}]
                )

    @patch("google.genai.Client")
    def test_batch_success_all_documents(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ExtractionBatchResponse(
            documents=[
                BatchDocumentExtraction(
                    idx=1,
                    document_type="receipt",
                    extraction={"vendor": "Store A", "total": 10.00},
                ),
                BatchDocumentExtraction(
                    idx=2,
                    document_type="invoice",
                    extraction={"issuer": "Corp B", "amount": 500.00},
                ),
            ]
        )
        mock_client.models.generate_content.return_value = mock_response

        documents = [
            {"raw_text": "receipt text", "doc_type": "receipt"},
            {"raw_text": "invoice text", "doc_type": "invoice"},
        ]
        results = structure_ocr_texts_gemini(documents, api_key="test-key")

        assert len(results) == 2
        assert results[0]["vendor"] == "Store A"
        assert results[0]["_pipeline"] == "gemini_batch_extraction"
        assert results[1]["issuer"] == "Corp B"

    @patch("google.genai.Client")
    def test_partial_failure_some_documents_have_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ExtractionBatchResponse(
            documents=[
                BatchDocumentExtraction(
                    idx=1,
                    extraction={"vendor": "Only First"},
                ),
                # idx=2 intentionally missing
            ]
        )
        mock_client.models.generate_content.return_value = mock_response

        documents = [
            {"raw_text": "text1", "doc_type": "receipt"},
            {"raw_text": "text2", "doc_type": "receipt"},
        ]
        results = structure_ocr_texts_gemini(documents, api_key="test-key")

        assert len(results) == 2
        assert results[0]["vendor"] == "Only First"
        assert results[1].get("_error") == "not_returned"

    @patch("google.genai.Client")
    def test_single_document_batch(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ExtractionBatchResponse(
            documents=[
                BatchDocumentExtraction(
                    idx=1, extraction={"vendor": "Solo Store", "total": 3.00}
                )
            ]
        )
        mock_client.models.generate_content.return_value = mock_response

        results = structure_ocr_texts_gemini(
            [{"raw_text": "solo text", "doc_type": "receipt"}],
            api_key="test-key",
        )

        assert len(results) == 1
        assert results[0]["vendor"] == "Solo Store"

    @patch("google.genai.Client")
    def test_api_exception_raises_gemini_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("Batch API down")

        with pytest.raises(GeminiExtractionError, match="Gemini batch extraction"):
            structure_ocr_texts_gemini(
                [{"raw_text": "text", "doc_type": "receipt"}],
                api_key="test-key",
            )

    @patch("google.genai.Client")
    def test_prompt_contains_all_document_texts(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ExtractionBatchResponse(documents=[])
        mock_client.models.generate_content.return_value = mock_response

        structure_ocr_texts_gemini(
            [
                {"raw_text": "RECEIPT TEXT UNIQUE_A", "doc_type": "receipt"},
                {"raw_text": "INVOICE TEXT UNIQUE_B", "doc_type": "invoice"},
            ],
            api_key="test-key",
        )

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args.args[1]
        assert "RECEIPT TEXT UNIQUE_A" in contents
        assert "INVOICE TEXT UNIQUE_B" in contents

    @patch("google.genai.Client")
    def test_prompt_contains_document_indices(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ExtractionBatchResponse(documents=[])
        mock_client.models.generate_content.return_value = mock_response

        structure_ocr_texts_gemini(
            [
                {"raw_text": "text1", "doc_type": "receipt"},
                {"raw_text": "text2", "doc_type": "invoice"},
            ],
            api_key="test-key",
        )

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args.args[1]
        assert "Document 1" in contents
        assert "Document 2" in contents

    @patch("google.genai.Client")
    def test_uses_specified_model(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ExtractionBatchResponse(documents=[])
        mock_client.models.generate_content.return_value = mock_response

        structure_ocr_texts_gemini(
            [{"raw_text": "text", "doc_type": "receipt"}],
            api_key="test-key",
            model="gemini-1.5-pro",
        )

        call_args = mock_client.models.generate_content.call_args
        model_arg = call_args.kwargs.get("model") or call_args.args[0]
        assert model_arg == "gemini-1.5-pro"

    @patch("google.genai.Client")
    def test_json_fallback_batch_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = (
            '{"documents": [{"idx": 1, "extraction": {"vendor": "FallbackVendor"}}]}'
        )
        mock_client.models.generate_content.return_value = mock_response

        results = structure_ocr_texts_gemini(
            [{"raw_text": "text", "doc_type": "receipt"}],
            api_key="test-key",
        )

        assert results[0]["vendor"] == "FallbackVendor"

    @patch("google.genai.Client")
    def test_doc_type_in_prompt_for_each_document(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ExtractionBatchResponse(documents=[])
        mock_client.models.generate_content.return_value = mock_response

        structure_ocr_texts_gemini(
            [
                {"raw_text": "text1", "doc_type": "receipt"},
                {"raw_text": "text2", "doc_type": "payment_confirmation"},
            ],
            api_key="test-key",
        )

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args.args[1]
        assert "receipt" in contents
        assert "payment_confirmation" in contents

    @patch("google.genai.Client")
    def test_missing_doc_type_defaults_to_receipt(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ExtractionBatchResponse(documents=[])
        mock_client.models.generate_content.return_value = mock_response

        structure_ocr_texts_gemini(
            [{"raw_text": "text without doc_type"}],
            api_key="test-key",
        )

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args.args[1]
        assert "receipt" in contents


# ===========================================================================
# Integration: structure_ocr_text() routing tests
# ===========================================================================


class TestStructurerIntegration:
    @patch("google.genai.Client")
    def test_routes_to_gemini_when_enabled(self, mock_client_cls):
        from alibi.config import reset_config
        from alibi.extraction.structurer import structure_ocr_text

        reset_config()
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ReceiptExtraction(
            vendor="GeminiVendor", total=25.00, currency="EUR"
        )
        mock_client.models.generate_content.return_value = mock_response

        with (
            patch("alibi.extraction.structurer.get_config") as mock_cfg,
            patch(
                "alibi.extraction.gemini_structurer._get_api_key",
                return_value="test-key",
            ),
            patch(
                "alibi.extraction.gemini_structurer._get_model",
                return_value="gemini-2.5-flash",
            ),
        ):
            mock_cfg.return_value.gemini_extraction_enabled = True
            mock_cfg.return_value.gemini_api_key = "test-key"
            mock_cfg.return_value.gemini_extraction_model = "gemini-2.5-flash"
            result = structure_ocr_text("receipt text", doc_type="receipt")

        assert result["vendor"] == "GeminiVendor"
        assert result["_pipeline"] == "gemini_extraction"
        reset_config()

    def test_skips_gemini_when_disabled(self):
        from alibi.config import reset_config
        from alibi.extraction.structurer import structure_ocr_text

        reset_config()
        with (
            patch("alibi.extraction.structurer.get_config") as mock_cfg,
            patch("alibi.extraction.structurer._call_ollama_text") as mock_ollama,
        ):
            mock_cfg.return_value.gemini_extraction_enabled = False
            mock_cfg.return_value.ollama_structure_model = "qwen3:8b"
            mock_cfg.return_value.ollama_url = "http://test:11434"
            mock_cfg.return_value.prompt_mode = "specialized"
            mock_ollama.return_value = {"response": '{"vendor": "OllamaVendor"}'}

            result = structure_ocr_text("text", doc_type="receipt")

        assert result["vendor"] == "OllamaVendor"
        mock_ollama.assert_called_once()
        reset_config()

    def test_falls_back_to_ollama_on_gemini_error(self):
        from alibi.config import reset_config
        from alibi.extraction.structurer import structure_ocr_text

        reset_config()
        with (
            patch("alibi.extraction.structurer.get_config") as mock_cfg,
            patch(
                "alibi.extraction.gemini_structurer.structure_ocr_text_gemini",
                side_effect=GeminiExtractionError("Gemini failed"),
            ),
            patch("alibi.extraction.structurer._call_ollama_text") as mock_ollama,
        ):
            mock_cfg.return_value.gemini_extraction_enabled = True
            mock_cfg.return_value.gemini_api_key = "test-key"
            mock_cfg.return_value.gemini_extraction_model = "gemini-2.5-flash"
            mock_cfg.return_value.ollama_structure_model = "qwen3:8b"
            mock_cfg.return_value.ollama_url = "http://test:11434"
            mock_cfg.return_value.prompt_mode = "specialized"
            mock_ollama.return_value = {"response": '{"vendor": "OllamaFallback"}'}

            result = structure_ocr_text("text", doc_type="receipt")

        assert result["vendor"] == "OllamaFallback"
        mock_ollama.assert_called_once()
        reset_config()

    def test_emphasis_prompt_bypasses_gemini(self):
        from alibi.config import reset_config
        from alibi.extraction.structurer import structure_ocr_text

        reset_config()
        with (
            patch("alibi.extraction.structurer.get_config") as mock_cfg,
            patch("alibi.extraction.structurer._call_ollama_text") as mock_ollama,
            patch(
                "alibi.extraction.gemini_structurer.structure_ocr_text_gemini"
            ) as mock_gemini,
        ):
            mock_cfg.return_value.gemini_extraction_enabled = True
            mock_cfg.return_value.gemini_api_key = "test-key"
            mock_cfg.return_value.ollama_structure_model = "qwen3:8b"
            mock_cfg.return_value.ollama_url = "http://test:11434"
            mock_cfg.return_value.prompt_mode = "specialized"
            mock_ollama.return_value = {"response": '{"vendor": "EmphasisResult"}'}

            structure_ocr_text(
                "text",
                doc_type="receipt",
                emphasis_prompt="Corrected prompt",
            )

        mock_gemini.assert_not_called()
        mock_ollama.assert_called_once()
        reset_config()

    def test_falls_back_to_ollama_on_unexpected_exception(self):
        from alibi.config import reset_config
        from alibi.extraction.structurer import structure_ocr_text

        reset_config()
        with (
            patch("alibi.extraction.structurer.get_config") as mock_cfg,
            patch(
                "alibi.extraction.gemini_structurer.structure_ocr_text_gemini",
                side_effect=RuntimeError("Unexpected crash"),
            ),
            patch("alibi.extraction.structurer._call_ollama_text") as mock_ollama,
        ):
            mock_cfg.return_value.gemini_extraction_enabled = True
            mock_cfg.return_value.gemini_api_key = "test-key"
            mock_cfg.return_value.gemini_extraction_model = "gemini-2.5-flash"
            mock_cfg.return_value.ollama_structure_model = "qwen3:8b"
            mock_cfg.return_value.ollama_url = "http://test:11434"
            mock_cfg.return_value.prompt_mode = "specialized"
            mock_ollama.return_value = {"response": '{"vendor": "OllamaAfterCrash"}'}

            result = structure_ocr_text("text", doc_type="receipt")

        assert result["vendor"] == "OllamaAfterCrash"
        mock_ollama.assert_called_once()
        reset_config()


# ===========================================================================
# extract_from_image_gemini tests
# ===========================================================================


@pytest.fixture
def tmp_jpeg(tmp_path):
    """Minimal valid JPEG bytes written to a tmp file."""
    # Smallest possible valid JPEG (1x1 white pixel)
    jpeg_bytes = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),\x01\x00\x00\x00"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4"
        b"\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xff\xd9"
    )
    p = tmp_path / "receipt.jpg"
    p.write_bytes(jpeg_bytes)
    return p


@pytest.fixture
def tmp_png(tmp_path):
    """Minimal 1x1 PNG file."""
    # Minimal valid PNG (1×1 red pixel)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    p = tmp_path / "invoice.png"
    p.write_bytes(png_bytes)
    return p


class TestExtractFromImageGemini:
    @patch("google.genai.Client")
    def test_success_structured_response(self, mock_client_cls, tmp_jpeg):
        """Parsed Pydantic model returned — result dict and _pipeline tag set."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ReceiptExtraction(
            vendor="VisionMart",
            total=42.00,
            currency="EUR",
            date="2026-02-24",
        )
        mock_client.models.generate_content.return_value = mock_response

        from alibi.extraction.gemini_structurer import extract_from_image_gemini

        result = extract_from_image_gemini(
            str(tmp_jpeg), doc_type="receipt", api_key="test-key"
        )

        assert result["vendor"] == "VisionMart"
        assert result["total"] == 42.00
        assert result["currency"] == "EUR"
        assert result["_pipeline"] == "gemini_vision"

    @patch("google.genai.Client")
    def test_json_fallback_when_parsed_none(self, mock_client_cls, tmp_jpeg):
        """When response.parsed is None, valid JSON in response.text is used."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = '{"vendor": "JsonVisionStore", "total": 7.50}'
        mock_client.models.generate_content.return_value = mock_response

        from alibi.extraction.gemini_structurer import extract_from_image_gemini

        result = extract_from_image_gemini(
            str(tmp_jpeg), doc_type="receipt", api_key="test-key"
        )

        assert result["vendor"] == "JsonVisionStore"
        assert result["total"] == 7.50
        assert result["_pipeline"] == "gemini_vision"

    def test_missing_api_key_raises_error(self, tmp_jpeg):
        """No API key configured → GeminiExtractionError."""
        from alibi.extraction.gemini_structurer import (
            GeminiExtractionError,
            extract_from_image_gemini,
        )

        with patch(
            "alibi.extraction.gemini_structurer._get_api_key", return_value=None
        ):
            with pytest.raises(GeminiExtractionError, match="ALIBI_GEMINI_API_KEY"):
                extract_from_image_gemini(str(tmp_jpeg), api_key=None)

    def test_image_file_not_found_raises_error(self, tmp_path):
        """Non-existent image path → GeminiExtractionError."""
        from alibi.extraction.gemini_structurer import (
            GeminiExtractionError,
            extract_from_image_gemini,
        )

        missing = str(tmp_path / "does_not_exist.jpg")
        with pytest.raises(GeminiExtractionError, match="Image file not found"):
            extract_from_image_gemini(missing, api_key="test-key")

    @patch("google.genai.Client")
    def test_api_failure_raises_gemini_error(self, mock_client_cls, tmp_jpeg):
        """API exception wrapped in GeminiExtractionError."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("Vision API down")

        from alibi.extraction.gemini_structurer import (
            GeminiExtractionError,
            extract_from_image_gemini,
        )

        with pytest.raises(
            GeminiExtractionError, match="Gemini vision extraction failed"
        ):
            extract_from_image_gemini(
                str(tmp_jpeg), doc_type="receipt", api_key="test-key"
            )

    def test_mime_type_png(self, tmp_png):
        """PNG extension → image/png MIME type passed to Part.from_bytes."""
        from alibi.extraction.gemini_structurer import extract_from_image_gemini

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.parsed = InvoiceExtraction(issuer="VisionCorp", amount=100.0)
            mock_client.models.generate_content.return_value = mock_response

            with patch(
                "alibi.extraction.gemini_structurer.extract_from_image_gemini",
                wraps=extract_from_image_gemini,
            ):
                # Patch types.Part.from_bytes to capture the mime_type argument
                with patch("google.genai.types") as mock_types:
                    mock_part = MagicMock()
                    mock_types.Part.from_bytes.return_value = mock_part
                    mock_types.GenerateContentConfig.return_value = MagicMock()

                    extract_from_image_gemini(
                        str(tmp_png), doc_type="invoice", api_key="test-key"
                    )

                    call_kwargs = mock_types.Part.from_bytes.call_args
                    assert call_kwargs is not None
                    mime_arg = (
                        call_kwargs.kwargs.get("mime_type") or call_kwargs.args[1]
                        if len(call_kwargs.args) > 1
                        else call_kwargs.kwargs.get("mime_type")
                    )
                    assert mime_arg == "image/png"

    def test_mime_type_jpeg(self, tmp_jpeg):
        """.jpg extension → image/jpeg MIME type."""
        from alibi.extraction.gemini_structurer import extract_from_image_gemini

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.parsed = ReceiptExtraction(vendor="MimeStore")
            mock_client.models.generate_content.return_value = mock_response

            with patch("google.genai.types") as mock_types:
                mock_part = MagicMock()
                mock_types.Part.from_bytes.return_value = mock_part
                mock_types.GenerateContentConfig.return_value = MagicMock()

                extract_from_image_gemini(
                    str(tmp_jpeg), doc_type="receipt", api_key="test-key"
                )

                call_kwargs = mock_types.Part.from_bytes.call_args
                assert call_kwargs is not None
                mime_arg = (
                    call_kwargs.kwargs.get("mime_type") or call_kwargs.args[1]
                    if len(call_kwargs.args) > 1
                    else call_kwargs.kwargs.get("mime_type")
                )
                assert mime_arg == "image/jpeg"

    def test_mime_type_pdf(self, tmp_path):
        """.pdf extension → application/pdf MIME type."""
        from alibi.extraction.gemini_structurer import extract_from_image_gemini

        pdf_file = tmp_path / "statement.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.parsed = PaymentExtraction(vendor="BankCorp")
            mock_client.models.generate_content.return_value = mock_response

            with patch("google.genai.types") as mock_types:
                mock_part = MagicMock()
                mock_types.Part.from_bytes.return_value = mock_part
                mock_types.GenerateContentConfig.return_value = MagicMock()

                extract_from_image_gemini(
                    str(pdf_file),
                    doc_type="payment_confirmation",
                    api_key="test-key",
                )

                call_kwargs = mock_types.Part.from_bytes.call_args
                assert call_kwargs is not None
                mime_arg = (
                    call_kwargs.kwargs.get("mime_type") or call_kwargs.args[1]
                    if len(call_kwargs.args) > 1
                    else call_kwargs.kwargs.get("mime_type")
                )
                assert mime_arg == "application/pdf"

    @patch("google.genai.Client")
    def test_receipt_doc_type_uses_receipt_model(self, mock_client_cls, tmp_jpeg):
        """doc_type='receipt' → ReceiptExtraction schema passed to API."""
        from alibi.extraction.gemini_structurer import extract_from_image_gemini

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ReceiptExtraction(vendor="SchemaReceipt", total=5.00)
        mock_client.models.generate_content.return_value = mock_response

        result = extract_from_image_gemini(
            str(tmp_jpeg), doc_type="receipt", api_key="test-key"
        )

        assert result["vendor"] == "SchemaReceipt"

        call_args = mock_client.models.generate_content.call_args
        config_arg = call_args.kwargs.get("config")
        assert config_arg is not None
        assert config_arg.response_schema is ReceiptExtraction

    @patch("google.genai.Client")
    def test_invoice_doc_type_uses_invoice_model(self, mock_client_cls, tmp_png):
        """doc_type='invoice' → InvoiceExtraction schema passed to API."""
        from alibi.extraction.gemini_structurer import extract_from_image_gemini

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = InvoiceExtraction(issuer="SchemaInvoicer", amount=200.0)
        mock_client.models.generate_content.return_value = mock_response

        result = extract_from_image_gemini(
            str(tmp_png), doc_type="invoice", api_key="test-key"
        )

        assert result["issuer"] == "SchemaInvoicer"

        call_args = mock_client.models.generate_content.call_args
        config_arg = call_args.kwargs.get("config")
        assert config_arg is not None
        assert config_arg.response_schema is InvoiceExtraction

    @patch("google.genai.Client")
    def test_uses_specified_model(self, mock_client_cls, tmp_jpeg):
        """model parameter is forwarded to generate_content."""
        from alibi.extraction.gemini_structurer import extract_from_image_gemini

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ReceiptExtraction()
        mock_client.models.generate_content.return_value = mock_response

        extract_from_image_gemini(
            str(tmp_jpeg),
            doc_type="receipt",
            api_key="test-key",
            model="gemini-1.5-pro",
        )

        call_args = mock_client.models.generate_content.call_args
        model_arg = call_args.kwargs.get("model") or call_args.args[0]
        assert model_arg == "gemini-1.5-pro"

    @patch("google.genai.Client")
    def test_api_key_passed_to_client(self, mock_client_cls, tmp_jpeg):
        """API key is forwarded to genai.Client constructor."""
        from alibi.extraction.gemini_structurer import extract_from_image_gemini

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = ReceiptExtraction()
        mock_client.models.generate_content.return_value = mock_response

        extract_from_image_gemini(
            str(tmp_jpeg), doc_type="receipt", api_key="vision-api-key"
        )

        mock_client_cls.assert_called_once_with(api_key="vision-api-key")


# ===========================================================================
# _fallback_vision tests
# ===========================================================================


class TestFallbackVision:
    def test_calls_gemini_when_enabled(self, tmp_jpeg):
        """When gemini_extraction_enabled=True, Gemini Vision is called."""
        from pathlib import Path

        from alibi.extraction.vision import _fallback_vision

        # extract_from_image_gemini is imported inside _fallback_vision, so patch
        # the function in its source module.
        with (
            patch("alibi.extraction.vision.get_config") as mock_cfg,
            patch(
                "alibi.extraction.gemini_structurer.extract_from_image_gemini",
                return_value={
                    "vendor": "GeminiVisionVendor",
                    "total": 99.0,
                    "_pipeline": "gemini_vision",
                },
            ) as mock_gemini,
        ):
            mock_cfg.return_value.gemini_extraction_enabled = True

            result = _fallback_vision(Path(str(tmp_jpeg)), doc_type="receipt")

        assert result["vendor"] == "GeminiVisionVendor"
        assert result["_pipeline"] == "gemini_vision"
        mock_gemini.assert_called_once()

    def test_falls_back_to_legacy_on_gemini_failure(self, tmp_jpeg):
        """When Gemini Vision raises GeminiExtractionError, legacy Ollama is used."""
        from pathlib import Path

        from alibi.extraction.gemini_structurer import GeminiExtractionError
        from alibi.extraction.vision import _fallback_vision

        with (
            patch("alibi.extraction.vision.get_config") as mock_cfg,
            patch(
                "alibi.extraction.gemini_structurer.extract_from_image_gemini",
                side_effect=GeminiExtractionError("Vision API down"),
            ),
            patch("alibi.extraction.vision._extract_from_image_legacy") as mock_legacy,
        ):
            mock_cfg.return_value.gemini_extraction_enabled = True
            mock_legacy.return_value = {
                "vendor": "LegacyOllamaVendor",
                "_pipeline": "legacy",
            }

            result = _fallback_vision(Path(str(tmp_jpeg)), doc_type="receipt")

        assert result["vendor"] == "LegacyOllamaVendor"
        mock_legacy.assert_called_once()

    def test_goes_directly_to_legacy_when_disabled(self, tmp_jpeg):
        """When gemini_extraction_enabled=False, legacy Ollama is called directly."""
        from pathlib import Path

        from alibi.extraction.vision import _fallback_vision

        # When Gemini is disabled the code never reaches the import of
        # extract_from_image_gemini.  We only need to verify that
        # _extract_from_image_legacy is called and Gemini is NOT invoked;
        # track the latter by monitoring the source-module function.
        with (
            patch("alibi.extraction.vision.get_config") as mock_cfg,
            patch("alibi.extraction.vision._extract_from_image_legacy") as mock_legacy,
            patch(
                "alibi.extraction.gemini_structurer.extract_from_image_gemini"
            ) as mock_gemini,
        ):
            mock_cfg.return_value.gemini_extraction_enabled = False
            mock_legacy.return_value = {
                "vendor": "DirectLegacyVendor",
                "_pipeline": "legacy",
            }

            result = _fallback_vision(Path(str(tmp_jpeg)), doc_type="receipt")

        assert result["vendor"] == "DirectLegacyVendor"
        mock_legacy.assert_called_once()
        mock_gemini.assert_not_called()
