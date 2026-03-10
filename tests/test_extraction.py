"""Tests for extraction modules."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alibi.extraction import (
    INVOICE_PROMPT,
    RECEIPT_PROMPT,
    STATEMENT_PROMPT,
    WARRANTY_PROMPT,
    VisionExtractionError,
    extract_json_from_response,
    get_prompt_for_type,
    get_text_structure_prompt,
)


class TestPrompts:
    """Tests for extraction prompts."""

    def test_get_prompt_for_receipt(self):
        """Test getting receipt prompt."""
        prompt = get_prompt_for_type("receipt")
        assert prompt == RECEIPT_PROMPT
        assert "vendor" in prompt.lower()
        assert "total" in prompt.lower()

    def test_get_prompt_for_invoice(self):
        """Test getting invoice prompt."""
        prompt = get_prompt_for_type("invoice")
        assert prompt == INVOICE_PROMPT
        assert "invoice" in prompt.lower()

    def test_get_prompt_for_statement(self):
        """Test getting statement prompt."""
        prompt = get_prompt_for_type("statement")
        assert prompt == STATEMENT_PROMPT
        assert "statement" in prompt.lower()

    def test_get_prompt_for_warranty(self):
        """Test getting warranty prompt."""
        prompt = get_prompt_for_type("warranty")
        assert prompt == WARRANTY_PROMPT
        assert "warranty" in prompt.lower()

    def test_get_prompt_for_unknown_defaults_to_receipt(self):
        """Test that unknown type defaults to receipt prompt."""
        prompt = get_prompt_for_type("unknown")
        assert prompt == RECEIPT_PROMPT

    def test_get_text_structure_prompt_no_tables(self):
        """Test text structure prompt without tables."""
        prompt = get_text_structure_prompt("Sample text content")
        assert "Sample text content" in prompt
        assert "None" in prompt  # No tables

    def test_get_text_structure_prompt_with_tables(self):
        """Test text structure prompt with tables."""
        tables: list[list[list[str | None]]] = [
            [["Header1", "Header2"], ["Value1", "Value2"]]
        ]
        prompt = get_text_structure_prompt("Sample text", tables)
        assert "Sample text" in prompt
        assert "Header1" in prompt
        assert "Value1" in prompt


class TestJsonExtraction:
    """Tests for JSON extraction from LLM responses."""

    def test_extract_json_direct(self):
        """Test extracting JSON directly from response."""
        response = '{"vendor": "Test Store", "total": 42.50}'
        result = extract_json_from_response(response)
        assert result["vendor"] == "Test Store"
        assert result["total"] == 42.50

    def test_extract_json_from_code_block(self):
        """Test extracting JSON from markdown code block."""
        response = """Here is the result:
```json
{"vendor": "Test Store", "total": 42.50}
```
That's all."""
        result = extract_json_from_response(response)
        assert result["vendor"] == "Test Store"

    def test_extract_json_from_plain_code_block(self):
        """Test extracting JSON from plain code block."""
        response = """```
{"vendor": "Test Store", "total": 42.50}
```"""
        result = extract_json_from_response(response)
        assert result["vendor"] == "Test Store"

    def test_extract_json_with_surrounding_text(self):
        """Test extracting JSON with surrounding text."""
        response = 'Based on the image, I found: {"vendor": "Store", "total": 10.00} That is my analysis.'
        result = extract_json_from_response(response)
        assert result["vendor"] == "Store"

    def test_extract_json_no_json_raises(self):
        """Test that missing JSON raises error."""
        response = "No JSON here at all"
        with pytest.raises(VisionExtractionError, match="No JSON found"):
            extract_json_from_response(response)

    def test_extract_json_invalid_json_raises(self):
        """Test that invalid JSON raises error."""
        response = '{"vendor": "Test", invalid}'
        with pytest.raises(VisionExtractionError, match="Invalid JSON"):
            extract_json_from_response(response)


class TestVisionExtraction:
    """Tests for vision extraction (mocked)."""

    @patch("alibi.extraction.vision.httpx.Client")
    def test_extract_from_image_success(self, mock_client_class):
        """Test successful image extraction."""
        from alibi.extraction.vision import extract_from_image

        # Create a minimal test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Write minimal PNG (1x1 pixel)
            f.write(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
                b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
                b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            tmp_path = Path(f.name)

        try:
            # Mock the HTTP client
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": '{"vendor": "Test Store", "total": 42.50}'
            }
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = extract_from_image(tmp_path, doc_type="receipt")
            assert result["vendor"] == "Test Store"
            assert result["total"] == 42.50
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_extract_from_image_file_not_found(self):
        """Test that missing file raises error."""
        from alibi.extraction.vision import extract_from_image

        with pytest.raises(VisionExtractionError, match="not found"):
            extract_from_image(Path("/nonexistent/image.png"))


class TestPDFExtraction:
    """Tests for PDF extraction."""

    def test_extract_text_from_pdf_file_not_found(self):
        """Test that missing file raises error."""
        from alibi.extraction.pdf import PDFExtractionError, extract_text_from_pdf

        with pytest.raises(PDFExtractionError, match="not found"):
            extract_text_from_pdf(Path("/nonexistent/file.pdf"))

    def test_get_pdf_page_count_file_not_found(self):
        """Test that missing file raises error."""
        from alibi.extraction.pdf import PDFExtractionError, get_pdf_page_count

        with pytest.raises(PDFExtractionError, match="not found"):
            get_pdf_page_count(Path("/nonexistent/file.pdf"))
