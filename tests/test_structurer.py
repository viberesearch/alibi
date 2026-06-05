"""Tests for structurer module (Stage 2)."""

import json
from unittest.mock import patch

import pytest

from alibi.extraction.structurer import structure_ocr_text
from alibi.extraction.vision import VisionExtractionError


class TestStructureOcrText:
    """Tests for text structuring."""

    @patch("alibi.extraction.structurer._call_ollama_text")
    def test_basic_structuring(self, mock_call):
        mock_call.return_value = {
            "response": json.dumps(
                {
                    "vendor": "SuperMarket",
                    "date": "2024-01-15",
                    "total": 12.50,
                    "currency": "EUR",
                    "line_items": [
                        {"name": "Bread", "total_price": 2.50},
                        {"name": "Milk", "total_price": 10.00},
                    ],
                }
            )
        }
        result = structure_ocr_text(
            "SuperMarket\nBread 2.50\nMilk 10.00\nTotal 12.50",
            doc_type="receipt",
            model="qwen3:30b",
            ollama_url="http://test:11434",
        )
        assert result["vendor"] == "SuperMarket"
        assert result["total"] == 12.50
        assert len(result["line_items"]) == 2

    @patch("alibi.extraction.structurer._call_ollama_text")
    def test_error_response(self, mock_call):
        mock_call.return_value = {"error": "model not loaded"}
        with pytest.raises(VisionExtractionError, match="Structure error"):
            structure_ocr_text(
                "text", model="qwen3:30b", ollama_url="http://test:11434"
            )

    @patch("alibi.extraction.structurer._call_ollama_text")
    def test_no_response_field(self, mock_call):
        mock_call.return_value = {"unexpected": "format"}
        with pytest.raises(VisionExtractionError, match="Unexpected"):
            structure_ocr_text(
                "text", model="qwen3:30b", ollama_url="http://test:11434"
            )

    @patch("alibi.extraction.structurer._call_ollama_text")
    def test_with_emphasis_prompt(self, mock_call):
        mock_call.return_value = {
            "response": json.dumps({"vendor": "Test", "total": 5.00})
        }
        result = structure_ocr_text(
            "text",
            model="qwen3:30b",
            ollama_url="http://test:11434",
            emphasis_prompt="Custom emphasis prompt with text\n{...}",
        )
        # Should use the emphasis prompt verbatim (reasoning is now disabled
        # via the think=false request field, not an in-prompt /no_think prefix).
        call_args = mock_call.call_args
        assert call_args[0][2] == "Custom emphasis prompt with text\n{...}"
        # Correction/emphasis retries are not schema-constrained.
        assert call_args.kwargs.get("response_format") is None
        assert result["vendor"] == "Test"

    @patch("alibi.extraction.structurer._call_ollama_text")
    def test_schema_enforced_for_default_path(self, mock_call):
        """Non-emphasis extraction constrains output to the JSON schema."""
        mock_call.return_value = {"response": json.dumps({"vendor": "Test"})}
        structure_ocr_text("text", model="qwen3:30b", ollama_url="http://test:11434")
        fmt = mock_call.call_args.kwargs.get("response_format")
        assert isinstance(fmt, dict)
        assert "line_items" in fmt.get("properties", {})

    @patch("alibi.extraction.structurer._call_ollama_text")
    def test_no_images_in_request(self, mock_call):
        """Verify the text-only API call doesn't include images field."""
        mock_call.return_value = {"response": json.dumps({"vendor": "Test"})}
        structure_ocr_text("text", model="qwen3:30b", ollama_url="http://test:11434")
        # The _call_ollama_text function doesn't accept images at all
        call_args = mock_call.call_args
        # It takes (ollama_url, model, prompt, timeout) — no images param
        assert len(call_args[0]) == 4

    @patch("alibi.extraction.structurer._call_ollama_text")
    def test_escalates_budget_on_truncation(self, mock_call):
        """A truncated (done_reason=length) response triggers one larger retry."""
        from alibi.config import get_config

        cfg = get_config()
        mock_call.side_effect = [
            {"response": "", "done_reason": "length", "eval_count": 4096},
            {"response": json.dumps({"vendor": "Big"}), "done_reason": "stop"},
        ]
        result = structure_ocr_text(
            "text", model="gemma4:12b", ollama_url="http://test:11434"
        )
        assert mock_call.call_count == 2
        retry = mock_call.call_args_list[1]
        assert retry.kwargs.get("num_predict") == cfg.ollama_num_predict_escalated
        assert retry.kwargs.get("num_ctx") == cfg.ollama_num_ctx_escalated
        assert result["vendor"] == "Big"

    @patch("alibi.extraction.structurer._call_ollama_text")
    def test_no_escalation_when_not_truncated(self, mock_call):
        """A normal (done_reason=stop) response is used directly, no retry."""
        mock_call.return_value = {
            "response": json.dumps({"vendor": "Small"}),
            "done_reason": "stop",
        }
        structure_ocr_text("text", model="gemma4:12b", ollama_url="http://test:11434")
        assert mock_call.call_count == 1

    @patch("alibi.extraction.structurer._call_ollama_text")
    def test_json_in_code_block(self, mock_call):
        mock_call.return_value = {
            "response": '```json\n{"vendor": "Test", "total": 5.00}\n```'
        }
        result = structure_ocr_text(
            "text", model="qwen3:30b", ollama_url="http://test:11434"
        )
        assert result["vendor"] == "Test"


class TestPromptConstruction:
    """Tests for prompt construction in structurer."""

    def test_text_extraction_prompt_contains_ocr_text(self):
        from alibi.extraction.prompts import get_text_extraction_prompt

        prompt = get_text_extraction_prompt("Hello World Receipt", "receipt")
        assert "Hello World Receipt" in prompt
        assert "BEGIN OCR TEXT" in prompt
        assert "END OCR TEXT" in prompt

    def test_text_extraction_prompt_strips_vision_preamble(self):
        from alibi.extraction.prompts import get_text_extraction_prompt

        prompt = get_text_extraction_prompt("test", "receipt")
        # Should not contain vision-specific instructions
        assert "Analyze this receipt image" not in prompt

    def test_text_extraction_prompt_universal_mode(self):
        from alibi.extraction.prompts import get_text_extraction_prompt

        prompt = get_text_extraction_prompt("test", "receipt", mode="universal")
        assert "test" in prompt
        # Universal mode uses different base prompt
        assert "BEGIN OCR TEXT" in prompt

    def test_text_extraction_prompt_mentions_doc_type(self):
        from alibi.extraction.prompts import get_text_extraction_prompt

        prompt = get_text_extraction_prompt("text", "invoice")
        assert "invoice" in prompt.lower()
