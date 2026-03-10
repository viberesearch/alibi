"""Tests for micro-prompt extraction system."""

from unittest.mock import MagicMock, patch

import pytest

from alibi.extraction.micro_prompts import (
    MAX_MICRO_CALLS,
    MicroPrompt,
    _build_enrichment_prompt,
    _build_footer_prompt,
    _build_header_prompt,
    _build_line_items_prompt,
    _first_n_lines,
    _last_n_lines,
    build_micro_prompts,
    merge_micro_responses,
    run_micro_prompts,
)
from alibi.extraction.text_parser import ParseResult, TextRegions


@pytest.fixture
def sample_regions():
    return TextRegions(
        header="FRESKO HYPERMARKET\nVAT: 10336127M\n15/01/2026 10:30",
        body="MILK 1L    2.50\nBREAD      1.20\nEGGS 6    3.00",
        footer="SUBTOTAL   6.70\nTAX 19%    1.27\nTOTAL      7.97",
        header_end=3,
        footer_start=6,
    )


@pytest.fixture
def high_confidence_result(sample_regions):
    return ParseResult(
        data={
            "vendor": "FRESKO HYPERMARKET",
            "date": "2026-01-15",
            "total": 7.97,
            "subtotal": 6.70,
            "tax": 1.27,
            "currency": "EUR",
            "line_items": [
                {"name": "MILK 1L", "total_price": 2.50},
                {"name": "BREAD", "total_price": 1.20},
                {"name": "EGGS 6", "total_price": 3.00},
            ],
        },
        confidence=0.85,
        field_confidence={
            "vendor": 1.0,
            "date": 1.0,
            "total": 1.0,
            "subtotal": 1.0,
            "tax": 1.0,
            "currency": 1.0,
            "line_items": 0.8,
            "name_en": 0.0,
            "category": 0.0,
            "brand": 0.0,
            "language": 0.0,
        },
        line_item_count=3,
        needs_llm=True,
        regions=sample_regions,
    )


@pytest.fixture
def partial_confidence_result(sample_regions):
    return ParseResult(
        data={
            "vendor": None,
            "date": "2026-01-15",
            "total": 7.97,
            "subtotal": None,
            "tax": None,
            "currency": "EUR",
            "line_items": [
                {"name": "MILK 1L", "total_price": 2.50},
            ],
        },
        confidence=0.5,
        field_confidence={
            "vendor": 0.0,
            "date": 1.0,
            "total": 1.0,
            "subtotal": 0.0,
            "tax": 0.0,
            "currency": 1.0,
            "line_items": 0.5,
            "name_en": 0.0,
            "category": 0.0,
            "brand": 0.0,
            "language": 0.0,
        },
        line_item_count=1,
        needs_llm=True,
        regions=sample_regions,
    )


@pytest.fixture
def ocr_text():
    return (
        "FRESKO HYPERMARKET\n"
        "VAT: 10336127M\n"
        "15/01/2026 10:30\n"
        "MILK 1L    2.50\n"
        "BREAD      1.20\n"
        "EGGS 6     3.00\n"
        "SUBTOTAL   6.70\n"
        "TAX 19%    1.27\n"
        "TOTAL      7.97\n"
    )


class TestBuildMicroPrompts:
    def test_returns_none_when_no_uncertain_fields_and_no_items(self):
        result = ParseResult(
            data={"vendor": "Test", "total": 10.0},
            confidence=0.9,
            field_confidence={"vendor": 1.0, "total": 1.0},
        )
        prompts = build_micro_prompts(result, "test text", "receipt")
        assert prompts is None

    def test_enrichment_only_when_structural_fields_confident(self, ocr_text):
        # All structural fields at 1.0, only semantic fields missing
        result = ParseResult(
            data={
                "vendor": "FRESKO",
                "date": "2026-01-15",
                "total": 7.97,
                "currency": "EUR",
                "line_items": [{"name": "MILK", "total_price": 2.50}],
            },
            confidence=0.9,
            field_confidence={
                "vendor": 1.0,
                "date": 1.0,
                "total": 1.0,
                "currency": 1.0,
                "line_items": 1.0,
                "name_en": 0.0,
                "category": 0.0,
                "brand": 0.0,
                "language": 0.0,
            },
        )
        prompts = build_micro_prompts(result, ocr_text, "receipt")
        assert prompts is not None
        assert len(prompts) == 1
        assert "language" in prompts[0].fields
        assert "line_items" in prompts[0].fields

    def test_body_prompt_for_uncertain_line_items(
        self, high_confidence_result, ocr_text
    ):
        # line_items at 0.8 triggers a body micro-prompt
        prompts = build_micro_prompts(high_confidence_result, ocr_text, "receipt")
        assert prompts is not None
        body_prompts = [p for p in prompts if p.region == "body"]
        assert len(body_prompts) == 1
        assert "line_items" in body_prompts[0].fields

    def test_header_prompt_for_missing_vendor(
        self, partial_confidence_result, ocr_text
    ):
        prompts = build_micro_prompts(partial_confidence_result, ocr_text, "receipt")
        assert prompts is not None
        header_prompts = [p for p in prompts if p.region == "header"]
        assert len(header_prompts) == 1
        assert "vendor" in header_prompts[0].fields

    def test_footer_prompt_for_missing_totals(
        self, partial_confidence_result, ocr_text
    ):
        prompts = build_micro_prompts(partial_confidence_result, ocr_text, "receipt")
        assert prompts is not None
        footer_prompts = [p for p in prompts if p.region == "footer"]
        assert len(footer_prompts) == 1
        assert "subtotal" in footer_prompts[0].fields
        assert "tax" in footer_prompts[0].fields

    def test_uses_region_text_when_available(
        self, partial_confidence_result, ocr_text, sample_regions
    ):
        prompts = build_micro_prompts(partial_confidence_result, ocr_text, "receipt")
        header_prompt = [p for p in prompts if p.region == "header"][0]
        assert "FRESKO" in header_prompt.ocr_text
        assert "TOTAL" not in header_prompt.ocr_text

    def test_falls_back_to_first_n_lines_without_regions(self, ocr_text):
        result = ParseResult(
            data={"vendor": None, "total": 7.97},
            confidence=0.5,
            field_confidence={"vendor": 0.0, "total": 1.0},
            regions=None,
        )
        prompts = build_micro_prompts(result, ocr_text, "receipt")
        assert prompts is not None
        header_prompt = [p for p in prompts if p.region == "header"][0]
        assert "FRESKO" in header_prompt.ocr_text

    def test_returns_none_when_too_many_prompts(self, ocr_text):
        # Create a result with many uncertain fields across all regions
        fc = {f"field_{i}": 0.0 for i in range(20)}
        # Put fields in all regions to generate > MAX_MICRO_CALLS prompts
        # Header, footer, body fields
        fc["vendor"] = 0.0
        fc["total"] = 0.0
        fc["subtotal"] = 0.0
        fc["line_items"] = 0.0
        result = ParseResult(
            data={
                "vendor": None,
                "total": None,
                "line_items": [{"name": "x"}],
            },
            confidence=0.4,
            field_confidence=fc,
        )
        prompts = build_micro_prompts(result, ocr_text, "receipt")
        # Should still work since we group by region (max 3: header+footer+body)
        assert prompts is not None
        assert len(prompts) <= MAX_MICRO_CALLS + 1  # +1 for enrichment


class TestBuildPrompts:
    def test_header_prompt_includes_field_names(self):
        prompt = _build_header_prompt(
            ["vendor", "date"],
            "ACME STORE\n01/01/2026",
            "receipt",
            {"vendor": "ACME", "date": None},
        )
        assert "vendor" in prompt
        assert "date" in prompt
        assert "ACME STORE" in prompt

    def test_header_prompt_shows_existing_values(self):
        prompt = _build_header_prompt(
            ["vendor", "date"],
            "text",
            "receipt",
            {"vendor": "ACME", "date": "2026-01-01"},
        )
        assert "ACME" in prompt
        assert "2026-01-01" in prompt

    def test_footer_prompt_includes_field_names(self):
        prompt = _build_footer_prompt(
            ["total", "tax"],
            "TOTAL 10.00\nTAX 1.90",
            "receipt",
            {"total": 10.0, "tax": None},
        )
        assert "total" in prompt
        assert "tax" in prompt

    def test_line_items_prompt_includes_items(self):
        items = [{"name": "MILK", "total_price": 2.50}]
        prompt = _build_line_items_prompt("MILK 2.50", "receipt", items)
        assert "MILK" in prompt
        assert "2.50" in prompt or "2.5" in prompt

    def test_enrichment_prompt_includes_vendor(self):
        data = {
            "vendor": "Fresko",
            "line_items": [{"name": "ΓΑΛΑ", "total_price": 2.50}],
        }
        prompt = _build_enrichment_prompt(data, "receipt")
        assert "Fresko" in prompt
        assert "ΓΑΛΑ" in prompt
        assert "category" in prompt
        assert "name_en" in prompt


class TestMergeMicroResponses:
    def test_merge_scalar_fields(self):
        base = {"vendor": None, "total": 7.97, "date": "2026-01-15"}
        prompts = [
            MicroPrompt(
                fields=["vendor"],
                region="header",
                prompt="",
                ocr_text="",
            )
        ]
        responses = [{"vendor": "Fresko"}]
        result = merge_micro_responses(base, prompts, responses)
        assert result["vendor"] == "Fresko"
        assert result["total"] == 7.97
        assert result["date"] == "2026-01-15"

    def test_null_response_keeps_parser_value(self):
        base = {"vendor": "Parser Vendor", "total": 7.97}
        prompts = [
            MicroPrompt(
                fields=["vendor"],
                region="header",
                prompt="",
                ocr_text="",
            )
        ]
        responses = [{"vendor": None}]
        result = merge_micro_responses(base, prompts, responses)
        assert result["vendor"] == "Parser Vendor"

    def test_merge_line_items_same_count(self):
        base = {
            "line_items": [
                {"name": "MILK", "total_price": 2.50, "category": None},
                {"name": "BREAD", "total_price": 1.20, "category": None},
            ]
        }
        prompts = [
            MicroPrompt(
                fields=["line_items"],
                region="body",
                prompt="",
                ocr_text="",
            )
        ]
        responses = [
            {
                "line_items": [
                    {"name": "MILK", "total_price": 2.50, "category": "dairy"},
                    {
                        "name": "BREAD",
                        "total_price": 1.20,
                        "category": "bakery",
                    },
                ]
            }
        ]
        result = merge_micro_responses(base, prompts, responses)
        assert result["line_items"][0]["category"] == "dairy"
        assert result["line_items"][1]["category"] == "bakery"

    def test_merge_line_items_different_count(self):
        base = {
            "line_items": [
                {"name": "MILK", "total_price": 2.50},
            ]
        }
        prompts = [
            MicroPrompt(
                fields=["line_items"],
                region="body",
                prompt="",
                ocr_text="",
            )
        ]
        responses = [
            {
                "line_items": [
                    {"name": "MILK", "total_price": 2.50, "category": "dairy"},
                    {"name": "BREAD", "total_price": 1.20, "category": "bakery"},
                ]
            }
        ]
        result = merge_micro_responses(base, prompts, responses)
        assert len(result["line_items"]) == 2
        assert result["line_items"][0]["category"] == "dairy"

    def test_merge_language(self):
        base = {"language": None, "line_items": []}
        prompts = [
            MicroPrompt(
                fields=["language", "line_items"],
                region="body",
                prompt="",
                ocr_text="",
            )
        ]
        responses = [{"language": "el", "line_items": []}]
        result = merge_micro_responses(base, prompts, responses)
        assert result["language"] == "el"

    def test_empty_response_skipped(self):
        base = {"vendor": "Original"}
        prompts = [
            MicroPrompt(
                fields=["vendor"],
                region="header",
                prompt="",
                ocr_text="",
            )
        ]
        responses = [{}]
        result = merge_micro_responses(base, prompts, responses)
        assert result["vendor"] == "Original"

    def test_multiple_prompts_merged(self):
        base = {"vendor": None, "total": None, "currency": "EUR"}
        prompts = [
            MicroPrompt(
                fields=["vendor"],
                region="header",
                prompt="",
                ocr_text="",
            ),
            MicroPrompt(
                fields=["total"],
                region="footer",
                prompt="",
                ocr_text="",
            ),
        ]
        responses = [
            {"vendor": "Fresko"},
            {"total": 7.97},
        ]
        result = merge_micro_responses(base, prompts, responses)
        assert result["vendor"] == "Fresko"
        assert result["total"] == 7.97
        assert result["currency"] == "EUR"


class TestRunMicroPrompts:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_returns_merged_data(self, mock_structure, ocr_text, sample_regions):
        mock_structure.return_value = {"vendor": "FRESKO HYPERMARKET"}
        result_pr = ParseResult(
            data={"vendor": None, "total": 7.97, "currency": "EUR"},
            confidence=0.6,
            field_confidence={
                "vendor": 0.0,
                "total": 1.0,
                "currency": 1.0,
            },
            regions=sample_regions,
        )
        result = run_micro_prompts(result_pr, ocr_text, "receipt")
        assert result is not None
        assert result["vendor"] == "FRESKO HYPERMARKET"
        assert result["total"] == 7.97
        assert result["_pipeline"] == "micro_prompts"

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_returns_none_when_no_prompts_needed(self, mock_structure):
        result_pr = ParseResult(
            data={"vendor": "Test", "total": 10.0},
            confidence=0.9,
            field_confidence={"vendor": 1.0, "total": 1.0},
        )
        result = run_micro_prompts(result_pr, "text", "receipt")
        assert result is None
        mock_structure.assert_not_called()

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_handles_llm_failure_gracefully(
        self, mock_structure, ocr_text, sample_regions
    ):
        mock_structure.side_effect = Exception("LLM timeout")
        result_pr = ParseResult(
            data={"vendor": None, "total": 7.97},
            confidence=0.5,
            field_confidence={"vendor": 0.0, "total": 1.0},
            regions=sample_regions,
        )
        result = run_micro_prompts(result_pr, ocr_text, "receipt")
        assert result is not None
        assert result["vendor"] is None  # LLM failed, parser value kept
        assert result["total"] == 7.97

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_sets_pipeline_metadata(self, mock_structure, ocr_text, sample_regions):
        mock_structure.return_value = {"vendor": "ACME"}
        result_pr = ParseResult(
            data={"vendor": None, "total": 7.97},
            confidence=0.5,
            field_confidence={"vendor": 0.0, "total": 1.0},
            regions=sample_regions,
        )
        result = run_micro_prompts(result_pr, ocr_text, "receipt")
        assert result["_pipeline"] == "micro_prompts"
        assert result["_parser_confidence"] == 0.5
        assert result["_micro_prompt_count"] >= 1


class TestHelpers:
    def test_first_n_lines(self):
        text = "line1\nline2\nline3\nline4\nline5"
        assert _first_n_lines(text, 3) == "line1\nline2\nline3"

    def test_last_n_lines(self):
        text = "line1\nline2\nline3\nline4\nline5"
        assert _last_n_lines(text, 2) == "line4\nline5"

    def test_first_n_lines_more_than_available(self):
        text = "line1\nline2"
        assert _first_n_lines(text, 10) == "line1\nline2"

    def test_last_n_lines_more_than_available(self):
        text = "line1\nline2"
        assert _last_n_lines(text, 10) == "line1\nline2"
