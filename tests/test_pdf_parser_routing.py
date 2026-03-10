"""Tests for PDF text -> heuristic parser routing (v3 pipeline).

Verifies that text-layer PDFs go through the heuristic parser first
instead of always hitting the LLM.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alibi.extraction.pdf import (
    _extract_pdf_with_parser,
    extract_from_pdf,
)


class TestPdfParserRouting:
    """Tests for parser-first PDF extraction."""

    def test_high_confidence_skips_llm(self):
        """When parser confidence >= threshold, LLM should be skipped."""
        text = (
            "FreSko\n"
            "FRESKO BUTANOLO LTD\n"
            "123 Main St\n"
            "VAT 10336127M\n"
            "17/02/2026 14:30\n"
            "Milk 1L            2.50 A\n"
            "Bread               1.20 B\n"
            "Butter              2.50 A\n"
            "SUBTOTAL            6.20\n"
            "VAT A 19%           0.80\n"
            "VAT B 9%            0.10\n"
            "TOTAL EUR           7.10\n"
            "VISA ****7201\n"
        )

        with patch("alibi.extraction.pdf.get_config") as mock_config:
            mock_config.return_value.skip_llm_threshold = 0.9

            result = _extract_pdf_with_parser(
                Path("test.pdf"), text, [], "receipt", True
            )

        assert result.get("_pipeline") == "pdf_parser_only"
        assert result.get("_text_source") == "pdfplumber"
        assert result.get("vendor") == "FreSko"
        assert "raw_text" in result

    def test_medium_confidence_uses_correction(self):
        """When 0.3 <= confidence < threshold, should use correction prompt."""
        text = "FreSko\n" "17/02/2026\n" "TOTAL EUR 4.20\n"

        mock_structure = MagicMock(return_value={"vendor": "FreSko", "total": 4.20})

        with (
            patch("alibi.extraction.pdf.get_config") as mock_config,
            patch(
                "alibi.extraction.structurer.structure_ocr_text", mock_structure
            ) as mock_llm,
        ):
            mock_config.return_value.skip_llm_threshold = 0.9
            mock_config.return_value.gemini_extraction_enabled = False

            result = _extract_pdf_with_parser(
                Path("test.pdf"), text, [], "receipt", True
            )

        # Should have called LLM with correction prompt
        assert mock_llm.called
        call_kwargs = mock_llm.call_args
        assert call_kwargs.kwargs.get("emphasis_prompt") is not None

    def test_low_confidence_uses_full_llm(self):
        """When confidence < 0.3, should use full LLM structuring."""
        text = "A" * 200  # Enough text length, but meaningless content

        mock_parse_result = MagicMock()
        mock_parse_result.confidence = 0.1
        mock_parse_result.data = {}
        mock_parse_result.gaps = ["vendor", "date", "total"]
        mock_parse_result.line_item_count = 0

        mock_llm = MagicMock(return_value={"vendor": "Unknown"})

        with (
            patch("alibi.extraction.pdf.get_config") as mock_config,
            patch(
                "alibi.extraction.text_parser.parse_ocr_text",
                return_value=mock_parse_result,
            ),
            patch("alibi.extraction.pdf.structure_text_with_llm", mock_llm),
        ):
            mock_config.return_value.skip_llm_threshold = 0.9

            result = _extract_pdf_with_parser(
                Path("test.pdf"), text, [], "receipt", True
            )

        assert mock_llm.called
        assert result.get("_pipeline") == "pdf_llm_full"

    def test_statement_type_passed_to_parser(self):
        """Document type should be forwarded to the parser."""
        text = (
            "EUROBANK STATEMENT\n"
            "Period: 01/01/2026 - 31/01/2026\n"
            "Opening Balance: 1500.00\n"
            "01/01 GROCERY STORE  -50.00  1450.00\n"
            "02/01 SALARY         +2000.00 3450.00\n"
            "Closing Balance: 3450.00\n"
        )

        with patch("alibi.extraction.pdf.get_config") as mock_config:
            mock_config.return_value.skip_llm_threshold = 0.9

            result = _extract_pdf_with_parser(
                Path("test.pdf"), text, [], "statement", True
            )

        # Parser should have been called with doc_type="statement"
        assert result.get("_text_source") == "pdfplumber"

    def test_extract_from_pdf_passes_doc_type(self):
        """extract_from_pdf should forward doc_type parameter."""
        mock_text = "A" * 200  # Enough chars to trigger text path

        with (
            patch(
                "alibi.extraction.pdf.extract_text_from_pdf",
                return_value=(mock_text, []),
            ),
            patch(
                "alibi.extraction.pdf._extract_pdf_with_parser",
                return_value={"_pipeline": "test"},
            ) as mock_parser,
        ):
            extract_from_pdf(Path("test.pdf"), doc_type="statement")

        mock_parser.assert_called_once()
        args = mock_parser.call_args
        assert args[0][3] == "statement"  # doc_type positional arg

    def test_short_text_falls_back_to_vision(self):
        """PDFs with < 100 chars of text should fall back to vision."""
        with (
            patch(
                "alibi.extraction.pdf.extract_text_from_pdf",
                return_value=("Short", []),
            ),
            patch(
                "alibi.extraction.pdf.extract_pdf_via_vision",
                return_value={"_pipeline": "vision"},
            ) as mock_vision,
        ):
            result = extract_from_pdf(Path("test.pdf"))

        mock_vision.assert_called_once()
        assert result["_pipeline"] == "vision"

    def test_parser_failure_falls_through_to_llm(self):
        """If parser crashes, should fall through to LLM."""
        text = "X" * 200

        mock_llm = MagicMock(return_value={"vendor": "Recovered"})

        with (
            patch("alibi.extraction.pdf.get_config") as mock_config,
            patch(
                "alibi.extraction.text_parser.parse_ocr_text",
                side_effect=Exception("parser crash"),
            ),
            patch("alibi.extraction.pdf.structure_text_with_llm", mock_llm),
        ):
            mock_config.return_value.skip_llm_threshold = 0.9

            result = _extract_pdf_with_parser(
                Path("test.pdf"), text, [], "invoice", True
            )

        # Should have fallen through to full LLM
        assert mock_llm.called
