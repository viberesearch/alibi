"""Tests for PDF extraction: per-page splitting, merge logic, and timeout handling."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from alibi.extraction.pdf import (
    PDFExtractionError,
    _MAX_TEXT_PER_LLM_CALL,
    _looks_like_transaction_confirmation_text,
    _merge_extractions,
    extract_from_pdf,
)


class TestMergeExtractions:
    """Tests for _merge_extractions() helper."""

    def test_merge_line_items(self):
        base: dict[str, Any] = {"vendor": "Store", "line_items": [{"name": "A"}]}
        extra = {"line_items": [{"name": "B"}, {"name": "C"}]}
        _merge_extractions(base, extra)
        assert len(base["line_items"]) == 3
        assert base["line_items"][2]["name"] == "C"

    def test_merge_items_key(self):
        """Also merges 'items' key (alternative naming)."""
        base = {"items": [{"name": "A"}]}
        extra = {"items": [{"name": "B"}]}
        _merge_extractions(base, extra)
        assert len(base["items"]) == 2

    def test_merge_creates_list_if_missing(self):
        base: dict[str, Any] = {"vendor": "Store"}
        extra = {"line_items": [{"name": "A"}]}
        _merge_extractions(base, extra)
        assert base["line_items"] == [{"name": "A"}]

    def test_merge_fills_missing_scalars(self):
        base = {"vendor": "Store", "date": None, "total": 10.0}
        extra = {"vendor": "Other Store", "date": "2025-03-15", "currency": "EUR"}
        _merge_extractions(base, extra)
        # vendor already set -> keep original
        assert base["vendor"] == "Store"
        # date was None/falsy -> filled from extra
        assert base["date"] == "2025-03-15"
        # currency missing -> filled
        assert base["currency"] == "EUR"

    def test_merge_does_not_overwrite_existing_scalars(self):
        base = {"vendor": "Store A", "total": 50.0}
        extra = {"vendor": "Store B", "total": 99.0}
        _merge_extractions(base, extra)
        assert base["vendor"] == "Store A"
        assert base["total"] == 50.0

    def test_merge_empty_extra_line_items(self):
        """Empty line_items in extra doesn't create empty list in base."""
        base = {"vendor": "Store"}
        extra: dict[str, Any] = {"line_items": []}
        _merge_extractions(base, extra)
        assert "line_items" not in base

    def test_merge_none_extra_line_items(self):
        """None line_items in extra doesn't create empty list in base."""
        base = {"vendor": "Store"}
        extra = {"line_items": None}
        _merge_extractions(base, extra)
        assert "line_items" not in base


class TestMaxTextConstant:
    """Verify the threshold constant."""

    def test_threshold_value(self):
        assert _MAX_TEXT_PER_LLM_CALL == 8000


class TestExtractFromPdfRouting:
    """Tests for extract_from_pdf() routing logic."""

    @patch("alibi.extraction.pdf.structure_text_with_llm")
    @patch("alibi.extraction.pdf.extract_text_from_pdf")
    def test_small_pdf_uses_single_call(self, mock_extract_text, mock_structure):
        """PDF with <8000 chars goes through single LLM call."""
        text = "A" * 500
        mock_extract_text.return_value = (text, [])
        mock_structure.return_value = {"vendor": "Test"}

        result = extract_from_pdf(Path("dummy.pdf"))

        mock_structure.assert_called_once_with(text, [])
        assert result["vendor"] == "Test"
        assert result["_pipeline"] == "pdf_llm_full"

    @patch("alibi.extraction.pdf._extract_pdf_per_page")
    @patch("alibi.extraction.pdf.extract_text_from_pdf")
    def test_large_pdf_uses_per_page(self, mock_extract_text, mock_per_page):
        """PDF with >8000 chars routes to per-page extraction."""
        text = "A" * 9000
        mock_extract_text.return_value = (text, [])
        mock_per_page.return_value = {"vendor": "Big Store"}

        result = extract_from_pdf(Path("dummy.pdf"))

        mock_per_page.assert_called_once_with(Path("dummy.pdf"))
        assert result == {"vendor": "Big Store"}

    @patch("alibi.extraction.pdf._extract_pdf_per_page")
    @patch("alibi.extraction.pdf.extract_text_from_pdf")
    def test_boundary_8000_uses_single_call(self, mock_extract_text, mock_per_page):
        """Exactly 8000 chars still uses single call (<=)."""
        text = "A" * 8000
        mock_extract_text.return_value = (text, [])

        with patch("alibi.extraction.pdf.structure_text_with_llm") as mock_structure:
            mock_structure.return_value = {"vendor": "Boundary"}
            result = extract_from_pdf(Path("dummy.pdf"))

        mock_per_page.assert_not_called()
        mock_structure.assert_called_once()

    @patch("alibi.extraction.pdf._extract_pdf_per_page")
    @patch("alibi.extraction.pdf.extract_text_from_pdf")
    def test_boundary_8001_uses_per_page(self, mock_extract_text, mock_per_page):
        """8001 chars routes to per-page extraction."""
        text = "A" * 8001
        mock_extract_text.return_value = (text, [])
        mock_per_page.return_value = {"vendor": "Over"}

        result = extract_from_pdf(Path("dummy.pdf"))

        mock_per_page.assert_called_once()

    @patch("alibi.extraction.pdf.extract_pdf_via_vision")
    @patch("alibi.extraction.pdf.extract_text_from_pdf")
    def test_short_text_falls_back_to_vision(self, mock_extract_text, mock_vision):
        """PDF with <100 chars of text falls back to vision."""
        mock_extract_text.return_value = ("short", [])
        mock_vision.return_value = {"vendor": "Vision"}

        result = extract_from_pdf(Path("dummy.pdf"))

        mock_vision.assert_called_once_with(
            Path("dummy.pdf"),
            doc_type="invoice",
            skip_llm_threshold=None,
            country=None,
            hints=None,
        )

    @patch("alibi.extraction.pdf.extract_text_from_pdf")
    def test_short_text_no_vision_raises(self, mock_extract_text):
        """Short text + vision fallback disabled raises error."""
        mock_extract_text.return_value = ("short", [])

        with pytest.raises(PDFExtractionError, match="insufficient text"):
            extract_from_pdf(Path("dummy.pdf"), use_vision_fallback=False)

    @patch("alibi.extraction.pdf.extract_pdf_via_vision")
    @patch("alibi.extraction.pdf.extract_text_from_pdf")
    def test_empty_text_falls_back_to_vision(self, mock_extract_text, mock_vision):
        """Empty text falls back to vision."""
        mock_extract_text.return_value = ("", [])
        mock_vision.return_value = {"vendor": "Vision"}

        result = extract_from_pdf(Path("dummy.pdf"))
        mock_vision.assert_called_once()

    @patch("alibi.extraction.pdf.extract_pdf_via_vision")
    @patch("alibi.extraction.pdf.extract_text_from_pdf")
    def test_whitespace_only_falls_back(self, mock_extract_text, mock_vision):
        """Whitespace-only text falls back to vision."""
        mock_extract_text.return_value = ("   \n\n   ", [])
        mock_vision.return_value = {"vendor": "Vision"}

        result = extract_from_pdf(Path("dummy.pdf"))
        mock_vision.assert_called_once()


class TestExtractPdfPerPage:
    """Tests for _extract_pdf_per_page()."""

    def _make_mock_page(self, text: str, tables: list[Any] | None = None):
        page = MagicMock()
        page.extract_text.return_value = text
        page.extract_tables.return_value = tables or []
        return page

    @patch("alibi.extraction.pdf.structure_text_with_llm")
    @patch("alibi.extraction.pdf.pdfplumber")
    def test_single_page(self, mock_pdfplumber, mock_structure):
        """Single-page PDF processes normally."""
        from alibi.extraction.pdf import _extract_pdf_per_page

        page = self._make_mock_page("Invoice from Store\nTotal: 100.00")
        mock_pdf = MagicMock()
        mock_pdf.pages = [page]
        mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)

        mock_structure.return_value = {"vendor": "Store", "total": 100.0}

        result = _extract_pdf_per_page(Path("test.pdf"))
        assert result == {"vendor": "Store", "total": 100.0}
        mock_structure.assert_called_once()

    @patch("alibi.extraction.pdf.structure_text_with_llm")
    @patch("alibi.extraction.pdf.pdfplumber")
    def test_multi_page_merges(self, mock_pdfplumber, mock_structure):
        """Multi-page PDF merges results from each page."""
        from alibi.extraction.pdf import _extract_pdf_per_page

        page1 = self._make_mock_page("Page 1 text")
        page2 = self._make_mock_page("Page 2 text")
        mock_pdf = MagicMock()
        mock_pdf.pages = [page1, page2]
        mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)

        mock_structure.side_effect = [
            {"vendor": "Store", "line_items": [{"name": "A"}]},
            {"line_items": [{"name": "B"}], "currency": "EUR"},
        ]

        result = _extract_pdf_per_page(Path("test.pdf"))
        assert result["vendor"] == "Store"
        assert len(result["line_items"]) == 2
        assert result["currency"] == "EUR"

    @patch("alibi.extraction.pdf.structure_text_with_llm")
    @patch("alibi.extraction.pdf.pdfplumber")
    def test_skips_empty_pages(self, mock_pdfplumber, mock_structure):
        """Empty pages are skipped."""
        from alibi.extraction.pdf import _extract_pdf_per_page

        page1 = self._make_mock_page("Page 1 content")
        page2 = self._make_mock_page("")  # empty
        page3 = self._make_mock_page("   ")  # whitespace only
        page4 = self._make_mock_page("Page 4 content")
        mock_pdf = MagicMock()
        mock_pdf.pages = [page1, page2, page3, page4]
        mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)

        mock_structure.side_effect = [
            {"vendor": "Store"},
            {"total": 50.0},
        ]

        result = _extract_pdf_per_page(Path("test.pdf"))
        assert mock_structure.call_count == 2  # Only page 1 and 4

    @patch("alibi.extraction.pdf.structure_text_with_llm")
    @patch("alibi.extraction.pdf.pdfplumber")
    def test_continues_on_page_failure(self, mock_pdfplumber, mock_structure):
        """Failed pages are skipped, others still processed."""
        from alibi.extraction.pdf import _extract_pdf_per_page

        page1 = self._make_mock_page("Page 1")
        page2 = self._make_mock_page("Page 2")
        mock_pdf = MagicMock()
        mock_pdf.pages = [page1, page2]
        mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)

        mock_structure.side_effect = [
            PDFExtractionError("Timeout"),
            {"vendor": "Store", "total": 25.0},
        ]

        result = _extract_pdf_per_page(Path("test.pdf"))
        assert result == {"vendor": "Store", "total": 25.0}

    @patch("alibi.extraction.pdf.structure_text_with_llm")
    @patch("alibi.extraction.pdf.pdfplumber")
    def test_all_pages_fail_raises(self, mock_pdfplumber, mock_structure):
        """If all pages fail, raises PDFExtractionError."""
        from alibi.extraction.pdf import _extract_pdf_per_page

        page1 = self._make_mock_page("Page 1")
        page2 = self._make_mock_page("Page 2")
        mock_pdf = MagicMock()
        mock_pdf.pages = [page1, page2]
        mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)

        mock_structure.side_effect = PDFExtractionError("Timeout")

        with pytest.raises(PDFExtractionError, match="No pages could be extracted"):
            _extract_pdf_per_page(Path("test.pdf"))

    @patch("alibi.extraction.pdf.pdfplumber")
    def test_no_pages_raises(self, mock_pdfplumber):
        """PDF with no pages raises error."""
        from alibi.extraction.pdf import _extract_pdf_per_page

        mock_pdf = MagicMock()
        mock_pdf.pages = []
        mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(PDFExtractionError, match="no pages"):
            _extract_pdf_per_page(Path("test.pdf"))

    @patch("alibi.extraction.pdf.pdfplumber")
    def test_open_failure_raises(self, mock_pdfplumber):
        """PDF open failure raises PDFExtractionError."""
        from alibi.extraction.pdf import _extract_pdf_per_page

        mock_pdfplumber.open.side_effect = Exception("Corrupt PDF")

        with pytest.raises(PDFExtractionError, match="Failed to open PDF"):
            _extract_pdf_per_page(Path("test.pdf"))

    @patch("alibi.extraction.pdf.structure_text_with_llm")
    @patch("alibi.extraction.pdf.pdfplumber")
    def test_passes_tables_to_llm(self, mock_pdfplumber, mock_structure):
        """Tables from each page are passed to LLM."""
        from alibi.extraction.pdf import _extract_pdf_per_page

        tables = [["Header", "Value"], ["Row1", "100"]]
        page = self._make_mock_page("Page with table", tables=[tables])
        mock_pdf = MagicMock()
        mock_pdf.pages = [page]
        mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)

        mock_structure.return_value = {"vendor": "Store"}

        _extract_pdf_per_page(Path("test.pdf"))
        # Should pass tables (non-empty list)
        args = mock_structure.call_args
        assert args[0][1] is not None  # tables argument


class TestStructureTextTimeout:
    """Tests for structure_text_with_llm() timeout parameter."""

    @patch("alibi.extraction.pdf.get_config")
    @patch("alibi.extraction.pdf.extract_json_from_response")
    @patch("httpx.Client")
    def test_default_timeout_180s(self, mock_client_cls, mock_extract, mock_config):
        """Default timeout is 180 seconds."""
        from alibi.extraction.pdf import structure_text_with_llm

        mock_config.return_value.ollama_url = "http://localhost:11434"
        mock_config.return_value.ollama_model = "test-model"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": '{"vendor": "Test"}'}
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_extract.return_value = {"vendor": "Test"}

        structure_text_with_llm("some text")

        mock_client_cls.assert_called_once_with(timeout=180.0)

    @patch("alibi.extraction.pdf.get_config")
    @patch("alibi.extraction.pdf.extract_json_from_response")
    @patch("httpx.Client")
    def test_custom_timeout(self, mock_client_cls, mock_extract, mock_config):
        """Custom timeout is passed through."""
        from alibi.extraction.pdf import structure_text_with_llm

        mock_config.return_value.ollama_url = "http://localhost:11434"
        mock_config.return_value.ollama_model = "test-model"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": '{"vendor": "Test"}'}
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_extract.return_value = {"vendor": "Test"}

        structure_text_with_llm("some text", timeout=300.0)

        mock_client_cls.assert_called_once_with(timeout=300.0)


# ---------------------------------------------------------------------------
# Transaction confirmation detection
# ---------------------------------------------------------------------------


class TestTransactionConfirmationDetection:
    """Tests for _looks_like_transaction_confirmation_text()."""

    def test_detects_bank_confirmation(self):
        """Text with confirmation header + transaction markers returns True."""
        text = (
            "Transaction Confirmation\n"
            "Date: 05/02/2026\n"
            "Beneficiary: PAPAS HYPERMARKET\n"
            "Transaction Reference: TXN-2026-00123\n"
            "Amount: EUR 85.69\n"
        )
        assert _looks_like_transaction_confirmation_text(text) is True

    def test_rejects_plain_invoice(self):
        """Text with invoice/total but no confirmation markers returns False."""
        text = (
            "ACME LTD\n"
            "Invoice No: INV-001\n"
            "Total: 500.00 EUR\n"
            "Due Date: 01/03/2026\n"
        )
        assert _looks_like_transaction_confirmation_text(text) is False

    def test_rejects_statement(self):
        """Text with confirmation header but debit+credit+period is a statement."""
        text = (
            "Transaction Confirmation\n"
            "Period: 01/01/2026\n"
            "Debit    Credit\n"
            "100.00   200.00\n"
        )
        assert _looks_like_transaction_confirmation_text(text) is False

    def test_requires_both_gates(self):
        """Confirmation header alone, without transaction markers, returns False."""
        text = "Transaction Confirmation\n" "Date: 05/02/2026\n" "Bank of Cyprus\n"
        assert _looks_like_transaction_confirmation_text(text) is False

    def test_detects_greek_bank_transfer(self):
        """Greek bank transfer (ΑΙΤΗΣΗ ΜΕΤΑΦΟΡΑΣ + Δικαιούχος) is detected."""
        text = (
            "ΑΙΤΗΣΗ ΜΕΤΑΦΟΡΑΣ ΚΕΦΑΛΑΙΩΝ\n"
            "Ημερομηνία: 10/02/2026\n"
            "Δικαιούχος: ACME LTD\n"
            "Ποσό: EUR 250.00\n"
        )
        assert _looks_like_transaction_confirmation_text(text) is True

    def test_detects_english_payment_order(self):
        """English PAYMENT ORDER with beneficiary is detected."""
        text = (
            "PAYMENT ORDER\n"
            "Date: 15/02/2026\n"
            "Beneficiary: ACME COMPANY\n"
            "Amount: EUR 1000.00\n"
        )
        assert _looks_like_transaction_confirmation_text(text) is True

    def test_detects_russian_payment_order(self):
        """Russian payment order (Платежное поручение + Получатель) is detected."""
        text = (
            "Платежное поручение\n"
            "Дата: 20/02/2026\n"
            "Получатель: ООО РОГА\n"
            "Сумма: 5000.00 руб\n"
        )
        assert _looks_like_transaction_confirmation_text(text) is True

    def test_detects_german_zahlungsauftrag(self):
        """German Zahlungsauftrag with Empfänger is detected."""
        text = (
            "Zahlungsauftrag\n"
            "Datum: 20/02/2026\n"
            "Empfänger: MUSTERMANN GMBH\n"
            "Betrag: EUR 500.00\n"
        )
        assert _looks_like_transaction_confirmation_text(text) is True

    def test_rejects_greek_text_without_transaction_markers(self):
        """Greek header alone without beneficiary marker returns False."""
        text = (
            "ΑΙΤΗΣΗ ΜΕΤΑΦΟΡΑΣ ΚΕΦΑΛΑΙΩΝ\n"
            "Ημερομηνία: 10/02/2026\n"
            "Ποσό: EUR 250.00\n"
        )
        assert _looks_like_transaction_confirmation_text(text) is False
