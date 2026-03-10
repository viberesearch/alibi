"""Tests for the doctr OCR integration module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alibi.extraction.doctr_ocr import (
    DoctrOcrResult,
    OcrLine,
    OcrWord,
    _build_result,
    compare_ocr,
    doctr_ocr_image,
    is_doctr_available,
)

# ---------------------------------------------------------------------------
# Mock doctr export structure (matches real doctr output format)
# ---------------------------------------------------------------------------

MOCK_EXPORT = {
    "pages": [
        {
            "blocks": [
                {
                    "geometry": ((0.0, 0.0), (1.0, 0.15)),
                    "lines": [
                        {
                            "geometry": ((0.05, 0.02), (0.95, 0.06)),
                            "words": [
                                {
                                    "value": "PAPAS",
                                    "confidence": 0.98,
                                    "geometry": ((0.05, 0.02), (0.35, 0.06)),
                                },
                                {
                                    "value": "HYPERMARKET",
                                    "confidence": 0.96,
                                    "geometry": ((0.38, 0.02), (0.95, 0.06)),
                                },
                            ],
                        },
                        {
                            "geometry": ((0.1, 0.07), (0.9, 0.10)),
                            "words": [
                                {
                                    "value": "Tombs",
                                    "confidence": 0.92,
                                    "geometry": ((0.1, 0.07), (0.35, 0.10)),
                                },
                                {
                                    "value": "of",
                                    "confidence": 0.99,
                                    "geometry": ((0.37, 0.07), (0.42, 0.10)),
                                },
                                {
                                    "value": "the",
                                    "confidence": 0.99,
                                    "geometry": ((0.44, 0.07), (0.52, 0.10)),
                                },
                                {
                                    "value": "Kings",
                                    "confidence": 0.95,
                                    "geometry": ((0.54, 0.07), (0.75, 0.10)),
                                },
                                {
                                    "value": "23",
                                    "confidence": 0.97,
                                    "geometry": ((0.77, 0.07), (0.9, 0.10)),
                                },
                            ],
                        },
                    ],
                },
                {
                    "geometry": ((0.0, 0.5), (1.0, 0.6)),
                    "lines": [
                        {
                            "geometry": ((0.05, 0.5), (0.95, 0.55)),
                            "words": [
                                {
                                    "value": "MILK",
                                    "confidence": 0.94,
                                    "geometry": ((0.05, 0.5), (0.4, 0.55)),
                                },
                                {
                                    "value": "2.69",
                                    "confidence": 0.91,
                                    "geometry": ((0.75, 0.5), (0.95, 0.55)),
                                },
                            ],
                        },
                    ],
                },
            ]
        }
    ]
}


def _mock_doc_result() -> MagicMock:
    """Create a mock doctr Document result."""
    mock = MagicMock()
    mock.export.return_value = MOCK_EXPORT
    mock.render.return_value = "PAPAS HYPERMARKET\nTombs of the Kings 23\nMILK 2.69"
    return mock


# ---------------------------------------------------------------------------
# is_doctr_available
# ---------------------------------------------------------------------------


class TestIsDoctrAvailable:
    def test_available_when_installed(self):
        import alibi.extraction.doctr_ocr as mod

        mod._doctr_available = None
        with patch.dict("sys.modules", {"doctr": MagicMock()}):
            assert mod.is_doctr_available() is True

    def test_unavailable_when_not_installed(self):
        import alibi.extraction.doctr_ocr as mod

        mod._doctr_available = None
        with patch.dict("sys.modules", {"doctr": None}):
            with patch("builtins.__import__", side_effect=ImportError("no doctr")):
                assert mod.is_doctr_available() is False

    def test_result_cached(self):
        import alibi.extraction.doctr_ocr as mod

        mod._doctr_available = True
        assert mod.is_doctr_available() is True
        mod._doctr_available = None  # Reset for other tests


# ---------------------------------------------------------------------------
# _build_result
# ---------------------------------------------------------------------------


class TestBuildResult:
    def test_builds_lines_and_words(self):
        mock_doc = _mock_doc_result()
        result = _build_result(mock_doc, "/test/img.jpg", 0.5, (1000, 800))

        assert isinstance(result, DoctrOcrResult)
        assert result.ocr_time_s == 0.5
        assert result.image_path == "/test/img.jpg"
        assert result.page_dimensions == (1000, 800)
        assert len(result.lines) == 3

    def test_first_line_text(self):
        mock_doc = _mock_doc_result()
        result = _build_result(mock_doc, "/test/img.jpg", 0.1, (100, 100))

        assert result.lines[0].text == "PAPAS HYPERMARKET"

    def test_words_in_line(self):
        mock_doc = _mock_doc_result()
        result = _build_result(mock_doc, "/test/img.jpg", 0.1, (100, 100))

        first_line = result.lines[0]
        assert len(first_line.words) == 2
        assert first_line.words[0].text == "PAPAS"
        assert first_line.words[1].text == "HYPERMARKET"

    def test_word_confidence(self):
        mock_doc = _mock_doc_result()
        result = _build_result(mock_doc, "/test/img.jpg", 0.1, (100, 100))

        assert result.lines[0].words[0].confidence == 0.98

    def test_word_geometry(self):
        mock_doc = _mock_doc_result()
        result = _build_result(mock_doc, "/test/img.jpg", 0.1, (100, 100))

        geo = result.lines[0].words[0].geometry
        assert geo == ((0.05, 0.02), (0.35, 0.06))

    def test_line_confidence_is_mean(self):
        mock_doc = _mock_doc_result()
        result = _build_result(mock_doc, "/test/img.jpg", 0.1, (100, 100))

        first_line = result.lines[0]
        expected = (0.98 + 0.96) / 2
        assert abs(first_line.confidence - expected) < 0.01

    def test_full_text(self):
        mock_doc = _mock_doc_result()
        result = _build_result(mock_doc, "/test/img.jpg", 0.1, (100, 100))

        assert "PAPAS HYPERMARKET" in result.text
        assert "MILK 2.69" in result.text

    def test_raw_export_preserved(self):
        mock_doc = _mock_doc_result()
        result = _build_result(mock_doc, "/test/img.jpg", 0.1, (100, 100))

        assert result.raw_export == MOCK_EXPORT

    def test_empty_page(self):
        mock_doc = MagicMock()
        mock_doc.export.return_value = {"pages": [{"blocks": []}]}
        result = _build_result(mock_doc, "/test/img.jpg", 0.1, (100, 100))

        assert result.text == ""
        assert result.lines == []


# ---------------------------------------------------------------------------
# doctr_ocr_image
# ---------------------------------------------------------------------------


class TestDoctrOcrImage:
    def test_import_error_when_unavailable(self):
        import alibi.extraction.doctr_ocr as mod

        mod._doctr_available = False
        with pytest.raises(ImportError, match="python-doctr"):
            doctr_ocr_image(Path("/fake/image.jpg"))
        mod._doctr_available = None

    def test_file_not_found(self):
        import alibi.extraction.doctr_ocr as mod

        mod._doctr_available = True
        with pytest.raises(FileNotFoundError):
            doctr_ocr_image(Path("/nonexistent/image.jpg"))
        mod._doctr_available = None

    @patch("alibi.extraction.doctr_ocr._get_predictor")
    @patch("alibi.extraction.doctr_ocr.is_doctr_available", return_value=True)
    def test_returns_result(self, mock_avail, mock_pred):
        mock_predictor = MagicMock()
        mock_predictor.return_value = _mock_doc_result()
        mock_pred.return_value = mock_predictor

        with patch("pathlib.Path.exists", return_value=True):
            with patch("PIL.Image.open") as mock_open:
                mock_img = MagicMock()
                mock_img.height = 1000
                mock_img.width = 800
                mock_open.return_value = mock_img

                with patch("numpy.array", return_value="fake_np"):
                    result = doctr_ocr_image(Path("/fake/receipt.jpg"))

        assert isinstance(result, DoctrOcrResult)
        assert len(result.lines) == 3
        assert result.page_dimensions == (1000, 800)
        assert "PAPAS" in result.text


# ---------------------------------------------------------------------------
# compare_ocr
# ---------------------------------------------------------------------------


class TestCompareOcr:
    @patch("alibi.extraction.doctr_ocr.doctr_ocr_image")
    @patch("alibi.extraction.doctr_ocr.is_doctr_available", return_value=True)
    def test_doctr_only(self, mock_avail, mock_doctr):
        mock_doctr.return_value = DoctrOcrResult(
            text="PAPAS HYPERMARKET",
            lines=[
                OcrLine(
                    text="PAPAS HYPERMARKET",
                    confidence=0.97,
                    geometry=((0, 0), (1, 0.1)),
                    words=[
                        OcrWord("PAPAS", 0.98, ((0, 0), (0.5, 0.1))),
                        OcrWord("HYPERMARKET", 0.96, ((0.5, 0), (1, 0.1))),
                    ],
                )
            ],
            ocr_time_s=0.3,
        )

        with patch(
            "alibi.extraction.ocr.ocr_image_with_retry",
            side_effect=Exception("no ollama"),
        ):
            result = compare_ocr(Path("/fake/img.jpg"))

        assert "text" in result["doctr"]
        assert "error" in result["ollama"]
        assert "comparison" not in result  # only one side worked

    @patch("alibi.extraction.doctr_ocr.is_doctr_available", return_value=False)
    def test_doctr_unavailable(self, mock_avail):
        with patch(
            "alibi.extraction.ocr.ocr_image_with_retry",
            side_effect=Exception("no ollama"),
        ):
            result = compare_ocr(Path("/fake/img.jpg"))

        assert "error" in result["doctr"]
        assert "not installed" in result["doctr"]["error"]

    @patch("alibi.extraction.doctr_ocr.doctr_ocr_image")
    @patch("alibi.extraction.doctr_ocr.is_doctr_available", return_value=True)
    def test_comparison_when_both_available(self, mock_avail, mock_doctr):
        mock_doctr.return_value = DoctrOcrResult(
            text="Receipt text from doctr",
            lines=[],
            ocr_time_s=0.2,
        )

        with patch(
            "alibi.extraction.ocr.ocr_image_with_retry",
            return_value=("Receipt text from ollama", False),
        ):
            result = compare_ocr(Path("/fake/img.jpg"))

        assert "comparison" in result
        comp = result["comparison"]
        assert comp["doctr_chars"] > 0
        assert comp["ollama_chars"] > 0
        assert comp["faster"] in ("doctr", "ollama")
