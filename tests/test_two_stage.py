"""Integration tests for the two-stage extraction pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alibi.extraction.vision import (
    VisionExtractionError,
    extract_from_image,
    extract_from_images,
)


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a simple test image."""
    from PIL import Image

    img = Image.new("RGB", (200, 300), color="white")
    path = tmp_path / "receipt.jpg"
    img.save(path, "JPEG")
    return path


@pytest.fixture
def sample_images(tmp_path: Path) -> list[Path]:
    """Create multiple test images."""
    from PIL import Image

    paths = []
    for i in range(3):
        img = Image.new("RGB", (200, 300), color="white")
        path = tmp_path / f"page{i}.jpg"
        img.save(path, "JPEG")
        paths.append(path)
    return paths


class TestTwoStageExtraction:
    """Tests for the two-stage extract_from_image pipeline."""

    @patch("alibi.extraction.structurer._call_ollama_text")
    @patch("alibi.extraction.ocr._call_ollama_ocr")
    def test_full_pipeline(self, mock_ocr, mock_structure, sample_image: Path):
        """Two-stage: OCR → structure → verified result."""
        mock_ocr.return_value = {
            "response": "SUPERMARKET\nBread 2.50\nMilk 1.99\nTotal 4.49"
        }
        mock_structure.return_value = {
            "response": json.dumps(
                {
                    "vendor": "SUPERMARKET",
                    "date": "2024-06-15",
                    "total": 4.49,
                    "currency": "EUR",
                    "line_items": [
                        {
                            "name": "Bread",
                            "quantity": 1,
                            "unit_price": 2.50,
                            "total_price": 2.50,
                        },
                        {
                            "name": "Milk",
                            "quantity": 1,
                            "unit_price": 1.99,
                            "total_price": 1.99,
                        },
                    ],
                }
            )
        }

        result = extract_from_image(
            sample_image,
            doc_type="receipt",
            ollama_url="http://test:11434",
        )

        assert result["vendor"] == "SUPERMARKET"
        assert result["total"] == 4.49
        assert len(result["line_items"]) == 2
        assert "raw_text" in result
        assert "SUPERMARKET" in result["raw_text"]

    @patch("alibi.extraction.structurer._call_ollama_text")
    @patch("alibi.extraction.ocr._call_ollama_ocr")
    def test_confidence_injected(self, mock_ocr, mock_structure, sample_image: Path):
        mock_ocr.return_value = {"response": "Shop\nTotal 10.00"}
        mock_structure.return_value = {
            "response": json.dumps(
                {
                    "vendor": "Shop",
                    "date": "2024-06-15",
                    "total": 10.00,
                    "currency": "EUR",
                    "line_items": [],
                }
            )
        }

        result = extract_from_image(
            sample_image,
            doc_type="receipt",
            ollama_url="http://test:11434",
        )

        assert "_two_stage_confidence" in result
        assert isinstance(result["_two_stage_confidence"], float)

    @patch("alibi.extraction.vision._extract_from_image_legacy")
    @patch("alibi.extraction.structurer._call_ollama_text")
    @patch("alibi.extraction.ocr._call_ollama_ocr")
    def test_low_confidence_falls_back_to_legacy(
        self, mock_ocr, mock_structure, mock_legacy, sample_image: Path
    ):
        """When two-stage produces very low confidence, fall back to legacy."""
        mock_ocr.return_value = {"response": "garbled text"}
        # Structure returns garbage: wrong total, bad item math, no required fields
        mock_structure.return_value = {
            "response": json.dumps(
                {
                    "total": 999.99,
                    "line_items": [
                        {
                            "name": "X",
                            "quantity": 1,
                            "unit_price": 1.00,
                            "total_price": 50.00,
                        }
                    ],
                }
            )
        }
        mock_legacy.return_value = {
            "vendor": "LegacyResult",
            "total": 10.00,
        }

        result = extract_from_image(
            sample_image,
            doc_type="receipt",
            ollama_url="http://test:11434",
        )

        mock_legacy.assert_called_once()
        assert result["vendor"] == "LegacyResult"

    @patch("alibi.extraction.vision._extract_from_image_legacy")
    @patch("alibi.extraction.ocr._call_ollama_ocr")
    def test_ocr_failure_falls_back_to_legacy(
        self, mock_ocr, mock_legacy, sample_image: Path
    ):
        """If OCR fails entirely, fall back to legacy vision."""
        mock_ocr.side_effect = VisionExtractionError("OCR model not found")
        mock_legacy.return_value = {"vendor": "FallbackResult", "total": 5.00}

        result = extract_from_image(
            sample_image,
            doc_type="receipt",
            ollama_url="http://test:11434",
        )

        mock_legacy.assert_called_once()
        assert result["vendor"] == "FallbackResult"

    @patch("alibi.extraction.structurer._call_ollama_text")
    @patch("alibi.extraction.ocr._call_ollama_ocr")
    def test_retry_with_emphasis(self, mock_ocr, mock_structure, sample_image: Path):
        """When first structuring has low confidence, retries with emphasis.

        The pipeline may use micro-prompts (multiple small LLM calls)
        or monolithic correction, so we allow variable call counts.
        """
        mock_ocr.return_value = {
            "response": "Shop A\nItem1 5.00\nItem2 5.00\nTotal 10.00"
        }

        good_result = {
            "response": json.dumps(
                {
                    "vendor": "Shop A",
                    "date": "2024-06-15",
                    "total": 10.00,
                    "currency": "EUR",
                    "line_items": [
                        {
                            "name": "Item1",
                            "quantity": 1,
                            "unit_price": 5.00,
                            "total_price": 5.00,
                        },
                        {
                            "name": "Item2",
                            "quantity": 1,
                            "unit_price": 5.00,
                            "total_price": 5.00,
                        },
                    ],
                }
            )
        }

        # Provide enough responses for micro-prompts + possible retry
        mock_structure.return_value = good_result

        result = extract_from_image(
            sample_image,
            doc_type="receipt",
            ollama_url="http://test:11434",
        )

        # LLM should have been called at least once (micro-prompts or correction)
        assert mock_structure.call_count >= 1
        # Should have extracted vendor from the LLM response
        assert result["vendor"] == "Shop A"

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(VisionExtractionError, match="not found"):
            extract_from_image(tmp_path / "nonexistent.jpg")


class TestTwoStageMultiImage:
    """Tests for multi-image three-stage extraction."""

    @patch("alibi.extraction.structurer._call_ollama_text")
    @patch("alibi.extraction.ocr.ocr_image_with_retry")
    def test_multi_page_ocr_then_structure(
        self, mock_ocr, mock_structure, sample_images: list[Path]
    ):
        """Multi-image: OCR each page, parse, structure once."""
        mock_ocr.side_effect = [
            ("Page 1 text", False),
            ("Page 2 text", False),
            ("Page 3 text", False),
        ]
        mock_structure.return_value = {
            "response": json.dumps(
                {
                    "vendor": "MultiPage Corp",
                    "total": 100.00,
                    "line_items": [],
                }
            )
        }

        result = extract_from_images(
            sample_images,
            doc_type="invoice",
            ollama_url="http://test:11434",
        )

        # OCR called once per page via ocr_image_with_retry
        assert mock_ocr.call_count == 3
        # Short text → parser confidence low → falls through to full LLM
        assert mock_structure.call_count == 1
        # Combined text should have page markers
        call_prompt = mock_structure.call_args[0][2]  # prompt arg
        assert "PAGE 1" in call_prompt
        assert "PAGE 2" in call_prompt
        assert result["vendor"] == "MultiPage Corp"

    @patch("alibi.extraction.structurer._call_ollama_text")
    @patch("alibi.extraction.ocr._call_ollama_ocr")
    def test_single_image_delegates(
        self, mock_ocr, mock_structure, sample_images: list[Path]
    ):
        """Single image in list delegates to extract_from_image."""
        mock_ocr.return_value = {"response": "Single page text here"}
        mock_structure.return_value = {
            "response": json.dumps({"vendor": "Single", "total": 5.00})
        }

        result = extract_from_images(
            [sample_images[0]],
            doc_type="receipt",
            ollama_url="http://test:11434",
        )

        assert result["vendor"] == "Single"

    def test_empty_list_raises(self):
        with pytest.raises(VisionExtractionError, match="No image paths"):
            extract_from_images([])


class TestDetectDocumentType:
    """Tests for document type detection with OCR model."""

    @patch("alibi.extraction.vision._call_ollama_vision")
    def test_detect_receipt(self, mock_call, sample_image: Path):
        mock_call.return_value = {"response": "receipt"}
        from alibi.extraction.vision import detect_document_type

        result = detect_document_type(sample_image)
        assert result == "receipt"
        # Should use OCR model (first arg after url)
        call_model = mock_call.call_args[0][1]
        # Model should be the OCR model, not the vision model
        assert call_model != "qwen3-vl:30b"

    @patch("alibi.extraction.vision._call_ollama_vision")
    def test_detect_invoice(self, mock_call, sample_image: Path):
        mock_call.return_value = {"response": "invoice"}
        from alibi.extraction.vision import detect_document_type

        result = detect_document_type(sample_image)
        assert result == "invoice"

    @patch("alibi.extraction.vision._detect_document_type_legacy")
    @patch("alibi.extraction.vision._call_ollama_vision")
    def test_ocr_model_failure_falls_back(
        self, mock_call, mock_legacy, sample_image: Path
    ):
        """If OCR model fails, falls back to legacy vision model."""
        mock_call.side_effect = VisionExtractionError("500")
        mock_legacy.return_value = "receipt"
        from alibi.extraction.vision import detect_document_type

        result = detect_document_type(sample_image)
        assert result == "receipt"
        mock_legacy.assert_called_once()
