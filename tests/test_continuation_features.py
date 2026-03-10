"""Tests for continuation features: type_map consolidation, Obsidian notes,
PDF type detection, statement vendor cleanup, and image slicing."""

import io
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alibi.db.models import (
    Artifact,
    DocumentStatus,
    DocumentType,
    RecordType,
)
from alibi.obsidian.notes import (
    generate_contract_note,
    generate_warranty_note,
)
from alibi.processing.pipeline import (
    ARTIFACT_TO_RECORD_TYPE,
    STR_TO_ARTIFACT_TYPE,
    ProcessingPipeline,
    _LLM_OVERRIDABLE_TYPES,
)


# ============================================================
# Task 1: STR_TO_ARTIFACT_TYPE consolidation
# ============================================================


class TestStrToDocumentType:
    """Tests for the consolidated STR_TO_ARTIFACT_TYPE mapping."""

    def test_contains_all_processable_types(self):
        expected = {
            "receipt",
            "invoice",
            "payment_confirmation",
            "statement",
            "warranty",
            "contract",
        }
        assert set(STR_TO_ARTIFACT_TYPE.keys()) == expected

    def test_maps_to_correct_enum_values(self):
        assert STR_TO_ARTIFACT_TYPE["receipt"] == DocumentType.RECEIPT
        assert STR_TO_ARTIFACT_TYPE["invoice"] == DocumentType.INVOICE
        assert STR_TO_ARTIFACT_TYPE["statement"] == DocumentType.STATEMENT
        assert STR_TO_ARTIFACT_TYPE["warranty"] == DocumentType.WARRANTY
        assert STR_TO_ARTIFACT_TYPE["contract"] == DocumentType.CONTRACT
        assert (
            STR_TO_ARTIFACT_TYPE["payment_confirmation"]
            == DocumentType.PAYMENT_CONFIRMATION
        )

    def test_excludes_other_and_policy(self):
        assert "other" not in STR_TO_ARTIFACT_TYPE
        assert "policy" not in STR_TO_ARTIFACT_TYPE

    def test_vision_type_map_is_same_object(self):
        assert ProcessingPipeline._VISION_TYPE_MAP is STR_TO_ARTIFACT_TYPE

    def test_llm_overridable_types_subset(self):
        for t in _LLM_OVERRIDABLE_TYPES:
            assert t in STR_TO_ARTIFACT_TYPE

    def test_llm_overridable_excludes_receipt(self):
        assert "receipt" not in _LLM_OVERRIDABLE_TYPES

    def test_llm_overridable_includes_statement(self):
        assert "statement" in _LLM_OVERRIDABLE_TYPES


class TestTypeToStr:
    """Tests for _type_to_str after consolidation."""

    def test_receipt(self):
        assert ProcessingPipeline._type_to_str(DocumentType.RECEIPT) == "receipt"

    def test_invoice(self):
        assert ProcessingPipeline._type_to_str(DocumentType.INVOICE) == "invoice"

    def test_contract(self):
        """Contract was missing before consolidation — now works."""
        assert ProcessingPipeline._type_to_str(DocumentType.CONTRACT) == "contract"

    def test_warranty(self):
        assert ProcessingPipeline._type_to_str(DocumentType.WARRANTY) == "warranty"

    def test_statement(self):
        assert ProcessingPipeline._type_to_str(DocumentType.STATEMENT) == "statement"

    def test_payment_confirmation(self):
        assert (
            ProcessingPipeline._type_to_str(DocumentType.PAYMENT_CONFIRMATION)
            == "payment_confirmation"
        )

    def test_other_falls_back_to_receipt(self):
        assert ProcessingPipeline._type_to_str(DocumentType.OTHER) == "receipt"


# ============================================================
# Task 2: Obsidian notes for contracts and warranties
# ============================================================


class TestGenerateContractNote:
    """Tests for generate_contract_note function."""

    @pytest.fixture
    def contract_artifact(self):
        return Artifact(
            id="art-contract-1",
            space_id="default",
            type=DocumentType.CONTRACT,
            file_path="/path/to/contract.pdf",
            file_hash="abc123",
            vendor="Acme Corp",
            vendor_address="123 Main St",
            document_id="CTR-2024-001",
            document_date=date(2024, 3, 1),
            amount=Decimal("1200.00"),
            currency="EUR",
            status=DocumentStatus.PROCESSED,
        )

    def test_generates_valid_frontmatter(self, contract_artifact):
        result = generate_contract_note(contract_artifact)
        assert "---" in result
        assert "type: contract" in result
        assert 'vendor: "Acme Corp"' in result
        assert "amount: 1200.00" in result

    def test_includes_contract_details(self, contract_artifact):
        data = {
            "payment_terms": "monthly",
            "renewal": "auto",
            "start_date": "2024-03-01",
            "end_date": "2025-03-01",
        }
        result = generate_contract_note(contract_artifact, extracted_data=data)
        assert "payment_terms: monthly" in result
        assert "renewal: auto" in result
        assert "start_date: 2024-03-01" in result
        assert "end_date: 2025-03-01" in result

    def test_includes_vendor_details(self, contract_artifact):
        result = generate_contract_note(contract_artifact)
        assert "**Address**: 123 Main St" in result

    def test_includes_document_id(self, contract_artifact):
        result = generate_contract_note(contract_artifact)
        assert "CTR-2024-001" in result

    def test_includes_line_items(self, contract_artifact):
        items = [{"name": "Consulting Services", "total_price": "1200.00"}]
        result = generate_contract_note(contract_artifact, line_items=items)
        assert "Consulting Services" in result

    def test_heading_format(self, contract_artifact):
        result = generate_contract_note(contract_artifact)
        assert "# Contract: Acme Corp" in result

    def test_formatted_amount(self, contract_artifact):
        result = generate_contract_note(contract_artifact)
        assert "\u20ac1,200.00" in result


class TestGenerateWarrantyNote:
    """Tests for generate_warranty_note function."""

    @pytest.fixture
    def warranty_artifact(self):
        return Artifact(
            id="art-warranty-1",
            space_id="default",
            type=DocumentType.WARRANTY,
            file_path="/path/to/warranty.pdf",
            file_hash="def456",
            vendor="Samsung",
            vendor_address="456 Tech Ave",
            document_id="WRN-2024-001",
            document_date=date(2024, 1, 15),
            amount=Decimal("899.00"),
            currency="EUR",
            status=DocumentStatus.PROCESSED,
        )

    def test_generates_valid_frontmatter(self, warranty_artifact):
        result = generate_warranty_note(warranty_artifact)
        assert "---" in result
        assert "type: warranty" in result
        assert 'vendor: "Samsung"' in result

    def test_includes_product_details(self, warranty_artifact):
        data = {
            "product": "Galaxy S24 Ultra",
            "model": "SM-S928B",
            "serial_number": "R5CW12345",
            "warranty_type": "manufacturer",
            "warranty_end": "2026-01-15",
        }
        result = generate_warranty_note(warranty_artifact, extracted_data=data)
        assert "Galaxy S24 Ultra" in result
        assert "SM-S928B" in result
        assert "R5CW12345" in result
        assert "manufacturer" in result
        assert "2026-01-15" in result

    def test_heading_uses_product_name(self, warranty_artifact):
        data = {"product": "Galaxy S24 Ultra"}
        result = generate_warranty_note(warranty_artifact, extracted_data=data)
        assert "# Warranty: Galaxy S24 Ultra" in result

    def test_heading_falls_back_to_vendor(self, warranty_artifact):
        result = generate_warranty_note(warranty_artifact)
        assert "# Warranty: Samsung" in result

    def test_coverage_section(self, warranty_artifact):
        data = {"coverage": "Screen, battery, and motherboard defects"}
        result = generate_warranty_note(warranty_artifact, extracted_data=data)
        assert "Screen, battery, and motherboard defects" in result

    def test_no_coverage_placeholder(self, warranty_artifact):
        result = generate_warranty_note(warranty_artifact)
        assert "_No coverage details_" in result


# ============================================================
# Task 3: PDF document type detection
# ============================================================


class TestPdfTypeDetection:
    """Tests for PDF type detection via first-page rendering."""

    @pytest.fixture
    def pipeline(self):
        with patch("alibi.processing.pipeline.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                get_unit_aliases_path=MagicMock(return_value=None),
                get_vendor_aliases_path=MagicMock(return_value=None),
            )
            with patch("alibi.processing.pipeline.init_unit_mappings"):
                with patch("alibi.processing.pipeline.init_vendor_mappings"):
                    return ProcessingPipeline(db=MagicMock())

    @patch("alibi.processing.pipeline.vision_detect_document_type")
    @patch("pdf2image.convert_from_path")
    def test_pdf_uses_vision_detection(self, mock_convert, mock_vision, pipeline):
        mock_vision.return_value = "statement"
        from PIL import Image

        img = Image.new("RGB", (100, 100))
        mock_convert.return_value = [img]
        result = pipeline._detect_pdf_type(Path("/tmp/test.pdf"))
        assert result == DocumentType.STATEMENT

    @patch("alibi.processing.pipeline.vision_detect_document_type")
    @patch("pdf2image.convert_from_path")
    def test_pdf_falls_back_to_invoice_on_failure(
        self, mock_convert, mock_vision, pipeline
    ):
        mock_vision.side_effect = Exception("Vision failed")
        from PIL import Image

        img = Image.new("RGB", (100, 100))
        mock_convert.return_value = [img]
        result = pipeline._detect_pdf_type(Path("/tmp/test.pdf"))
        assert result == DocumentType.INVOICE

    @patch("pdf2image.convert_from_path")
    def test_pdf_falls_back_if_no_images(self, mock_convert, pipeline):
        mock_convert.return_value = []
        result = pipeline._detect_pdf_type(Path("/tmp/test.pdf"))
        assert result == DocumentType.INVOICE

    @patch("alibi.processing.pipeline.vision_detect_document_type")
    @patch("pdf2image.convert_from_path")
    def test_pdf_detects_contract(self, mock_convert, mock_vision, pipeline):
        mock_vision.return_value = "contract"
        from PIL import Image

        img = Image.new("RGB", (100, 100))
        mock_convert.return_value = [img]
        result = pipeline._detect_pdf_type(Path("/tmp/test.pdf"))
        assert result == DocumentType.CONTRACT

    @patch("alibi.processing.pipeline.vision_detect_document_type")
    @patch("pdf2image.convert_from_path")
    def test_pdf_detects_warranty(self, mock_convert, mock_vision, pipeline):
        mock_vision.return_value = "warranty"
        from PIL import Image

        img = Image.new("RGB", (100, 100))
        mock_convert.return_value = [img]
        result = pipeline._detect_pdf_type(Path("/tmp/test.pdf"))
        assert result == DocumentType.WARRANTY

    @patch("alibi.processing.pipeline.vision_detect_document_type")
    @patch("pdf2image.convert_from_path")
    def test_pdf_other_returns_invoice(self, mock_convert, mock_vision, pipeline):
        mock_vision.return_value = "other"
        from PIL import Image

        img = Image.new("RGB", (100, 100))
        mock_convert.return_value = [img]
        result = pipeline._detect_pdf_type(Path("/tmp/test.pdf"))
        assert result == DocumentType.INVOICE

    def test_detect_document_type_routes_pdf(self, pipeline):
        with patch.object(
            pipeline, "_detect_pdf_type", return_value=DocumentType.STATEMENT
        ) as mock_pdf:
            result = pipeline._detect_document_type(Path("/tmp/test.pdf"))
        assert result == DocumentType.STATEMENT
        mock_pdf.assert_called_once()

    def test_detect_document_type_routes_image(self, pipeline):
        with patch.object(
            pipeline, "_detect_image_type", return_value=DocumentType.RECEIPT
        ) as mock_img:
            result = pipeline._detect_document_type(Path("/tmp/test.jpg"))
        assert result == DocumentType.RECEIPT
        mock_img.assert_called_once()


# ============================================================
# Task 4: Statement vendor cleanup
# ============================================================


class TestDimensionJitter:
    """Tests for _compute_dimensions and _dimension_jitter."""

    def test_compute_dimensions_downscale(self):
        from alibi.extraction.vision import _compute_dimensions

        w, h = _compute_dimensions(278, 1280, 1344)
        assert w % 28 == 0
        assert h % 28 == 0
        assert h <= 1344

    def test_compute_dimensions_no_scale_needed(self):
        from alibi.extraction.vision import _compute_dimensions

        w, h = _compute_dimensions(280, 560, 1344)
        assert w == 280
        assert h == 560

    def test_compute_dimensions_rounds_to_28(self):
        from alibi.extraction.vision import _compute_dimensions

        w, h = _compute_dimensions(100, 200, 1344)
        assert w % 28 == 0
        assert h % 28 == 0

    def test_dimension_jitter_returns_variants(self):
        from alibi.extraction.vision import _dimension_jitter

        variants = _dimension_jitter(280, 1344)
        assert len(variants) == 6
        # +28 variants
        assert (308, 1344) in variants
        assert (280, 1372) in variants
        assert (308, 1372) in variants
        # +56 variants
        assert (336, 1344) in variants
        assert (280, 1400) in variants
        assert (336, 1400) in variants

    def test_dimension_jitter_minimum_dimension(self):
        from alibi.extraction.vision import _dimension_jitter

        variants = _dimension_jitter(28, 28)
        # All should be valid (>= 28)
        for w, h in variants:
            assert w >= 28
            assert h >= 28

    def test_extract_from_image_tries_jitter_on_500(self, tmp_path):
        """Legacy vision extraction tries jittered dimensions after 500 error."""
        from alibi.extraction.vision import (
            VisionExtractionError,
            _extract_from_image_legacy,
        )

        # Create a real small image
        from PIL import Image

        img = Image.new("RGB", (100, 200))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        call_count = 0

        def mock_call(url, model, prompt, images, timeout):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise VisionExtractionError("HTTP error: 500")
            return {"response": '{"vendor": "test"}'}

        with (
            patch("alibi.extraction.vision._call_ollama_vision", side_effect=mock_call),
            patch("alibi.extraction.vision.get_prompt_for_type", return_value="prompt"),
        ):
            result = _extract_from_image_legacy(img_path)
            assert result == {"vendor": "test"}
            assert call_count == 2  # First failed, second (jittered) succeeded

    def test_detect_document_type_jitter_fallback(self, tmp_path):
        """detect_document_type falls back to legacy with jittered dims on OCR 500."""
        from alibi.extraction.vision import VisionExtractionError, detect_document_type

        from PIL import Image

        img = Image.new("RGB", (100, 200))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        call_count = 0

        def mock_call(url, model, prompt, images, timeout):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                # First call (OCR model) fails with 500
                raise VisionExtractionError("HTTP error: 500")
            # Second call (legacy fallback) succeeds
            return {"response": "receipt"}

        with patch(
            "alibi.extraction.vision._call_ollama_vision", side_effect=mock_call
        ):
            doc_type = detect_document_type(img_path)
            assert doc_type == "receipt"
            assert call_count >= 2  # OCR model failed, legacy succeeded


# ============================================================
# Image Slicing for Extreme Aspect Ratios
# ============================================================


class TestNeedsSlicing:
    """Tests for _needs_slicing aspect ratio check."""

    def test_extreme_tall_image(self):
        from alibi.extraction.vision import _needs_slicing

        # 278x1280 = ratio 4.6:1
        assert _needs_slicing(278, 1280) is True

    def test_extreme_wide_image(self):
        from alibi.extraction.vision import _needs_slicing

        assert _needs_slicing(1280, 278) is True

    def test_normal_aspect_ratio(self):
        from alibi.extraction.vision import _needs_slicing

        # 800x1200 = ratio 1.5:1
        assert _needs_slicing(800, 1200) is False

    def test_square_image(self):
        from alibi.extraction.vision import _needs_slicing

        assert _needs_slicing(500, 500) is False

    def test_borderline_ratio(self):
        from alibi.extraction.vision import _needs_slicing

        # Exactly 3:1 should NOT trigger (> not >=)
        assert _needs_slicing(100, 300) is False
        # Just over 3:1 should trigger
        assert _needs_slicing(100, 301) is True

    def test_zero_dimensions(self):
        from alibi.extraction.vision import _needs_slicing

        assert _needs_slicing(0, 100) is False
        assert _needs_slicing(100, 0) is False


class TestCreateImageBands:
    """Tests for _create_image_bands slicing."""

    def test_tall_image_produces_multiple_bands(self, tmp_path):
        from PIL import Image

        from alibi.extraction.vision import _create_image_bands

        # 278x1280 — extreme tall
        img = Image.new("RGB", (278, 1280), color="white")
        img_path = tmp_path / "tall.jpg"
        img.save(img_path)

        bands = _create_image_bands(img_path)
        assert len(bands) >= 2
        # Each band should be valid JPEG
        for band_bytes in bands:
            band_img = Image.open(io.BytesIO(band_bytes))
            assert band_img.format == "JPEG"
            # Band aspect ratio should be manageable
            ratio = max(band_img.width, band_img.height) / min(
                band_img.width, band_img.height
            )
            assert ratio <= 3.0  # Well within limits

    def test_wide_image_produces_multiple_bands(self, tmp_path):
        from PIL import Image

        from alibi.extraction.vision import _create_image_bands

        # 1280x200 — extreme wide
        img = Image.new("RGB", (1280, 200), color="white")
        img_path = tmp_path / "wide.jpg"
        img.save(img_path)

        bands = _create_image_bands(img_path)
        assert len(bands) >= 2

    def test_band_count_for_papas_receipt(self, tmp_path):
        from PIL import Image

        from alibi.extraction.vision import _create_image_bands

        # Simulate PAPAS HYPERMARKET receipt dimensions
        img = Image.new("RGB", (278, 1280), color="white")
        img_path = tmp_path / "papas.jpg"
        img.save(img_path)

        bands = _create_image_bands(img_path)
        # band_h = 278 * 2 = 556, overlap = 556 * 0.15 = 83
        # Band 0: 0-556, Band 1: 473-1029, Band 2: 946-1280
        assert len(bands) == 3

    def test_single_band_if_within_ratio(self, tmp_path):
        from PIL import Image

        from alibi.extraction.vision import _create_image_bands

        # 400x600 — ratio 1.5, band_h = 400*2 = 800 > 600 → single band
        img = Image.new("RGB", (400, 600), color="white")
        img_path = tmp_path / "normal.jpg"
        img.save(img_path)

        bands = _create_image_bands(img_path)
        assert len(bands) == 1
