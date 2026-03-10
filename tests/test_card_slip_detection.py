"""Tests for card terminal slip detection and reclassification.

Tests that the pipeline correctly classifies card terminal slips as
payment_confirmation instead of receipt, via both LLM vision detection
and post-extraction heuristic reclassification.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from alibi.db.connection import DatabaseManager
from alibi.db.models import DocumentType, RecordType
from alibi.processing.pipeline import ProcessingPipeline, ProcessingResult


@pytest.fixture
def db_manager(tmp_path: Path) -> DatabaseManager:
    """Create a fresh DB for testing."""
    from alibi.config import Config

    config = Config(db_path=tmp_path / "test.db")
    db = DatabaseManager(config)
    db.initialize()
    return db


@pytest.fixture
def pipeline(db_manager: DatabaseManager) -> ProcessingPipeline:
    """Create a pipeline with the test DB."""
    return ProcessingPipeline(db=db_manager)


def _mock_extraction(
    vendor: str = "Nut Cracker House",
    total: float = 12.45,
    doc_date: str = "2025-01-15",
    time: str = "14:30:00",
    payment_method: str | None = None,
    card_last4: str | None = None,
    authorization_code: str | None = None,
    terminal_id: str | None = None,
    document_type: str | None = None,
    line_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a mock extraction result."""
    result: dict[str, Any] = {
        "vendor": vendor,
        "total": total,
        "date": doc_date,
        "time": time,
        "currency": "EUR",
        "raw_text": f"Document from {vendor}",
        "line_items": line_items or [],
    }
    if payment_method:
        result["payment_method"] = payment_method
    if card_last4:
        result["card_last4"] = card_last4
    if authorization_code:
        result["authorization_code"] = authorization_code
    if terminal_id:
        result["terminal_id"] = terminal_id
    if document_type:
        result["document_type"] = document_type
    return result


def _process_with_mock(
    pipeline: ProcessingPipeline,
    tmp_path: Path,
    filename: str,
    extraction: dict[str, Any],
    vision_type: str = "receipt",
) -> ProcessingResult:
    """Process a mock file through the pipeline."""
    file_path = tmp_path / filename
    file_path.write_text(f"mock content for {filename}")

    with (
        patch.object(pipeline, "_extract_document", return_value=extraction),
        patch(
            "alibi.processing.pipeline.vision_detect_document_type",
            return_value=vision_type,
        ),
        patch(
            "alibi.matching.duplicates.compute_perceptual_hash",
            return_value="0000000000000000",
        ),
    ):
        return pipeline.process_file(file_path)


class TestVisionTypeDetection:
    """Test that vision-based type detection classifies card slips correctly."""

    def test_vision_detects_payment_confirmation(
        self, pipeline: ProcessingPipeline, tmp_path: Path
    ):
        """When vision returns payment_confirmation, record_type is PAYMENT."""
        extraction = _mock_extraction(
            payment_method="card",
            card_last4="7514",
        )
        result = _process_with_mock(
            pipeline,
            tmp_path,
            "card_slip.jpg",
            extraction,
            vision_type="payment_confirmation",
        )

        assert result.success
        assert result.document_id is not None
        assert result.record_type == RecordType.PAYMENT

    def test_vision_detects_receipt(self, pipeline: ProcessingPipeline, tmp_path: Path):
        """When vision returns receipt, record_type is PURCHASE."""
        extraction = _mock_extraction(
            line_items=[
                {
                    "name": "Milk",
                    "quantity": 1,
                    "unit_price": 2.50,
                    "total_price": 2.50,
                },
            ]
        )
        result = _process_with_mock(
            pipeline,
            tmp_path,
            "receipt.jpg",
            extraction,
            vision_type="receipt",
        )

        assert result.success
        assert result.record_type == RecordType.PURCHASE

    def test_vision_failure_falls_back_to_receipt(
        self, pipeline: ProcessingPipeline, tmp_path: Path
    ):
        """If vision detection fails, default to receipt (PURCHASE)."""
        extraction = _mock_extraction(
            line_items=[
                {
                    "name": "Bread",
                    "quantity": 1,
                    "unit_price": 1.80,
                    "total_price": 1.80,
                },
            ]
        )
        file_path = tmp_path / "receipt.jpg"
        file_path.write_text("mock content")

        with (
            patch.object(pipeline, "_extract_document", return_value=extraction),
            patch(
                "alibi.processing.pipeline.vision_detect_document_type",
                side_effect=Exception("Ollama down"),
            ),
            patch(
                "alibi.matching.duplicates.compute_perceptual_hash",
                return_value="0000000000000000",
            ),
        ):
            result = pipeline.process_file(file_path)

        assert result.success
        assert result.record_type == RecordType.PURCHASE

    def test_vision_other_falls_back_to_receipt(
        self, pipeline: ProcessingPipeline, tmp_path: Path
    ):
        """If vision returns 'other', default to receipt for images."""
        extraction = _mock_extraction(
            line_items=[
                {
                    "name": "Eggs",
                    "quantity": 1,
                    "unit_price": 3.00,
                    "total_price": 3.00,
                },
            ]
        )
        result = _process_with_mock(
            pipeline,
            tmp_path,
            "photo.jpg",
            extraction,
            vision_type="other",
        )

        assert result.success
        assert result.record_type == RecordType.PURCHASE

    def test_pdf_still_defaults_to_invoice(self, pipeline: ProcessingPipeline):
        """PDFs should still default to invoice (no vision detection)."""
        doc_type = pipeline._detect_document_type(Path("/fake/doc.pdf"))
        assert doc_type == DocumentType.INVOICE

    def test_txt_returns_other(self, pipeline: ProcessingPipeline):
        """Unsupported file types return OTHER."""
        doc_type = pipeline._detect_document_type(Path("/fake/doc.txt"))
        assert doc_type == DocumentType.OTHER


class TestHeuristicReclassification:
    """Test the post-extraction heuristic that catches misclassified card slips."""

    def test_no_items_with_card_details_reclassified(
        self, pipeline: ProcessingPipeline, tmp_path: Path
    ):
        """Card slip with no real items + card_last4 gets reclassified."""
        extraction = _mock_extraction(
            payment_method="contactless",
            card_last4="4321",
            line_items=[],
        )
        result = _process_with_mock(
            pipeline,
            tmp_path,
            "card_slip.jpg",
            extraction,
            vision_type="receipt",  # Vision got it wrong
        )

        assert result.success
        assert result.record_type == RecordType.PAYMENT

    def test_phantom_items_with_card_details_reclassified(
        self, pipeline: ProcessingPipeline, tmp_path: Path
    ):
        """Card slip with phantom items (PURCHASE, SALE) + card_last4 reclassified."""
        extraction = _mock_extraction(
            payment_method="card",
            card_last4="9876",
            line_items=[
                {"name": "PURCHASE", "total_price": 12.45},
                {"name": "TOTAL", "total_price": 12.45},
            ],
        )
        result = _process_with_mock(
            pipeline,
            tmp_path,
            "terminal.jpg",
            extraction,
            vision_type="receipt",
        )

        assert result.success
        assert result.record_type == RecordType.PAYMENT

    def test_terminal_id_triggers_reclassification(
        self, pipeline: ProcessingPipeline, tmp_path: Path
    ):
        """Terminal ID alone (without card_last4) should trigger reclassification."""
        extraction = _mock_extraction(
            terminal_id="TID-12345",
            line_items=[],
        )
        result = _process_with_mock(
            pipeline,
            tmp_path,
            "slip.jpg",
            extraction,
            vision_type="receipt",
        )

        assert result.success
        assert result.record_type == RecordType.PAYMENT

    def test_auth_code_triggers_reclassification(
        self, pipeline: ProcessingPipeline, tmp_path: Path
    ):
        """Authorization code without card details triggers reclassification."""
        extraction = _mock_extraction(
            authorization_code="AUTH-789",
            line_items=[],
        )
        result = _process_with_mock(
            pipeline,
            tmp_path,
            "auth.jpg",
            extraction,
            vision_type="receipt",
        )

        assert result.success
        assert result.record_type == RecordType.PAYMENT

    def test_real_receipt_not_reclassified(
        self, pipeline: ProcessingPipeline, tmp_path: Path
    ):
        """A real receipt with items should NOT be reclassified."""
        extraction = _mock_extraction(
            payment_method="card",
            card_last4="1234",
            line_items=[
                {
                    "name": "Milk",
                    "quantity": 2,
                    "unit_price": 1.50,
                    "total_price": 3.00,
                },
                {
                    "name": "Bread",
                    "quantity": 1,
                    "unit_price": 2.00,
                    "total_price": 2.00,
                },
            ],
        )
        result = _process_with_mock(
            pipeline,
            tmp_path,
            "receipt.jpg",
            extraction,
            vision_type="receipt",
        )

        assert result.success
        assert result.record_type == RecordType.PURCHASE

    def test_no_payment_details_not_reclassified(
        self, pipeline: ProcessingPipeline, tmp_path: Path
    ):
        """Empty receipt without payment details stays as receipt."""
        extraction = _mock_extraction(
            line_items=[],
        )
        result = _process_with_mock(
            pipeline,
            tmp_path,
            "empty.jpg",
            extraction,
            vision_type="receipt",
        )

        assert result.success
        assert result.record_type == RecordType.PURCHASE


class TestLooksLikePaymentConfirmation:
    """Unit tests for the _looks_like_payment_confirmation static method."""

    def test_empty_data(self):
        assert not ProcessingPipeline._looks_like_payment_confirmation({})

    def test_none_like(self):
        assert not ProcessingPipeline._looks_like_payment_confirmation(
            {"line_items": None}
        )

    def test_no_items_with_card(self):
        assert ProcessingPipeline._looks_like_payment_confirmation(
            {"line_items": [], "card_last4": "1234"}
        )

    def test_phantom_items_with_card(self):
        assert ProcessingPipeline._looks_like_payment_confirmation(
            {
                "line_items": [
                    {"name": "PURCHASE"},
                    {"name": "TOTAL"},
                ],
                "card_last4": "5678",
            }
        )

    def test_real_items_with_card_false(self):
        """Real items present -> not a payment confirmation."""
        assert not ProcessingPipeline._looks_like_payment_confirmation(
            {
                "line_items": [
                    {"name": "Milk"},
                    {"name": "PURCHASE"},
                ],
                "card_last4": "1234",
            }
        )

    def test_no_items_no_payment_details_false(self):
        """No items but no payment details -> not a payment confirmation."""
        assert not ProcessingPipeline._looks_like_payment_confirmation(
            {"line_items": []}
        )

    def test_terminal_id_only(self):
        assert ProcessingPipeline._looks_like_payment_confirmation(
            {"line_items": [], "terminal_id": "TID-001"}
        )

    def test_authorization_code_only(self):
        assert ProcessingPipeline._looks_like_payment_confirmation(
            {"line_items": [], "authorization_code": "AUTH-001"}
        )

    def test_payment_and_sale_phantom(self):
        """PAYMENT and SALE are phantom names."""
        assert ProcessingPipeline._looks_like_payment_confirmation(
            {
                "line_items": [
                    {"name": "Payment"},
                    {"name": "sale"},
                ],
                "card_last4": "0000",
            }
        )


class TestVisionTypeMap:
    """Test the _VISION_TYPE_MAP class attribute."""

    def test_all_expected_types_mapped(self):
        assert ProcessingPipeline._VISION_TYPE_MAP["receipt"] == DocumentType.RECEIPT
        assert ProcessingPipeline._VISION_TYPE_MAP["invoice"] == DocumentType.INVOICE
        assert (
            ProcessingPipeline._VISION_TYPE_MAP["payment_confirmation"]
            == DocumentType.PAYMENT_CONFIRMATION
        )
        assert (
            ProcessingPipeline._VISION_TYPE_MAP["statement"] == DocumentType.STATEMENT
        )
        assert ProcessingPipeline._VISION_TYPE_MAP["warranty"] == DocumentType.WARRANTY

    def test_other_not_in_map(self):
        assert "other" not in ProcessingPipeline._VISION_TYPE_MAP
