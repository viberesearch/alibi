"""Tests for Gemini batch verification."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from alibi.extraction.gemini_verifier import (
    ReceiptVerification,
    VerificationBatchResponse,
    VerificationResult,
    _build_batch_prompt,
    verify_batch,
    verify_documents,
)


class TestBuildBatchPrompt:
    def test_single_document(self) -> None:
        docs = [{"vendor": "Shop A", "total": "10.00", "items": []}]
        prompt = _build_batch_prompt(docs)
        assert "Receipt 0" in prompt
        assert "Shop A" in prompt

    def test_multiple_documents(self) -> None:
        docs = [
            {"vendor": "Shop A", "total": "10.00"},
            {"vendor": "Shop B", "total": "20.00"},
        ]
        prompt = _build_batch_prompt(docs)
        assert "Receipt 0" in prompt
        assert "Receipt 1" in prompt
        assert "Shop A" in prompt
        assert "Shop B" in prompt

    def test_ocr_text_included(self) -> None:
        docs = [{"vendor": "X", "ocr_text": "OCR CONTENT HERE"}]
        prompt = _build_batch_prompt(docs)
        assert "OCR CONTENT HERE" in prompt
        assert "OCR Text:" in prompt

    def test_ocr_text_truncated(self) -> None:
        docs = [{"vendor": "X", "ocr_text": "A" * 5000}]
        prompt = _build_batch_prompt(docs)
        assert len(prompt) < 5000

    def test_excludes_internal_fields(self) -> None:
        docs = [{"vendor": "X", "doc_id": "abc", "yaml_path": "/tmp/x.yaml"}]
        prompt = _build_batch_prompt(docs)
        assert "doc_id" not in prompt
        assert "yaml_path" not in prompt

    def test_skips_none_values(self) -> None:
        docs = [{"vendor": "X", "total": None}]
        prompt = _build_batch_prompt(docs)
        assert "null" not in prompt


class TestVerifyBatch:
    def test_no_api_key_returns_empty(self) -> None:
        with patch("alibi.extraction.gemini_verifier._get_api_key", return_value=None):
            result = verify_batch([{"vendor": "X"}])
        assert result == []

    def test_structured_response(self) -> None:
        mock_response = MagicMock()
        mock_parsed = VerificationBatchResponse(
            receipts=[ReceiptVerification(doc_idx=0, vendor_ok=True, total_ok=True)]
        )
        mock_response.parsed = mock_parsed

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with (
            patch(
                "alibi.extraction.gemini_verifier._get_api_key",
                return_value="test-key",
            ),
            patch(
                "alibi.extraction.gemini_verifier._get_model",
                return_value="gemini-2.0-flash",
            ),
            patch("google.genai.Client", return_value=mock_client),
        ):
            results = verify_batch([{"vendor": "Shop A", "total": "10.00"}])

        assert len(results) == 1
        assert results[0].vendor_ok is True

    def test_json_fallback(self) -> None:
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = json.dumps(
            {"receipts": [{"doc_idx": 0, "vendor_ok": False}]}
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with (
            patch(
                "alibi.extraction.gemini_verifier._get_api_key",
                return_value="test-key",
            ),
            patch(
                "alibi.extraction.gemini_verifier._get_model",
                return_value="gemini-2.0-flash",
            ),
            patch("google.genai.Client", return_value=mock_client),
        ):
            results = verify_batch([{"vendor": "Shop A"}])

        assert len(results) == 1
        assert results[0].vendor_ok is False

    def test_api_error_returns_empty(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("boom")

        with (
            patch(
                "alibi.extraction.gemini_verifier._get_api_key",
                return_value="test-key",
            ),
            patch(
                "alibi.extraction.gemini_verifier._get_model",
                return_value="gemini-2.0-flash",
            ),
            patch("google.genai.Client", return_value=mock_client),
        ):
            results = verify_batch([{"vendor": "Shop A"}])

        assert results == []


class TestVerifyDocuments:
    def test_empty_db(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = []
        results = verify_documents(mock_db, limit=10)
        assert results == []

    def test_specific_doc_ids(self, mock_db: MagicMock, tmp_path) -> None:
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("vendor: TestShop\ntotal: '10.00'\n")

        mock_db.fetchone.return_value = ("doc1", str(yaml_file))

        with patch("alibi.extraction.gemini_verifier.verify_batch") as mock_verify:
            mock_verify.return_value = [
                ReceiptVerification(doc_idx=0, vendor_ok=True, total_ok=True)
            ]
            results = verify_documents(mock_db, doc_ids=["doc1"])

        assert len(results) == 1
        assert results[0].doc_id == "doc1"
        assert results[0].all_ok is True

    def test_issues_detected(self, mock_db: MagicMock, tmp_path) -> None:
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("vendor: WrongVendor\ntotal: '999.99'\n")

        mock_db.fetchone.return_value = ("doc1", str(yaml_file))

        with patch("alibi.extraction.gemini_verifier.verify_batch") as mock_verify:
            mock_verify.return_value = [
                ReceiptVerification(
                    doc_idx=0,
                    vendor_ok=False,
                    total_ok=False,
                    suggested_vendor="CorrectVendor",
                    suggested_total="10.00",
                    note="Vendor and total mismatch",
                )
            ]
            results = verify_documents(mock_db, doc_ids=["doc1"])

        assert len(results) == 1
        assert results[0].all_ok is False
        assert len(results[0].issues) == 2
        fields = {i["field"] for i in results[0].issues}
        assert "vendor" in fields
        assert "total" in fields

    def test_missing_yaml_skipped(self, mock_db: MagicMock) -> None:
        mock_db.fetchone.return_value = ("doc1", "/nonexistent/path.yaml")

        with patch("alibi.extraction.gemini_verifier.verify_batch") as mock_verify:
            mock_verify.return_value = []
            results = verify_documents(mock_db, doc_ids=["doc1"])

        assert results == []
