"""Tests for Telegram bot handlers: upload, correction, annotation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from alibi.db.models import DocumentType
from alibi.services.ingestion import (
    _DOCTYPE_TO_FOLDER,
    _FALLBACK_DIR,
    _resolve_upload_dir,
    persist_upload,
)
from alibi.telegram.handlers.upload import (
    _COMMAND_TYPE_MAP,
    _PENDING_TTL,
    _expire_pending,
    _format_result,
    _parse_type_and_hint,
    _pending_uploads,
    _pop_pending,
)
from alibi.telegram.handlers.correction import (
    _extract_fact_id_from_reply,
    _looks_like_id,
)
from alibi.telegram.handlers.annotation import _parse_tag_args


@dataclass
class MockResult:
    success: bool = True
    file_path: Path = field(default_factory=lambda: Path("/tmp/test.jpg"))
    document_id: str | None = "fact-abc-123"
    is_duplicate: bool = False
    duplicate_of: str | None = None
    error: str | None = None
    extracted_data: dict[str, Any] | None = None
    refined_data: dict[str, Any] | None = None
    line_items: list[dict[str, Any]] = field(default_factory=list)
    record_type: Any = None


def _make_reply_message(text: str | None = None) -> MagicMock:
    """Create a mock Message with reply_to_message set."""
    reply = MagicMock()
    reply.text = text
    message = MagicMock()
    message.reply_to_message = reply
    return message


# ---------------------------------------------------------------------------
# Upload parsing tests
# ---------------------------------------------------------------------------


class TestUploadParsing:
    """Tests for _parse_type_and_hint."""

    def test_type_command_with_vendor_hint(self) -> None:
        """Short type command with vendor hint returns type and hint."""
        doc_type, hint = _parse_type_and_hint("receipt", "fresko")
        assert doc_type == DocumentType.RECEIPT
        assert hint == "fresko"

    def test_type_command_empty_args(self) -> None:
        """Short type command with empty args returns type and no hint."""
        doc_type, hint = _parse_type_and_hint("receipt", "")
        assert doc_type == DocumentType.RECEIPT
        assert hint is None

    def test_upload_command_with_type_and_hint(self) -> None:
        """/upload with valid type token and vendor hint parses both."""
        doc_type, hint = _parse_type_and_hint("upload", "receipt fresko")
        assert doc_type == DocumentType.RECEIPT
        assert hint == "fresko"

    def test_upload_command_empty_args(self) -> None:
        """/upload with no args returns (None, None)."""
        doc_type, hint = _parse_type_and_hint("upload", "")
        assert doc_type is None
        assert hint is None

    def test_upload_command_unknown_type_token(self) -> None:
        """/upload where first token is not a known type returns no type, full args as hint."""
        doc_type, hint = _parse_type_and_hint("upload", "mystery vendor")
        assert doc_type is None
        assert hint == "mystery vendor"

    def test_all_short_type_commands_resolve(self) -> None:
        """All short type aliases map to the expected DocumentType values."""
        expected = {
            "receipt": DocumentType.RECEIPT,
            "invoice": DocumentType.INVOICE,
            "payment": DocumentType.PAYMENT_CONFIRMATION,
            "statement": DocumentType.STATEMENT,
            "warranty": DocumentType.WARRANTY,
            "contract": DocumentType.CONTRACT,
        }
        for cmd, expected_type in expected.items():
            doc_type, _ = _parse_type_and_hint(cmd, "")
            assert (
                doc_type == expected_type
            ), f"Command {cmd!r} did not resolve correctly"

    def test_command_type_map_keys_match_short_commands(self) -> None:
        """_COMMAND_TYPE_MAP contains exactly the expected short commands."""
        expected_keys = {
            "receipt",
            "invoice",
            "payment",
            "statement",
            "warranty",
            "contract",
        }
        assert set(_COMMAND_TYPE_MAP.keys()) == expected_keys

    def test_upload_type_only_no_hint(self) -> None:
        """/upload receipt with no trailing hint returns type and no hint."""
        doc_type, hint = _parse_type_and_hint("upload", "receipt")
        assert doc_type == DocumentType.RECEIPT
        assert hint is None

    def test_upload_type_with_whitespace_hint(self) -> None:
        """/upload invoice with whitespace-only hint returns no hint."""
        doc_type, hint = _parse_type_and_hint("upload", "invoice   ")
        assert doc_type == DocumentType.INVOICE
        assert hint is None

    def test_type_command_whitespace_only_args(self) -> None:
        """Short type command with whitespace-only args returns no hint."""
        doc_type, hint = _parse_type_and_hint("payment", "   ")
        assert doc_type == DocumentType.PAYMENT_CONFIRMATION
        assert hint is None


# ---------------------------------------------------------------------------
# Upload pending-state machine tests
# ---------------------------------------------------------------------------


class TestUploadState:
    """Tests for the _pending_uploads state machine."""

    @pytest.fixture(autouse=True)
    def clear_pending(self) -> None:
        """Clear module-level pending state before every test."""
        _pending_uploads.clear()

    def test_set_and_pop_pending(self) -> None:
        """Setting pending state and popping it returns the stored values."""
        chat_id = 1001
        _pending_uploads[chat_id] = (DocumentType.RECEIPT, "fresko", time.time())
        result = _pop_pending(chat_id)
        assert result is not None
        doc_type, hint = result
        assert doc_type == DocumentType.RECEIPT
        assert hint == "fresko"

    def test_pop_removes_entry(self) -> None:
        """After pop the entry is gone."""
        chat_id = 1002
        _pending_uploads[chat_id] = (DocumentType.INVOICE, None, time.time())
        _pop_pending(chat_id)
        assert chat_id not in _pending_uploads

    def test_pop_missing_returns_none(self) -> None:
        """Popping a chat_id with no pending state returns None."""
        result = _pop_pending(9999)
        assert result is None

    def test_expired_entry_is_discarded(self) -> None:
        """An entry older than _PENDING_TTL is discarded and pop returns None."""
        chat_id = 1003
        expired_ts = time.time() - _PENDING_TTL - 1.0
        _pending_uploads[chat_id] = (DocumentType.RECEIPT, "old", expired_ts)
        result = _pop_pending(chat_id)
        assert result is None
        assert chat_id not in _pending_uploads

    def test_expire_pending_removes_stale_entry(self) -> None:
        """_expire_pending removes an entry that has exceeded TTL."""
        chat_id = 1004
        expired_ts = time.time() - _PENDING_TTL - 5.0
        _pending_uploads[chat_id] = (DocumentType.STATEMENT, None, expired_ts)
        _expire_pending(chat_id)
        assert chat_id not in _pending_uploads

    def test_expire_pending_keeps_fresh_entry(self) -> None:
        """_expire_pending does not remove an entry within TTL."""
        chat_id = 1005
        _pending_uploads[chat_id] = (DocumentType.WARRANTY, "acme", time.time())
        _expire_pending(chat_id)
        assert chat_id in _pending_uploads

    def test_expire_pending_noop_when_absent(self) -> None:
        """_expire_pending on a missing chat_id does nothing."""
        _expire_pending(88888)
        assert 88888 not in _pending_uploads

    def test_pending_with_none_type(self) -> None:
        """Pending state can store None as the document type."""
        chat_id = 1006
        _pending_uploads[chat_id] = (None, "hint only", time.time())
        result = _pop_pending(chat_id)
        assert result is not None
        doc_type, hint = result
        assert doc_type is None
        assert hint == "hint only"

    def test_multiple_chats_are_independent(self) -> None:
        """Pending state for different chat IDs is stored independently."""
        _pending_uploads[2001] = (DocumentType.RECEIPT, "alpha", time.time())
        _pending_uploads[2002] = (DocumentType.INVOICE, "beta", time.time())
        r1 = _pop_pending(2001)
        r2 = _pop_pending(2002)
        assert r1 is not None and r1[1] == "alpha"
        assert r2 is not None and r2[1] == "beta"


# ---------------------------------------------------------------------------
# Upload result formatting tests
# ---------------------------------------------------------------------------


class TestUploadFormat:
    """Tests for _format_result."""

    def test_success_result_contains_processed_successfully(self) -> None:
        """Successful result message contains the success phrase."""
        result = MockResult(
            success=True,
            extracted_data={
                "vendor": "Fresko",
                "total": 12.50,
                "currency": "EUR",
                "date": "2026-02-01",
            },
            document_id="fact-abc-123",
        )
        text = _format_result(result)
        assert "Document processed successfully" in text

    def test_success_result_contains_vendor(self) -> None:
        """Successful result message includes the vendor name."""
        result = MockResult(
            success=True,
            extracted_data={"vendor": "Fresko", "total": 12.50, "currency": "EUR"},
            document_id="fact-abc-123",
        )
        text = _format_result(result)
        assert "Fresko" in text

    def test_success_result_contains_fact_id(self) -> None:
        """Successful result message includes the fact ID."""
        result = MockResult(
            success=True,
            extracted_data={},
            document_id="fact-unique-id-999",
        )
        text = _format_result(result)
        assert "fact-unique-id-999" in text

    def test_failure_result_contains_processing_failed(self) -> None:
        """Failed result message contains the failure phrase."""
        result = MockResult(
            success=False,
            error="OCR timeout",
        )
        text = _format_result(result)
        assert "Processing failed" in text

    def test_failure_result_sanitized_no_raw_error(self) -> None:
        """Failed result does not expose raw error details to user."""
        result = MockResult(
            success=False,
            error="OCR timeout",
        )
        text = _format_result(result)
        assert "OCR timeout" not in text
        assert "Try a clearer photo" in text

    def test_failure_result_no_error_field(self) -> None:
        """Failed result with no error field still shows user-friendly message."""
        result = MockResult(success=False, error=None)
        text = _format_result(result)
        assert "Processing failed" in text

    def test_duplicate_result_contains_duplicate(self) -> None:
        """Duplicate result message contains the word 'duplicate'."""
        result = MockResult(
            success=True,
            is_duplicate=True,
            duplicate_of="fact-original-001",
        )
        text = _format_result(result)
        assert "duplicate" in text.lower()

    def test_duplicate_result_contains_original_id(self) -> None:
        """Duplicate result message includes the original document ID."""
        result = MockResult(
            success=True,
            is_duplicate=True,
            duplicate_of="fact-original-001",
        )
        text = _format_result(result)
        assert "fact-original-001" in text

    def test_duplicate_result_without_duplicate_of(self) -> None:
        """Duplicate result with no duplicate_of still contains duplicate phrase."""
        result = MockResult(
            success=True,
            is_duplicate=True,
            duplicate_of=None,
        )
        text = _format_result(result)
        assert "duplicate" in text.lower()

    def test_success_result_line_item_count(self) -> None:
        """Successful result message includes item count."""
        items = [{"name": "Milk"}, {"name": "Bread"}, {"name": "Eggs"}]
        result = MockResult(
            success=True,
            extracted_data={},
            line_items=items,
        )
        text = _format_result(result)
        assert "3" in text

    def test_success_result_na_for_missing_fields(self) -> None:
        """When extracted_data is empty, fallback values appear."""
        result = MockResult(
            success=True,
            extracted_data={},
            document_id=None,
        )
        text = _format_result(result)
        assert "N/A" in text

    def test_success_result_record_type_fallback(self) -> None:
        """When document_type absent from data, record_type.value is used."""
        mock_type = MagicMock()
        mock_type.value = "receipt"
        result = MockResult(
            success=True,
            extracted_data={},
            record_type=mock_type,
        )
        text = _format_result(result)
        assert "receipt" in text


# ---------------------------------------------------------------------------
# Correction handler parsing tests
# ---------------------------------------------------------------------------


class TestCorrectionParsing:
    """Tests for _looks_like_id and _extract_fact_id_from_reply."""

    def test_looks_like_id_with_hyphen(self) -> None:
        """A token containing a hyphen is treated as an ID."""
        assert _looks_like_id("abc-def-123") is True

    def test_looks_like_id_plain_word(self) -> None:
        """A plain word without hyphen and under min length is not an ID."""
        assert _looks_like_id("vendor") is False

    def test_looks_like_id_uuid_length(self) -> None:
        """A string >= 36 chars is treated as an ID even without hyphen."""
        long_token = "a" * 36
        assert _looks_like_id(long_token) is True

    def test_looks_like_id_short_no_hyphen(self) -> None:
        """Short tokens without hyphens are not treated as IDs."""
        assert _looks_like_id("short") is False
        assert _looks_like_id("fix") is False
        assert _looks_like_id("") is False

    def test_looks_like_id_uuid_format(self) -> None:
        """A full UUID string is recognized as an ID."""
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        assert _looks_like_id(uuid) is True

    def test_extract_fact_id_from_reply_found(self) -> None:
        """Extracts fact ID from a message containing the standard pattern."""
        message = _make_reply_message(
            "Document processed successfully!\nFact ID: `fact-abc-123`\nUse /fix to correct."
        )
        result = _extract_fact_id_from_reply(message)
        assert result == "fact-abc-123"

    def test_extract_fact_id_from_reply_not_found(self) -> None:
        """Returns None when reply text has no Fact ID pattern."""
        message = _make_reply_message("Some unrelated message text.")
        result = _extract_fact_id_from_reply(message)
        assert result is None

    def test_extract_fact_id_no_reply(self) -> None:
        """Returns None when message has no reply_to_message."""
        message = MagicMock()
        message.reply_to_message = None
        result = _extract_fact_id_from_reply(message)
        assert result is None

    def test_extract_fact_id_reply_has_no_text(self) -> None:
        """Returns None when the replied-to message has no text."""
        message = _make_reply_message(None)
        result = _extract_fact_id_from_reply(message)
        assert result is None

    def test_extract_fact_id_extracts_uuid(self) -> None:
        """Extracts a full UUID-format fact ID correctly."""
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        message = _make_reply_message(
            f"Vendor updated.\nFact ID: `{uuid}`\nNew vendor: Fresko"
        )
        result = _extract_fact_id_from_reply(message)
        assert result == uuid


# ---------------------------------------------------------------------------
# Annotation handler parsing tests
# ---------------------------------------------------------------------------


class TestAnnotationParsing:
    """Tests for _parse_tag_args."""

    def test_fact_id_key_quoted_value(self) -> None:
        """fact_id + key + quoted value parses all three fields."""
        fact_id, key, value = _parse_tag_args('abc-123 project "kitchen renovation"')
        assert fact_id == "abc-123"
        assert key == "project"
        assert value == "kitchen renovation"

    def test_key_quoted_value_no_fact_id(self) -> None:
        """key + quoted value without fact_id returns None for fact_id."""
        fact_id, key, value = _parse_tag_args('project "kitchen renovation"')
        assert fact_id is None
        assert key == "project"
        assert value == "kitchen renovation"

    def test_fact_id_key_plain_value(self) -> None:
        """fact_id + key + unquoted value parses all three fields."""
        fact_id, key, value = _parse_tag_args("abc-123 person Maria")
        assert fact_id == "abc-123"
        assert key == "person"
        assert value == "Maria"

    def test_empty_string_returns_all_none(self) -> None:
        """Empty input returns (None, None, None)."""
        fact_id, key, value = _parse_tag_args("")
        assert fact_id is None
        assert key is None
        assert value is None

    def test_single_token_returns_all_none(self) -> None:
        """Single token without key or value returns (None, None, None)."""
        fact_id, key, value = _parse_tag_args("single")
        assert fact_id is None
        assert key is None
        assert value is None

    def test_key_plain_value_no_fact_id(self) -> None:
        """key + plain value without fact_id returns None for fact_id."""
        fact_id, key, value = _parse_tag_args("person Maria")
        assert fact_id is None
        assert key == "person"
        assert value == "Maria"

    def test_fact_id_key_multi_word_plain_value(self) -> None:
        """fact_id + key + multi-word unquoted value joins words."""
        fact_id, key, value = _parse_tag_args("abc-123 note this is a long note")
        assert fact_id == "abc-123"
        assert key == "note"
        assert value == "this is a long note"

    def test_uuid_fact_id_detected(self) -> None:
        """Full UUID is detected as fact_id."""
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        fact_id, key, value = _parse_tag_args(f"{uuid} category food")
        assert fact_id == uuid
        assert key == "category"
        assert value == "food"

    def test_two_tokens_no_id_returns_key_value(self) -> None:
        """Two plain tokens without hyphen: first is key, second is value, no fact_id."""
        fact_id, key, value = _parse_tag_args("category food")
        assert fact_id is None
        assert key == "category"
        assert value == "food"

    def test_whitespace_only_returns_all_none(self) -> None:
        """Whitespace-only input returns (None, None, None)."""
        fact_id, key, value = _parse_tag_args("   ")
        assert fact_id is None
        assert key is None
        assert value is None


# ---------------------------------------------------------------------------
# Upload file persistence tests
# ---------------------------------------------------------------------------


class TestDocTypeFolderMapping:
    """Tests for _DOCTYPE_TO_FOLDER mapping."""

    def test_receipt_maps_to_receipts(self) -> None:
        assert _DOCTYPE_TO_FOLDER[DocumentType.RECEIPT] == "receipts"

    def test_invoice_maps_to_invoices(self) -> None:
        assert _DOCTYPE_TO_FOLDER[DocumentType.INVOICE] == "invoices"

    def test_payment_maps_to_payments(self) -> None:
        assert _DOCTYPE_TO_FOLDER[DocumentType.PAYMENT_CONFIRMATION] == "payments"

    def test_all_command_types_have_folders(self) -> None:
        """Every DocumentType in _COMMAND_TYPE_MAP has a folder mapping."""
        for doc_type in _COMMAND_TYPE_MAP.values():
            assert doc_type in _DOCTYPE_TO_FOLDER


class TestResolveUploadDir:
    """Tests for _resolve_upload_dir."""

    def test_receipt_with_vault(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With vault configured, receipt goes to inbox/documents/receipts."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")  # allow .env loading of vault
        result = _resolve_upload_dir(DocumentType.RECEIPT)
        assert result == tmp_path / "inbox" / "documents" / "receipts"

    def test_none_type_with_vault(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With vault configured, None type goes to inbox/documents/unsorted."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")
        result = _resolve_upload_dir(None)
        assert result == tmp_path / "inbox" / "documents" / "unsorted"

    def test_fallback_without_vault(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without vault, falls back to data/uploads/."""
        monkeypatch.delenv("ALIBI_VAULT_PATH", raising=False)
        result = _resolve_upload_dir(DocumentType.INVOICE)
        assert result == _FALLBACK_DIR / "invoices"

    def test_fallback_none_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without vault, None type uses data/uploads/unsorted."""
        monkeypatch.delenv("ALIBI_VAULT_PATH", raising=False)
        result = _resolve_upload_dir(None)
        assert result == _FALLBACK_DIR / "unsorted"


class TestPersistUpload:
    """Tests for persist_upload (service layer)."""

    def test_writes_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """File is written to disk with source prefix."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")
        from alibi.processing.folder_router import FolderContext

        ctx = FolderContext(doc_type=DocumentType.RECEIPT, source="telegram")
        content = b"fake-image-data"
        path = persist_upload(content, "photo123.jpg", ctx)
        assert path.exists()
        assert path.read_bytes() == content
        assert path.name.startswith("telegram_")
        assert "photo123" in path.name
        assert path.suffix == ".jpg"

    def test_creates_parent_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parent directories are created automatically."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")
        from alibi.processing.folder_router import FolderContext

        ctx = FolderContext(doc_type=DocumentType.INVOICE, source="api")
        path = persist_upload(b"data", "test.pdf", ctx)
        assert path.parent.exists()
        assert "invoices" in str(path.parent)

    def test_unsorted_for_none_type(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """None doc_type saves to unsorted/ subfolder."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")
        path = persist_upload(b"data", "mystery.jpg", None)
        assert "unsorted" in str(path.parent)
