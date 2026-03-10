"""Tests for Telegram media group buffering in the upload handler.

Covers:
- _peek_pending: read pending without consuming
- _persist_group_page: page naming, extension preservation
- attachment_handler: media group buffering, timer management, single-photo path
- _process_media_group: multi-page vs single-page routing, sort order, folder creation
- _media_group_timer: fires and processes, cancellation suppresses processing
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alibi.db.models import DocumentType
from alibi.services.ingestion import persist_upload_group
from alibi.telegram.handlers.upload import (
    _MEDIA_GROUP_TIMEOUT,
    _MediaGroupBuffer,
    _media_groups,
    _peek_pending,
    _pending_uploads,
    _process_media_group,
    attachment_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(
    chat_id: int = 1001,
    message_id: int = 100,
    media_group_id: str | None = None,
    tg_user_id: int = 42,
    has_photo: bool = True,
) -> AsyncMock:
    """Build a minimal mock Telegram Message for upload tests."""
    msg = AsyncMock()
    msg.chat = MagicMock()
    msg.chat.do = AsyncMock()
    msg.chat.id = chat_id
    msg.message_id = message_id
    msg.media_group_id = media_group_id
    msg.from_user = MagicMock()
    msg.from_user.id = tg_user_id
    msg.text = None
    msg.reply_to_message = None
    msg.reply = AsyncMock()
    msg.bot = AsyncMock()

    if has_photo:
        photo_size = MagicMock()
        photo_size.file_id = f"file_{message_id}"
        msg.photo = [photo_size]
        msg.document = None
    else:
        msg.photo = []
        msg.document = None

    # Default bot stubs: get_file returns a file_path, download_file writes bytes
    mock_file = MagicMock()
    mock_file.file_path = f"photos/{message_id}.jpg"
    msg.bot.get_file = AsyncMock(return_value=mock_file)
    msg.bot.download_file = AsyncMock(
        side_effect=lambda path, buf: buf.write(b"fake image data")
    )

    return msg


# ---------------------------------------------------------------------------
# Autouse fixture — isolate module-level state between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_state():
    """Clear _pending_uploads and _media_groups before every test."""
    _pending_uploads.clear()
    _media_groups.clear()
    yield
    _pending_uploads.clear()
    _media_groups.clear()


# ---------------------------------------------------------------------------
# TestPeekPending
# ---------------------------------------------------------------------------


class TestPeekPending:
    """Tests for _peek_pending — reads pending without consuming."""

    def test_peek_returns_state_without_consuming(self):
        """_peek_pending returns the stored state and leaves it in the dict."""
        chat_id = 1001
        _pending_uploads[chat_id] = (DocumentType.RECEIPT, "fresko", time.time())

        result = _peek_pending(chat_id)

        assert result is not None
        doc_type, vendor_hint = result
        assert doc_type == DocumentType.RECEIPT
        assert vendor_hint == "fresko"
        # Entry must still be present
        assert chat_id in _pending_uploads

    def test_peek_returns_none_when_empty(self):
        """_peek_pending returns None when no entry exists for chat_id."""
        result = _peek_pending(9999)
        assert result is None

    def test_peek_expires_old_entries(self):
        """_peek_pending returns None and removes entries older than TTL."""
        from alibi.telegram.handlers.upload import _PENDING_TTL

        chat_id = 1002
        expired_ts = time.time() - _PENDING_TTL - 1.0
        _pending_uploads[chat_id] = (DocumentType.INVOICE, None, expired_ts)

        result = _peek_pending(chat_id)

        assert result is None
        assert chat_id not in _pending_uploads


# ---------------------------------------------------------------------------
# TestPersistGroupPage
# ---------------------------------------------------------------------------


class TestPersistUploadGroup:
    """Tests for persist_upload_group (service layer) — writes pages to a subfolder."""

    def test_writes_pages_with_correct_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Pages are named page_000.jpg, page_001.jpg, etc."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")
        from alibi.processing.folder_router import FolderContext

        ctx = FolderContext(doc_type=DocumentType.RECEIPT, source="telegram")
        pages = [(b"page 0", "photo_a.jpg"), (b"page 1", "photo_b.jpg")]
        paths = persist_upload_group(pages, ctx)

        assert len(paths) == 2
        assert paths[0].name == "page_000.jpg"
        assert paths[1].name == "page_001.jpg"
        assert paths[0].read_bytes() == b"page 0"
        assert paths[1].read_bytes() == b"page 1"

    def test_preserves_suffix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Suffix from filename is preserved in the page file name."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")
        pages = [(b"pdf", "scan.pdf"), (b"png", "img.png")]
        paths = persist_upload_group(pages)

        assert paths[0].suffix == ".pdf"
        assert paths[1].suffix == ".png"

    def test_creates_subfolder_with_source_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Pages are placed in a source-prefixed timestamped subfolder."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")
        from alibi.processing.folder_router import FolderContext

        ctx = FolderContext(source="telegram")
        pages = [(b"x", "a.jpg")]
        paths = persist_upload_group(pages, ctx)

        parent = paths[0].parent
        assert parent.name.startswith("telegram_")

    def test_falls_back_to_jpg_when_no_suffix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Filenames without an extension fall back to .jpg."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")
        pages = [(b"x", "no_suffix")]
        paths = persist_upload_group(pages)
        assert paths[0].suffix == ".jpg"


# ---------------------------------------------------------------------------
# TestAttachmentHandlerMediaGroup
# ---------------------------------------------------------------------------


class TestAttachmentHandlerMediaGroup:
    """Tests for attachment_handler when messages carry a media_group_id."""

    @pytest.mark.asyncio
    async def test_first_message_creates_buffer(self):
        """First media group message creates a new _MediaGroupBuffer."""
        msg = _make_message(chat_id=1001, message_id=100, media_group_id="mg123")

        await attachment_handler(msg)

        assert "mg123" in _media_groups
        buf = _media_groups["mg123"]
        assert buf.chat_id == 1001
        assert buf.messages == [msg]

    @pytest.mark.asyncio
    async def test_second_message_appends_to_buffer(self):
        """Second message with same media_group_id appends to existing buffer."""
        msg1 = _make_message(chat_id=1001, message_id=100, media_group_id="mg123")
        msg2 = _make_message(chat_id=1001, message_id=101, media_group_id="mg123")

        await attachment_handler(msg1)
        await attachment_handler(msg2)

        buf = _media_groups["mg123"]
        assert len(buf.messages) == 2
        assert msg2 in buf.messages

    @pytest.mark.asyncio
    async def test_pending_type_applied_to_first_message(self):
        """Pending type command is consumed and set on the buffer for the first message."""
        chat_id = 1001
        _pending_uploads[chat_id] = (DocumentType.INVOICE, "acme", time.time())

        msg = _make_message(chat_id=chat_id, message_id=100, media_group_id="mg456")
        await attachment_handler(msg)

        buf = _media_groups["mg456"]
        assert buf.doc_type == DocumentType.INVOICE
        assert buf.vendor_hint == "acme"
        # Pending state consumed
        assert chat_id not in _pending_uploads

    @pytest.mark.asyncio
    async def test_pending_consumed_only_on_first_message(self):
        """Pending state is only consumed by the first message; second leaves it untouched."""
        chat_id = 1001
        _pending_uploads[chat_id] = (DocumentType.RECEIPT, "fresko", time.time())

        msg1 = _make_message(chat_id=chat_id, message_id=100, media_group_id="mg789")
        msg2 = _make_message(chat_id=chat_id, message_id=101, media_group_id="mg789")

        # First message consumes pending
        await attachment_handler(msg1)
        assert chat_id not in _pending_uploads

        # Second message does not re-read any stale pending
        await attachment_handler(msg2)
        buf = _media_groups["mg789"]
        # doc_type was set by first message
        assert buf.doc_type == DocumentType.RECEIPT
        assert len(buf.messages) == 2

    @pytest.mark.asyncio
    async def test_no_media_group_processes_single(self):
        """A message without media_group_id is processed as a single attachment."""
        msg = _make_message(chat_id=1001, message_id=200, media_group_id=None)

        mock_db = MagicMock()
        mock_db.is_initialized.return_value = True

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.is_duplicate = False
        mock_result.extracted_data = {"vendor": "Fresko", "total": 5.0}
        mock_result.document_id = "fact-001"
        mock_result.line_items = []
        mock_result.record_type = None

        with (
            patch("alibi.telegram.handlers.get_db", return_value=mock_db),
            patch(
                "alibi.telegram.handlers.upload.process_file",
                return_value=mock_result,
            ),
            patch(
                "alibi.telegram.handlers.upload._resolve_telegram_user",
                return_value="system",
            ),
        ):
            await attachment_handler(msg)

        # Must not have created any media group buffer
        assert len(_media_groups) == 0
        # reply was called (processing feedback)
        assert msg.reply.called

    @pytest.mark.asyncio
    async def test_timer_created_and_cancelled_on_new_message(self):
        """Each new message cancels the previous timer and creates a fresh one."""
        msg1 = _make_message(chat_id=1001, message_id=100, media_group_id="mg_t")
        msg2 = _make_message(chat_id=1001, message_id=101, media_group_id="mg_t")

        await attachment_handler(msg1)
        first_task = _media_groups["mg_t"].timer_task
        assert first_task is not None

        await attachment_handler(msg2)
        second_task = _media_groups["mg_t"].timer_task

        # A new task was created
        assert second_task is not first_task

        # cancel() schedules cancellation but the coroutine only transitions to
        # cancelled state after it runs and handles the CancelledError.  Check
        # that cancellation has been requested (cancelling() > 0) or wait for
        # the event loop to process it and then confirm task.cancelled().
        assert first_task.cancelling() > 0 or first_task.cancelled()

        # Clean up: cancel the second task to avoid warnings
        second_task.cancel()
        try:
            await second_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# TestProcessMediaGroup
# ---------------------------------------------------------------------------


class TestProcessMediaGroup:
    """Tests for _process_media_group — the core download/persist/process logic."""

    def _make_buf(
        self,
        messages: list,
        doc_type: DocumentType | None = None,
        vendor_hint: str | None = None,
    ) -> _MediaGroupBuffer:
        first = messages[0] if messages else _make_message()
        return _MediaGroupBuffer(
            chat_id=1001,
            first_message=first,
            messages=list(messages),
            doc_type=doc_type,
            vendor_hint=vendor_hint,
        )

    @pytest.mark.asyncio
    async def test_processes_multi_page_as_group(self, tmp_path: Path, monkeypatch):
        """Multi-page group calls process_document_group with all page paths."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")

        msg1 = _make_message(message_id=10, media_group_id="mg")
        msg2 = _make_message(message_id=11, media_group_id="mg")
        buf = self._make_buf([msg1, msg2])

        mock_db = MagicMock()
        mock_db.is_initialized.return_value = True

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.is_duplicate = False
        mock_result.extracted_data = {}
        mock_result.document_id = "fact-grp"
        mock_result.line_items = []
        mock_result.record_type = None

        with (
            patch("alibi.telegram.handlers.get_db", return_value=mock_db),
            patch(
                "alibi.telegram.handlers.upload.process_document_group",
                return_value=mock_result,
            ) as mock_pdg,
            patch(
                "alibi.telegram.handlers.upload._resolve_telegram_user",
                return_value="system",
            ),
        ):
            await _process_media_group(buf)

        mock_pdg.assert_called_once()
        call_args = mock_pdg.call_args
        paths = call_args[0][1]  # second positional arg is the list of paths
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    @pytest.mark.asyncio
    async def test_single_page_processes_as_file(self, tmp_path: Path, monkeypatch):
        """Single-page group calls process_file instead of process_document_group."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")

        msg = _make_message(message_id=20, media_group_id="mg_single")
        buf = self._make_buf([msg])

        mock_db = MagicMock()
        mock_db.is_initialized.return_value = True

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.is_duplicate = False
        mock_result.extracted_data = {}
        mock_result.document_id = "fact-single"
        mock_result.line_items = []
        mock_result.record_type = None

        with (
            patch("alibi.telegram.handlers.get_db", return_value=mock_db),
            patch(
                "alibi.telegram.handlers.upload.process_file",
                return_value=mock_result,
            ) as mock_pf,
            patch("alibi.telegram.handlers.upload.process_document_group") as mock_pdg,
            patch(
                "alibi.telegram.handlers.upload._resolve_telegram_user",
                return_value="system",
            ),
        ):
            await _process_media_group(buf)

        mock_pf.assert_called_once()
        mock_pdg.assert_not_called()

    @pytest.mark.asyncio
    async def test_pages_sorted_by_message_id(self, tmp_path: Path, monkeypatch):
        """Pages are written in message_id order regardless of arrival order."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")

        # Arrive out of order: message_id 30 first, 29 second
        msg_hi = _make_message(message_id=30, media_group_id="mg_order")
        msg_lo = _make_message(message_id=29, media_group_id="mg_order")

        # Give each a distinct file_id so we can tell them apart
        photo_lo = MagicMock()
        photo_lo.file_id = "file_lo"
        msg_lo.photo = [photo_lo]

        file_lo = MagicMock()
        file_lo.file_path = "lo.jpg"
        msg_lo.bot.get_file = AsyncMock(return_value=file_lo)
        msg_lo.bot.download_file = AsyncMock(
            side_effect=lambda path, buf: buf.write(b"page lo")
        )

        photo_hi = MagicMock()
        photo_hi.file_id = "file_hi"
        msg_hi.photo = [photo_hi]

        file_hi = MagicMock()
        file_hi.file_path = "hi.jpg"
        msg_hi.bot.get_file = AsyncMock(return_value=file_hi)
        msg_hi.bot.download_file = AsyncMock(
            side_effect=lambda path, buf: buf.write(b"page hi")
        )

        buf = self._make_buf([msg_hi, msg_lo])  # arrival order: hi then lo

        mock_db = MagicMock()
        mock_db.is_initialized.return_value = True
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.is_duplicate = False
        mock_result.extracted_data = {}
        mock_result.document_id = "f"
        mock_result.line_items = []
        mock_result.record_type = None

        captured_paths: list[Path] = []

        def capture_paths(db, paths, folder_context=None):
            captured_paths.extend(paths)
            return mock_result

        with (
            patch("alibi.telegram.handlers.get_db", return_value=mock_db),
            patch(
                "alibi.telegram.handlers.upload.process_document_group",
                side_effect=capture_paths,
            ),
            patch(
                "alibi.telegram.handlers.upload._resolve_telegram_user",
                return_value="system",
            ),
        ):
            await _process_media_group(buf)

        assert len(captured_paths) == 2
        # page_000 should contain "page lo" (message_id=29), page_001 "page hi"
        assert captured_paths[0].read_bytes() == b"page lo"
        assert captured_paths[1].read_bytes() == b"page hi"

    @pytest.mark.asyncio
    async def test_pages_saved_to_subfolder(self, tmp_path: Path, monkeypatch):
        """Pages are written into a telegram_<timestamp> subfolder."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")

        msg1 = _make_message(message_id=40, media_group_id="mg_dir")
        msg2 = _make_message(message_id=41, media_group_id="mg_dir")
        buf = self._make_buf([msg1, msg2])

        mock_db = MagicMock()
        mock_db.is_initialized.return_value = True
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.is_duplicate = False
        mock_result.extracted_data = {}
        mock_result.document_id = "f"
        mock_result.line_items = []
        mock_result.record_type = None

        captured_paths: list[Path] = []

        def capture_paths(db, paths, folder_context=None):
            captured_paths.extend(paths)
            return mock_result

        with (
            patch("alibi.telegram.handlers.get_db", return_value=mock_db),
            patch(
                "alibi.telegram.handlers.upload.process_document_group",
                side_effect=capture_paths,
            ),
            patch(
                "alibi.telegram.handlers.upload._resolve_telegram_user",
                return_value="system",
            ),
        ):
            await _process_media_group(buf)

        # All paths share the same parent directory (subfolder)
        assert len(captured_paths) == 2
        parent = captured_paths[0].parent
        assert parent == captured_paths[1].parent
        assert parent.exists()

    @pytest.mark.asyncio
    async def test_folder_context_set_correctly(self, tmp_path: Path, monkeypatch):
        """FolderContext passed to process_document_group carries source=telegram and doc_type."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")

        msg1 = _make_message(message_id=50, media_group_id="mg_ctx")
        msg2 = _make_message(message_id=51, media_group_id="mg_ctx")
        buf = self._make_buf(
            [msg1, msg2], doc_type=DocumentType.INVOICE, vendor_hint="acme"
        )

        mock_db = MagicMock()
        mock_db.is_initialized.return_value = True
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.is_duplicate = False
        mock_result.extracted_data = {}
        mock_result.document_id = "f"
        mock_result.line_items = []
        mock_result.record_type = None

        captured_ctx = []

        def capture_ctx(db, paths, folder_context=None):
            captured_ctx.append(folder_context)
            return mock_result

        with (
            patch("alibi.telegram.handlers.get_db", return_value=mock_db),
            patch(
                "alibi.telegram.handlers.upload.process_document_group",
                side_effect=capture_ctx,
            ),
            patch(
                "alibi.telegram.handlers.upload._resolve_telegram_user",
                return_value="system",
            ),
        ):
            await _process_media_group(buf)

        assert len(captured_ctx) == 1
        ctx = captured_ctx[0]
        assert ctx.source == "telegram"
        assert ctx.doc_type == DocumentType.INVOICE
        assert ctx.vendor_hint == "acme"

    @pytest.mark.asyncio
    async def test_db_not_initialized_replies_error(self):
        """When db is not initialized, first_message.reply is called with an error."""
        msg = _make_message(message_id=60, media_group_id="mg_nodb")
        buf = self._make_buf([msg])

        mock_db = MagicMock()
        mock_db.is_initialized.return_value = False

        with patch("alibi.telegram.handlers.get_db", return_value=mock_db):
            await _process_media_group(buf)

        msg.answer.assert_called_once()
        reply_text = msg.answer.call_args[0][0]
        assert (
            "not initialized" in reply_text.lower() or "database" in reply_text.lower()
        )

    @pytest.mark.asyncio
    async def test_download_failure_handled(self, tmp_path: Path, monkeypatch):
        """When photo download raises, first_message.reply is called with error text."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")

        msg = _make_message(message_id=70, media_group_id="mg_err")
        msg.bot.get_file = AsyncMock(side_effect=RuntimeError("network error"))
        buf = self._make_buf([msg])

        mock_db = MagicMock()
        mock_db.is_initialized.return_value = True

        with (
            patch("alibi.telegram.handlers.get_db", return_value=mock_db),
            patch(
                "alibi.telegram.handlers.upload._resolve_telegram_user",
                return_value="system",
            ),
        ):
            await _process_media_group(buf)

        msg.reply.assert_called()
        all_calls = " ".join(str(c) for c in msg.reply.call_args_list)
        assert "error" in all_calls.lower() or "download" in all_calls.lower()


# ---------------------------------------------------------------------------
# TestMediaGroupTimer
# ---------------------------------------------------------------------------


class TestMediaGroupTimer:
    """Tests for _media_group_timer — sleep, then process."""

    @pytest.mark.asyncio
    async def test_timer_fires_and_processes(self, tmp_path: Path, monkeypatch):
        """After the sleep completes, _process_media_group is called for the buffer."""
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("ALIBI_TESTING", "")

        from alibi.telegram.handlers.upload import _media_group_timer

        msg = _make_message(message_id=80, media_group_id="mg_timer")
        buf = _MediaGroupBuffer(
            chat_id=1001,
            first_message=msg,
            messages=[msg],
        )
        _media_groups["mg_timer"] = buf

        processed: list[_MediaGroupBuffer] = []

        async def fake_process(b):
            processed.append(b)

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch(
                "alibi.telegram.handlers.upload._process_media_group",
                side_effect=fake_process,
            ),
        ):
            await _media_group_timer("mg_timer")

        assert len(processed) == 1
        assert processed[0] is buf
        # Buffer should have been removed from _media_groups
        assert "mg_timer" not in _media_groups

    @pytest.mark.asyncio
    async def test_timer_cancellation_does_not_process(self):
        """When the timer task is cancelled, _process_media_group is not called."""
        from alibi.telegram.handlers.upload import _media_group_timer

        msg = _make_message(message_id=90, media_group_id="mg_cancel")
        buf = _MediaGroupBuffer(
            chat_id=1001,
            first_message=msg,
            messages=[msg],
        )
        _media_groups["mg_cancel"] = buf

        processed: list = []

        async def fake_process(b):
            processed.append(b)

        async def cancellable_sleep(duration):
            raise asyncio.CancelledError()

        with (
            patch("asyncio.sleep", side_effect=cancellable_sleep),
            patch(
                "alibi.telegram.handlers.upload._process_media_group",
                side_effect=fake_process,
            ),
        ):
            await _media_group_timer("mg_cancel")

        assert len(processed) == 0
