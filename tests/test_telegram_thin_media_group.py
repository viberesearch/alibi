"""Tests for the thin (API-client) Telegram media-group path.

When several photos arrive as a Telegram media group (album), they are
buffered and forwarded to the host API as one multi-page document. This file
covers :func:`alibi.telegram.handlers.upload._process_media_group`: it
downloads each page via ``_get_attachment``, calls
``_client.process_document_group`` for >1 page, replies with the formatted
result, and primes the pending-location flow (see
``docs/TELEGRAM_THIN_BOT_PLAN.md``).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from alibi.telegram.api_client import ProcessResult
from alibi.telegram.handlers import upload


def _ok_result(**kw) -> ProcessResult:
    base = dict(
        success=True,
        document_id="doc1",
        fact_id="fact1",
        vendor="Lidl",
        amount="12.50",
        currency="EUR",
        date="2026-06-10",
        document_type="receipt",
        items_count=3,
    )
    base.update(kw)
    return ProcessResult(**base)


def _make_page_message(message_id: int) -> MagicMock:
    """A buffered media-group page message."""
    msg = MagicMock()
    msg.message_id = message_id
    msg.from_user = None  # -> api_key None (default user)
    return msg


def _make_first_message(chat_id: int = 99) -> MagicMock:
    """The first message of the group, used for chat actions + replies."""
    msg = MagicMock()
    msg.message_id = 1
    msg.from_user = None
    msg.chat.id = chat_id
    msg.chat.do = AsyncMock()
    msg.reply = AsyncMock()
    return msg


def _make_buffer(pages: int, chat_id: int = 99) -> upload._MediaGroupBuffer:
    first = _make_first_message(chat_id)
    messages = [first] + [_make_page_message(i + 2) for i in range(pages - 1)]
    return upload._MediaGroupBuffer(
        chat_id=chat_id,
        first_message=first,
        messages=messages,
        doc_type="receipt",
        vendor_hint="Lidl",
    )


@pytest.mark.asyncio
async def test_two_page_group_calls_process_document_group(monkeypatch):
    client = MagicMock()
    client.process_document_group = AsyncMock(return_value=_ok_result())
    client.process_document = AsyncMock()
    monkeypatch.setattr(upload, "_client", client)
    monkeypatch.setattr(
        upload, "_get_attachment", AsyncMock(return_value=(b"bytes", "p.jpg"))
    )
    upload._pending_location.clear()

    buf = _make_buffer(pages=2, chat_id=99)
    await upload._process_media_group(buf)

    # Multi-page -> group endpoint, not the single-document one.
    client.process_document_group.assert_awaited_once()
    client.process_document.assert_not_awaited()

    pages_arg = client.process_document_group.await_args.args[0]
    assert len(pages_arg) == 2
    kwargs = client.process_document_group.await_args.kwargs
    assert kwargs["doc_type"] == "receipt"
    assert kwargs["vendor_hint"] == "Lidl"
    assert kwargs["api_key"] is None

    # Reply carries the formatted result on the first message.
    reply_texts = [c.args[0] for c in buf.first_message.reply.await_args_list]
    assert any("Vendor: Lidl" in t for t in reply_texts)

    # fact_id from the result primes the pending-location flow.
    assert upload._pending_location[99][0] == "fact1"


@pytest.mark.asyncio
async def test_single_page_group_uses_process_document(monkeypatch):
    client = MagicMock()
    client.process_document_group = AsyncMock()
    client.process_document = AsyncMock(return_value=_ok_result(fact_id=None))
    monkeypatch.setattr(upload, "_client", client)
    monkeypatch.setattr(
        upload, "_get_attachment", AsyncMock(return_value=(b"bytes", "p.jpg"))
    )
    upload._pending_location.clear()

    buf = _make_buffer(pages=1, chat_id=77)
    await upload._process_media_group(buf)

    # One page collapses to the single-document endpoint.
    client.process_document.assert_awaited_once()
    client.process_document_group.assert_not_awaited()
    # No fact_id -> no pending location set.
    assert 77 not in upload._pending_location
