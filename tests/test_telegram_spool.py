"""Tests for the thin bot's offline upload spool + retry drain.

The spool closes a live boot-order race: the bot container and host API both
auto-start with no ordering guarantee, so an upload can arrive before the API
is reachable. Connection failures (``AlibiAPIConnectionError``) are spooled to
disk and replayed by a background drain loop; genuine HTTP errors are not.
See ``docs/TELEGRAM_THIN_BOT_PLAN.md``.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from alibi.telegram import spool as spool_mod
from alibi.telegram.api_client import (
    AlibiAPIConnectionError,
    AlibiAPIError,
    ProcessResult,
)
from alibi.telegram.handlers import upload
from alibi.telegram.spool import Spool


@pytest.fixture
def tmp_spool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Spool:
    """Isolate the spool singleton to a temp directory."""
    store = Spool(tmp_path / "spool")
    monkeypatch.setattr(spool_mod, "_default_spool", store)
    return store


def _make_message(user_id: int = 42, chat_id: int = 99, msg_id: int = 7) -> MagicMock:
    msg = MagicMock()
    msg.from_user.id = user_id
    msg.chat.id = chat_id
    msg.message_id = msg_id
    msg.chat.do = AsyncMock()
    msg.reply = AsyncMock()
    return msg


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


# --- Spool persistence -----------------------------------------------------


def test_add_single_roundtrips(tmp_spool: Spool):
    entry_id = tmp_spool.add(
        [(b"imgbytes", "r.jpg")],
        kind="single",
        api_key="key1",
        doc_type="receipt",
        vendor_hint="Lidl",
        chat_id=99,
        reply_to_message_id=7,
    )
    entries = list(tmp_spool.iter_pending())
    assert len(entries) == 1
    e = entries[0]
    assert e.id == entry_id
    assert e.kind == "single"
    assert e.api_key == "key1"
    assert e.doc_type == "receipt"
    assert e.vendor_hint == "Lidl"
    assert e.chat_id == 99
    assert e.reply_to_message_id == 7
    assert e.pages == [(b"imgbytes", "r.jpg")]


def test_add_group_preserves_pages_in_order(tmp_spool: Spool):
    pages = [(b"p0", "a.jpg"), (b"p1", "b.jpg"), (b"p2", "c.jpg")]
    tmp_spool.add(
        pages,
        kind="group",
        api_key=None,
        doc_type=None,
        vendor_hint=None,
        chat_id=1,
        reply_to_message_id=None,
    )
    (e,) = list(tmp_spool.iter_pending())
    assert e.kind == "group"
    assert e.pages == pages
    assert e.api_key is None


def test_add_rejects_empty_pages(tmp_spool: Spool):
    with pytest.raises(ValueError):
        tmp_spool.add(
            [],
            kind="single",
            api_key=None,
            doc_type=None,
            vendor_hint=None,
            chat_id=1,
            reply_to_message_id=None,
        )


def test_remove_deletes_entry(tmp_spool: Spool):
    entry_id = tmp_spool.add(
        [(b"x", "r.jpg")],
        kind="single",
        api_key=None,
        doc_type=None,
        vendor_hint=None,
        chat_id=1,
        reply_to_message_id=None,
    )
    assert tmp_spool.pending_count() == 1
    assert tmp_spool.remove(entry_id) is True
    assert tmp_spool.pending_count() == 0
    assert tmp_spool.remove(entry_id) is False


def test_iter_pending_orders_oldest_first(tmp_spool: Spool, monkeypatch):
    # Patching spool_mod.time.time patches the *global* time module, so logging
    # inside add() also calls it -- use a never-exhausting monotonic counter
    # (not a fixed-length iterator) to avoid an incidental StopIteration.
    counter = itertools.count(100, 100)
    monkeypatch.setattr(spool_mod.time, "time", lambda: float(next(counter)))
    first = tmp_spool.add(
        [(b"a", "a.jpg")],
        kind="single",
        api_key=None,
        doc_type=None,
        vendor_hint=None,
        chat_id=1,
        reply_to_message_id=None,
    )
    second = tmp_spool.add(
        [(b"b", "b.jpg")],
        kind="single",
        api_key=None,
        doc_type=None,
        vendor_hint=None,
        chat_id=1,
        reply_to_message_id=None,
    )
    ids = [e.id for e in tmp_spool.iter_pending()]
    assert ids == [first, second]


# --- Handler spool-on-connection-error -------------------------------------


@pytest.mark.asyncio
async def test_connection_error_spools_the_upload(tmp_spool: Spool, monkeypatch):
    client = MagicMock()
    client.process_document = AsyncMock(side_effect=AlibiAPIConnectionError("down"))
    monkeypatch.setattr(upload, "_client", client)

    msg = _make_message()
    await upload._process_attachment(msg, b"imgbytes", "r.jpg", "receipt", "Lidl")

    entries = list(tmp_spool.iter_pending())
    assert len(entries) == 1
    assert entries[0].pages == [(b"imgbytes", "r.jpg")]
    assert entries[0].doc_type == "receipt"
    # User is told it was saved, not that it failed.
    assert "Saved" in msg.reply.await_args.args[0]


@pytest.mark.asyncio
async def test_http_error_does_not_spool(tmp_spool: Spool, monkeypatch):
    client = MagicMock()
    client.process_document = AsyncMock(side_effect=AlibiAPIError("400: bad"))
    monkeypatch.setattr(upload, "_client", client)

    msg = _make_message()
    await upload._process_attachment(msg, b"imgbytes", "r.jpg", None, None)

    assert tmp_spool.pending_count() == 0
    assert "error" in msg.reply.await_args.args[0].lower()


# --- Drain loop ------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_success_replies_and_removes(tmp_spool: Spool, monkeypatch):
    tmp_spool.add(
        [(b"imgbytes", "r.jpg")],
        kind="single",
        api_key="key1",
        doc_type="receipt",
        vendor_hint="Lidl",
        chat_id=99,
        reply_to_message_id=7,
    )
    client = MagicMock()
    client.process_document = AsyncMock(return_value=_ok_result())
    monkeypatch.setattr(upload, "_client", client)
    upload._pending_location.clear()

    bot = MagicMock()
    bot.send_message = AsyncMock()

    processed = await upload.drain_spool_once(bot)

    assert processed == 1
    client.process_document.assert_awaited_once()
    assert client.process_document.await_args.kwargs["api_key"] == "key1"
    bot.send_message.assert_awaited_once()
    sent = bot.send_message.await_args.kwargs
    assert sent["chat_id"] == 99
    assert sent["reply_to_message_id"] == 7
    assert "Lidl" in sent["text"]
    # Entry cleared, and the map/location flow is primed for the chat.
    assert tmp_spool.pending_count() == 0
    assert upload._pending_location[99][0] == "fact1"


@pytest.mark.asyncio
async def test_drain_group_uses_group_endpoint(tmp_spool: Spool, monkeypatch):
    pages = [(b"p0", "a.jpg"), (b"p1", "b.jpg")]
    tmp_spool.add(
        pages,
        kind="group",
        api_key=None,
        doc_type=None,
        vendor_hint=None,
        chat_id=5,
        reply_to_message_id=None,
    )
    client = MagicMock()
    client.process_document_group = AsyncMock(return_value=_ok_result())
    client.process_document = AsyncMock()
    monkeypatch.setattr(upload, "_client", client)

    bot = MagicMock()
    bot.send_message = AsyncMock()

    await upload.drain_spool_once(bot)

    client.process_document_group.assert_awaited_once()
    client.process_document.assert_not_awaited()
    assert tmp_spool.pending_count() == 0


@pytest.mark.asyncio
async def test_drain_still_down_keeps_entry(tmp_spool: Spool, monkeypatch):
    tmp_spool.add(
        [(b"x", "r.jpg")],
        kind="single",
        api_key=None,
        doc_type=None,
        vendor_hint=None,
        chat_id=1,
        reply_to_message_id=None,
    )
    client = MagicMock()
    client.process_document = AsyncMock(side_effect=AlibiAPIConnectionError("down"))
    monkeypatch.setattr(upload, "_client", client)

    bot = MagicMock()
    bot.send_message = AsyncMock()

    processed = await upload.drain_spool_once(bot)

    assert processed == 0
    assert tmp_spool.pending_count() == 1  # entry kept for next pass
    bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_drain_http_error_drops_and_notifies(tmp_spool: Spool, monkeypatch):
    tmp_spool.add(
        [(b"x", "r.jpg")],
        kind="single",
        api_key=None,
        doc_type=None,
        vendor_hint=None,
        chat_id=1,
        reply_to_message_id=None,
    )
    client = MagicMock()
    client.process_document = AsyncMock(side_effect=AlibiAPIError("422: unprocessable"))
    monkeypatch.setattr(upload, "_client", client)

    bot = MagicMock()
    bot.send_message = AsyncMock()

    processed = await upload.drain_spool_once(bot)

    assert processed == 1
    assert tmp_spool.pending_count() == 0  # permanent failure dropped
    bot.send_message.assert_awaited_once()
    assert "could not be processed" in bot.send_message.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_drain_reply_falls_back_when_original_gone(tmp_spool: Spool, monkeypatch):
    tmp_spool.add(
        [(b"x", "r.jpg")],
        kind="single",
        api_key=None,
        doc_type=None,
        vendor_hint=None,
        chat_id=1,
        reply_to_message_id=123,
    )
    client = MagicMock()
    client.process_document = AsyncMock(return_value=_ok_result(fact_id=None))
    monkeypatch.setattr(upload, "_client", client)

    bot = MagicMock()
    # First send (with reply_to) fails as if the message was deleted; retry ok.
    bot.send_message = AsyncMock(
        side_effect=[Exception("message to reply not found"), None]
    )

    await upload.drain_spool_once(bot)

    assert bot.send_message.await_count == 2
    assert tmp_spool.pending_count() == 0


# --- Timeout handling (server has the doc; never spool / never resend) ------


@pytest.mark.asyncio
async def test_timeout_does_not_spool_and_says_dont_resend(
    tmp_spool: Spool, monkeypatch
):
    from alibi.telegram.api_client import AlibiAPITimeoutError

    client = MagicMock()
    client.process_document = AsyncMock(side_effect=AlibiAPITimeoutError("420s"))
    monkeypatch.setattr(upload, "_client", client)

    msg = _make_message()
    await upload._process_attachment(msg, b"imgbytes", "r.jpg", None, None)

    # Spooling would replay the upload against a server that already has it.
    assert tmp_spool.pending_count() == 0
    reply = msg.reply.await_args.args[0].lower()
    assert "resend" in reply
    assert "error" not in reply


@pytest.mark.asyncio
async def test_drain_timeout_clears_entry_without_failure_notice(
    tmp_spool: Spool, monkeypatch
):
    from alibi.telegram.api_client import AlibiAPITimeoutError

    tmp_spool.add(
        [(b"x", "a.jpg")],
        kind="single",
        api_key=None,
        doc_type=None,
        vendor_hint=None,
        chat_id=1,
        reply_to_message_id=None,
    )
    client = MagicMock()
    client.process_document = AsyncMock(side_effect=AlibiAPITimeoutError("420s"))
    monkeypatch.setattr(upload, "_client", client)

    bot = MagicMock()
    bot.send_message = AsyncMock()
    cleared = await upload.drain_spool_once(bot)

    # Entry cleared (replaying would duplicate), user NOT told it failed.
    assert cleared == 1
    assert tmp_spool.pending_count() == 0
    bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_timeout_raises_timeout_error(monkeypatch):
    """A read timeout after delivery maps to AlibiAPITimeoutError, no retries."""
    import httpx

    from alibi.telegram.api_client import AlibiAPIClient, AlibiAPITimeoutError

    calls = itertools.count()

    class _FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, *a, **kw):
            next(calls)
            raise httpx.ReadTimeout("read timed out")

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    client = AlibiAPIClient(base_url="http://test")
    with pytest.raises(AlibiAPITimeoutError):
        await client._request("POST", "/x", api_key=None, timeout=1.0)
    assert next(calls) == 1  # no retry loop for delivered-but-slow requests
