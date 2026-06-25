"""Tests for the thin (API-client) Telegram upload + account handlers.

These replace the in-process pipeline tests: the bot no longer touches the DB
or runs the pipeline, so handlers are verified by asserting the right
:class:`AlibiAPIClient` calls are made and replies are formatted from the API
response. Companion thin suites cover the query/correction/annotation handlers
(test_telegram_thin_*); the media-group path lives in
test_telegram_thin_media_group. See docs/TELEGRAM_THIN_BOT_PLAN.md.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from alibi.telegram import keystore as keystore_mod
from alibi.telegram.api_client import ProcessResult
from alibi.telegram.handlers import upload


@pytest.fixture
def tmp_keystore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate the keystore singleton to a temp file."""
    store = keystore_mod.TelegramKeystore(tmp_path / "keys.json")
    monkeypatch.setattr(keystore_mod, "_default_store", store)
    return store


def _make_message(text: str = "", user_id: int = 42, chat_id: int = 99) -> MagicMock:
    """Build a mock aiogram Message with async reply/chat helpers."""
    msg = MagicMock()
    msg.text = text
    msg.from_user.id = user_id
    msg.chat.id = chat_id
    msg.chat.do = AsyncMock()
    msg.reply = AsyncMock()
    msg.photo = None
    msg.document = None
    msg.reply_to_message = None
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


def test_parse_type_and_hint_short_alias():
    assert upload._parse_type_and_hint("receipt", "Lidl") == ("receipt", "Lidl")
    assert upload._parse_type_and_hint("receipt", "") == ("receipt", None)


def test_parse_type_and_hint_upload_with_type():
    assert upload._parse_type_and_hint("upload", "invoice ACME") == ("invoice", "ACME")
    # First token not a valid type -> whole string is the hint
    assert upload._parse_type_and_hint("upload", "ACME Corp") == (None, "ACME Corp")


def test_format_result_success_includes_fact_and_currency():
    text = upload._format_result(_ok_result())
    assert "Vendor: Lidl" in text
    assert "12.50 EUR" in text
    assert "`fact1`" in text
    assert "/fix `fact1`" in text


def test_format_result_duplicate():
    text = upload._format_result(
        ProcessResult(success=True, is_duplicate=True, duplicate_of="docX")
    )
    assert "duplicate" in text.lower()
    assert "docX" in text


def test_format_result_failure():
    text = upload._format_result(ProcessResult(success=False))
    assert "failed" in text.lower()


@pytest.mark.asyncio
async def test_process_attachment_forwards_to_api_and_sets_pending_location(
    tmp_keystore, monkeypatch
):
    tmp_keystore.set(42, "mnemonic-key")
    client = MagicMock()
    client.process_document = AsyncMock(return_value=_ok_result())
    monkeypatch.setattr(upload, "_client", client)
    upload._pending_location.clear()

    msg = _make_message()
    await upload._process_attachment(msg, b"imgbytes", "r.jpg", "receipt", "Lidl")

    client.process_document.assert_awaited_once()
    kwargs = client.process_document.await_args.kwargs
    assert kwargs["api_key"] == "mnemonic-key"
    assert kwargs["doc_type"] == "receipt"
    assert kwargs["vendor_hint"] == "Lidl"
    # fact_id from the response primes the pending-location flow
    assert upload._pending_location[99][0] == "fact1"


@pytest.mark.asyncio
async def test_process_attachment_no_key_uses_default_user(tmp_keystore, monkeypatch):
    client = MagicMock()
    client.process_document = AsyncMock(return_value=_ok_result(fact_id=None))
    monkeypatch.setattr(upload, "_client", client)

    msg = _make_message()
    await upload._process_attachment(msg, b"x", "r.jpg", None, None)

    assert client.process_document.await_args.kwargs["api_key"] is None


@pytest.mark.asyncio
async def test_link_valid_key_stores_in_keystore(tmp_keystore, monkeypatch):
    client = MagicMock()
    client.whoami = AsyncMock(return_value={"id": "u1", "name": "Alice"})
    monkeypatch.setattr(upload, "_client", client)

    msg = _make_message(text="/link some mnemonic words", user_id=7)
    await upload.link_handler(msg)

    assert tmp_keystore.get(7) == "some mnemonic words"
    assert "Alice" in msg.reply.await_args.args[0]


@pytest.mark.asyncio
async def test_link_invalid_key_not_stored(tmp_keystore, monkeypatch):
    client = MagicMock()
    client.whoami = AsyncMock(return_value=None)
    monkeypatch.setattr(upload, "_client", client)

    msg = _make_message(text="/link bad key", user_id=8)
    await upload.link_handler(msg)

    assert tmp_keystore.get(8) is None
    assert "Invalid" in msg.reply.await_args.args[0]


@pytest.mark.asyncio
async def test_unlink_removes_key(tmp_keystore, monkeypatch):
    tmp_keystore.set(9, "k")
    monkeypatch.setattr(upload, "_client", MagicMock())

    msg = _make_message(user_id=9)
    await upload.unlink_handler(msg)

    assert tmp_keystore.get(9) is None


@pytest.mark.asyncio
async def test_store_location_calls_api(tmp_keystore, monkeypatch):
    tmp_keystore.set(42, "k")
    client = MagicMock()
    client.set_fact_location = AsyncMock(return_value={"place_name": "Lidl Berlin"})
    monkeypatch.setattr(upload, "_client", client)

    msg = _make_message()
    await upload._store_location(msg, "fact1", "https://maps.google.com/?q=1,2")

    client.set_fact_location.assert_awaited_once()
    assert client.set_fact_location.await_args.args[0] == "fact1"
    assert "Lidl Berlin" in msg.reply.await_args.args[0]
