"""Tests for the thin (API-client) Telegram correction handlers.

The bot no longer touches the DB or runs the pipeline. The correction handlers
(``/fix``, ``/barcode``, ``/merge`` + the ``yes`` confirm) are thin HTTP clients
of the host API, so they are verified by asserting the right
:class:`AlibiAPIClient` calls are made and replies are formatted from the API
response (see ``docs/TELEGRAM_THIN_BOT_PLAN.md``).
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from alibi.telegram.handlers import correction


def _make_message(
    text: str = "", user_id: int = 42, chat_id: int = 99, reply_text: str | None = None
) -> MagicMock:
    """Build a mock aiogram Message with an async ``answer`` helper."""
    msg = MagicMock()
    msg.text = text
    msg.from_user.id = user_id
    msg.chat.id = chat_id
    msg.answer = AsyncMock()
    if reply_text is None:
        msg.reply_to_message = None
    else:
        reply = MagicMock()
        reply.text = reply_text
        msg.reply_to_message = reply
    return msg


def _mock_client(**methods) -> MagicMock:
    """Build a MagicMock client whose named methods are AsyncMocks."""
    client = MagicMock()
    for name, value in methods.items():
        setattr(client, name, AsyncMock(return_value=value))
    return client


# ---------------------------------------------------------------------------
# /fix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fix_vendor_happy(monkeypatch):
    client = _mock_client(get_fact={"vendor": "Old Vendor"}, correct_vendor=True)
    monkeypatch.setattr(correction, "client", client)

    fid = "11111111-2222-3333-4444-555555555555"
    msg = _make_message(text=f"/fix {fid} vendor Fresko")
    await correction.fix_handler(msg)

    client.correct_vendor.assert_awaited_once()
    assert client.correct_vendor.await_args.args[0] == fid
    assert client.correct_vendor.await_args.args[1] == "Fresko"
    out = msg.answer.await_args.args[0]
    assert "Old Vendor" in out and "Fresko" in out


@pytest.mark.asyncio
async def test_fix_amount_happy(monkeypatch):
    client = _mock_client(get_fact={"total_amount": "10.00"}, update_fact=True)
    monkeypatch.setattr(correction, "client", client)

    fid = "11111111-2222-3333-4444-555555555555"
    msg = _make_message(text=f"/fix {fid} amount 12.50")
    await correction.fix_handler(msg)

    client.update_fact.assert_awaited_once()
    assert client.update_fact.await_args.args[0] == fid
    assert client.update_fact.await_args.args[1] == {"amount": "12.50"}
    assert "12.50" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_fix_date_happy(monkeypatch):
    client = _mock_client(get_fact={"event_date": "2026-01-01"}, update_fact=True)
    monkeypatch.setattr(correction, "client", client)

    fid = "11111111-2222-3333-4444-555555555555"
    msg = _make_message(text=f"/fix {fid} date 2026-06-10")
    await correction.fix_handler(msg)

    client.update_fact.assert_awaited_once()
    assert client.update_fact.await_args.args[1] == {"date": "2026-06-10"}
    assert "2026-06-10" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_fix_not_found(monkeypatch):
    client = _mock_client(get_fact=None, correct_vendor=False)
    monkeypatch.setattr(correction, "client", client)

    fid = "11111111-2222-3333-4444-555555555555"
    msg = _make_message(text=f"/fix {fid} vendor Fresko")
    await correction.fix_handler(msg)

    assert "not found" in msg.answer.await_args.args[0].lower()


@pytest.mark.asyncio
async def test_fix_invalid_amount(monkeypatch):
    client = _mock_client(get_fact=None, update_fact=True)
    monkeypatch.setattr(correction, "client", client)

    fid = "11111111-2222-3333-4444-555555555555"
    msg = _make_message(text=f"/fix {fid} amount notanumber")
    await correction.fix_handler(msg)

    assert "Invalid amount" in msg.answer.await_args.args[0]
    client.update_fact.assert_not_awaited()


@pytest.mark.asyncio
async def test_fix_invalid_date(monkeypatch):
    client = _mock_client(get_fact=None, update_fact=True)
    monkeypatch.setattr(correction, "client", client)

    fid = "11111111-2222-3333-4444-555555555555"
    msg = _make_message(text=f"/fix {fid} date 06-10-2026")
    await correction.fix_handler(msg)

    assert "Invalid date" in msg.answer.await_args.args[0]
    client.update_fact.assert_not_awaited()


@pytest.mark.asyncio
async def test_fix_fact_id_from_reply(monkeypatch):
    client = _mock_client(get_fact={"vendor": "Old"}, correct_vendor=True)
    monkeypatch.setattr(correction, "client", client)

    fid = "11111111-2222-3333-4444-555555555555"
    msg = _make_message(
        text="/fix vendor Fresko",
        reply_text=f"Document processed.\nFact ID: `{fid}`\nUse /fix to edit.",
    )
    await correction.fix_handler(msg)

    client.correct_vendor.assert_awaited_once()
    assert client.correct_vendor.await_args.args[0] == fid


# ---------------------------------------------------------------------------
# /barcode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_barcode_happy(monkeypatch):
    client = _mock_client(update_line_item=True)
    monkeypatch.setattr(correction, "client", client)

    iid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    msg = _make_message(text=f"/barcode {iid} 4006381333931")
    await correction.barcode_handler(msg)

    client.update_line_item.assert_awaited_once()
    assert client.update_line_item.await_args.args[0] == iid
    assert client.update_line_item.await_args.args[1] == {"barcode": "4006381333931"}
    assert "Barcode set" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_barcode_invalid_format(monkeypatch):
    client = _mock_client(update_line_item=True)
    monkeypatch.setattr(correction, "client", client)

    iid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    msg = _make_message(text=f"/barcode {iid} ABC123")
    await correction.barcode_handler(msg)

    assert "Invalid barcode" in msg.answer.await_args.args[0]
    client.update_line_item.assert_not_awaited()


@pytest.mark.asyncio
async def test_barcode_not_found(monkeypatch):
    client = _mock_client(update_line_item=False)
    monkeypatch.setattr(correction, "client", client)

    iid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    msg = _make_message(text=f"/barcode {iid} 4006381333931")
    await correction.barcode_handler(msg)

    assert "not found" in msg.answer.await_args.args[0].lower()


# ---------------------------------------------------------------------------
# /merge + yes confirm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_then_confirm_happy(monkeypatch):
    client = _mock_client(
        get_identity={"name": "Lidl"},
        merge_identities=True,
    )
    monkeypatch.setattr(correction, "client", client)
    correction._pending_merges.clear()

    ida = "11111111-1111-1111-1111-111111111111"
    idb = "22222222-2222-2222-2222-222222222222"
    msg = _make_message(text=f"/merge {ida} {idb}", chat_id=99)
    await correction.merge_handler(msg)

    # Pending merge stored, no merge yet.
    assert 99 in correction._pending_merges
    client.merge_identities.assert_not_awaited()
    assert "confirm" in msg.answer.await_args.args[0].lower()

    confirm = _make_message(text="yes", chat_id=99)
    await correction.merge_confirm_handler(confirm)

    client.merge_identities.assert_awaited_once()
    assert client.merge_identities.await_args.args[0] == ida
    assert client.merge_identities.await_args.args[1] == idb
    assert "merged" in confirm.answer.await_args.args[0].lower()
    assert 99 not in correction._pending_merges


@pytest.mark.asyncio
async def test_merge_confirm_expired(monkeypatch):
    client = _mock_client(merge_identities=True)
    monkeypatch.setattr(correction, "client", client)

    ida = "11111111-1111-1111-1111-111111111111"
    idb = "22222222-2222-2222-2222-222222222222"
    # Stash a pending merge with a stale timestamp (> TTL ago).
    correction._pending_merges.clear()
    correction._pending_merges[99] = (
        ida,
        idb,
        time.time() - correction._MERGE_CONFIRM_TTL - 5,
    )

    confirm = _make_message(text="yes", chat_id=99)
    await correction.merge_confirm_handler(confirm)

    client.merge_identities.assert_not_awaited()
    assert "expired" in confirm.answer.await_args.args[0].lower()


@pytest.mark.asyncio
async def test_merge_confirm_no_pending(monkeypatch):
    client = _mock_client(merge_identities=True)
    monkeypatch.setattr(correction, "client", client)
    correction._pending_merges.clear()

    confirm = _make_message(text="yes", chat_id=99)
    await correction.merge_confirm_handler(confirm)

    # No pending merge -> ignored silently (no merge, no reply).
    client.merge_identities.assert_not_awaited()
    confirm.answer.assert_not_awaited()
