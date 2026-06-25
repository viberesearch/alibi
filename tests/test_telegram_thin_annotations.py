"""Tests for the thin (API-client) Telegram annotation handlers.

``/tag`` and ``/untag`` are thin HTTP clients of the host ``/annotations``
endpoints: they are verified by asserting the right :class:`AlibiAPIClient`
calls and reply formatting (see ``docs/TELEGRAM_THIN_BOT_PLAN.md``).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from alibi.telegram.handlers import annotation


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
    client = MagicMock()
    for name, value in methods.items():
        setattr(client, name, AsyncMock(return_value=value))
    return client


_FID = "11111111-2222-3333-4444-555555555555"


@pytest.mark.asyncio
async def test_tag_inline_fact_id(monkeypatch):
    client = _mock_client(annotate_fact="ann-1")
    monkeypatch.setattr(annotation, "client", client)

    msg = _make_message(text=f"/tag {_FID} person Maria")
    await annotation.tag_handler(msg)

    client.annotate_fact.assert_awaited_once()
    assert client.annotate_fact.await_args.args[0] == _FID
    kwargs = client.annotate_fact.await_args.kwargs
    assert kwargs["annotation_type"] == "user_tag"
    assert kwargs["key"] == "person"
    assert kwargs["value"] == "Maria"
    out = msg.answer.await_args.args[0]
    assert "Tag added" in out and "ann-1" in out


@pytest.mark.asyncio
async def test_tag_quoted_value(monkeypatch):
    client = _mock_client(annotate_fact="ann-2")
    monkeypatch.setattr(annotation, "client", client)

    msg = _make_message(text=f'/tag {_FID} project "kitchen renovation"')
    await annotation.tag_handler(msg)

    kwargs = client.annotate_fact.await_args.kwargs
    assert kwargs["key"] == "project"
    assert kwargs["value"] == "kitchen renovation"


@pytest.mark.asyncio
async def test_tag_fact_id_from_reply(monkeypatch):
    client = _mock_client(annotate_fact="ann-3")
    monkeypatch.setattr(annotation, "client", client)

    msg = _make_message(
        text="/tag person Maria",
        reply_text=f"Document processed.\nFact ID: `{_FID}`",
    )
    await annotation.tag_handler(msg)

    client.annotate_fact.assert_awaited_once()
    assert client.annotate_fact.await_args.args[0] == _FID


@pytest.mark.asyncio
async def test_tag_usage_error_missing_args(monkeypatch):
    client = _mock_client(annotate_fact="ann-x")
    monkeypatch.setattr(annotation, "client", client)

    # fact_id present but no value -> usage error, no API call.
    msg = _make_message(text=f"/tag {_FID} key-only")
    await annotation.tag_handler(msg)

    client.annotate_fact.assert_not_awaited()
    assert "Usage" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_untag_happy(monkeypatch):
    client = _mock_client(delete_annotation=True)
    monkeypatch.setattr(annotation, "client", client)

    msg = _make_message(text="/untag ann-1")
    await annotation.untag_handler(msg)

    client.delete_annotation.assert_awaited_once()
    assert client.delete_annotation.await_args.args[0] == "ann-1"
    assert "deleted" in msg.answer.await_args.args[0].lower()


@pytest.mark.asyncio
async def test_untag_not_found(monkeypatch):
    client = _mock_client(delete_annotation=False)
    monkeypatch.setattr(annotation, "client", client)

    msg = _make_message(text="/untag missing")
    await annotation.untag_handler(msg)

    assert "not found" in msg.answer.await_args.args[0].lower()
