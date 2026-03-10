"""Tests for Telegram user allowlist middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alibi.telegram.middleware import AllowedUsersMiddleware


def _make_update(user_id: int | None = None) -> MagicMock:
    """Create a mock aiogram Update with a message from a user."""
    from aiogram.types import Update

    update = MagicMock(spec=Update)
    update.callback_query = None
    update.inline_query = None
    update.edited_message = None

    if user_id is not None:
        update.message = MagicMock()
        update.message.from_user = MagicMock()
        update.message.from_user.id = user_id
    else:
        update.message = None

    return update


class TestAllowedUsersMiddleware:
    def test_parses_allowed_users_from_config(self):
        with patch("alibi.telegram.middleware.get_config") as mock_cfg:
            mock_cfg.return_value.telegram_allowed_users = "111,222,333"
            mw = AllowedUsersMiddleware()
            assert mw.allowed_ids == {111, 222, 333}

    def test_empty_allowlist_allows_all(self):
        with patch("alibi.telegram.middleware.get_config") as mock_cfg:
            mock_cfg.return_value.telegram_allowed_users = ""
            mw = AllowedUsersMiddleware()
            assert mw.allowed_ids == set()

    def test_handles_whitespace_in_config(self):
        with patch("alibi.telegram.middleware.get_config") as mock_cfg:
            mock_cfg.return_value.telegram_allowed_users = " 111 , 222 , "
            mw = AllowedUsersMiddleware()
            assert mw.allowed_ids == {111, 222}

    @pytest.mark.asyncio
    async def test_allows_permitted_user(self):
        with patch("alibi.telegram.middleware.get_config") as mock_cfg:
            mock_cfg.return_value.telegram_allowed_users = "12345,67890"
            mw = AllowedUsersMiddleware()

        handler = AsyncMock(return_value="ok")
        update = _make_update(user_id=12345)

        result = await mw(handler, update, {})
        assert result == "ok"
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_blocks_unknown_user(self):
        with patch("alibi.telegram.middleware.get_config") as mock_cfg:
            mock_cfg.return_value.telegram_allowed_users = "12345"
            mw = AllowedUsersMiddleware()

        handler = AsyncMock(return_value="ok")
        update = _make_update(user_id=99999)

        result = await mw(handler, update, {})
        assert result is None
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_open_mode_when_no_allowlist(self):
        with patch("alibi.telegram.middleware.get_config") as mock_cfg:
            mock_cfg.return_value.telegram_allowed_users = ""
            mw = AllowedUsersMiddleware()

        handler = AsyncMock(return_value="ok")
        update = _make_update(user_id=99999)

        result = await mw(handler, update, {})
        assert result == "ok"
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_blocks_update_without_user(self):
        with patch("alibi.telegram.middleware.get_config") as mock_cfg:
            mock_cfg.return_value.telegram_allowed_users = "12345"
            mw = AllowedUsersMiddleware()

        handler = AsyncMock(return_value="ok")
        update = _make_update(user_id=None)

        result = await mw(handler, update, {})
        assert result is None
        handler.assert_not_awaited()
