"""Tests for new Telegram bot handlers (language).

The budget and lineitem handlers are now thin API clients and are covered by
``test_telegram_thin_queries``. The language handler is unchanged (no DB/API
access) and remains tested here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def mock_message():
    """Create a mock Telegram Message object."""
    msg = AsyncMock()
    msg.answer = AsyncMock()
    return msg


# ---------------------------------------------------------------------------
# Language handler tests
# ---------------------------------------------------------------------------


class TestLanguageHandler:
    """Tests for /language command."""

    @pytest.mark.asyncio
    async def test_language_show(self, mock_message):
        """Test /language shows current language setting."""
        mock_message.text = "/language"

        with patch("alibi.telegram.handlers.language.get_config") as mock_cfg:
            mock_cfg.return_value.display_language = "original"

            from alibi.telegram.handlers.language import language_command

            await language_command(mock_message)

        mock_message.answer.assert_called_once()
        response = mock_message.answer.call_args[0][0]
        assert "original" in response
        assert "Supported" in response

    @pytest.mark.asyncio
    async def test_language_set_valid(self, mock_message):
        """Test /language en sets language."""
        mock_message.text = "/language en"

        from alibi.telegram.handlers.language import language_command

        await language_command(mock_message)

        mock_message.answer.assert_called_once()
        response = mock_message.answer.call_args[0][0]
        assert "preference set to" in response
        assert "display names" in response

    @pytest.mark.asyncio
    async def test_language_invalid(self, mock_message):
        """Test /language xx shows error for unsupported language."""
        mock_message.text = "/language xx"

        from alibi.telegram.handlers.language import language_command

        await language_command(mock_message)

        mock_message.answer.assert_called_once()
        response = mock_message.answer.call_args[0][0]
        assert "Unsupported language" in response
        assert "xx" in response
