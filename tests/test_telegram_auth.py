"""Tests for Telegram auth handlers: /link, /whoami, user resolution."""

from __future__ import annotations

from unittest.mock import MagicMock

from alibi.services.auth import create_api_key, create_user, link_telegram
from alibi.telegram.handlers.upload import _resolve_telegram_user


def _make_message(telegram_user_id: int | None = None) -> MagicMock:
    """Create a mock Message with a from_user."""
    msg = MagicMock()
    if telegram_user_id is not None:
        msg.from_user = MagicMock()
        msg.from_user.id = telegram_user_id
    else:
        msg.from_user = None
    return msg


class TestResolveTelegramUser:
    def test_returns_system_when_no_from_user(self, db):
        msg = _make_message(telegram_user_id=None)
        assert _resolve_telegram_user(db, msg) == "system"

    def test_returns_system_when_not_linked(self, db):
        msg = _make_message(telegram_user_id=12345)
        assert _resolve_telegram_user(db, msg) == "system"

    def test_returns_user_id_when_linked(self, db):
        user = create_user(db, "Alice")
        link_telegram(db, user["id"], "12345")

        msg = _make_message(telegram_user_id=12345)
        assert _resolve_telegram_user(db, msg) == user["id"]


class TestLinkFlow:
    """Test the link flow: create user, create key, validate key, link telegram."""

    def test_full_link_flow(self, db):
        # Create user and key
        user = create_user(db, "Alice")
        key = create_api_key(db, user["id"])

        # Validate the key (as /link handler would)
        from alibi.services.auth import validate_api_key

        validated = validate_api_key(db, key["mnemonic"])
        assert validated is not None
        assert validated["id"] == user["id"]

        # Link telegram
        assert link_telegram(db, validated["id"], "67890")

        # Now resolve should return the user
        msg = _make_message(telegram_user_id=67890)
        assert _resolve_telegram_user(db, msg) == user["id"]
