"""Telegram bot middleware for access control."""

import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Update

from alibi.config import get_config

logger = logging.getLogger(__name__)


class AllowedUsersMiddleware(BaseMiddleware):
    """Reject messages from Telegram users not in the allowlist.

    When ALIBI_TELEGRAM_ALLOWED_USERS is set (comma-separated user IDs),
    only those users can interact with the bot. All others are silently
    ignored (no response, no processing).

    When the allowlist is empty, the bot is open to all users (single-user
    or development mode).
    """

    def __init__(self) -> None:
        config = get_config()
        raw = config.telegram_allowed_users.strip()
        if raw:
            self.allowed_ids: set[int] = {
                int(uid.strip()) for uid in raw.split(",") if uid.strip().isdigit()
            }
        else:
            self.allowed_ids = set()

        if self.allowed_ids:
            logger.info(
                "Telegram allowlist active: %d user(s) permitted",
                len(self.allowed_ids),
            )
        else:
            logger.warning(
                "ALIBI_TELEGRAM_ALLOWED_USERS is not set -- "
                "bot is open to ALL Telegram users. "
                "Set this variable to restrict access."
            )

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # If no allowlist configured, pass everything through
        if not self.allowed_ids:
            return await handler(event, data)

        # Extract user ID from the update
        user_id = self._extract_user_id(event)
        if user_id is None:
            # No user context (e.g. channel post) -- block by default
            return None

        if user_id not in self.allowed_ids:
            logger.debug("Blocked Telegram user %d (not in allowlist)", user_id)
            return None

        return await handler(event, data)

    @staticmethod
    def _extract_user_id(event: TelegramObject) -> int | None:
        """Extract the sender's user ID from any update type."""
        if not isinstance(event, Update):
            return None

        # Check all update types that carry a user
        if event.message and event.message.from_user:
            return event.message.from_user.id
        if event.callback_query and event.callback_query.from_user:
            return event.callback_query.from_user.id
        if event.inline_query and event.inline_query.from_user:
            return event.inline_query.from_user.id
        if event.edited_message and event.edited_message.from_user:
            return event.edited_message.from_user.id

        return None
