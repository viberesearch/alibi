"""Shared helpers for the thin Telegram handlers.

Every handler is a thin HTTP client of the host API: it resolves the sender's
``X-API-Key`` from the local keystore and forwards the request. These helpers
centralise the shared client instance and the per-message key lookup so each
handler module stays small (see ``docs/TELEGRAM_THIN_BOT_PLAN.md``).
"""

from __future__ import annotations

from aiogram.types import Message

from alibi.telegram.api_client import AlibiAPIClient
from alibi.telegram.keystore import get_keystore

# One shared async client for all handlers (base URL from ALIBI_API_URL).
client = AlibiAPIClient()


def api_key_for(message: Message) -> str | None:
    """Return the API key linked to the message sender, or None (default user).

    When a user has not run ``/link``, None is returned and the host API falls
    back to single-user/default attribution -- matching the pre-thin behaviour.
    """
    if message.from_user:
        return get_keystore().get(message.from_user.id)
    return None
