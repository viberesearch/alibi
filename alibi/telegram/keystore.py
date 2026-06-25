"""Telegram-user -> API-key store for the thin bot.

In the thin (containerised) model the bot has no DB access, so it cannot use
``find_user_by_telegram``. Instead it keeps its own mapping of Telegram user id
-> the user's mnemonic API key, established when the user runs ``/link <key>``.
Each upload/query then sends that key as ``X-API-Key`` and the host API resolves
it to the right Alibi user, preserving per-user attribution.

The store is a small JSON file persisted to ``ALIBI_TELEGRAM_KEYSTORE`` (default
``~/.alibi/telegram_keys.json``). Mount that path as a Docker volume so links
survive container restarts. The file holds API keys, so it is created with
owner-only permissions and should be treated as a secret.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_KEYSTORE_PATH = "~/.alibi/telegram_keys.json"


def _keystore_path() -> Path:
    raw = os.environ.get("ALIBI_TELEGRAM_KEYSTORE", DEFAULT_KEYSTORE_PATH)
    return Path(raw).expanduser()


class TelegramKeystore:
    """Thread-safe persistent ``{telegram_id -> api_key}`` map.

    Telegram ids are stored as strings. All access is guarded by a lock because
    aiogram may dispatch handlers from worker threads.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _keystore_path()
        self._lock = threading.Lock()
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            if isinstance(raw, dict):
                self._data = {str(k): str(v) for k, v in raw.items()}
        except (OSError, ValueError) as exc:
            logger.warning("Could not read keystore %s: %s", self.path, exc)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    def get(self, telegram_id: str | int) -> str | None:
        """Return the API key linked to a Telegram id, or None."""
        with self._lock:
            return self._data.get(str(telegram_id))

    def set(self, telegram_id: str | int, api_key: str) -> None:
        """Link a Telegram id to an API key and persist."""
        with self._lock:
            self._data[str(telegram_id)] = api_key
            self._save()

    def remove(self, telegram_id: str | int) -> bool:
        """Unlink a Telegram id. Returns True if it was present."""
        with self._lock:
            existed = str(telegram_id) in self._data
            self._data.pop(str(telegram_id), None)
            if existed:
                self._save()
            return existed

    def is_linked(self, telegram_id: str | int) -> bool:
        with self._lock:
            return str(telegram_id) in self._data


_default_store: TelegramKeystore | None = None
_default_lock = threading.Lock()


def get_keystore() -> TelegramKeystore:
    """Return the process-wide keystore singleton."""
    global _default_store
    with _default_lock:
        if _default_store is None:
            _default_store = TelegramKeystore()
        return _default_store
