"""Offline upload spool for the thin Telegram bot.

The bot container (``restart: unless-stopped``) and the host API (launchd
``KeepAlive``) both auto-start on boot with no ordering guarantee. If the
container wins the race, the API is briefly unreachable and any document the
user sends in that window would otherwise be lost -- Telegram already delivered
the update, so its ~24h server buffer does NOT cover it.

The spool closes that gap: when an upload fails with
:class:`~alibi.telegram.api_client.AlibiAPIConnectionError` (connection refused,
*not* an HTTP 4xx), the raw bytes plus the metadata needed to retry and reply
are persisted to disk. A background drain loop in ``main.py`` periodically
retries spooled entries against the API and, on success, sends the normal
formatted reply.

Layout (one directory per entry under ``ALIBI_TELEGRAM_SPOOL``, default
``/data/spool`` so it lives on the same mounted volume as the keystore)::

    spool/
      <entry_id>/
        meta.json        # SpoolEntry metadata + page filename list
        page0.bin        # raw document bytes (one per page)
        page1.bin

Like the keystore, entries hold user data (document bytes, API keys) so files
are created owner-only (0600) under owner-only directories (0700). This module
is deliberately DB/Ollama/pipeline free -- it is part of the thin bot package.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

DEFAULT_SPOOL_PATH = "/data/spool"

_META_NAME = "meta.json"


def _spool_path() -> Path:
    raw = os.environ.get("ALIBI_TELEGRAM_SPOOL", DEFAULT_SPOOL_PATH)
    return Path(raw).expanduser()


@dataclass
class SpoolEntry:
    """A single spooled upload, ready to be retried and replied to.

    ``pages`` is the in-memory list of ``(bytes, filename)`` tuples; for a
    single-document upload it has one element, for a media group it has one per
    page. ``kind`` records which API call to replay (``"single"`` vs
    ``"group"``).
    """

    id: str
    api_key: Optional[str]
    doc_type: Optional[str]
    vendor_hint: Optional[str]
    chat_id: int
    reply_to_message_id: Optional[int]
    kind: str
    ts: float
    pages: list[tuple[bytes, str]]


class Spool:
    """Persistent on-disk queue of failed uploads awaiting retry.

    Not internally locked: in the thin bot a single asyncio drain task is the
    only reader and handlers are the only writers, all on one event loop, so
    disk writes do not race. Each entry lives in its own directory and is
    written via a temp dir + atomic rename so a partially written entry is never
    observed by ``iter_pending``.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.dir = path or _spool_path()

    def _ensure_root(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.dir, 0o700)
        except OSError:  # pragma: no cover -- best effort on exotic mounts
            pass

    def add(
        self,
        pages: list[tuple[bytes, str]],
        *,
        kind: str,
        api_key: Optional[str],
        doc_type: Optional[str],
        vendor_hint: Optional[str],
        chat_id: int,
        reply_to_message_id: Optional[int],
    ) -> str:
        """Persist a failed upload and return its entry id.

        ``pages`` must be non-empty. The write is staged in a sibling ``.tmp``
        directory and atomically renamed into place so a concurrent
        ``iter_pending`` never sees a half-written entry.
        """
        if not pages:
            raise ValueError("Spool.add requires at least one page")

        self._ensure_root()
        entry_id = uuid.uuid4().hex
        ts = time.time()
        final = self.dir / entry_id
        staging = self.dir / f".{entry_id}.tmp"
        staging.mkdir(parents=True, exist_ok=True)
        os.chmod(staging, 0o700)

        page_names: list[str] = []
        for i, (data, filename) in enumerate(pages):
            page_file = f"page{i}.bin"
            page_path = staging / page_file
            page_path.write_bytes(data)
            os.chmod(page_path, 0o600)
            page_names.append(page_file)

        meta = {
            "id": entry_id,
            "api_key": api_key,
            "doc_type": doc_type,
            "vendor_hint": vendor_hint,
            "chat_id": chat_id,
            "reply_to_message_id": reply_to_message_id,
            "kind": kind,
            "ts": ts,
            "pages": [
                {"file": name, "filename": fn}
                for name, (_, fn) in zip(page_names, pages)
            ],
        }
        meta_path = staging / _META_NAME
        meta_path.write_text(json.dumps(meta, indent=2))
        os.chmod(meta_path, 0o600)

        staging.replace(final)
        logger.info(
            "Spooled %s upload %s (%d page(s), chat %s)",
            kind,
            entry_id,
            len(pages),
            chat_id,
        )
        return entry_id

    def iter_pending(self) -> Iterator[SpoolEntry]:
        """Yield spooled entries oldest-first, skipping unreadable ones.

        Loads each entry's page bytes into memory. A corrupt or partially
        written entry (bad JSON, missing page file) is logged and skipped rather
        than aborting the whole drain.
        """
        if not self.dir.exists():
            return

        entries: list[SpoolEntry] = []
        for child in self.dir.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            entry = self._load(child)
            if entry is not None:
                entries.append(entry)

        entries.sort(key=lambda e: e.ts)
        yield from entries

    def _load(self, entry_dir: Path) -> Optional[SpoolEntry]:
        meta_path = entry_dir / _META_NAME
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable spool entry %s: %s", entry_dir, exc)
            return None
        try:
            pages: list[tuple[bytes, str]] = []
            for page in meta["pages"]:
                data = (entry_dir / page["file"]).read_bytes()
                pages.append((data, page["filename"]))
            return SpoolEntry(
                id=str(meta["id"]),
                api_key=meta.get("api_key"),
                doc_type=meta.get("doc_type"),
                vendor_hint=meta.get("vendor_hint"),
                chat_id=int(meta["chat_id"]),
                reply_to_message_id=meta.get("reply_to_message_id"),
                kind=str(meta.get("kind", "single")),
                ts=float(meta.get("ts", 0.0)),
                pages=pages,
            )
        except (OSError, KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed spool entry %s: %s", entry_dir, exc)
            return None

    def remove(self, entry_id: str) -> bool:
        """Delete a spooled entry by id. Returns True if it existed."""
        entry_dir = self.dir / entry_id
        if not entry_dir.is_dir():
            return False
        for child in entry_dir.iterdir():
            try:
                child.unlink()
            except OSError:  # pragma: no cover -- best effort
                pass
        try:
            entry_dir.rmdir()
        except OSError as exc:  # pragma: no cover
            logger.warning("Could not remove spool entry %s: %s", entry_id, exc)
            return False
        return True

    def pending_count(self) -> int:
        """Number of spooled entries (cheap; does not read page bytes)."""
        if not self.dir.exists():
            return 0
        return sum(
            1 for c in self.dir.iterdir() if c.is_dir() and not c.name.startswith(".")
        )


_default_spool: Spool | None = None


def get_spool() -> Spool:
    """Return the process-wide spool singleton."""
    global _default_spool
    if _default_spool is None:
        _default_spool = Spool()
    return _default_spool
