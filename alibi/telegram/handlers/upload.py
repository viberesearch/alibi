"""Document upload handler with type commands and media group support.

Supports /receipt, /invoice, /payment, /statement, /warranty, /contract,
and /upload commands. Each command optionally carries a vendor hint and
can be followed by a photo or document attachment, or sent first to
prime a 60-second upload window.

When multiple photos are sent as a media group (album), they are
automatically buffered and processed as pages of a single document
via ``process_document_group()``.

Files are persisted to the inbox before processing so that the optimized
image remains on disk after the pipeline completes.
"""

import asyncio
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import Message

from alibi.db.connection import DatabaseManager
from alibi.db.models import DocumentType
from alibi.processing.folder_router import FolderContext
from alibi.telegram.handlers import require_db
from alibi.services.ingestion import (
    persist_upload,
    persist_upload_group,
    process_document_group,
    process_file,
)

logger = logging.getLogger(__name__)

router = Router()

# Command name -> DocumentType
_COMMAND_TYPE_MAP: dict[str, DocumentType] = {
    "receipt": DocumentType.RECEIPT,
    "invoice": DocumentType.INVOICE,
    "payment": DocumentType.PAYMENT_CONFIRMATION,
    "statement": DocumentType.STATEMENT,
    "warranty": DocumentType.WARRANTY,
    "contract": DocumentType.CONTRACT,
}

# Per-chat pending upload state: chat_id -> (doc_type, vendor_hint, timestamp)
_pending_uploads: dict[int, tuple[DocumentType | None, str | None, float]] = {}

# Per-chat pending location state: chat_id -> (fact_id, timestamp)
_pending_location: dict[int, tuple[str, float]] = {}

# Seconds before a pending upload state expires
_PENDING_TTL = 60.0

# Seconds to wait for additional photos in a media group
_MEDIA_GROUP_TIMEOUT = 2.0


@dataclass
class _MediaGroupBuffer:
    """Accumulates messages belonging to the same Telegram media group."""

    chat_id: int
    first_message: Message
    messages: list[Message] = field(default_factory=list)
    doc_type: DocumentType | None = None
    vendor_hint: str | None = None
    timer_task: asyncio.Task | None = field(default=None, repr=False)  # type: ignore[type-arg]


# Active media group buffers: media_group_id -> buffer
_media_groups: dict[str, _MediaGroupBuffer] = {}


def _expire_pending(chat_id: int) -> None:
    """Remove the pending state for a chat if it has expired."""
    entry = _pending_uploads.get(chat_id)
    if entry and time.time() - entry[2] > _PENDING_TTL:
        del _pending_uploads[chat_id]


def _pop_pending(chat_id: int) -> tuple[DocumentType | None, str | None] | None:
    """Return and remove the pending state for a chat, or None if absent/expired."""
    _expire_pending(chat_id)
    entry = _pending_uploads.pop(chat_id, None)
    if entry:
        return entry[0], entry[1]
    return None


def _peek_pending(chat_id: int) -> tuple[DocumentType | None, str | None] | None:
    """Return pending state without removing it, or None if absent/expired."""
    _expire_pending(chat_id)
    entry = _pending_uploads.get(chat_id)
    if entry:
        return entry[0], entry[1]
    return None


def _parse_type_and_hint(
    command: str, raw_args: str
) -> tuple[DocumentType | None, str | None]:
    """Resolve DocumentType and vendor hint from command name and argument string.

    For short type aliases (/receipt, /invoice, etc.) the entire args string
    is the vendor hint.

    For /upload the first token is the type name (optional) and the remainder
    is the vendor hint.

    Returns (doc_type, vendor_hint). Both may be None.
    """
    args = raw_args.strip()

    if command == "upload":
        parts = args.split(maxsplit=1)
        if not parts:
            return None, None
        type_name = parts[0].lower()
        doc_type = _COMMAND_TYPE_MAP.get(type_name)
        if doc_type is not None:
            vendor_hint = parts[1].strip() if len(parts) > 1 else None
        else:
            # First token was not a valid type name — no type, whole args is hint
            doc_type = None
            vendor_hint = args if args else None
        return doc_type, vendor_hint or None

    # Short command aliases
    doc_type = _COMMAND_TYPE_MAP.get(command)
    vendor_hint = args if args else None
    return doc_type, vendor_hint


def _format_result(result) -> str:  # type: ignore[no-untyped-def]
    """Format a ProcessingResult into a human-readable reply."""
    if not result.success:
        return "Processing failed. Try a clearer photo or different angle."

    if result.is_duplicate:
        lines = ["Document already processed (duplicate)."]
        if result.duplicate_of:
            lines.append(f"Duplicate of document: {result.duplicate_of}")
        return "\n".join(lines)

    data = result.extracted_data or {}
    vendor = data.get("vendor") or "Unknown"
    total = data.get("total")
    amount_str = str(total) if total is not None else "N/A"
    currency = data.get("currency") or ""
    date_val = data.get("date") or "N/A"
    fact_type = data.get("document_type") or (
        result.record_type.value if result.record_type else "N/A"
    )
    item_count = len(result.line_items) if result.line_items else 0
    fact_id = result.document_id or "N/A"

    lines = [
        "Document processed successfully!",
        "",
        f"Vendor: {vendor}",
        f"Amount: {amount_str} {currency}".rstrip(),
        f"Date: {date_val}",
        f"Type: {fact_type}",
        f"Items: {item_count}",
        f"Fact ID: `{fact_id}`",
        "",
        f"Use /fix `{fact_id}` to correct any field.",
        "Send a Google Maps URL for this location, or /skip.",
    ]
    return "\n".join(lines)


async def _download_photo(message: Message) -> tuple[bytes, str] | None:
    """Download the largest photo from a message. Returns (data, filename) or None."""
    if not message.photo:
        return None
    bot = message.bot
    if bot is None:
        return None
    largest = message.photo[-1]
    file = await bot.get_file(largest.file_id)
    if file.file_path is None:
        return None
    buf = io.BytesIO()
    await bot.download_file(file.file_path, buf)
    data = buf.getvalue()
    filename = f"telegram_{largest.file_id}.jpg"
    return data, filename


async def _download_document(message: Message) -> tuple[bytes, str] | None:
    """Download a document attachment from a message. Returns (data, filename) or None."""
    if not message.document:
        return None
    bot = message.bot
    if bot is None:
        return None
    doc = message.document
    file = await bot.get_file(doc.file_id)
    if file.file_path is None:
        return None
    buf = io.BytesIO()
    await bot.download_file(file.file_path, buf)
    data = buf.getvalue()
    filename = doc.file_name or f"telegram_{doc.file_id}"
    return data, filename


async def _get_attachment(
    message: Message,
) -> tuple[bytes, str] | None:
    """Return (data, filename) from the first available attachment in a message.

    Checks photo first, then document.
    """
    result = await _download_photo(message)
    if result:
        return result
    return await _download_document(message)


def _resolve_telegram_user(db: DatabaseManager, message: Message) -> str:
    """Resolve the alibi user_id for a Telegram message sender.

    Returns the linked user ID, or "system" if not linked.
    """
    from alibi.services.auth import find_user_by_telegram

    if message.from_user:
        tg_user_id = str(message.from_user.id)
        user = find_user_by_telegram(db, tg_user_id)
        if user:
            return str(user["id"])
    return "system"


async def _process_attachment(
    message: Message,
    data: bytes,
    filename: str,
    doc_type: DocumentType | None,
    vendor_hint: str | None,
) -> None:
    """Persist upload to inbox, then run process_file in a thread."""
    ctx = FolderContext(
        doc_type=doc_type,
        vendor_hint=vendor_hint,
        source="telegram",
    )

    db = await require_db(message)
    if db is None:
        return

    # Resolve user identity from Telegram account
    ctx.user_id = await asyncio.to_thread(_resolve_telegram_user, db, message)

    await message.chat.do(ChatAction.TYPING)
    await message.reply("Processing document...")

    try:
        saved_path = await asyncio.to_thread(persist_upload, data, filename, ctx)
        result = await asyncio.to_thread(
            process_file, db, saved_path, folder_context=ctx
        )
    except Exception as exc:
        logger.exception("Processing raised an exception for %s", filename)
        await message.reply(
            "Processing error. Please try again or use a different file format."
        )
        return

    reply = _format_result(result)
    await message.reply(reply, parse_mode="Markdown")

    # Set pending location state so user can send a map URL
    if result.success and result.document_id:
        _pending_location[message.chat.id] = (result.document_id, time.time())


async def _process_media_group(buf: _MediaGroupBuffer) -> None:
    """Download, persist, and process all pages in a media group as one document.

    Pages are stored in a timestamped subfolder under the type directory
    (e.g. ``receipts/telegram_1234567890/page_000.jpg``) via the shared
    ``persist_upload_group()`` service. This mirrors the CLI convention
    where a subfolder represents a multi-page document.
    """
    db = await require_db(buf.first_message)
    if db is None:
        return

    ctx = FolderContext(
        doc_type=buf.doc_type,
        vendor_hint=buf.vendor_hint,
        source="telegram",
    )
    ctx.user_id = await asyncio.to_thread(_resolve_telegram_user, db, buf.first_message)

    # Sort messages by message_id to preserve page order
    sorted_msgs = sorted(buf.messages, key=lambda m: m.message_id)

    page_count = len(sorted_msgs)
    await buf.first_message.chat.do(ChatAction.TYPING)
    await buf.first_message.reply(f"Processing {page_count}-page document...")

    # Download all pages
    pages: list[tuple[bytes, str]] = []
    try:
        for msg in sorted_msgs:
            attachment = await _get_attachment(msg)
            if attachment is None:
                logger.warning(
                    "Could not download page from message %d", msg.message_id
                )
                continue
            pages.append(attachment)
    except Exception as exc:
        logger.exception("Failed downloading media group pages")
        await buf.first_message.reply(
            "Could not download the file. Please try sending again."
        )
        return

    if not pages:
        await buf.first_message.reply("Could not download any pages.")
        return

    if len(pages) == 1:
        # Single page — persist as single file and process
        data, filename = pages[0]
        try:
            saved_path = await asyncio.to_thread(persist_upload, data, filename, ctx)
            result = await asyncio.to_thread(
                process_file, db, saved_path, folder_context=ctx
            )
        except Exception as exc:
            logger.exception("Processing failed for single-page media group")
            await buf.first_message.reply(
                "Processing error. Please try again or use a different file format."
            )
            return
    else:
        # Multi-page — persist to subfolder and process as group
        try:
            saved_paths = await asyncio.to_thread(persist_upload_group, pages, ctx)
            result = await asyncio.to_thread(
                process_document_group, db, saved_paths, folder_context=ctx
            )
        except Exception as exc:
            logger.exception("Processing failed for media group")
            await buf.first_message.reply(
                "Processing error. Please try again or use a different file format."
            )
            return

    reply = _format_result(result)
    await buf.first_message.reply(reply, parse_mode="Markdown")

    # Set pending location state
    if result.success and result.document_id:
        _pending_location[buf.chat_id] = (result.document_id, time.time())


async def _media_group_timer(media_group_id: str) -> None:
    """Wait for the buffer timeout, then process the collected media group."""
    try:
        await asyncio.sleep(_MEDIA_GROUP_TIMEOUT)
    except asyncio.CancelledError:
        return

    buf = _media_groups.pop(media_group_id, None)
    if buf is None:
        return

    try:
        await _process_media_group(buf)
    except Exception:
        logger.exception("Unexpected error processing media group %s", media_group_id)


@router.message(
    Command(
        "receipt",
        "invoice",
        "payment",
        "statement",
        "warranty",
        "contract",
        "upload",
    )
)
async def type_command_handler(message: Message) -> None:
    """Handle type commands and optional inline attachments.

    Parses the command name and any trailing arguments to determine the
    document type and vendor hint, then either processes an attached
    file immediately or stores pending state waiting for the next upload.
    """
    if not message.text:
        return

    # Extract the bare command name (strip leading slash and any @bot suffix)
    first_token = message.text.split()[0]
    command = first_token.lstrip("/").split("@")[0].lower()

    # Everything after the command token is raw args
    raw_args = message.text[len(first_token) :].strip()

    doc_type, vendor_hint = _parse_type_and_hint(command, raw_args)

    chat_id = message.chat.id

    # Check for inline attachment on this message
    attachment = await _get_attachment(message)
    if attachment:
        data, filename = attachment
        _pending_uploads.pop(chat_id, None)
        await _process_attachment(message, data, filename, doc_type, vendor_hint)
        return

    # Check for a replied-to message with an attachment
    if message.reply_to_message:
        attachment = await _get_attachment(message.reply_to_message)
        if attachment:
            data, filename = attachment
            _pending_uploads.pop(chat_id, None)
            await _process_attachment(message, data, filename, doc_type, vendor_hint)
            return

    # No attachment present — store pending state
    _pending_uploads[chat_id] = (doc_type, vendor_hint, time.time())

    type_label = (
        command if command != "upload" else (doc_type.value if doc_type else "auto")
    )
    hint_suffix = f" (vendor: {vendor_hint})" if vendor_hint else ""
    await message.reply(f"Send the document now. Type: {type_label}{hint_suffix}")


@router.message(F.photo | F.document)
async def attachment_handler(message: Message) -> None:
    """Handle bare photo or document messages.

    If the message belongs to a media group (album), it is buffered with
    other messages sharing the same ``media_group_id``. After a short
    timeout (no new messages), the group is processed as a multi-page
    document via ``process_document_group()``.

    Single photos without a media group ID are processed immediately.
    Uses pending state from a prior type command if available.
    """
    chat_id = message.chat.id
    media_group_id = message.media_group_id

    if media_group_id is not None:
        # --- Media group path: buffer and wait for more pages ---
        buf = _media_groups.get(media_group_id)

        if buf is None:
            # First message in this media group
            pending = _pop_pending(chat_id)
            doc_type, vendor_hint = pending if pending else (None, None)

            buf = _MediaGroupBuffer(
                chat_id=chat_id,
                first_message=message,
                messages=[message],
                doc_type=doc_type,
                vendor_hint=vendor_hint,
            )
            _media_groups[media_group_id] = buf
            logger.info("Media group %s: first page (chat %d)", media_group_id, chat_id)
        else:
            # Additional message in existing media group
            buf.messages.append(message)
            logger.info("Media group %s: page %d", media_group_id, len(buf.messages))

        # Cancel existing timer and start a new one
        if buf.timer_task is not None and not buf.timer_task.done():
            buf.timer_task.cancel()
        buf.timer_task = asyncio.create_task(_media_group_timer(media_group_id))
        return

    # --- Single attachment path (no media group) ---
    pending = _pop_pending(chat_id)

    if pending is not None:
        doc_type, vendor_hint = pending
    else:
        doc_type, vendor_hint = None, None

    attachment = await _get_attachment(message)
    if not attachment:
        await message.reply("Could not download the attachment.")
        return

    data, filename = attachment
    await _process_attachment(message, data, filename, doc_type, vendor_hint)


@router.message(Command("link"))
async def link_handler(message: Message) -> None:
    """Link a Telegram account to an Alibi user via mnemonic API key.

    Usage: /link word1 word2 word3 word4 word5 word6
    """
    if not message.text or not message.from_user:
        return

    parts = message.text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply("Usage: /link word1 word2 word3 word4 word5 word6")
        return

    mnemonic = parts[1].strip()
    tg_user_id = str(message.from_user.id)

    db = await require_db(message)
    if db is None:
        return

    from alibi.services.auth import add_contact, get_display_name, validate_api_key

    # Validate the key to find the user
    user = await asyncio.to_thread(validate_api_key, db, mnemonic)
    if not user:
        await message.reply("Invalid key. Please check and try again.")
        return

    # Link the Telegram account to the user
    try:
        result = await asyncio.to_thread(
            add_contact,
            db,
            user["id"],
            "telegram",
            tg_user_id,
        )
    except Exception:
        result = None
    if result:
        display = get_display_name(user)
        await message.reply(f"Linked! Welcome, {display}.")
    else:
        await message.reply("Could not link account. Already linked?")


@router.message(Command("whoami"))
async def whoami_handler(message: Message) -> None:
    """Show the linked Alibi user for this Telegram account."""
    if not message.from_user:
        await message.reply("Not linked.")
        return

    tg_user_id = str(message.from_user.id)
    db = await require_db(message)
    if db is None:
        return

    from alibi.services.auth import (
        find_user_by_telegram,
        get_display_name,
        list_contacts,
    )

    user = await asyncio.to_thread(find_user_by_telegram, db, tg_user_id)
    if user:
        display = get_display_name(user)
        contacts = await asyncio.to_thread(list_contacts, db, user["id"])
        lines = [f"You are: {display} (id: {user['id'][:8]}...)"]
        for c in contacts:
            label = f" ({c['label']})" if c.get("label") else ""
            lines.append(f"  {c['contact_type']}: {c['value']}{label}")
        await message.reply("\n".join(lines))
    else:
        await message.reply("Not linked. Use /link <mnemonic> to connect.")


@router.message(Command("unlink"))
async def unlink_handler(message: Message) -> None:
    """Unlink the current Telegram account from Alibi user."""
    if not message.from_user:
        return

    tg_user_id = str(message.from_user.id)
    db = await require_db(message)
    if db is None:
        return

    from alibi.services.auth import remove_contact_by_value

    ok = await asyncio.to_thread(
        remove_contact_by_value,
        db,
        "telegram",
        tg_user_id,
    )
    if ok:
        await message.reply("Unlinked. Your uploads will be attributed to 'system'.")
    else:
        await message.reply("This account is not linked.")


@router.message(Command("setname"))
async def setname_handler(message: Message) -> None:
    """Set your display name. Usage: /setname Alice"""
    if not message.text or not message.from_user:
        return

    parts = message.text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply("Usage: /setname <display name>")
        return

    new_name = parts[1].strip()
    tg_user_id = str(message.from_user.id)

    db = await require_db(message)
    if db is None:
        return

    from alibi.services.auth import find_user_by_telegram, update_user

    user = await asyncio.to_thread(find_user_by_telegram, db, tg_user_id)
    if not user:
        await message.reply("Not linked. Use /link <mnemonic> first.")
        return

    ok = await asyncio.to_thread(update_user, db, user["id"], name=new_name)
    if ok:
        await message.reply(f"Name set to: {new_name}")
    else:
        await message.reply("Could not update name.")


# ---------------------------------------------------------------------------
# Location handlers: /map, /skip, and automatic map URL detection
# ---------------------------------------------------------------------------


def _pop_pending_location(chat_id: int) -> str | None:
    """Return and remove pending location fact_id, or None if absent/expired."""
    entry = _pending_location.get(chat_id)
    if entry and time.time() - entry[1] > _PENDING_TTL:
        _pending_location.pop(chat_id, None)
        return None
    if entry:
        _pending_location.pop(chat_id, None)
        return entry[0]
    return None


async def _store_location(message: Message, fact_id: str, map_url: str) -> None:
    """Parse map URL and store location annotation on a fact."""
    db = await require_db(message)
    if db is None:
        return

    from alibi.services.correction import set_fact_location

    result = await asyncio.to_thread(set_fact_location, db, fact_id, map_url)
    if result:
        place = result.get("place_name") or "location"
        await message.reply(f"Location saved: {place}")
    else:
        await message.reply("Could not parse that URL. Send a Google Maps link.")


@router.message(Command("skip"))
async def skip_location_handler(message: Message) -> None:
    """Clear pending location state without storing a location."""
    chat_id = message.chat.id
    if _pending_location.pop(chat_id, None):
        await message.reply("Skipped location.")
    else:
        await message.reply("Nothing to skip.")


@router.message(Command("map"))
async def map_command_handler(message: Message) -> None:
    """Set location on a fact. Usage: /map [fact_id] <google_maps_url>

    If fact_id is omitted, uses the pending location from the last upload.
    """
    if not message.text:
        return

    parts = message.text.split(maxsplit=2)
    # /map <url> — use pending fact_id
    # /map <fact_id> <url>
    chat_id = message.chat.id

    if len(parts) < 2:
        await message.reply("Usage: /map [fact_id] <google_maps_url>")
        return

    from alibi.utils.map_url import is_map_url

    if len(parts) == 2:
        # /map <url_or_fact_id>
        arg = parts[1].strip()
        if is_map_url(arg):
            # It's a URL — use pending fact_id
            fact_id = _pop_pending_location(chat_id)
            if not fact_id:
                await message.reply("No recent upload. Use: /map <fact_id> <url>")
                return
            await _store_location(message, fact_id, arg)
        else:
            await message.reply("Usage: /map [fact_id] <google_maps_url>")
        return

    # /map <fact_id> <url>
    fact_id = parts[1].strip()
    url = parts[2].strip()
    _pending_location.pop(chat_id, None)
    await _store_location(message, fact_id, url)


@router.message(F.text)
async def map_url_auto_handler(message: Message) -> None:
    """Auto-detect Google Maps URLs in text messages.

    When a pending location state is active (after an upload), a plain
    text message containing a maps URL is treated as the location for
    the most recent upload.

    This handler has lower priority than commands (registered after them).
    """
    if not message.text:
        return

    chat_id = message.chat.id
    text = message.text.strip()

    # Only trigger on maps URLs when we have a pending location
    from alibi.utils.map_url import is_map_url

    if not is_map_url(text):
        return

    fact_id = _pop_pending_location(chat_id)
    if not fact_id:
        return  # No pending location — ignore

    await _store_location(message, fact_id, text)
