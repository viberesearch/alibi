"""Document upload handler with type commands and media group support.

Supports /receipt, /invoice, /payment, /statement, /warranty, /contract,
and /upload commands. Each command optionally carries a vendor hint and
can be followed by a photo or document attachment, or sent first to
prime an upload window.

When multiple photos are sent as a media group (album), they are
automatically buffered and processed as pages of a single document.

This handler is **thin**: it never touches the DB, Ollama or the pipeline.
Documents are forwarded to the host API (``POST /process`` and
``/process/group``) via :class:`AlibiAPIClient`, and per-user attribution is
carried by the sender's ``X-API-Key`` resolved from the local keystore (see
``docs/TELEGRAM_THIN_BOT_PLAN.md``).
"""

import asyncio
import io
import logging
import time
from dataclasses import dataclass, field

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import Message

from alibi.telegram.api_client import (
    AlibiAPIClient,
    AlibiAPIConnectionError,
    AlibiAPIError,
    ProcessResult,
)
from alibi.telegram.keystore import get_keystore
from alibi.telegram.spool import get_spool

logger = logging.getLogger(__name__)

router = Router()

# Shared async client (base URL from ALIBI_API_URL).
_client = AlibiAPIClient()

# Document type names accepted by the API ``type`` parameter. The bot's command
# names map 1:1 to these strings, so no DocumentType enum is needed here.
_DOC_TYPES = {"receipt", "invoice", "payment", "statement", "warranty", "contract"}

# Per-chat pending upload state: chat_id -> (doc_type, vendor_hint, timestamp)
_pending_uploads: dict[int, tuple[str | None, str | None, float]] = {}

# Per-chat pending location state: chat_id -> (fact_id, timestamp)
_pending_location: dict[int, tuple[str, float]] = {}

# Seconds before a pending upload state expires. Generous so a user has time to
# open Maps, find the place and paste the share link after an upload.
_PENDING_TTL = 600.0

# Seconds to wait for additional photos in a media group
_MEDIA_GROUP_TIMEOUT = 2.0


@dataclass
class _MediaGroupBuffer:
    """Accumulates messages belonging to the same Telegram media group."""

    chat_id: int
    first_message: Message
    messages: list[Message] = field(default_factory=list)
    doc_type: str | None = None
    vendor_hint: str | None = None
    timer_task: asyncio.Task | None = field(default=None, repr=False)  # type: ignore[type-arg]


# Active media group buffers: media_group_id -> buffer
_media_groups: dict[str, _MediaGroupBuffer] = {}


def _api_key_for(message: Message) -> str | None:
    """Return the API key linked to the message sender, or None (default user)."""
    if message.from_user:
        return get_keystore().get(message.from_user.id)
    return None


def _expire_pending(chat_id: int) -> None:
    """Remove the pending state for a chat if it has expired."""
    entry = _pending_uploads.get(chat_id)
    if entry and time.time() - entry[2] > _PENDING_TTL:
        del _pending_uploads[chat_id]


def _pop_pending(chat_id: int) -> tuple[str | None, str | None] | None:
    """Return and remove the pending state for a chat, or None if absent/expired."""
    _expire_pending(chat_id)
    entry = _pending_uploads.pop(chat_id, None)
    if entry:
        return entry[0], entry[1]
    return None


def _parse_type_and_hint(command: str, raw_args: str) -> tuple[str | None, str | None]:
    """Resolve document type name and vendor hint from command and args.

    For short type aliases (/receipt, /invoice, etc.) the entire args string
    is the vendor hint. For /upload the first token is the type name (optional)
    and the remainder is the vendor hint. Both returned values may be None.
    """
    args = raw_args.strip()

    if command == "upload":
        parts = args.split(maxsplit=1)
        if not parts:
            return None, None
        type_name = parts[0].lower()
        if type_name in _DOC_TYPES:
            vendor_hint = parts[1].strip() if len(parts) > 1 else None
            return type_name, vendor_hint or None
        # First token was not a valid type name — no type, whole args is hint
        return None, (args or None)

    # Short command aliases: command name is the type name
    doc_type = command if command in _DOC_TYPES else None
    vendor_hint = args if args else None
    return doc_type, vendor_hint


def _format_result(result: ProcessResult) -> str:
    """Format an API ProcessResult into a human-readable reply."""
    if not result.success:
        return "Processing failed. Try a clearer photo or different angle."

    if result.is_duplicate:
        lines = ["Document already processed (duplicate)."]
        if result.duplicate_of:
            lines.append(f"Duplicate of document: {result.duplicate_of}")
        return "\n".join(lines)

    vendor = result.vendor or "Unknown"
    amount_str = result.amount or "N/A"
    currency = result.currency or ""
    date_val = result.date or "N/A"
    fact_type = result.document_type or "N/A"
    item_count = result.items_count
    fact_id = result.fact_id or result.document_id or "N/A"

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


async def _reply_result(message: Message, result: ProcessResult) -> None:
    """Reply with the formatted result.

    Falls back to plain text if Telegram rejects the Markdown -- e.g. a vendor
    name containing an unbalanced ``*``/``_``/``[``/`` ` `` which legacy Markdown
    cannot escape. Without this fallback the reply raises ``TelegramBadRequest``
    and the user is left staring at the earlier "Processing document..." line
    even though the document was saved successfully.
    """
    text = _format_result(result)
    try:
        await message.reply(text, parse_mode="Markdown")
    except TelegramBadRequest:
        await message.reply(text, parse_mode=None)


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


async def _get_attachment(message: Message) -> tuple[bytes, str] | None:
    """Return (data, filename) from the first attachment (photo, then document)."""
    result = await _download_photo(message)
    if result:
        return result
    return await _download_document(message)


async def _process_attachment(
    message: Message,
    data: bytes,
    filename: str,
    doc_type: str | None,
    vendor_hint: str | None,
) -> None:
    """Forward a single document to the API and reply with the result."""
    api_key = _api_key_for(message)

    await message.chat.do(ChatAction.TYPING)
    await message.reply("Processing document...")

    try:
        result = await _client.process_document(
            data,
            filename,
            api_key=api_key,
            doc_type=doc_type,
            vendor_hint=vendor_hint,
        )
    except AlibiAPIConnectionError:
        # API unreachable (e.g. boot-order race with the host). Persist the
        # document so the drain loop can process it once the API is back.
        get_spool().add(
            [(data, filename)],
            kind="single",
            api_key=api_key,
            doc_type=doc_type,
            vendor_hint=vendor_hint,
            chat_id=message.chat.id,
            reply_to_message_id=message.message_id,
        )
        logger.warning("API unreachable; spooled %s for later", filename)
        await message.reply(
            "Saved — the service is starting up. I'll process this and reply "
            "as soon as it's back."
        )
        return
    except AlibiAPIError:
        logger.exception("API processing failed for %s", filename)
        await message.reply(
            "Processing error. The service may be busy — please try again."
        )
        return

    await _reply_result(message, result)

    # Set pending location state keyed on the real fact id.
    if result.fact_id:
        _pending_location[message.chat.id] = (result.fact_id, time.time())


async def _process_media_group(buf: _MediaGroupBuffer) -> None:
    """Download all pages in a media group and forward them as one document."""
    api_key = _api_key_for(buf.first_message)

    # Sort messages by message_id to preserve page order
    sorted_msgs = sorted(buf.messages, key=lambda m: m.message_id)

    page_count = len(sorted_msgs)
    await buf.first_message.chat.do(ChatAction.TYPING)
    await buf.first_message.reply(f"Processing {page_count}-page document...")

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
    except Exception:
        logger.exception("Failed downloading media group pages")
        await buf.first_message.reply(
            "Could not download the file. Please try sending again."
        )
        return

    if not pages:
        await buf.first_message.reply("Could not download any pages.")
        return

    single = len(pages) == 1
    try:
        if single:
            data, filename = pages[0]
            result = await _client.process_document(
                data,
                filename,
                api_key=api_key,
                doc_type=buf.doc_type,
                vendor_hint=buf.vendor_hint,
            )
        else:
            result = await _client.process_document_group(
                pages,
                api_key=api_key,
                doc_type=buf.doc_type,
                vendor_hint=buf.vendor_hint,
            )
    except AlibiAPIConnectionError:
        # API unreachable -- spool every page so the drain loop can replay it.
        get_spool().add(
            pages,
            kind="single" if single else "group",
            api_key=api_key,
            doc_type=buf.doc_type,
            vendor_hint=buf.vendor_hint,
            chat_id=buf.chat_id,
            reply_to_message_id=buf.first_message.message_id,
        )
        logger.warning("API unreachable; spooled %d-page document", len(pages))
        await buf.first_message.reply(
            "Saved — the service is starting up. I'll process this and reply "
            "as soon as it's back."
        )
        return
    except AlibiAPIError:
        logger.exception("API processing failed for media group")
        await buf.first_message.reply(
            "Processing error. The service may be busy — please try again."
        )
        return

    await _reply_result(buf.first_message, result)

    if result.fact_id:
        _pending_location[buf.chat_id] = (result.fact_id, time.time())


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
    """Handle type commands and optional inline attachments."""
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

    type_label = doc_type or "auto"
    hint_suffix = f" (vendor: {vendor_hint})" if vendor_hint else ""
    await message.reply(f"Send the document now. Type: {type_label}{hint_suffix}")


@router.message(F.photo | F.document)
async def attachment_handler(message: Message) -> None:
    """Handle bare photo or document messages, buffering media groups."""
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
    doc_type, vendor_hint = pending if pending is not None else (None, None)

    attachment = await _get_attachment(message)
    if not attachment:
        await message.reply("Could not download the attachment.")
        return

    data, filename = attachment
    await _process_attachment(message, data, filename, doc_type, vendor_hint)


# ---------------------------------------------------------------------------
# Account linking: /link, /whoami, /unlink, /setname
# ---------------------------------------------------------------------------


def _display_name(user: dict) -> str:  # type: ignore[type-arg]
    """Best-effort display name from a /users/me payload."""
    return user.get("name") or "there"


@router.message(Command("link"))
async def link_handler(message: Message) -> None:
    """Link a Telegram account to an Alibi user via mnemonic API key.

    Usage: /link word1 word2 word3 word4 word5 word6

    The mnemonic *is* the API key: it is validated against the host API and, if
    valid, stored in the bot keystore so subsequent requests carry it.
    """
    if not message.text or not message.from_user:
        return

    parts = message.text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply("Usage: /link word1 word2 word3 word4 word5 word6")
        return

    mnemonic = parts[1].strip()
    tg_user_id = message.from_user.id

    try:
        user = await _client.whoami(mnemonic)
    except AlibiAPIError:
        logger.exception("API unreachable during /link")
        await message.reply("Service unavailable — please try again shortly.")
        return

    if not user:
        await message.reply("Invalid key. Please check and try again.")
        return

    get_keystore().set(tg_user_id, mnemonic)
    await message.reply(f"Linked! Welcome, {_display_name(user)}.")


@router.message(Command("whoami"))
async def whoami_handler(message: Message) -> None:
    """Show the linked Alibi user for this Telegram account."""
    if not message.from_user:
        await message.reply("Not linked.")
        return

    api_key = get_keystore().get(message.from_user.id)
    if not api_key:
        await message.reply("Not linked. Use /link <mnemonic> to connect.")
        return

    try:
        user = await _client.whoami(api_key)
    except AlibiAPIError:
        await message.reply("Service unavailable — please try again shortly.")
        return

    if not user:
        get_keystore().remove(message.from_user.id)
        await message.reply("Your key is no longer valid. Use /link to reconnect.")
        return

    uid = str(user.get("id", ""))[:8]
    lines = [f"You are: {_display_name(user)} (id: {uid}...)"]
    for c in user.get("contacts", []):
        label = f" ({c['label']})" if c.get("label") else ""
        lines.append(f"  {c.get('contact_type')}: {c.get('value')}{label}")
    await message.reply("\n".join(lines))


@router.message(Command("unlink"))
async def unlink_handler(message: Message) -> None:
    """Unlink the current Telegram account from Alibi user."""
    if not message.from_user:
        return

    if get_keystore().remove(message.from_user.id):
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
    api_key = get_keystore().get(message.from_user.id)
    if not api_key:
        await message.reply("Not linked. Use /link <mnemonic> first.")
        return

    try:
        user = await _client.whoami(api_key)
        if not user:
            get_keystore().remove(message.from_user.id)
            await message.reply("Your key is no longer valid. Use /link to reconnect.")
            return
        await _client.update_user_name(str(user["id"]), new_name, api_key=api_key)
    except AlibiAPIError:
        logger.exception("API error during /setname")
        await message.reply("Could not update name. Please try again.")
        return

    await message.reply(f"Name set to: {new_name}")


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
    """Attach a location to a fact via the API."""
    api_key = _api_key_for(message)
    try:
        result = await _client.set_fact_location(fact_id, map_url, api_key=api_key)
    except AlibiAPIError as exc:
        if str(exc).startswith("400"):
            await message.reply("Could not parse that URL. Send a Google Maps link.")
        else:
            await message.reply("Could not save location. Please try again.")
        return
    place = result.get("place_name") or "location"
    await message.reply(f"Location saved: {place}")


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
    chat_id = message.chat.id

    if len(parts) < 2:
        await message.reply("Usage: /map [fact_id] <google_maps_url>")
        return

    from alibi.utils.map_url import is_map_url

    if len(parts) == 2:
        # /map <url_or_fact_id>
        arg = parts[1].strip()
        if is_map_url(arg):
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
    """Auto-detect Google Maps URLs in text messages after an upload."""
    if not message.text:
        return

    chat_id = message.chat.id
    text = message.text.strip()

    from alibi.utils.map_url import is_map_url

    if not is_map_url(text):
        return

    fact_id = _pop_pending_location(chat_id)
    if not fact_id:
        return  # No pending location — ignore

    await _store_location(message, fact_id, text)


# ---------------------------------------------------------------------------
# Offline spool drain
# ---------------------------------------------------------------------------
#
# Uploads that failed because the host API was unreachable (boot-order race;
# AlibiAPIConnectionError) are persisted by the spool. This background loop,
# started alongside polling in main.main, retries them once the API is back and
# sends the same reply the live path would have. It needs the bot handle to
# reach the originating chat, so main passes it in.

_SPOOL_DRAIN_INTERVAL = 30.0


async def _send_drained_reply(bot, entry, result: ProcessResult) -> None:  # type: ignore[no-untyped-def]
    """Send the formatted result for a drained entry and prime the map flow."""
    text = _format_result(result)
    try:
        await bot.send_message(
            chat_id=entry.chat_id,
            text=text,
            reply_to_message_id=entry.reply_to_message_id,
            parse_mode="Markdown",
        )
    except Exception:
        # Either the Markdown was rejected (e.g. a vendor name with an
        # unbalanced * or _) or the anchored message was deleted. Resend as
        # plain text without the reply anchor so the confirmation always lands.
        logger.warning("Result reply failed for chat %s; sending plain", entry.chat_id)
        await bot.send_message(chat_id=entry.chat_id, text=text)

    if result.fact_id:
        _pending_location[entry.chat_id] = (result.fact_id, time.time())


async def drain_spool_once(bot) -> int:  # type: ignore[no-untyped-def]
    """Retry every spooled upload once. Returns how many entries were cleared.

    A connection failure means the API is still down for everyone, so the pass
    stops early and leaves the rest for next time. An HTTP error (4xx/5xx) will
    never succeed on retry, so the entry is dropped and the chat is notified.
    """
    spool = get_spool()
    processed = 0
    for entry in spool.iter_pending():
        try:
            if entry.kind == "group" and len(entry.pages) > 1:
                result = await _client.process_document_group(
                    entry.pages,
                    api_key=entry.api_key,
                    doc_type=entry.doc_type,
                    vendor_hint=entry.vendor_hint,
                )
            else:
                data, filename = entry.pages[0]
                result = await _client.process_document(
                    data,
                    filename,
                    api_key=entry.api_key,
                    doc_type=entry.doc_type,
                    vendor_hint=entry.vendor_hint,
                )
        except AlibiAPIConnectionError:
            logger.info("Spool drain: API still unreachable, will retry later")
            break
        except AlibiAPIError:
            logger.exception("Spooled entry %s failed permanently; dropping", entry.id)
            spool.remove(entry.id)
            try:
                await bot.send_message(
                    chat_id=entry.chat_id,
                    text="A document you sent earlier could not be processed.",
                )
            except Exception:
                logger.exception("Could not notify chat %s of failure", entry.chat_id)
            processed += 1
            continue

        await _send_drained_reply(bot, entry, result)
        spool.remove(entry.id)
        processed += 1
        logger.info("Drained spooled upload %s", entry.id)
    return processed


async def run_spool_drain(bot, interval: float = _SPOOL_DRAIN_INTERVAL) -> None:  # type: ignore[no-untyped-def]
    """Background loop draining the spool every ``interval`` seconds.

    Resilient: an unexpected error in one pass is logged and the loop continues.
    """
    while True:
        try:
            await drain_spool_once(bot)
        except Exception:
            logger.exception("Unexpected error in spool drain loop")
        await asyncio.sleep(interval)
