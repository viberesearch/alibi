"""Correction command handlers — thin client over the host API.

``/fix`` edits a fact (vendor/amount/date), ``/barcode`` sets a barcode on a
fact item, and ``/merge`` merges two vendor identities. All route through the
host API; the bot holds no DB.
"""

from __future__ import annotations

import logging
import re
import time
from decimal import Decimal, InvalidOperation

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from alibi.telegram.api_client import AlibiAPIError
from alibi.telegram.handlers._common import api_key_for, client

router = Router()
logger = logging.getLogger(__name__)

_FACT_ID_RE = re.compile(r"Fact ID: `([^`]+)`")

# A fact_id looks like a UUID: 8-4-4-4-12 hex chars with hyphens, or any
# string >= 36 chars. We treat the first arg as a fact_id if it contains
# a hyphen (all UUIDs do) or is at least 36 characters long.
_UUID_MIN_LEN = 36

# Pending merge confirmations: chat_id -> (identity_id_a, identity_id_b, timestamp)
_pending_merges: dict[int, tuple[str, str, float]] = {}
_MERGE_CONFIRM_TTL = 60.0


def _looks_like_id(token: str) -> bool:
    """Return True if token looks like a UUID or similar identifier."""
    return "-" in token or len(token) >= _UUID_MIN_LEN


def _extract_fact_id_from_reply(message: Message) -> str | None:
    """Pull the fact_id out of the replied-to message's text, if present."""
    reply = message.reply_to_message
    if not reply or not reply.text:
        return None
    m = _FACT_ID_RE.search(reply.text)
    return m.group(1) if m else None


@router.message(Command("fix"))
async def fix_handler(message: Message) -> None:
    """Handle /fix command - correct a field on a fact.

    Usage:
        /fix <fact_id> vendor <new_vendor>
        /fix <fact_id> amount <amount>
        /fix <fact_id> date <YYYY-MM-DD>

    When replying to a bot message that contains "Fact ID: `xxx`", the
    fact_id may be omitted:
        /fix vendor Fresko
        /fix amount 12.50
        /fix date 2026-01-15
    """
    if not message.text:
        await message.answer("Usage: /fix [fact_id] <vendor|amount|date> <value>")
        return

    raw = message.text.split(maxsplit=1)
    if len(raw) < 2:
        await message.answer("Usage: /fix [fact_id] <vendor|amount|date> <value>")
        return

    parts = raw[1].split()

    fact_id: str | None
    if parts and _looks_like_id(parts[0]):
        fact_id = parts[0]
        field_parts = parts[1:]
    else:
        fact_id = _extract_fact_id_from_reply(message)
        field_parts = parts

    if not fact_id:
        await message.answer(
            "No fact ID provided and no replied-to message with a Fact ID."
        )
        return

    if len(field_parts) < 2:
        await message.answer("Usage: /fix [fact_id] <vendor|amount|date> <value>")
        return

    field = field_parts[0].lower()
    value = " ".join(field_parts[1:])
    api_key = api_key_for(message)

    if field not in ("vendor", "amount", "date"):
        await message.answer(
            f"Unknown field: {field!r}. Supported fields: vendor, amount, date."
        )
        return

    # Validate amount/date before any round-trip.
    if field == "amount":
        try:
            Decimal(value)
        except InvalidOperation:
            await message.answer(f"Invalid amount: {value!r}")
            return
    elif field == "date":
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", value):
            await message.answer(
                f"Invalid date format: {value!r}. Expected YYYY-MM-DD."
            )
            return

    # Fetch existing fact to show old values in the confirmation.
    try:
        existing = await client.get_fact(fact_id, api_key=api_key)
    except AlibiAPIError:
        existing = None

    try:
        if field == "vendor":
            old = (existing or {}).get("vendor", "Unknown")
            ok = await client.correct_vendor(fact_id, value, api_key=api_key)
            label = "Vendor"
        elif field == "amount":
            old = (existing or {}).get("total_amount", "N/A")
            ok = await client.update_fact(fact_id, {"amount": value}, api_key=api_key)
            label = "Amount"
        else:  # date
            old = (existing or {}).get("event_date", "N/A")
            ok = await client.update_fact(fact_id, {"date": value}, api_key=api_key)
            label = "Date"
    except AlibiAPIError:
        logger.exception("Failed to apply /fix to %s", fact_id)
        await message.answer("Could not apply the fix. Please try again.")
        return

    if ok:
        await message.answer(
            f"{label} updated: {old} -> {value}\nFact ID: `{fact_id}`",
            parse_mode="Markdown",
        )
    else:
        await message.answer(f"Fact not found: `{fact_id}`", parse_mode="Markdown")


_ITEM_ID_RE = re.compile(r"Item ID: `([^`]+)`")


def _extract_item_id_from_reply(message: Message) -> str | None:
    """Pull the item_id out of the replied-to message's text, if present."""
    reply = message.reply_to_message
    if not reply or not reply.text:
        return None
    m = _ITEM_ID_RE.search(reply.text)
    return m.group(1) if m else None


@router.message(Command("barcode"))
async def barcode_handler(message: Message) -> None:
    """Handle /barcode command - set barcode on a fact item.

    Usage:
        /barcode <item_id> <barcode_value>

    When replying to a bot message that contains "Item ID: `xxx`":
        /barcode <barcode_value>
    """
    if not message.text:
        await message.answer("Usage: /barcode [item_id] <barcode_value>")
        return

    raw = message.text.split(maxsplit=1)
    if len(raw) < 2:
        await message.answer("Usage: /barcode [item_id] <barcode_value>")
        return

    parts = raw[1].split()

    item_id: str | None
    if len(parts) >= 2 and _looks_like_id(parts[0]):
        item_id = parts[0]
        barcode_value = parts[1]
    else:
        item_id = _extract_item_id_from_reply(message)
        barcode_value = parts[0]

    if not item_id:
        await message.answer(
            "No item ID provided and no replied-to message with an Item ID."
        )
        return

    if not re.match(r"^\d{7,14}$", barcode_value):
        await message.answer(
            f"Invalid barcode format: {barcode_value!r}. "
            "Expected 7-14 digit numeric code (EAN-8/EAN-13/UPC-A)."
        )
        return

    try:
        ok = await client.update_line_item(
            item_id, {"barcode": barcode_value}, api_key=api_key_for(message)
        )
    except AlibiAPIError:
        logger.exception("Failed to set barcode on %s", item_id)
        await message.answer("Could not set barcode. Please try again.")
        return

    if ok:
        await message.answer(
            f"Barcode set.\nItem ID: `{item_id}`\nBarcode: {barcode_value}",
            parse_mode="Markdown",
        )
    else:
        await message.answer(f"Item not found: `{item_id}`", parse_mode="Markdown")


@router.message(Command("merge"))
async def merge_handler(message: Message) -> None:
    """Handle /merge command - merge two vendor identities with confirmation.

    Usage: /merge <identity_id_a> <identity_id_b>

    First call shows what will be merged and asks for confirmation.
    Reply 'yes' within 60 seconds to execute the merge.
    """
    if not message.text:
        await message.answer("Usage: /merge <identity_id_a> <identity_id_b>")
        return

    parts = message.text.split()
    if len(parts) != 3:
        await message.answer("Usage: /merge <identity_id_a> <identity_id_b>")
        return

    identity_id_a, identity_id_b = parts[1], parts[2]
    api_key = api_key_for(message)

    try:
        ident_a = await client.get_identity(identity_id_a, api_key=api_key)
        ident_b = await client.get_identity(identity_id_b, api_key=api_key)
    except AlibiAPIError:
        logger.exception("Failed to look up identities for /merge")
        await message.answer("Could not look up identities. Please try again.")
        return

    if not ident_a or not ident_b:
        missing = []
        if not ident_a:
            missing.append(f"A: `{identity_id_a}`")
        if not ident_b:
            missing.append(f"B: `{identity_id_b}`")
        await message.answer(
            "Identity not found:\n" + "\n".join(missing),
            parse_mode="Markdown",
        )
        return

    name_a = ident_a.get("name") or identity_id_a
    name_b = ident_b.get("name") or identity_id_b

    chat_id = message.chat.id
    _pending_merges[chat_id] = (identity_id_a, identity_id_b, time.time())

    await message.answer(
        f"Merge confirmation:\n"
        f"Keep: {name_a} (`{identity_id_a}`)\n"
        f"Remove: {name_b} (`{identity_id_b}`)\n\n"
        f"All members from '{name_b}' will be moved into '{name_a}'.\n"
        f"Reply 'yes' within 60s to confirm.",
        parse_mode="Markdown",
    )


@router.message(lambda m: m.text and m.text.strip().lower() == "yes")
async def merge_confirm_handler(message: Message) -> None:
    """Handle 'yes' reply to confirm a pending merge."""
    chat_id = message.chat.id
    pending = _pending_merges.pop(chat_id, None)

    if not pending:
        return  # No pending merge -- ignore silently

    identity_id_a, identity_id_b, ts = pending
    if time.time() - ts > _MERGE_CONFIRM_TTL:
        await message.answer("Merge confirmation expired. Please run /merge again.")
        return

    try:
        ok = await client.merge_identities(
            identity_id_a, identity_id_b, api_key=api_key_for(message)
        )
    except AlibiAPIError:
        logger.exception("Merge failed")
        await message.answer("Merge failed. Please try again.")
        return

    if ok:
        await message.answer(
            f"Identities merged.\n"
            f"Kept: `{identity_id_a}`\n"
            f"Removed: `{identity_id_b}`",
            parse_mode="Markdown",
        )
    else:
        await message.answer(
            f"Merge failed. One or both identities may have been modified.\n"
            f"A: `{identity_id_a}`\n"
            f"B: `{identity_id_b}`",
            parse_mode="Markdown",
        )
