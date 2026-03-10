"""Correction command handlers for inline fact and vendor corrections."""

from __future__ import annotations

import logging
import re
import time
from decimal import Decimal, InvalidOperation

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from alibi.db.connection import get_db
from alibi.services import correction, identity

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

    # Everything after "/fix"
    raw = message.text.split(maxsplit=1)
    if len(raw) < 2:
        await message.answer("Usage: /fix [fact_id] <vendor|amount|date> <value>")
        return

    parts = raw[1].split()

    # Determine fact_id and where the field/value start
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

    db = get_db()
    if not db.is_initialized():
        await message.answer("Database not initialized. Please run 'lt init' first.")
        return

    # Fetch existing fact to show old values in confirmation
    from alibi.db.v2_store import get_fact_by_id

    existing_fact = get_fact_by_id(db, fact_id)

    if field == "vendor":
        old_vendor = (existing_fact or {}).get("vendor", "Unknown")
        ok = correction.correct_vendor(db, fact_id, value)
        if ok:
            await message.answer(
                f"Vendor updated: {old_vendor} -> {value}\n" f"Fact ID: `{fact_id}`",
                parse_mode="Markdown",
            )
        else:
            await message.answer(f"Fact not found: `{fact_id}`", parse_mode="Markdown")

    elif field == "amount":
        try:
            amount = Decimal(value)
        except InvalidOperation:
            await message.answer(f"Invalid amount: {value!r}")
            return
        old_amount = (existing_fact or {}).get("total_amount", "N/A")
        ok = correction.update_fact(db, fact_id, {"amount": amount})
        if ok:
            await message.answer(
                f"Amount updated: {old_amount} -> {amount}\n" f"Fact ID: `{fact_id}`",
                parse_mode="Markdown",
            )
        else:
            await message.answer(f"Fact not found: `{fact_id}`", parse_mode="Markdown")

    elif field == "date":
        # Basic format validation before handing to the service layer.
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if not date_pattern.match(value):
            await message.answer(
                f"Invalid date format: {value!r}. Expected YYYY-MM-DD."
            )
            return
        old_date = (existing_fact or {}).get("event_date", "N/A")
        ok = correction.update_fact(db, fact_id, {"date": value})
        if ok:
            await message.answer(
                f"Date updated: {old_date} -> {value}\n" f"Fact ID: `{fact_id}`",
                parse_mode="Markdown",
            )
        else:
            await message.answer(f"Fact not found: `{fact_id}`", parse_mode="Markdown")

    else:
        await message.answer(
            f"Unknown field: {field!r}. Supported fields: vendor, amount, date."
        )


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

    # Determine item_id and barcode value
    item_id: str | None
    if len(parts) >= 2 and _looks_like_id(parts[0]):
        item_id = parts[0]
        barcode_value = parts[1]
    elif len(parts) == 1:
        item_id = _extract_item_id_from_reply(message)
        barcode_value = parts[0]
    else:
        # Two+ parts, first doesn't look like ID — try reply context
        item_id = _extract_item_id_from_reply(message)
        barcode_value = parts[0]

    if not item_id:
        await message.answer(
            "No item ID provided and no replied-to message with an Item ID."
        )
        return

    # Validate barcode format (EAN-8, EAN-13, UPC-A, or similar)
    if not re.match(r"^\d{7,14}$", barcode_value):
        await message.answer(
            f"Invalid barcode format: {barcode_value!r}. "
            "Expected 7-14 digit numeric code (EAN-8/EAN-13/UPC-A)."
        )
        return

    db = get_db()
    if not db.is_initialized():
        await message.answer("Database not initialized. Please run 'lt init' first.")
        return

    ok = correction.update_fact_item(db, item_id, {"barcode": barcode_value})
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

    identity_id_a = parts[1]
    identity_id_b = parts[2]

    db = get_db()
    if not db.is_initialized():
        await message.answer("Database not initialized. Please run 'lt init' first.")
        return

    # Look up both identities to show names
    ident_a = identity.get_identity(db, identity_id_a)
    ident_b = identity.get_identity(db, identity_id_b)

    if not ident_a or not ident_b:
        missing = []
        if not ident_a:
            missing.append(f"A: `{identity_id_a}`")
        if not ident_b:
            missing.append(f"B: `{identity_id_b}`")
        await message.answer(
            f"Identity not found:\n" + "\n".join(missing),
            parse_mode="Markdown",
        )
        return

    name_a = ident_a.get("name") or identity_id_a
    name_b = ident_b.get("name") or identity_id_b

    # Store pending merge
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

    db = get_db()
    if not db.is_initialized():
        await message.answer("Database not initialized.")
        return

    ok = identity.merge_vendors(db, identity_id_a, identity_id_b)
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
