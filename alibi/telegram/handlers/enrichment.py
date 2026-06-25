"""Enrichment review handler — thin client over the host ``/enrichment`` endpoints."""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from alibi.telegram.api_client import AlibiAPIError
from alibi.telegram.handlers._common import api_key_for, client

router = Router()
logger = logging.getLogger(__name__)


class EnrichAction(CallbackData, prefix="enrich"):
    """Callback data for enrichment review actions."""

    action: str  # "confirm" or "reject"
    item_id: str  # fact_item_id (UUID)


def _esc(text: str) -> str:
    """Escape Markdown special characters (Markdown mode: _, *, `, [)."""
    if not text:
        return ""
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


def _fmt_conf(conf: float | None) -> str:
    """Format confidence as percentage string."""
    if conf is None:
        return "—"
    return f"{conf * 100:.0f}%"


def _api_key_for_callback(callback: CallbackQuery) -> str | None:
    """Resolve the acting user's API key from a callback query."""
    from alibi.telegram.keystore import get_keystore

    if callback.from_user:
        return get_keystore().get(callback.from_user.id)
    return None


@router.message(Command("enrich"))
async def enrich_handler(message: Message) -> None:
    """Handle /enrich command — show enrichment review queue with inline keyboards.

    Displays up to 5 pending enrichment items (those with confidence < 0.8),
    each with Confirm / Reject buttons.
    """
    api_key = api_key_for(message)
    try:
        queue = await client.enrichment_review(api_key=api_key, limit=5)
    except AlibiAPIError:
        logger.exception("Failed to load enrichment queue")
        await message.answer("Could not load the review queue. Please try again.")
        return

    if not queue:
        await message.answer("No items pending enrichment review.")
        return

    try:
        stats = await client.enrichment_stats(api_key=api_key)
        pending = stats.get("pending_review", len(queue))
    except AlibiAPIError:
        pending = len(queue)

    await message.answer(
        f"*Enrichment Review* ({pending} pending)\n\n", parse_mode="Markdown"
    )

    for item in queue:
        text = (
            f"*{_esc(item['name'])}*\n"
            f"Brand: {_esc(item.get('brand') or '—')}\n"
            f"Category: {_esc(item.get('category') or '—')}\n"
            f"Source: `{item['enrichment_source']}` "
            f"({_fmt_conf(item['enrichment_confidence'])})\n"
            f"Vendor: {_esc(item.get('vendor') or '—')}"
        )
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="✓ Confirm",
                        callback_data=EnrichAction(
                            action="confirm", item_id=item["id"]
                        ).pack(),
                    ),
                    InlineKeyboardButton(
                        text="✗ Reject",
                        callback_data=EnrichAction(
                            action="reject", item_id=item["id"]
                        ).pack(),
                    ),
                ]
            ]
        )
        await message.answer(text, reply_markup=keyboard, parse_mode="Markdown")


@router.callback_query(EnrichAction.filter(F.action == "confirm"))
async def handle_confirm(callback: CallbackQuery, callback_data: EnrichAction) -> None:
    """Handle confirm button press — set enrichment_source to user_confirmed."""
    try:
        ok = await client.confirm_enrichment(
            callback_data.item_id, api_key=_api_key_for_callback(callback)
        )
    except AlibiAPIError:
        await callback.answer("Service unavailable", show_alert=True)
        return

    msg = callback.message
    if ok and isinstance(msg, Message):
        await msg.edit_text(
            (msg.text or "") + "\n\n✓ *Confirmed*",
            reply_markup=None,
            parse_mode="Markdown",
        )
    elif not ok:
        await callback.answer("Item not found", show_alert=True)
    await callback.answer()


@router.callback_query(EnrichAction.filter(F.action == "reject"))
async def handle_reject(callback: CallbackQuery, callback_data: EnrichAction) -> None:
    """Handle reject button press — clear enrichment data from item."""
    try:
        ok = await client.reject_enrichment(
            callback_data.item_id, api_key=_api_key_for_callback(callback)
        )
    except AlibiAPIError:
        await callback.answer("Service unavailable", show_alert=True)
        return

    msg = callback.message
    if ok and isinstance(msg, Message):
        await msg.edit_text(
            (msg.text or "") + "\n\n✗ *Rejected*",
            reply_markup=None,
            parse_mode="Markdown",
        )
    elif not ok:
        await callback.answer("Item not found", show_alert=True)
    await callback.answer()
