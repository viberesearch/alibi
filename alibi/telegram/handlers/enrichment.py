"""Enrichment review handler with inline keyboard."""

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

from alibi.db.connection import get_db
from alibi.services import enrichment_review

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


@router.message(Command("enrich"))
async def enrich_handler(message: Message) -> None:
    """Handle /enrich command — show enrichment review queue with inline keyboards.

    Displays up to 5 pending enrichment items (those with confidence < 0.8),
    each with Confirm / Reject buttons.
    """
    db = get_db()
    queue = enrichment_review.get_review_queue(db, limit=5)

    if not queue:
        await message.answer("No items pending enrichment review.")
        return

    stats = enrichment_review.get_review_stats(db)
    header = f"*Enrichment Review* ({stats['pending_review']} pending)\n\n"
    await message.answer(header, parse_mode="Markdown")

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
    db = get_db()
    ok = enrichment_review.confirm_enrichment(db, callback_data.item_id)
    msg = callback.message
    if ok and isinstance(msg, Message):
        original = msg.text or ""
        await msg.edit_text(
            original + "\n\n✓ *Confirmed*",
            reply_markup=None,
            parse_mode="Markdown",
        )
    elif not ok:
        await callback.answer("Item not found", show_alert=True)
    await callback.answer()


@router.callback_query(EnrichAction.filter(F.action == "reject"))
async def handle_reject(callback: CallbackQuery, callback_data: EnrichAction) -> None:
    """Handle reject button press — clear enrichment data from item."""
    db = get_db()
    ok = enrichment_review.reject_enrichment(db, callback_data.item_id)
    msg = callback.message
    if ok and isinstance(msg, Message):
        original = msg.text or ""
        await msg.edit_text(
            original + "\n\n✗ *Rejected*",
            reply_markup=None,
            parse_mode="Markdown",
        )
    elif not ok:
        await callback.answer("Item not found", show_alert=True)
    await callback.answer()
