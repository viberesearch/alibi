"""Telegram barcode scanning handler — detect barcodes from photos."""

from __future__ import annotations

import asyncio
import io
import json
import logging

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from alibi.db.connection import DatabaseManager, get_db
from alibi.extraction.barcode_detector import (
    BarcodeResult,
    detect_barcodes,
    has_barcode_support,
)

router = Router()
logger = logging.getLogger(__name__)


class ScanAction(CallbackData, prefix="scan"):
    """Callback data for scan result actions."""

    action: str  # "enrich" or "lookup"
    barcode: str  # barcode value


def _esc(text: str) -> str:
    """Escape Markdown special characters."""
    if not text:
        return ""
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


async def _download_photo(message: Message) -> bytes | None:
    """Download the largest photo from a message."""
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
    return buf.getvalue()


@router.message(Command("scan"))
async def scan_handler(message: Message) -> None:
    """Handle /scan command — detect barcodes from attached or replied-to photo.

    Usage:
        - Send a photo with /scan as caption
        - Reply to a photo with /scan
    """
    if not has_barcode_support():
        await message.answer(
            "Barcode scanning not available — pyzbar library not installed."
        )
        return

    # Try to get photo from: 1) this message, 2) replied-to message
    photo_data: bytes | None = None

    if message.photo:
        photo_data = await _download_photo(message)
    elif message.reply_to_message and message.reply_to_message.photo:
        photo_data = await _download_photo(message.reply_to_message)

    if photo_data is None:
        await message.answer(
            "Send a photo with /scan as caption, or reply to a photo with /scan."
        )
        return

    await message.chat.do(ChatAction.TYPING)
    status_msg = await message.answer("Scanning for barcodes...")

    # Run detection in thread to avoid blocking
    results: list[BarcodeResult] = await asyncio.to_thread(detect_barcodes, photo_data)

    if not results:
        await status_msg.edit_text("No barcodes detected in this image.")
        return

    # Clean up status message before sending results
    try:
        await status_msg.delete()
    except Exception:
        pass

    # Look up each barcode
    db = get_db()
    for result in results:
        text = f"*Barcode detected:* `{result.data}`\nType: {result.type}"

        if result.valid_ean:
            # Look up in product cache
            product = _lookup_product(db, result.data)
            if product:
                text += (
                    f"\n\n*Product found:*\n"
                    f"Name: {_esc(product.get('product_name', '—'))}\n"
                    f"Brand: {_esc(product.get('brand', '—'))}\n"
                    f"Category: {_esc(product.get('category', '—'))}"
                )
            else:
                text += "\n\nProduct not in local cache."

            text += "\n\nEAN check digit: valid"
        else:
            text += "\n\nNot a valid EAN barcode."

        # Inline keyboard: Enrich items with this barcode
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Enrich matching items",
                        callback_data=ScanAction(
                            action="enrich", barcode=result.data
                        ).pack(),
                    ),
                    InlineKeyboardButton(
                        text="Look up product",
                        callback_data=ScanAction(
                            action="lookup", barcode=result.data
                        ).pack(),
                    ),
                ]
            ]
        )

        await message.answer(text, reply_markup=keyboard, parse_mode="Markdown")


def _lookup_product(db: DatabaseManager, barcode: str) -> dict[str, str] | None:
    """Look up barcode in product_cache table."""
    row = db.fetchone(
        "SELECT data FROM product_cache WHERE barcode = ?",
        (barcode,),
    )
    if not row:
        return None
    try:
        data: dict[str, object] = json.loads(row["data"])
        if data.get("_not_found"):
            return None
        tags = data.get("categories_tags")
        category = (
            ", ".join(str(t) for t in tags[:3])  # type: ignore[index]
            if isinstance(tags, list) and tags
            else ""
        )
        return {
            "product_name": str(data.get("product_name", "")),
            "brand": str(data.get("brands", "")),
            "category": category,
        }
    except (json.JSONDecodeError, TypeError):
        return None


@router.callback_query(ScanAction.filter(F.action == "enrich"))
async def handle_enrich(callback: CallbackQuery, callback_data: ScanAction) -> None:
    """Handle 'enrich matching items' button — run barcode enrichment."""
    db = get_db()
    barcode = callback_data.barcode

    from alibi.enrichment.service import enrich_item

    # Find all fact_items with this barcode that haven't been enriched
    rows = db.fetchall(
        "SELECT id FROM fact_items "
        "WHERE barcode = ? AND (brand IS NULL OR brand = '')",
        (barcode,),
    )

    if not rows:
        await callback.answer(
            f"No unenriched items found with barcode {barcode}",
            show_alert=True,
        )
        return

    enriched = 0
    for row in rows:
        result = await asyncio.to_thread(enrich_item, db, row["id"], barcode)
        if result.success:
            enriched += 1

    msg = callback.message
    if isinstance(msg, Message):
        original = msg.text or ""
        await msg.edit_text(
            original + f"\n\nEnriched {enriched}/{len(rows)} items",
            reply_markup=None,
            parse_mode="Markdown",
        )
    await callback.answer()


@router.callback_query(ScanAction.filter(F.action == "lookup"))
async def handle_lookup(callback: CallbackQuery, callback_data: ScanAction) -> None:
    """Handle 'look up product' button — fetch from Open Food Facts."""
    barcode = callback_data.barcode

    from alibi.enrichment.off_client import lookup_barcode

    await callback.answer("Looking up product...")

    try:
        product = await asyncio.to_thread(lookup_barcode, barcode)
    except Exception:
        logger.exception("OFF lookup failed for %s", barcode)
        await callback.answer("Lookup failed", show_alert=True)
        return

    msg = callback.message
    if product and isinstance(msg, Message):
        nutriscore = str(product.get("nutriscore_grade", "—")).upper()
        text = (
            f"*Open Food Facts:*\n"
            f"Name: {_esc(str(product.get('product_name', '—')))}\n"
            f"Brand: {_esc(str(product.get('brands', '—')))}\n"
            f"Nutriscore: {nutriscore}\n"
        )
        categories = product.get("categories_tags", [])
        if isinstance(categories, list) and categories:
            cleaned = [str(c).replace("en:", "") for c in categories[:5]]
            text += f"Categories: {', '.join(cleaned)}"

        original = msg.text or ""
        await msg.edit_text(
            original + "\n\n" + text,
            reply_markup=None,
            parse_mode="Markdown",
        )
    elif isinstance(msg, Message):
        original = msg.text or ""
        await msg.edit_text(
            original + "\n\nProduct not found in Open Food Facts.",
            reply_markup=None,
            parse_mode="Markdown",
        )
    await callback.answer()
