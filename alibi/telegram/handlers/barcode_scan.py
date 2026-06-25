"""Telegram barcode scanning handler — thin client over the host API.

Barcode detection (pyzbar) and product lookup live on the host. The bot
forwards the photo bytes to ``POST /items/barcode/scan`` and renders the
result; the inline buttons forward enrich / lookup actions to the host.
"""

from __future__ import annotations

import io
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

from alibi.telegram.api_client import AlibiAPIError
from alibi.telegram.handlers._common import api_key_for, client

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


def _api_key_for_callback(callback: CallbackQuery) -> str | None:
    """Resolve the acting user's API key from a callback query."""
    from alibi.telegram.keystore import get_keystore

    if callback.from_user:
        return get_keystore().get(callback.from_user.id)
    return None


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

    try:
        body = await client.scan_barcode(
            photo_data, "scan.jpg", api_key=api_key_for(message)
        )
    except AlibiAPIError as exc:
        if str(exc).startswith("503"):
            await status_msg.edit_text("Barcode scanning not available on the server.")
        else:
            logger.exception("Barcode scan failed")
            await status_msg.edit_text("Scan failed. Please try again.")
        return

    barcodes = body.get("barcodes", [])
    if not barcodes:
        await status_msg.edit_text("No barcodes detected in this image.")
        return

    try:
        await status_msg.delete()
    except Exception:
        pass

    for result in barcodes:
        data = result.get("data", "")
        text = f"*Barcode detected:* `{data}`\nType: {result.get('type', '?')}"

        if result.get("valid_ean"):
            product = result.get("product")
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

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Enrich matching items",
                        callback_data=ScanAction(action="enrich", barcode=data).pack(),
                    ),
                    InlineKeyboardButton(
                        text="Look up product",
                        callback_data=ScanAction(action="lookup", barcode=data).pack(),
                    ),
                ]
            ]
        )
        await message.answer(text, reply_markup=keyboard, parse_mode="Markdown")


@router.callback_query(ScanAction.filter(F.action == "enrich"))
async def handle_enrich(callback: CallbackQuery, callback_data: ScanAction) -> None:
    """Handle 'enrich matching items' button — run barcode enrichment on the host."""
    barcode = callback_data.barcode
    try:
        result = await client.enrich_by_barcode(
            barcode, api_key=_api_key_for_callback(callback)
        )
    except AlibiAPIError:
        logger.exception("enrich-by-barcode failed for %s", barcode)
        await callback.answer("Enrichment failed", show_alert=True)
        return

    matched = result.get("matched", 0)
    enriched = result.get("enriched", 0)
    if matched == 0:
        await callback.answer(
            f"No unenriched items found with barcode {barcode}", show_alert=True
        )
        return

    msg = callback.message
    if isinstance(msg, Message):
        await msg.edit_text(
            (msg.text or "") + f"\n\nEnriched {enriched}/{matched} items",
            reply_markup=None,
            parse_mode="Markdown",
        )
    await callback.answer()


@router.callback_query(ScanAction.filter(F.action == "lookup"))
async def handle_lookup(callback: CallbackQuery, callback_data: ScanAction) -> None:
    """Handle 'look up product' button — fetch from Open Food Facts via the host."""
    barcode = callback_data.barcode
    await callback.answer("Looking up product...")

    try:
        product = await client.barcode_lookup(
            barcode, api_key=_api_key_for_callback(callback)
        )
    except AlibiAPIError:
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
        await msg.edit_text(
            (msg.text or "") + "\n\n" + text,
            reply_markup=None,
            parse_mode="Markdown",
        )
    elif isinstance(msg, Message):
        await msg.edit_text(
            (msg.text or "") + "\n\nProduct not found in Open Food Facts.",
            reply_markup=None,
            parse_mode="Markdown",
        )
