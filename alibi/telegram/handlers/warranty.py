"""Warranty command handler — thin client over the host items endpoint."""

import logging
from datetime import date

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from alibi.telegram.api_client import AlibiAPIError
from alibi.telegram.handlers._common import api_key_for, client

router = Router()
logger = logging.getLogger(__name__)


@router.message(Command("warranty"))
async def warranty_handler(message: Message) -> None:
    """Handle /warranty command - list items with warranty expiring soon.

    Shows items with warranty expiring in the next 90 days (and any that
    lapsed in the last 30), soonest first.
    """
    try:
        rows = await client.list_warranty_expiring(
            api_key=api_key_for(message), ahead_days=90, expired_days=30
        )
    except AlibiAPIError:
        logger.exception("Failed to load warranties")
        await message.answer("Could not load warranties. Please try again.")
        return

    if not rows:
        await message.answer("No items with warranty expiring in the next 90 days.")
        return

    today = date.today()
    response = "*Items with Warranty Expiring Soon*\n\n"

    for row in rows:
        item_name = (row.get("name") or "Unknown")[:30]
        category = row.get("category") or "Uncategorized"
        warranty_type = row.get("warranty_type") or "Standard"
        currency = row.get("currency") or "EUR"
        price = float(row["purchase_price"]) if row.get("purchase_price") else None

        expires_raw = row.get("warranty_expires")
        if not expires_raw:
            continue
        expires = date.fromisoformat(expires_raw)
        days_left = (expires - today).days
        if days_left < 0:
            status = f"Expired {-days_left}d ago"
        elif days_left <= 30:
            status = f"Expires in {days_left}d"
        else:
            status = f"{days_left}d remaining"

        response += f"*{item_name}*\n"
        response += f"  Category: {category}\n"
        response += f"  Expires: {expires} ({status})\n"
        response += f"  Type: {warranty_type}\n"
        if price:
            response += f"  Value: {price:.2f} {currency}\n"
        response += "\n"

    await message.answer(response, parse_mode="Markdown")
