"""Line item query handler — thin client over the host ``/line-items`` endpoint."""

import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from alibi.telegram.api_client import AlibiAPIError
from alibi.telegram.handlers._common import api_key_for, client

router = Router()
logger = logging.getLogger(__name__)


@router.message(Command("lineitem"))
async def lineitem_command(message: Message) -> None:
    """Query line items.

    /lineitem - Show recent line items
    /lineitem <category> - Show recent items in category
    /lineitem search <term> - Search line items by name
    """
    parts = (message.text or "").split(maxsplit=2)
    category: str | None = None
    name: str | None = None

    if len(parts) >= 3 and parts[1] == "search":
        name = parts[2]
    elif len(parts) >= 2:
        category = parts[1]

    try:
        body = await client.list_line_items(
            api_key=api_key_for(message),
            category=category,
            name=name,
            per_page=10,
        )
    except AlibiAPIError:
        logger.exception("Failed to query line items")
        await message.answer("Could not retrieve line items. Please try again.")
        return

    rows = body.get("fact_items", [])[:10]
    if not rows:
        await message.answer("No line items found.")
        return

    lines = ["*Line Items*\n"]
    for row in rows:
        name_val = row.get("name") or "Unknown"
        cat = row.get("category") or ""
        price = f"{float(row['total_price']):.2f}" if row.get("total_price") else "?"
        currency = row.get("currency") or "EUR"
        lines.append(f"• {name_val} [{cat}] {price} {currency}")

    await message.answer("\n".join(lines), parse_mode="Markdown")
