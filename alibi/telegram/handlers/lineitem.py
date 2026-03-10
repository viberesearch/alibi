"""Line item query Telegram commands."""

import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from alibi.services.query import list_fact_items_with_fact
from alibi.telegram.handlers import require_db

router = Router()
logger = logging.getLogger(__name__)


@router.message(Command("lineitem"))
async def lineitem_command(message: Message) -> None:
    """Query line items.

    /lineitem - Show recent line items
    /lineitem <category> - Show recent items in category
    /lineitem search <term> - Search line items by name
    """
    db = await require_db(message)
    if db is None:
        return

    parts = (message.text or "").split(maxsplit=2)
    filters: dict[str, str] = {}

    if len(parts) >= 3 and parts[1] == "search":
        filters["name"] = parts[2]
    elif len(parts) >= 2:
        filters["category"] = parts[1]

    try:
        rows = list_fact_items_with_fact(db, filters if filters else None)
    except Exception:
        logger.exception("Failed to query line items")
        await message.answer("Could not retrieve line items. Please try again.")
        return

    # Limit to 10 most recent
    rows = rows[:10]

    if not rows:
        await message.answer("No line items found.")
        return

    lines = ["*Line Items*\n"]
    for row in rows:
        name = row.get("name") or "Unknown"
        cat = row.get("category") or ""
        price = f"{float(row['total_price']):.2f}" if row.get("total_price") else "?"
        currency = row.get("currency") or "EUR"
        lines.append(f"\u2022 {name} [{cat}] {price} {currency}")

    await message.answer("\n".join(lines), parse_mode="Markdown")
