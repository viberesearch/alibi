"""Warranty command handler."""

from datetime import date, timedelta

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from alibi.telegram.handlers import require_db

router = Router()


@router.message(Command("warranty"))
async def warranty_handler(message: Message) -> None:
    """Handle /warranty command - list items with warranty expiring soon.

    Shows items with warranty expiring in the next 90 days.
    """
    db = await require_db(message)
    if db is None:
        return

    # Get items with warranty expiring in next 90 days
    today = date.today()
    warning_date = today + timedelta(days=90)

    # Include items expired up to 30 days ago and expiring within 90 days
    expired_cutoff = today - timedelta(days=30)

    sql = """
        SELECT
            name,
            category,
            warranty_expires,
            warranty_type,
            purchase_price,
            currency
        FROM items
        WHERE warranty_expires IS NOT NULL
        AND warranty_expires <= ?
        AND warranty_expires >= ?
        AND status = 'active'
        ORDER BY warranty_expires ASC
    """

    rows = db.fetchall(sql, (warning_date.isoformat(), expired_cutoff.isoformat()))

    if not rows:
        await message.answer("No items with warranty expiring in the next 90 days.")
        return

    # Format response
    response = "*Items with Warranty Expiring Soon*\n\n"

    for row in rows:
        item_name = row[0]
        category = row[1] or "Uncategorized"
        expires = date.fromisoformat(row[2]) if row[2] else None
        warranty_type = row[3] or "Standard"
        price = float(row[4]) if row[4] else None
        currency = row[5] or "EUR"

        if expires:
            days_left = (expires - today).days
            if days_left < 0:
                status = f"Expired {-days_left}d ago"
            elif days_left <= 30:
                status = f"Expires in {days_left}d"
            else:
                status = f"{days_left}d remaining"

            response += f"*{item_name[:30]}*\n"
            response += f"  Category: {category}\n"
            response += f"  Expires: {expires} ({status})\n"
            response += f"  Type: {warranty_type}\n"
            if price:
                response += f"  Value: {price:.2f} {currency}\n"
            response += "\n"

    await message.answer(response, parse_mode="Markdown")
