"""Expenses command handler — thin client over the host ``/facts`` endpoint."""

import logging
import math
from datetime import date, timedelta

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from alibi.telegram.api_client import AlibiAPIError
from alibi.telegram.handlers._common import api_key_for, client

router = Router()
logger = logging.getLogger(__name__)

_EXPENSE_TYPES = {"purchase", "subscription_payment"}


@router.message(Command("expenses"))
async def expenses_handler(message: Message) -> None:
    """Handle /expenses command - show recent expenses.

    Usage: /expenses [days] [page N]
    Example: /expenses 30
    Example: /expenses 7 page 2
    """
    days = 7
    page = 1
    per_page = 10

    if message.text:
        parts = message.text.split()
        page_idx = None
        for i, part in enumerate(parts):
            if part.lower() == "page" and i + 1 < len(parts):
                try:
                    page = int(parts[i + 1])
                    if page < 1:
                        page = 1
                    page_idx = i
                except ValueError:
                    pass

        if len(parts) > 1 and (page_idx is None or page_idx != 1):
            try:
                days = int(parts[1])
                if days <= 0 or days > 365:
                    await message.answer("Days must be between 1 and 365")
                    return
            except ValueError:
                if page_idx is None:
                    await message.answer(
                        "Invalid parameter. Usage: /expenses [days] [page N]"
                    )
                    return

    since_date = date.today() - timedelta(days=days)

    try:
        body = await client.list_facts(
            api_key=api_key_for(message),
            date_from=since_date.isoformat(),
            per_page=200,
        )
    except AlibiAPIError:
        logger.exception("Failed to list facts for expenses")
        await message.answer("Could not load expenses. Please try again.")
        return

    all_facts = body.get("items", [])

    # The API filters on a single exact fact_type, so fetch broadly and narrow
    # to the expense-relevant types here.
    facts = [f for f in all_facts if f.get("fact_type") in _EXPENSE_TYPES]

    if not facts:
        await message.answer(f"No expenses found in the last {days} days.")
        return

    total = sum(float(f["total_amount"]) for f in facts if f.get("total_amount"))
    currency = facts[0].get("currency") or "EUR"
    total_pages = math.ceil(len(facts) / per_page)

    if page > total_pages:
        page = total_pages

    start = (page - 1) * per_page
    page_facts = facts[start : start + per_page]

    response = f"*Expenses - Last {days} days*\n\n"
    for fact in page_facts:
        vendor = (fact.get("vendor") or "Unknown")[:20]
        txn_date = fact.get("event_date") or ""
        amount = float(fact["total_amount"]) if fact.get("total_amount") else 0.0
        fact_type = fact.get("fact_type") or ""
        response += f"{txn_date} - {vendor}: {amount:.2f}\n"
        if fact_type == "subscription_payment":
            response += f"  _{fact_type}_\n"

    response += f"\n*Total: {total:.2f} {currency}*"
    response += f"\nPage {page}/{total_pages}."
    if page < total_pages:
        response += f" Use `/expenses {days} page {page + 1}` for next."

    await message.answer(response, parse_mode="Markdown")
