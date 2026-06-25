"""Summary command handler — thin client over the host analytics endpoints."""

import logging
from datetime import date

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from alibi.telegram.api_client import AlibiAPIError
from alibi.telegram.handlers._common import api_key_for, client

router = Router()
logger = logging.getLogger(__name__)


def _f(value: object) -> float:
    """Coerce an API numeric (Decimal serialised as str) to float."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


@router.message(Command("summary"))
async def summary_handler(message: Message) -> None:
    """Handle /summary command - show monthly spending summary.

    Shows the current month total, recent monthly trend, top vendors,
    and any detected subscription patterns.
    """
    api_key = api_key_for(message)
    today = date.today()
    month_start = date(today.year, today.month, 1)

    try:
        monthly = await client.spending_summary(api_key=api_key, period="month")
    except AlibiAPIError:
        logger.exception("spending summary failed")
        await message.answer("Could not load summary. Please try again.")
        return

    if not monthly:
        await message.answer(f"No transactions found for {today.strftime('%B %Y')}")
        return

    current_month_key = f"{today.year}-{today.month:02d}"
    current_month = next(
        (m for m in monthly if m.get("month") == current_month_key), None
    )

    response = f"*Monthly Summary - {today.strftime('%B %Y')}*\n\n"

    if current_month:
        response += (
            f"*This month:* {_f(current_month.get('total')):.2f}"
            f" ({current_month.get('count', 0)} transactions,"
            f" avg {_f(current_month.get('avg_amount')):.2f})\n"
        )
    else:
        response += "_No purchases recorded this month yet_\n"

    prior_months = [m for m in monthly if m.get("month") != current_month_key][-3:]
    if prior_months:
        response += "\n*Recent months:*\n"
        for m in prior_months:
            response += (
                f"  {m.get('month')}: {_f(m.get('total')):.2f}"
                f" ({m.get('count', 0)} txns)\n"
            )

    # Top vendors by spend (current month only)
    try:
        vendors = await client.spending_summary(
            api_key=api_key,
            period="vendor",
            date_from=month_start.isoformat(),
            limit=5,
        )
    except AlibiAPIError:
        vendors = []
    if vendors:
        response += "\n*Top vendors this month:*\n"
        for v in vendors:
            response += (
                f"  {(v.get('vendor') or 'Unknown')[:25]}: {_f(v.get('total')):.2f}"
                f" ({v.get('count', 0)} txns, {_f(v.get('share_pct')):.0f}%)\n"
            )

    # Detected subscriptions
    try:
        subscriptions = await client.detect_subscriptions(api_key=api_key)
        if subscriptions:
            response += "\n*Subscriptions detected:*\n"
            for sub in subscriptions[:5]:
                response += (
                    f"  {(sub.get('vendor') or 'Unknown')[:25]}:"
                    f" {_f(sub.get('avg_amount')):.2f}"
                    f" {sub.get('period_type', '')}"
                    f" (next: {sub.get('next_expected', '?')})\n"
                )
    except AlibiAPIError:
        logger.warning("Subscription detection failed, skipping", exc_info=True)

    await message.answer(response, parse_mode="Markdown")
