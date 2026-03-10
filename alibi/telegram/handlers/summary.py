"""Summary command handler."""

import logging
from datetime import date

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from alibi.services.analytics import (
    detect_subscriptions,
    spending_by_vendor,
    spending_summary,
)
from alibi.telegram.handlers import require_db

router = Router()
logger = logging.getLogger(__name__)


@router.message(Command("summary"))
async def summary_handler(message: Message) -> None:
    """Handle /summary command - show monthly spending summary.

    Shows the current month total, recent monthly trend, top vendors,
    and any detected subscription patterns.
    """
    db = await require_db(message)
    if db is None:
        return

    today = date.today()
    month_start = date(today.year, today.month, 1)

    # Monthly totals across all history (chronological, purchase facts only)
    from alibi.analytics.spending import MonthlySpend

    raw_monthly = spending_summary(db, period="month")
    monthly: list[MonthlySpend] = list(raw_monthly)  # type: ignore[arg-type]

    if not monthly:
        await message.answer(f"No transactions found for {today.strftime('%B %Y')}")
        return

    # Current month entry (last entry if it matches this month)
    current_month_key = f"{today.year}-{today.month:02d}"
    current_month = next((m for m in monthly if m.month == current_month_key), None)

    response = f"*Monthly Summary - {today.strftime('%B %Y')}*\n\n"

    if current_month:
        response += (
            f"*This month:* {float(current_month.total):.2f}"
            f" ({current_month.count} transactions,"
            f" avg {float(current_month.avg_amount):.2f})\n"
        )
    else:
        response += "_No purchases recorded this month yet_\n"

    # Show the last 3 months for context (excluding current to avoid duplication)
    prior_months = [m for m in monthly if m.month != current_month_key][-3:]
    if prior_months:
        response += "\n*Recent months:*\n"
        for m in prior_months:
            response += f"  {m.month}: {float(m.total):.2f} ({m.count} txns)\n"

    # Top vendors by spend (current month only)
    vendors = spending_by_vendor(db, filters={"date_from": month_start, "limit": 5})
    if vendors:
        response += "\n*Top vendors this month:*\n"
        for v in vendors:
            response += (
                f"  {v.vendor[:25]}: {float(v.total):.2f}"
                f" ({v.count} txns, {v.share_pct:.0f}%)\n"
            )

    # Detected subscriptions
    try:
        subscriptions = detect_subscriptions(db)
        if subscriptions:
            response += "\n*Subscriptions detected:*\n"
            for sub in subscriptions[:5]:
                response += (
                    f"  {sub.vendor[:25]}: {float(sub.avg_amount):.2f}"
                    f" {sub.period_type}"
                    f" (next: {sub.next_expected.isoformat()})\n"
                )
    except Exception:
        logger.warning("Subscription detection failed, skipping", exc_info=True)

    await message.answer(response, parse_mode="Markdown")
