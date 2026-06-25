"""Budget Telegram commands — thin client over the host ``/budgets`` endpoints."""

import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from alibi.telegram.api_client import AlibiAPIError
from alibi.telegram.handlers._common import api_key_for, client

router = Router()
logger = logging.getLogger(__name__)


def _f(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


@router.message(Command("budget"))
async def budget_command(message: Message) -> None:
    """Show budget overview or compare scenarios.

    /budget - Show available budget scenarios
    /budget compare <base_id> <compare_id> - Compare two scenarios
    """
    api_key = api_key_for(message)
    parts = (message.text or "").split()

    if len(parts) >= 4 and parts[1] == "compare":
        base_id, compare_id = parts[2], parts[3]
        try:
            comparisons = await client.compare_budgets(
                base_id, compare_id, api_key=api_key
            )
        except AlibiAPIError as exc:
            await message.answer(f"Error: {exc}")
            return
        if not comparisons:
            await message.answer("No comparison data found.")
            return
        lines = ["*Budget Comparison*\n"]
        for item in comparisons[:10]:
            variance = _f(item.get("variance"))
            indicator = "\U0001f534" if variance < 0 else "\U0001f7e2"
            lines.append(
                f"{indicator} {item.get('category')}: "
                f"{_f(item.get('compare_amount')):.2f} /"
                f" {_f(item.get('base_amount')):.2f}"
            )
        await message.answer("\n".join(lines), parse_mode="Markdown")
        return

    # Show scenarios list (default space)
    try:
        scenarios = await client.list_budget_scenarios(api_key=api_key)
    except AlibiAPIError:
        scenarios = []
    if not scenarios:
        await message.answer("No budget scenarios found. Create one via the API.")
        return
    lines = ["*Budget Scenarios*\n"]
    for s in scenarios[:10]:
        lines.append(f"• {s.get('name')} ({s.get('data_type')})")
    await message.answer("\n".join(lines), parse_mode="Markdown")
