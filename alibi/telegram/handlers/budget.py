"""Budget-related Telegram commands."""

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from alibi.budgets.service import BudgetService
from alibi.telegram.handlers import require_db

router = Router()


@router.message(Command("budget"))
async def budget_command(message: Message) -> None:
    """Show budget overview or compare scenarios.

    /budget - Show available budget scenarios
    /budget compare <base_id> <compare_id> - Compare two scenarios
    """
    db = await require_db(message)
    if db is None:
        return
    service = BudgetService(db)

    parts = (message.text or "").split()

    if len(parts) >= 4 and parts[1] == "compare":
        base_id = parts[2]
        compare_id = parts[3]
        try:
            comparisons = service.compare(base_id, compare_id)
            if not comparisons:
                await message.answer("No comparison data found.")
                return
            lines = ["*Budget Comparison*\n"]
            for item in comparisons[:10]:
                indicator = "\U0001f534" if item.variance < 0 else "\U0001f7e2"
                lines.append(
                    f"{indicator} {item.category}: "
                    f"{item.compare_amount:.2f} / {item.base_amount:.2f}"
                )
            await message.answer("\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            await message.answer(f"Error: {e}")
    else:
        # Show scenarios list (default space)
        try:
            scenarios = service.list_scenarios(space_id="default")
        except Exception:
            scenarios = []
        if not scenarios:
            await message.answer("No budget scenarios found. Create one via the API.")
            return
        lines = ["*Budget Scenarios*\n"]
        for s in scenarios[:10]:
            lines.append(f"\u2022 {s.name} ({s.data_type.value})")
        await message.answer("\n".join(lines), parse_mode="Markdown")
