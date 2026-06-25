"""Tests for the thin (API-client) Telegram query handlers.

The query handlers (find, summary, expenses, budget, lineitem, warranty) no
longer touch the DB: each is a thin HTTP client of the host API. These tests
verify that the right :class:`AlibiAPIClient` method is awaited and the reply
is formatted from the (mocked) API response. See ``test_telegram_thin_upload``
for the established mocking style and ``docs/TELEGRAM_THIN_BOT_PLAN.md``.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_message(text: str = "", user_id: int = 42, chat_id: int = 99) -> MagicMock:
    """Build a mock aiogram Message with async answer/chat helpers."""
    msg = MagicMock()
    msg.text = text
    msg.from_user.id = user_id
    msg.chat.id = chat_id
    msg.chat.do = AsyncMock()
    msg.answer = AsyncMock()
    return msg


def _mock_client(**methods) -> MagicMock:
    """Build a MagicMock client whose named async methods are AsyncMocks."""
    client = MagicMock()
    for name, return_value in methods.items():
        setattr(client, name, AsyncMock(return_value=return_value))
    return client


def _reply_text(msg: MagicMock) -> str:
    """Return the text passed to the last ``message.answer`` call."""
    return msg.answer.await_args.args[0]


# ---------------------------------------------------------------------------
# /find
# ---------------------------------------------------------------------------


class TestFindHandler:
    @pytest.mark.asyncio
    async def test_find_happy_path(self):
        from alibi.telegram.handlers import find

        body = {
            "query": "lidl",
            "total": 2,
            "results": [
                {
                    "id": "f1",
                    "result_type": "fact",
                    "title": "Lidl Berlin",
                    "subtype": "purchase",
                    "document_date": "2026-06-01",
                    "amount": 12.5,
                },
                {
                    "id": "i1",
                    "result_type": "item",
                    "title": "Milk",
                    "subtype": "dairy",
                    "document_date": "2026-06-01",
                    "amount": 1.2,
                },
            ],
        }
        client = _mock_client(search=body)
        msg = _make_message(text="/find lidl")

        with patch("alibi.telegram.handlers.find.client", new=client):
            await find.find_handler(msg)

        client.search.assert_awaited_once()
        assert client.search.await_args.args[0] == "lidl"
        text = _reply_text(msg)
        assert "Search Results for 'lidl'" in text
        assert "Lidl Berlin" in text
        assert "12.50" in text
        assert "Milk" in text

    @pytest.mark.asyncio
    async def test_find_empty(self):
        from alibi.telegram.handlers import find

        client = _mock_client(search={"query": "zzz", "total": 0, "results": []})
        msg = _make_message(text="/find zzz")

        with patch("alibi.telegram.handlers.find.client", new=client):
            await find.find_handler(msg)

        assert "No results found for 'zzz'" in _reply_text(msg)

    @pytest.mark.asyncio
    async def test_find_missing_query(self):
        from alibi.telegram.handlers import find

        client = _mock_client(search={"results": []})
        msg = _make_message(text="/find")

        with patch("alibi.telegram.handlers.find.client", new=client):
            await find.find_handler(msg)

        client.search.assert_not_awaited()
        assert "search query" in _reply_text(msg).lower()


# ---------------------------------------------------------------------------
# /summary
# ---------------------------------------------------------------------------


class TestSummaryHandler:
    @pytest.mark.asyncio
    async def test_summary_happy_path(self):
        from alibi.telegram.handlers import summary

        today = date.today()
        cur_key = f"{today.year}-{today.month:02d}"
        monthly = [
            {"month": cur_key, "total": "150.00", "count": 3, "avg_amount": "50.00"},
            {"month": "2026-01", "total": "90.00", "count": 2, "avg_amount": "45.00"},
        ]
        vendors = [
            {"vendor": "Lidl", "total": "99.00", "count": 2, "share_pct": "40.0"}
        ]
        client = MagicMock()
        client.spending_summary = AsyncMock(side_effect=[monthly, vendors])
        client.detect_subscriptions = AsyncMock(
            return_value=[
                {
                    "vendor": "Netflix",
                    "avg_amount": "12.99",
                    "period_type": "monthly",
                    "next_expected": "2026-07-01",
                }
            ]
        )
        msg = _make_message(text="/summary")

        with patch("alibi.telegram.handlers.summary.client", new=client):
            await summary.summary_handler(msg)

        assert client.spending_summary.await_count == 2
        client.detect_subscriptions.assert_awaited_once()
        text = _reply_text(msg)
        assert "Monthly Summary" in text
        assert "150.00" in text
        assert "Lidl" in text
        assert "Netflix" in text

    @pytest.mark.asyncio
    async def test_summary_empty(self):
        from alibi.telegram.handlers import summary

        client = MagicMock()
        client.spending_summary = AsyncMock(return_value=[])
        msg = _make_message(text="/summary")

        with patch("alibi.telegram.handlers.summary.client", new=client):
            await summary.summary_handler(msg)

        assert "No transactions found" in _reply_text(msg)

    @pytest.mark.asyncio
    async def test_summary_no_current_month(self):
        from alibi.telegram.handlers import summary

        monthly = [
            {"month": "2025-01", "total": "90.00", "count": 2, "avg_amount": "45.00"}
        ]
        client = MagicMock()
        client.spending_summary = AsyncMock(side_effect=[monthly, []])
        client.detect_subscriptions = AsyncMock(return_value=[])
        msg = _make_message(text="/summary")

        with patch("alibi.telegram.handlers.summary.client", new=client):
            await summary.summary_handler(msg)

        text = _reply_text(msg)
        assert "No purchases recorded this month yet" in text
        assert "Recent months" in text


# ---------------------------------------------------------------------------
# /expenses
# ---------------------------------------------------------------------------


class TestExpensesHandler:
    @pytest.mark.asyncio
    async def test_expenses_happy_path(self):
        from alibi.telegram.handlers import expenses

        body = {
            "items": [
                {
                    "vendor": "Lidl",
                    "event_date": "2026-06-09",
                    "total_amount": 12.5,
                    "fact_type": "purchase",
                    "currency": "EUR",
                },
                {
                    "vendor": "Netflix",
                    "event_date": "2026-06-08",
                    "total_amount": 12.99,
                    "fact_type": "subscription_payment",
                    "currency": "EUR",
                },
                {
                    "vendor": "BankFee",
                    "event_date": "2026-06-07",
                    "total_amount": 5.0,
                    "fact_type": "statement_line",
                    "currency": "EUR",
                },
            ],
            "total": 3,
        }
        client = _mock_client(list_facts=body)
        msg = _make_message(text="/expenses 30")

        with patch("alibi.telegram.handlers.expenses.client", new=client):
            await expenses.expenses_handler(msg)

        client.list_facts.assert_awaited_once()
        # 30-day window honoured in the date_from filter
        expected_from = (date.today() - timedelta(days=30)).isoformat()
        assert client.list_facts.await_args.kwargs["date_from"] == expected_from
        text = _reply_text(msg)
        assert "Last 30 days" in text
        assert "Lidl" in text
        assert "Netflix" in text
        # statement_line is filtered out (not an expense type)
        assert "BankFee" not in text
        # Total is only the two expense rows: 12.5 + 12.99 = 25.49
        assert "25.49" in text

    @pytest.mark.asyncio
    async def test_expenses_empty(self):
        from alibi.telegram.handlers import expenses

        body = {
            "items": [
                {
                    "vendor": "BankFee",
                    "event_date": "2026-06-07",
                    "total_amount": 5.0,
                    "fact_type": "statement_line",
                    "currency": "EUR",
                }
            ],
            "total": 1,
        }
        client = _mock_client(list_facts=body)
        msg = _make_message(text="/expenses")

        with patch("alibi.telegram.handlers.expenses.client", new=client):
            await expenses.expenses_handler(msg)

        # default window is 7 days when no arg given
        assert "No expenses found in the last 7 days." in _reply_text(msg)

    @pytest.mark.asyncio
    async def test_expenses_invalid_days(self):
        from alibi.telegram.handlers import expenses

        client = _mock_client(list_facts={"items": []})
        msg = _make_message(text="/expenses 999")

        with patch("alibi.telegram.handlers.expenses.client", new=client):
            await expenses.expenses_handler(msg)

        client.list_facts.assert_not_awaited()
        assert "between 1 and 365" in _reply_text(msg)


# ---------------------------------------------------------------------------
# /budget
# ---------------------------------------------------------------------------


class TestBudgetHandler:
    @pytest.mark.asyncio
    async def test_budget_list_scenarios(self):
        from alibi.telegram.handlers import budget

        scenarios = [
            {"name": "Monthly Target", "data_type": "target"},
            {"name": "Actual Spending", "data_type": "actual"},
        ]
        client = _mock_client(list_budget_scenarios=scenarios)
        msg = _make_message(text="/budget")

        with patch("alibi.telegram.handlers.budget.client", new=client):
            await budget.budget_command(msg)

        client.list_budget_scenarios.assert_awaited_once()
        text = _reply_text(msg)
        assert "Budget Scenarios" in text
        assert "Monthly Target" in text
        assert "target" in text

    @pytest.mark.asyncio
    async def test_budget_compare(self):
        from alibi.telegram.handlers import budget

        comparisons = [
            {
                "category": "groceries",
                "base_amount": "200.00",
                "compare_amount": "250.00",
                "variance": "50.00",
            }
        ]
        client = _mock_client(compare_budgets=comparisons)
        msg = _make_message(text="/budget compare base1 comp1")

        with patch("alibi.telegram.handlers.budget.client", new=client):
            await budget.budget_command(msg)

        client.compare_budgets.assert_awaited_once()
        assert client.compare_budgets.await_args.args[:2] == ("base1", "comp1")
        text = _reply_text(msg)
        assert "Budget Comparison" in text
        assert "groceries" in text
        assert "250.00" in text

    @pytest.mark.asyncio
    async def test_budget_empty(self):
        from alibi.telegram.handlers import budget

        client = _mock_client(list_budget_scenarios=[])
        msg = _make_message(text="/budget")

        with patch("alibi.telegram.handlers.budget.client", new=client):
            await budget.budget_command(msg)

        assert "No budget scenarios found" in _reply_text(msg)


# ---------------------------------------------------------------------------
# /lineitem
# ---------------------------------------------------------------------------


class TestLineitemHandler:
    @pytest.mark.asyncio
    async def test_lineitem_by_category(self):
        from alibi.telegram.handlers import lineitem

        body = {
            "fact_items": [
                {
                    "name": "Milk",
                    "category": "groceries",
                    "total_price": 2.5,
                    "currency": "EUR",
                }
            ],
            "total": 1,
        }
        client = _mock_client(list_line_items=body)
        msg = _make_message(text="/lineitem groceries")

        with patch("alibi.telegram.handlers.lineitem.client", new=client):
            await lineitem.lineitem_command(msg)

        client.list_line_items.assert_awaited_once()
        assert client.list_line_items.await_args.kwargs["category"] == "groceries"
        assert client.list_line_items.await_args.kwargs["name"] is None
        text = _reply_text(msg)
        assert "Milk" in text
        assert "2.50" in text

    @pytest.mark.asyncio
    async def test_lineitem_search(self):
        from alibi.telegram.handlers import lineitem

        body = {
            "fact_items": [
                {
                    "name": "Organic Milk",
                    "category": "dairy",
                    "total_price": 3.2,
                    "currency": "EUR",
                }
            ],
            "total": 1,
        }
        client = _mock_client(list_line_items=body)
        msg = _make_message(text="/lineitem search milk")

        with patch("alibi.telegram.handlers.lineitem.client", new=client):
            await lineitem.lineitem_command(msg)

        # "search <term>" routes to the name filter, not category
        assert client.list_line_items.await_args.kwargs["name"] == "milk"
        assert client.list_line_items.await_args.kwargs["category"] is None
        assert "Organic Milk" in _reply_text(msg)

    @pytest.mark.asyncio
    async def test_lineitem_empty(self):
        from alibi.telegram.handlers import lineitem

        client = _mock_client(list_line_items={"fact_items": [], "total": 0})
        msg = _make_message(text="/lineitem")

        with patch("alibi.telegram.handlers.lineitem.client", new=client):
            await lineitem.lineitem_command(msg)

        assert "No line items found" in _reply_text(msg)


# ---------------------------------------------------------------------------
# /warranty
# ---------------------------------------------------------------------------


class TestWarrantyHandler:
    @pytest.mark.asyncio
    async def test_warranty_happy_path(self):
        from alibi.telegram.handlers import warranty

        future = (date.today() + timedelta(days=60)).isoformat()
        rows = [
            {
                "name": "Laptop",
                "category": "electronics",
                "warranty_type": "Manufacturer",
                "currency": "EUR",
                "purchase_price": 999.0,
                "warranty_expires": future,
            }
        ]
        client = _mock_client(list_warranty_expiring=rows)
        msg = _make_message(text="/warranty")

        with patch("alibi.telegram.handlers.warranty.client", new=client):
            await warranty.warranty_handler(msg)

        client.list_warranty_expiring.assert_awaited_once()
        text = _reply_text(msg)
        assert "Warranty Expiring Soon" in text
        assert "Laptop" in text
        assert "60d remaining" in text
        assert "999.00" in text

    @pytest.mark.asyncio
    async def test_warranty_expired(self):
        from alibi.telegram.handlers import warranty

        past = (date.today() - timedelta(days=5)).isoformat()
        rows = [
            {
                "name": "Phone",
                "category": "electronics",
                "warranty_type": "Standard",
                "currency": "EUR",
                "purchase_price": 500.0,
                "warranty_expires": past,
            }
        ]
        client = _mock_client(list_warranty_expiring=rows)
        msg = _make_message(text="/warranty")

        with patch("alibi.telegram.handlers.warranty.client", new=client):
            await warranty.warranty_handler(msg)

        assert "Expired 5d ago" in _reply_text(msg)

    @pytest.mark.asyncio
    async def test_warranty_empty(self):
        from alibi.telegram.handlers import warranty

        client = _mock_client(list_warranty_expiring=[])
        msg = _make_message(text="/warranty")

        with patch("alibi.telegram.handlers.warranty.client", new=client):
            await warranty.warranty_handler(msg)

        assert "No items with warranty expiring" in _reply_text(msg)
