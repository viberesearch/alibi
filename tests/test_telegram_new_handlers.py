"""Tests for new Telegram bot handlers (budget, lineitem, language)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alibi.budgets.models import BudgetComparison, BudgetScenario
from alibi.db.models import DataType


@pytest.fixture
def mock_message():
    """Create a mock Telegram Message object."""
    msg = AsyncMock()
    msg.answer = AsyncMock()
    return msg


@pytest.fixture
def mock_db():
    """Create a mock DatabaseManager."""
    db = MagicMock()
    db.is_initialized.return_value = True
    db.fetchall.return_value = []
    return db


# ---------------------------------------------------------------------------
# Budget handler tests
# ---------------------------------------------------------------------------


class TestBudgetHandler:
    """Tests for /budget command."""

    @pytest.mark.asyncio
    async def test_budget_list_shows_scenarios(self, mock_message, mock_db):
        """Test /budget shows scenario list."""
        scenarios = [
            BudgetScenario(
                id="s1",
                space_id="default",
                name="Monthly Target",
                data_type=DataType.TARGET,
            ),
            BudgetScenario(
                id="s2",
                space_id="default",
                name="Actual Spending",
                data_type=DataType.ACTUAL,
            ),
        ]

        mock_message.text = "/budget"

        with (
            patch("alibi.telegram.handlers.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.budget.BudgetService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.list_scenarios.return_value = scenarios

            from alibi.telegram.handlers.budget import budget_command

            await budget_command(mock_message)

        mock_message.answer.assert_called_once()
        response = mock_message.answer.call_args[0][0]
        assert "Budget Scenarios" in response
        assert "Monthly Target" in response
        assert "target" in response

    @pytest.mark.asyncio
    async def test_budget_compare(self, mock_message, mock_db):
        """Test /budget compare shows comparison."""
        comparisons = [
            BudgetComparison(
                category="groceries",
                period="2025-06",
                base_amount=Decimal("200.00"),
                compare_amount=Decimal("250.00"),
                variance=Decimal("50.00"),
            ),
        ]

        mock_message.text = "/budget compare base1 comp1"

        with (
            patch("alibi.telegram.handlers.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.budget.BudgetService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.compare.return_value = comparisons

            from alibi.telegram.handlers.budget import budget_command

            await budget_command(mock_message)

        mock_message.answer.assert_called_once()
        response = mock_message.answer.call_args[0][0]
        assert "Budget Comparison" in response
        assert "groceries" in response

    @pytest.mark.asyncio
    async def test_budget_empty(self, mock_message, mock_db):
        """Test /budget with no scenarios shows helpful message."""
        mock_message.text = "/budget"

        with (
            patch("alibi.telegram.handlers.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.budget.BudgetService") as mock_service_cls,
        ):
            mock_service = mock_service_cls.return_value
            mock_service.list_scenarios.return_value = []

            from alibi.telegram.handlers.budget import budget_command

            await budget_command(mock_message)

        mock_message.answer.assert_called_once()
        response = mock_message.answer.call_args[0][0]
        assert "No budget scenarios found" in response


# ---------------------------------------------------------------------------
# Line item handler tests
# ---------------------------------------------------------------------------


class TestLineitemHandler:
    """Tests for /lineitem command."""

    @pytest.mark.asyncio
    async def test_lineitem_by_category(self, mock_message, mock_db):
        """Test /lineitem groceries shows items in category."""
        rows = [
            {
                "name": "Milk",
                "category": "groceries",
                "total_price": "2.50",
                "currency": "EUR",
            }
        ]

        mock_message.text = "/lineitem groceries"

        with (
            patch("alibi.telegram.handlers.get_db", return_value=mock_db),
            patch(
                "alibi.telegram.handlers.lineitem.list_fact_items_with_fact",
                return_value=rows,
            ),
        ):
            from alibi.telegram.handlers.lineitem import lineitem_command

            await lineitem_command(mock_message)

        mock_message.answer.assert_called_once()
        response = mock_message.answer.call_args[0][0]
        assert "Milk" in response
        assert "2.50" in response

    @pytest.mark.asyncio
    async def test_lineitem_search(self, mock_message, mock_db):
        """Test /lineitem search milk finds matching items."""
        rows = [
            {
                "name": "Organic Milk",
                "category": "dairy",
                "total_price": "3.20",
                "currency": "EUR",
            }
        ]

        mock_message.text = "/lineitem search milk"

        with (
            patch("alibi.telegram.handlers.get_db", return_value=mock_db),
            patch(
                "alibi.telegram.handlers.lineitem.list_fact_items_with_fact",
                return_value=rows,
            ),
        ):
            from alibi.telegram.handlers.lineitem import lineitem_command

            await lineitem_command(mock_message)

        mock_message.answer.assert_called_once()
        response = mock_message.answer.call_args[0][0]
        assert "Organic Milk" in response

    @pytest.mark.asyncio
    async def test_lineitem_empty(self, mock_message, mock_db):
        """Test /lineitem with no results shows message."""
        mock_message.text = "/lineitem"

        with (
            patch("alibi.telegram.handlers.get_db", return_value=mock_db),
            patch(
                "alibi.telegram.handlers.lineitem.list_fact_items_with_fact",
                return_value=[],
            ),
        ):
            from alibi.telegram.handlers.lineitem import lineitem_command

            await lineitem_command(mock_message)

        mock_message.answer.assert_called_once()
        response = mock_message.answer.call_args[0][0]
        assert "No line items found" in response


# ---------------------------------------------------------------------------
# Language handler tests
# ---------------------------------------------------------------------------


class TestLanguageHandler:
    """Tests for /language command."""

    @pytest.mark.asyncio
    async def test_language_show(self, mock_message):
        """Test /language shows current language setting."""
        mock_message.text = "/language"

        with patch("alibi.telegram.handlers.language.get_config") as mock_cfg:
            mock_cfg.return_value.display_language = "original"

            from alibi.telegram.handlers.language import language_command

            await language_command(mock_message)

        mock_message.answer.assert_called_once()
        response = mock_message.answer.call_args[0][0]
        assert "original" in response
        assert "Supported" in response

    @pytest.mark.asyncio
    async def test_language_set_valid(self, mock_message):
        """Test /language en sets language."""
        mock_message.text = "/language en"

        from alibi.telegram.handlers.language import language_command

        await language_command(mock_message)

        mock_message.answer.assert_called_once()
        response = mock_message.answer.call_args[0][0]
        assert "preference set to" in response
        assert "display names" in response

    @pytest.mark.asyncio
    async def test_language_invalid(self, mock_message):
        """Test /language xx shows error for unsupported language."""
        mock_message.text = "/language xx"

        from alibi.telegram.handlers.language import language_command

        await language_command(mock_message)

        mock_message.answer.assert_called_once()
        response = mock_message.answer.call_args[0][0]
        assert "Unsupported language" in response
        assert "xx" in response
