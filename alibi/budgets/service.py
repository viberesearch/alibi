"""Budget service for CRUD operations and scenario management.

Provides the primary interface for creating, reading, and comparing
budget scenarios. Integrates with the transaction database to build
actual spending entries from tagged transactions.
"""

import uuid
from decimal import Decimal
from typing import Optional

from alibi.budgets.merge import compute_variance, merge_entries
from alibi.budgets.models import BudgetComparison, BudgetEntry, BudgetScenario
from alibi.db.connection import DatabaseManager
from alibi.db.models import DataType


class BudgetService:
    """Service layer for budget scenario management.

    Handles CRUD for scenarios and entries, plus higher-level operations
    like building actual spending from transactions, materializing merged
    views, and comparing scenarios.
    """

    def __init__(self, db: DatabaseManager) -> None:
        """Initialize the budget service.

        Args:
            db: DatabaseManager instance for database access.
        """
        self.db = db

    # -----------------------------------------------------------------------
    # Scenario CRUD
    # -----------------------------------------------------------------------

    def create_scenario(self, scenario: BudgetScenario) -> str:
        """Create a new budget scenario.

        Args:
            scenario: The BudgetScenario to persist.

        Returns:
            The scenario ID.
        """
        with self.db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO budgets (id, space_id, name, description, data_type,
                                     parent_id, period_start, period_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scenario.id,
                    scenario.space_id,
                    scenario.name,
                    scenario.description,
                    scenario.data_type.value,
                    scenario.parent_id,
                    (
                        scenario.period_start.isoformat()
                        if scenario.period_start
                        else None
                    ),
                    scenario.period_end.isoformat() if scenario.period_end else None,
                ),
            )
        return scenario.id

    def get_scenario(self, scenario_id: str) -> Optional[BudgetScenario]:
        """Get a budget scenario by ID.

        Args:
            scenario_id: The scenario ID to look up.

        Returns:
            The BudgetScenario if found, None otherwise.
        """
        row = self.db.fetchone(
            "SELECT * FROM budgets WHERE id = ?",
            (scenario_id,),
        )
        if row is None:
            return None
        return self._row_to_scenario(row)

    def list_scenarios(self, space_id: str) -> list[BudgetScenario]:
        """List all budget scenarios for a space.

        Args:
            space_id: The space to list scenarios for.

        Returns:
            List of BudgetScenario objects.
        """
        rows = self.db.fetchall(
            "SELECT * FROM budgets WHERE space_id = ? ORDER BY created_at",
            (space_id,),
        )
        return [self._row_to_scenario(row) for row in rows]

    def delete_scenario(self, scenario_id: str) -> bool:
        """Delete a budget scenario and its entries.

        Args:
            scenario_id: The scenario ID to delete.

        Returns:
            True if the scenario was deleted, False if not found.
        """
        with self.db.transaction() as cursor:
            # Delete entries first (foreign key)
            cursor.execute(
                "DELETE FROM budget_entries WHERE scenario_id = ?",
                (scenario_id,),
            )
            cursor.execute(
                "DELETE FROM budgets WHERE id = ?",
                (scenario_id,),
            )
            return cursor.rowcount > 0

    # -----------------------------------------------------------------------
    # Entry CRUD
    # -----------------------------------------------------------------------

    def add_entry(self, entry: BudgetEntry) -> str:
        """Add a budget entry to a scenario.

        Args:
            entry: The BudgetEntry to persist.

        Returns:
            The entry ID.
        """
        with self.db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO budget_entries (id, scenario_id, category, amount,
                                           currency, period, note)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.scenario_id,
                    entry.category,
                    str(entry.amount),
                    entry.currency,
                    entry.period,
                    entry.note,
                ),
            )
        return entry.id

    def get_entries(
        self,
        scenario_id: str,
        period: Optional[str] = None,
    ) -> list[BudgetEntry]:
        """Get budget entries for a scenario, optionally filtered by period.

        Args:
            scenario_id: The scenario to get entries for.
            period: Optional YYYY-MM period filter.

        Returns:
            List of BudgetEntry objects.
        """
        if period is not None:
            rows = self.db.fetchall(
                """
                SELECT * FROM budget_entries
                WHERE scenario_id = ? AND period = ?
                ORDER BY category
                """,
                (scenario_id, period),
            )
        else:
            rows = self.db.fetchall(
                """
                SELECT * FROM budget_entries
                WHERE scenario_id = ?
                ORDER BY period, category
                """,
                (scenario_id,),
            )
        return [self._row_to_entry(row) for row in rows]

    # -----------------------------------------------------------------------
    # Higher-level operations
    # -----------------------------------------------------------------------

    def get_actual_spending(
        self,
        space_id: str,
        period: str,
    ) -> list[BudgetEntry]:
        """Build actual spending from transaction data, grouped by category tag.

        Queries transactions for the given period (YYYY-MM), joins with tags
        that have type='category', and aggregates amounts per category.

        Args:
            space_id: The space to query transactions for.
            period: YYYY-MM period string.

        Returns:
            List of BudgetEntry objects representing actual spending.
        """
        # Extract year-month from event dates and group by fact_items category
        rows = self.db.fetchall(
            """
            SELECT COALESCE(fi.category, 'uncategorized') AS category,
                   SUM(fi.total_price) AS total_amount,
                   f.currency
            FROM fact_items fi
            JOIN facts f ON fi.fact_id = f.id
            WHERE strftime('%Y-%m', f.event_date) = ?
              AND f.fact_type IN ('purchase', 'subscription_payment')
            GROUP BY COALESCE(fi.category, 'uncategorized'), f.currency
            ORDER BY category
            """,
            (period,),
        )

        entries: list[BudgetEntry] = []
        for row in rows:
            entries.append(
                BudgetEntry(
                    id=str(uuid.uuid4()),
                    scenario_id="__actual__",
                    category=row["category"],
                    amount=Decimal(str(row["total_amount"])),
                    currency=row["currency"],
                    period=period,
                )
            )
        return entries

    def get_materialized(
        self,
        scenario_id: str,
        period: str,
    ) -> list[BudgetEntry]:
        """Merge actual spending with scenario overrides to produce materialized view.

        Fetches actual spending from transactions and merges with any
        overrides stored in the scenario's budget entries.

        Args:
            scenario_id: The scenario to materialize.
            period: YYYY-MM period string.

        Returns:
            Merged list of BudgetEntry objects.
        """
        scenario = self.get_scenario(scenario_id)
        if scenario is None:
            return []

        actual = self.get_actual_spending(scenario.space_id, period)
        overrides = self.get_entries(scenario_id, period)
        return merge_entries(actual, overrides)

    def compare(
        self,
        base_id: str,
        compare_id: str,
        period: Optional[str] = None,
    ) -> list[BudgetComparison]:
        """Compare two scenarios, returning per-category variance.

        Args:
            base_id: The base scenario ID.
            compare_id: The comparison scenario ID.
            period: Optional YYYY-MM period filter.

        Returns:
            List of BudgetComparison objects with variance data.
        """
        base_entries = self.get_entries(base_id, period)
        compare_entries = self.get_entries(compare_id, period)
        return compute_variance(base_entries, compare_entries)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _row_to_scenario(row: object) -> BudgetScenario:
        """Convert a database row to a BudgetScenario model.

        Args:
            row: A sqlite3.Row or similar mapping.

        Returns:
            BudgetScenario instance.
        """
        # sqlite3.Row supports dict-like access
        r = dict(row)  # type: ignore[call-overload]
        return BudgetScenario(
            id=r["id"],
            space_id=r["space_id"],
            name=r["name"],
            description=r.get("description"),
            data_type=DataType(r["data_type"]),
            parent_id=r.get("parent_id"),
            period_start=r.get("period_start"),
            period_end=r.get("period_end"),
        )

    @staticmethod
    def _row_to_entry(row: object) -> BudgetEntry:
        """Convert a database row to a BudgetEntry model.

        Args:
            row: A sqlite3.Row or similar mapping.

        Returns:
            BudgetEntry instance.
        """
        r = dict(row)  # type: ignore[call-overload]
        return BudgetEntry(
            id=r["id"],
            scenario_id=r["scenario_id"],
            category=r["category"],
            amount=Decimal(str(r["amount"])),
            currency=r["currency"],
            period=r["period"],
            note=r.get("note"),
        )
