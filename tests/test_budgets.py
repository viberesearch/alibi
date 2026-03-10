"""Tests for budget scenarios with temporal modeling.

Covers:
- Model creation and validation
- Merge logic (actual + overrides)
- Variance computation
- Service CRUD with temp database
- Materialized view tests
- Compare scenario tests
"""

import uuid
from collections.abc import Generator
from decimal import Decimal
from pathlib import Path

import pytest

from alibi.budgets.merge import compute_variance, merge_entries
from alibi.budgets.models import BudgetComparison, BudgetEntry, BudgetScenario
from alibi.budgets.service import BudgetService
from alibi.config import Config
from alibi.db.connection import DatabaseManager
from alibi.db.models import DataType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary database path."""
    path = tmp_path / "test_budgets.db"
    yield path


@pytest.fixture
def db_manager(temp_db_path: Path) -> Generator[DatabaseManager, None, None]:
    """Create a database manager with temporary database."""
    config = Config(db_path=temp_db_path)
    manager = DatabaseManager(config)
    manager.initialize()
    yield manager
    manager.close()


@pytest.fixture
def seeded_db(db_manager: DatabaseManager) -> DatabaseManager:
    """Database with v2 seed data for testing (facts, fact_items with categories)."""
    from uuid import uuid4

    conn = db_manager.get_connection()

    # Create a user and space (needed for budget scenarios)
    conn.execute(
        "INSERT INTO users (id, name) VALUES (?, ?)",
        ("user-1", "Test User"),
    )
    conn.execute(
        "INSERT INTO spaces (id, name, type, owner_id) VALUES (?, ?, ?, ?)",
        ("space-1", "Personal", "private", "user-1"),
    )

    # Create a document (needed for atom FK)
    doc_id = str(uuid4())
    conn.execute(
        "INSERT INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
        (doc_id, "/test/receipt.jpg", "abc123"),
    )

    # Create facts (replacing transactions) for 2025-01 and 2025-02
    facts_data = [
        ("fact-1", "purchase", "Supermarket", "150.00", "2025-01-05", "groceries"),
        ("fact-2", "purchase", "Restaurant", "45.00", "2025-01-10", "dining"),
        ("fact-3", "purchase", "Metro", "30.00", "2025-01-15", "transport"),
        ("fact-4", "purchase", "Supermarket", "120.00", "2025-02-03", "groceries"),
        ("fact-5", "purchase", "Restaurant", "60.00", "2025-02-14", "dining"),
    ]

    for fact_id, fact_type, vendor, amount, event_date, category in facts_data:
        cloud_id = str(uuid4())
        conn.execute(
            "INSERT INTO clouds (id, status) VALUES (?, ?)",
            (cloud_id, "collapsed"),
        )
        conn.execute(
            """INSERT INTO facts (id, cloud_id, fact_type, vendor,
               total_amount, currency, event_date, status)
               VALUES (?, ?, ?, ?, ?, 'EUR', ?, 'confirmed')""",
            (fact_id, cloud_id, fact_type, vendor, amount, event_date),
        )
        # Create fact_item with category (replacing transaction_tags+tags)
        atom_id = str(uuid4())
        fi_id = str(uuid4())
        conn.execute(
            "INSERT INTO atoms (id, document_id, atom_type, data) VALUES (?, ?, ?, ?)",
            (atom_id, doc_id, "item", "{}"),
        )
        conn.execute(
            """INSERT INTO fact_items (id, fact_id, atom_id, name,
               quantity, total_price, category)
               VALUES (?, ?, ?, ?, 1, ?, ?)""",
            (fi_id, fact_id, atom_id, vendor, amount, category),
        )

    conn.commit()
    return db_manager


@pytest.fixture
def service(seeded_db: DatabaseManager) -> BudgetService:
    """BudgetService with seeded database."""
    return BudgetService(seeded_db)


@pytest.fixture
def sample_scenario() -> BudgetScenario:
    """Create a sample budget scenario."""
    return BudgetScenario(
        id="scenario-1",
        space_id="space-1",
        name="Q1 2025 Budget",
        description="First quarter budget plan",
        data_type=DataType.TARGET,
    )


def _make_entry(
    category: str,
    amount: str,
    period: str = "2025-01",
    scenario_id: str = "scenario-1",
    entry_id: str | None = None,
) -> BudgetEntry:
    """Helper to create a BudgetEntry."""
    return BudgetEntry(
        id=entry_id or str(uuid.uuid4()),
        scenario_id=scenario_id,
        category=category,
        amount=Decimal(amount),
        period=period,
    )


# ===========================================================================
# Model creation and validation tests
# ===========================================================================


class TestBudgetModels:
    """Tests for Pydantic model creation and validation."""

    def test_scenario_creation_minimal(self):
        """Test creating a scenario with minimal fields."""
        scenario = BudgetScenario(
            id="s1",
            space_id="sp1",
            name="Test",
            data_type=DataType.ACTUAL,
        )
        assert scenario.id == "s1"
        assert scenario.description is None
        assert scenario.parent_id is None
        assert scenario.period_start is None
        assert scenario.period_end is None

    def test_scenario_creation_full(self):
        """Test creating a scenario with all fields."""
        from datetime import date

        scenario = BudgetScenario(
            id="s2",
            space_id="sp1",
            name="Full Scenario",
            description="A complete scenario",
            data_type=DataType.PROJECTED,
            parent_id="s1",
            period_start=date(2025, 1, 1),
            period_end=date(2025, 3, 31),
        )
        assert scenario.description == "A complete scenario"
        assert scenario.parent_id == "s1"
        assert scenario.period_start == date(2025, 1, 1)
        assert scenario.period_end == date(2025, 3, 31)

    def test_scenario_data_type_values(self):
        """Test that DataType enum values work correctly."""
        for dt in [DataType.ACTUAL, DataType.PROJECTED, DataType.TARGET]:
            scenario = BudgetScenario(id="s", space_id="sp", name="T", data_type=dt)
            assert scenario.data_type == dt

    def test_entry_creation(self):
        """Test creating a budget entry."""
        entry = BudgetEntry(
            id="e1",
            scenario_id="s1",
            category="groceries",
            amount=Decimal("250.00"),
            period="2025-01",
        )
        assert entry.category == "groceries"
        assert entry.amount == Decimal("250.00")
        assert entry.currency == "EUR"
        assert entry.note is None

    def test_entry_with_note(self):
        """Test creating a budget entry with a note."""
        entry = BudgetEntry(
            id="e2",
            scenario_id="s1",
            category="dining",
            amount=Decimal("100.50"),
            period="2025-02",
            note="Reduced dining budget",
        )
        assert entry.note == "Reduced dining budget"

    def test_entry_custom_currency(self):
        """Test creating a budget entry with custom currency."""
        entry = BudgetEntry(
            id="e3",
            scenario_id="s1",
            category="transport",
            amount=Decimal("50.00"),
            currency="USD",
            period="2025-01",
        )
        assert entry.currency == "USD"

    def test_comparison_creation(self):
        """Test creating a budget comparison."""
        comp = BudgetComparison(
            category="groceries",
            period="2025-01",
            base_amount=Decimal("200.00"),
            compare_amount=Decimal("250.00"),
            variance=Decimal("50.00"),
            variance_pct=Decimal("25.00"),
        )
        assert comp.variance == Decimal("50.00")
        assert comp.variance_pct == Decimal("25.00")

    def test_comparison_no_pct(self):
        """Test creating a comparison without percentage."""
        comp = BudgetComparison(
            category="dining",
            period="2025-01",
            base_amount=Decimal("0"),
            compare_amount=Decimal("100.00"),
            variance=Decimal("100.00"),
        )
        assert comp.variance_pct is None


# ===========================================================================
# Merge logic tests
# ===========================================================================


class TestMergeEntries:
    """Tests for the merge_entries function."""

    def test_merge_no_overrides(self):
        """Merging with empty overrides returns actuals unchanged."""
        actuals = [
            _make_entry("groceries", "200.00"),
            _make_entry("dining", "100.00"),
        ]
        result = merge_entries(actuals, [])
        assert len(result) == 2

    def test_merge_no_actuals(self):
        """Merging with empty actuals returns overrides."""
        overrides = [
            _make_entry("groceries", "250.00"),
        ]
        result = merge_entries([], overrides)
        assert len(result) == 1
        assert result[0].amount == Decimal("250.00")

    def test_merge_override_replaces_actual(self):
        """Override for same category replaces actual."""
        actuals = [_make_entry("groceries", "200.00")]
        overrides = [_make_entry("groceries", "250.00")]
        result = merge_entries(actuals, overrides)
        assert len(result) == 1
        assert result[0].amount == Decimal("250.00")

    def test_merge_preserves_non_overridden(self):
        """Categories not overridden pass through from actuals."""
        actuals = [
            _make_entry("groceries", "200.00"),
            _make_entry("dining", "100.00"),
        ]
        overrides = [_make_entry("groceries", "250.00")]
        result = merge_entries(actuals, overrides)
        assert len(result) == 2
        # dining should still be 100
        dining = [e for e in result if e.category == "dining"]
        assert len(dining) == 1
        assert dining[0].amount == Decimal("100.00")

    def test_merge_adds_new_categories(self):
        """Override for new category adds it to the result."""
        actuals = [_make_entry("groceries", "200.00")]
        overrides = [_make_entry("entertainment", "75.00")]
        result = merge_entries(actuals, overrides)
        assert len(result) == 2
        categories = {e.category for e in result}
        assert categories == {"groceries", "entertainment"}

    def test_merge_respects_period(self):
        """Entries with same category but different periods are separate."""
        actuals = [_make_entry("groceries", "200.00", period="2025-01")]
        overrides = [_make_entry("groceries", "250.00", period="2025-02")]
        result = merge_entries(actuals, overrides)
        assert len(result) == 2

    def test_merge_sorted_output(self):
        """Merged result is sorted by (period, category)."""
        actuals = [
            _make_entry("transport", "50.00", period="2025-02"),
            _make_entry("groceries", "200.00", period="2025-01"),
        ]
        overrides = [
            _make_entry("dining", "100.00", period="2025-01"),
        ]
        result = merge_entries(actuals, overrides)
        keys = [(e.period, e.category) for e in result]
        assert keys == sorted(keys)

    def test_merge_both_empty(self):
        """Merging two empty lists returns empty."""
        result = merge_entries([], [])
        assert result == []


# ===========================================================================
# Variance computation tests
# ===========================================================================


class TestComputeVariance:
    """Tests for the compute_variance function."""

    def test_variance_positive(self):
        """Compare amount > base amount yields positive variance."""
        base = [_make_entry("groceries", "200.00")]
        compare = [_make_entry("groceries", "250.00")]
        result = compute_variance(base, compare)
        assert len(result) == 1
        assert result[0].variance == Decimal("50.00")
        assert result[0].variance_pct == Decimal("25.00")

    def test_variance_negative(self):
        """Compare amount < base amount yields negative variance."""
        base = [_make_entry("groceries", "200.00")]
        compare = [_make_entry("groceries", "150.00")]
        result = compute_variance(base, compare)
        assert result[0].variance == Decimal("-50.00")
        assert result[0].variance_pct == Decimal("-25.00")

    def test_variance_zero(self):
        """Same amounts yield zero variance."""
        base = [_make_entry("groceries", "200.00")]
        compare = [_make_entry("groceries", "200.00")]
        result = compute_variance(base, compare)
        assert result[0].variance == Decimal("0")
        assert result[0].variance_pct == Decimal("0.00")

    def test_variance_base_zero(self):
        """Zero base amount yields None for percentage."""
        base = [_make_entry("groceries", "0")]
        compare = [_make_entry("groceries", "100.00")]
        result = compute_variance(base, compare)
        assert result[0].variance == Decimal("100.00")
        assert result[0].variance_pct is None

    def test_variance_missing_in_compare(self):
        """Category only in base gets zero in compare."""
        base = [_make_entry("groceries", "200.00")]
        compare: list[BudgetEntry] = []
        result = compute_variance(base, compare)
        assert len(result) == 1
        assert result[0].compare_amount == Decimal("0")
        assert result[0].variance == Decimal("-200.00")

    def test_variance_missing_in_base(self):
        """Category only in compare gets zero in base."""
        base: list[BudgetEntry] = []
        compare = [_make_entry("dining", "100.00")]
        result = compute_variance(base, compare)
        assert len(result) == 1
        assert result[0].base_amount == Decimal("0")
        assert result[0].variance == Decimal("100.00")

    def test_variance_multiple_categories(self):
        """Variance computed correctly for multiple categories."""
        base = [
            _make_entry("groceries", "200.00"),
            _make_entry("dining", "100.00"),
        ]
        compare = [
            _make_entry("groceries", "180.00"),
            _make_entry("dining", "120.00"),
        ]
        result = compute_variance(base, compare)
        assert len(result) == 2
        # sorted by (period, category)
        assert result[0].category == "dining"
        assert result[0].variance == Decimal("20.00")
        assert result[1].category == "groceries"
        assert result[1].variance == Decimal("-20.00")

    def test_variance_multiple_periods(self):
        """Variance computed per-period."""
        base = [
            _make_entry("groceries", "200.00", period="2025-01"),
            _make_entry("groceries", "210.00", period="2025-02"),
        ]
        compare = [
            _make_entry("groceries", "250.00", period="2025-01"),
            _make_entry("groceries", "200.00", period="2025-02"),
        ]
        result = compute_variance(base, compare)
        assert len(result) == 2
        jan = [r for r in result if r.period == "2025-01"][0]
        feb = [r for r in result if r.period == "2025-02"][0]
        assert jan.variance == Decimal("50.00")
        assert feb.variance == Decimal("-10.00")

    def test_variance_both_empty(self):
        """Variance of two empty sets is empty."""
        result = compute_variance([], [])
        assert result == []


# ===========================================================================
# Service CRUD tests
# ===========================================================================


class TestBudgetServiceCrud:
    """Tests for BudgetService CRUD operations."""

    def test_create_and_get_scenario(self, service, sample_scenario):
        """Create a scenario and retrieve it."""
        service.create_scenario(sample_scenario)
        retrieved = service.get_scenario("scenario-1")
        assert retrieved is not None
        assert retrieved.name == "Q1 2025 Budget"
        assert retrieved.data_type == DataType.TARGET

    def test_get_nonexistent_scenario(self, service):
        """Getting a nonexistent scenario returns None."""
        assert service.get_scenario("nonexistent") is None

    def test_list_scenarios_empty(self, service):
        """Listing scenarios for space with none returns empty list."""
        result = service.list_scenarios("space-1")
        assert result == []

    def test_list_scenarios(self, service, sample_scenario):
        """List scenarios returns all scenarios for a space."""
        service.create_scenario(sample_scenario)

        scenario2 = BudgetScenario(
            id="scenario-2",
            space_id="space-1",
            name="Conservative Q1",
            data_type=DataType.TARGET,
        )
        service.create_scenario(scenario2)

        result = service.list_scenarios("space-1")
        assert len(result) == 2

    def test_list_scenarios_space_isolation(self, service, sample_scenario):
        """Scenarios are isolated per space."""
        service.create_scenario(sample_scenario)
        result = service.list_scenarios("other-space")
        assert result == []

    def test_delete_scenario(self, service, sample_scenario):
        """Delete a scenario and verify it's gone."""
        service.create_scenario(sample_scenario)
        assert service.delete_scenario("scenario-1") is True
        assert service.get_scenario("scenario-1") is None

    def test_delete_nonexistent_scenario(self, service):
        """Deleting a nonexistent scenario returns False."""
        assert service.delete_scenario("nonexistent") is False

    def test_delete_scenario_cascades_entries(self, service, sample_scenario):
        """Deleting a scenario also deletes its entries."""
        service.create_scenario(sample_scenario)
        entry = _make_entry("groceries", "200.00", entry_id="e1")
        service.add_entry(entry)

        service.delete_scenario("scenario-1")
        entries = service.get_entries("scenario-1")
        assert entries == []

    def test_add_and_get_entries(self, service, sample_scenario):
        """Add entries and retrieve them."""
        service.create_scenario(sample_scenario)

        e1 = _make_entry("groceries", "200.00", entry_id="e1")
        e2 = _make_entry("dining", "100.00", entry_id="e2")
        service.add_entry(e1)
        service.add_entry(e2)

        entries = service.get_entries("scenario-1")
        assert len(entries) == 2

    def test_get_entries_by_period(self, service, sample_scenario):
        """Get entries filtered by period."""
        service.create_scenario(sample_scenario)

        e1 = _make_entry("groceries", "200.00", period="2025-01", entry_id="e1")
        e2 = _make_entry("groceries", "210.00", period="2025-02", entry_id="e2")
        service.add_entry(e1)
        service.add_entry(e2)

        jan = service.get_entries("scenario-1", period="2025-01")
        assert len(jan) == 1
        assert jan[0].period == "2025-01"

    def test_get_entries_empty(self, service, sample_scenario):
        """Get entries for scenario with none returns empty list."""
        service.create_scenario(sample_scenario)
        entries = service.get_entries("scenario-1")
        assert entries == []


# ===========================================================================
# Actual spending from transactions
# ===========================================================================


class TestActualSpending:
    """Tests for building actual spending from transaction data."""

    def test_actual_spending_jan(self, service):
        """Get actual spending for January 2025."""
        entries = service.get_actual_spending("space-1", "2025-01")
        assert len(entries) == 3
        categories = {e.category for e in entries}
        assert categories == {"groceries", "dining", "transport"}

    def test_actual_spending_feb(self, service):
        """Get actual spending for February 2025."""
        entries = service.get_actual_spending("space-1", "2025-02")
        assert len(entries) == 2
        categories = {e.category for e in entries}
        assert categories == {"groceries", "dining"}

    def test_actual_spending_amounts_jan(self, service):
        """Verify actual spending amounts for January."""
        entries = service.get_actual_spending("space-1", "2025-01")
        amounts = {e.category: e.amount for e in entries}
        assert amounts["groceries"] == Decimal("150.00")
        assert amounts["dining"] == Decimal("45.00")
        assert amounts["transport"] == Decimal("30.00")

    def test_actual_spending_empty_period(self, service):
        """No transactions in period returns empty."""
        entries = service.get_actual_spending("space-1", "2025-06")
        assert entries == []

    def test_actual_spending_space_ignored(self, service):
        """V2 facts don't have space_id, so space param is ignored."""
        entries = service.get_actual_spending("other-space", "2025-01")
        # Same results regardless of space_id since v2 facts are global
        assert len(entries) == 3


# ===========================================================================
# Materialized view tests
# ===========================================================================


class TestMaterializedView:
    """Tests for materialized views (actual + overrides)."""

    def test_materialized_no_overrides(self, service, sample_scenario):
        """Materialized view with no overrides equals actual spending."""
        service.create_scenario(sample_scenario)
        result = service.get_materialized("scenario-1", "2025-01")
        # Should have 3 categories from actual spending
        assert len(result) == 3

    def test_materialized_with_override(self, service, sample_scenario):
        """Materialized view with override replaces matching category."""
        service.create_scenario(sample_scenario)
        override = _make_entry(
            "groceries", "300.00", period="2025-01", entry_id="override-1"
        )
        service.add_entry(override)

        result = service.get_materialized("scenario-1", "2025-01")
        groceries = [e for e in result if e.category == "groceries"]
        assert len(groceries) == 1
        assert groceries[0].amount == Decimal("300.00")

    def test_materialized_nonexistent_scenario(self, service):
        """Materializing a nonexistent scenario returns empty."""
        result = service.get_materialized("nonexistent", "2025-01")
        assert result == []


# ===========================================================================
# Compare scenario tests
# ===========================================================================


class TestCompareScenarios:
    """Tests for comparing two budget scenarios."""

    def test_compare_two_scenarios(self, service):
        """Compare two scenarios with different amounts."""
        s1 = BudgetScenario(
            id="base-scenario",
            space_id="space-1",
            name="Base",
            data_type=DataType.TARGET,
        )
        s2 = BudgetScenario(
            id="aggressive-scenario",
            space_id="space-1",
            name="Aggressive",
            data_type=DataType.TARGET,
        )
        service.create_scenario(s1)
        service.create_scenario(s2)

        # Base entries
        service.add_entry(
            _make_entry(
                "groceries", "200.00", scenario_id="base-scenario", entry_id="b1"
            )
        )
        service.add_entry(
            _make_entry("dining", "100.00", scenario_id="base-scenario", entry_id="b2")
        )

        # Aggressive entries (reduced)
        service.add_entry(
            _make_entry(
                "groceries", "150.00", scenario_id="aggressive-scenario", entry_id="a1"
            )
        )
        service.add_entry(
            _make_entry(
                "dining", "50.00", scenario_id="aggressive-scenario", entry_id="a2"
            )
        )

        comparisons = service.compare("base-scenario", "aggressive-scenario")
        assert len(comparisons) == 2

        # All variances should be negative (aggressive < base)
        for comp in comparisons:
            assert comp.variance < 0

    def test_compare_with_period_filter(self, service):
        """Compare scenarios with period filter."""
        s1 = BudgetScenario(
            id="sc-a", space_id="space-1", name="A", data_type=DataType.TARGET
        )
        s2 = BudgetScenario(
            id="sc-b", space_id="space-1", name="B", data_type=DataType.TARGET
        )
        service.create_scenario(s1)
        service.create_scenario(s2)

        # Entries for two periods
        service.add_entry(
            _make_entry(
                "groceries",
                "200.00",
                period="2025-01",
                scenario_id="sc-a",
                entry_id="p1",
            )
        )
        service.add_entry(
            _make_entry(
                "groceries",
                "180.00",
                period="2025-02",
                scenario_id="sc-a",
                entry_id="p2",
            )
        )
        service.add_entry(
            _make_entry(
                "groceries",
                "220.00",
                period="2025-01",
                scenario_id="sc-b",
                entry_id="p3",
            )
        )
        service.add_entry(
            _make_entry(
                "groceries",
                "190.00",
                period="2025-02",
                scenario_id="sc-b",
                entry_id="p4",
            )
        )

        # Filter to January only
        comparisons = service.compare("sc-a", "sc-b", period="2025-01")
        assert len(comparisons) == 1
        assert comparisons[0].period == "2025-01"
        assert comparisons[0].variance == Decimal("20.00")

    def test_compare_empty_scenarios(self, service):
        """Comparing empty scenarios returns empty."""
        s1 = BudgetScenario(
            id="empty-a", space_id="space-1", name="Empty A", data_type=DataType.TARGET
        )
        s2 = BudgetScenario(
            id="empty-b", space_id="space-1", name="Empty B", data_type=DataType.TARGET
        )
        service.create_scenario(s1)
        service.create_scenario(s2)

        comparisons = service.compare("empty-a", "empty-b")
        assert comparisons == []


# ===========================================================================
# Migration tests
# ===========================================================================


class TestBudgetMigration:
    """Tests for the budget_entries migration."""

    def test_budget_entries_table_exists(self, db_manager):
        """Verify budget_entries table is created."""
        rows = db_manager.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='budget_entries'"
        )
        assert len(rows) == 1

    def test_schema_version_includes_3(self, db_manager):
        """Verify schema version 3 is recorded."""
        row = db_manager.fetchone(
            "SELECT version FROM schema_version WHERE version = 3"
        )
        assert row is not None
        assert row[0] == 3

    def test_budget_entries_indexes(self, db_manager):
        """Verify indexes are created on budget_entries."""
        rows = db_manager.fetchall(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='budget_entries'"
        )
        index_names = {row[0] for row in rows}
        assert "idx_budget_entries_scenario" in index_names
        assert "idx_budget_entries_period" in index_names
