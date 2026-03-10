"""Tests for MCP server tools and resources.

Tests the underlying functions directly with a test database,
following the same fixture pattern as other alibi tests.
"""

import json
import uuid
from collections.abc import Generator
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from alibi.config import Config
from alibi.db.connection import DatabaseManager
from alibi.mcp.resources import (
    get_expiring_warranties,
    get_monthly_report,
    get_recent_transactions,
)
from alibi.mcp.tools import (
    analyze_spending_patterns,
    annotate_entity,
    correct_fact_vendor,
    get_budget_comparison,
    get_fact_detail,
    get_line_items,
    get_recurring_expenses,
    get_spending_summary,
    ingest_document,
    inspect_fact_detail,
    list_unassigned_bundles,
    move_fact_bundle,
    search_transactions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary database path."""
    path = tmp_path / "test_mcp.db"
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
    """Database with seed data for testing.

    Creates v2 clouds, facts, fact_items, and items with
    warranties for comprehensive testing.
    """
    from uuid import uuid4

    conn = db_manager.get_connection()

    # Create user and space (still needed for items/budgets)
    conn.execute(
        "INSERT INTO users (id, name) VALUES (?, ?)",
        ("user-1", "Test User"),
    )
    conn.execute(
        "INSERT INTO spaces (id, name, type, owner_id) VALUES (?, ?, ?, ?)",
        ("space-1", "Personal", "private", "user-1"),
    )

    today = date.today()

    # Create a dummy document for atoms
    doc_id = str(uuid4())
    conn.execute(
        "INSERT INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
        (doc_id, "/test/receipt.jpg", "abc123"),
    )

    # Create v2 clouds and facts
    facts_data = [
        # Recent facts (within 30 days)
        (
            "purchase",
            "Supermarket A",
            "150.00",
            "EUR",
            (today - timedelta(days=5)).isoformat(),
        ),
        (
            "purchase",
            "Restaurant B",
            "45.00",
            "EUR",
            (today - timedelta(days=10)).isoformat(),
        ),
        (
            "purchase",
            "Metro Transit",
            "30.00",
            "EUR",
            (today - timedelta(days=15)).isoformat(),
        ),
        (
            "refund",
            "Employer Corp",
            "3000.00",
            "EUR",
            (today - timedelta(days=20)).isoformat(),
        ),
        # Older facts (for spending patterns)
        (
            "purchase",
            "Supermarket A",
            "120.00",
            "EUR",
            (today - timedelta(days=35)).isoformat(),
        ),
        (
            "purchase",
            "Supermarket A",
            "135.00",
            "EUR",
            (today - timedelta(days=65)).isoformat(),
        ),
        (
            "purchase",
            "Electric Co",
            "80.00",
            "EUR",
            (today - timedelta(days=45)).isoformat(),
        ),
        (
            "purchase",
            "Restaurant B",
            "25.00",
            "EUR",
            (today - timedelta(days=50)).isoformat(),
        ),
    ]

    fact_ids = []
    for i, (fact_type, vendor, amount, currency, event_date) in enumerate(facts_data):
        cloud_id = str(uuid4())
        fact_id = f"fact-{i + 1}"
        conn.execute(
            "INSERT INTO clouds (id, status) VALUES (?, ?)",
            (cloud_id, "collapsed"),
        )
        conn.execute(
            """INSERT INTO facts
               (id, cloud_id, fact_type, vendor, total_amount, currency, event_date, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact_id,
                cloud_id,
                fact_type,
                vendor,
                amount,
                currency,
                event_date,
                "confirmed",
            ),
        )
        fact_ids.append(fact_id)

    # Create atoms and fact_items for fact 1 (Supermarket A, 150.00)
    item_data = [
        (fact_ids[0], "Milk 1L", "milk", 2, "pcs", 1.50, 3.00, None, "dairy"),
        (fact_ids[0], "Bread", "bread", 1, "pcs", 2.50, 2.50, None, "bakery"),
        (
            fact_ids[0],
            "Chicken breast",
            "chicken breast",
            0.5,
            "kg",
            8.00,
            4.00,
            None,
            "meat",
        ),
        # fact_item for fact 5 (Supermarket A, 120.00)
        (fact_ids[4], "Olive oil", "olive oil", 1, "pcs", 6.00, 6.00, None, "pantry"),
    ]

    for fact_id, name, name_norm, qty, unit, up, tp, brand, cat in item_data:
        atom_id = str(uuid4())
        conn.execute(
            "INSERT INTO atoms (id, document_id, atom_type, data) VALUES (?, ?, ?, ?)",
            (atom_id, doc_id, "item", "{}"),
        )
        fi_id = str(uuid4())
        conn.execute(
            """INSERT INTO fact_items
               (id, fact_id, atom_id, name, name_normalized,
                quantity, unit, unit_price, total_price, brand, category)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fi_id, fact_id, atom_id, name, name_norm, qty, unit, up, tp, brand, cat),
        )

    # Create items with warranties (still uses v1 items table)
    items = [
        (
            "item-1",
            "space-1",
            "Laptop",
            "electronics",
            "ThinkPad X1",
            "SN123",
            (today - timedelta(days=300)).isoformat(),
            1200.00,
            900.00,
            "EUR",
            "active",
            (today + timedelta(days=30)).isoformat(),
            "manufacturer",
        ),
        (
            "item-2",
            "space-1",
            "Washing Machine",
            "appliances",
            "Bosch WAX",
            "SN456",
            (today - timedelta(days=600)).isoformat(),
            800.00,
            500.00,
            "EUR",
            "active",
            (today + timedelta(days=60)).isoformat(),
            "extended",
        ),
        (
            "item-3",
            "space-1",
            "Old Phone",
            "electronics",
            "iPhone 12",
            "SN789",
            (today - timedelta(days=900)).isoformat(),
            999.00,
            200.00,
            "EUR",
            "active",
            (today - timedelta(days=30)).isoformat(),
            "manufacturer",
        ),
    ]

    for item in items:
        conn.execute(
            """INSERT INTO items
               (id, space_id, name, category, model, serial_number,
                purchase_date, purchase_price, current_value, currency,
                status, warranty_expires, warranty_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            item,
        )

    conn.commit()
    return db_manager


@pytest.fixture
def seeded_db_with_budgets(seeded_db: DatabaseManager) -> DatabaseManager:
    """Database with budget scenario data added."""
    conn = seeded_db.get_connection()

    today = date.today()
    current_period = today.strftime("%Y-%m")

    # Create a budget scenario
    conn.execute(
        """INSERT INTO budgets
           (id, space_id, name, description, data_type,
            period_start, period_end)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            "budget-1",
            "space-1",
            "Monthly Budget",
            "Target spending for current month",
            "target",
            f"{today.year}-{today.month:02d}-01",
            f"{today.year}-{today.month:02d}-28",
        ),
    )

    # Create budget entries
    entries = [
        ("be-1", "budget-1", "groceries", "200.00", "EUR", current_period),
        ("be-2", "budget-1", "dining", "100.00", "EUR", current_period),
        ("be-3", "budget-1", "transport", "50.00", "EUR", current_period),
        ("be-4", "budget-1", "utilities", "150.00", "EUR", current_period),
    ]

    for entry in entries:
        conn.execute(
            """INSERT INTO budget_entries
               (id, scenario_id, category, amount, currency, period)
               VALUES (?, ?, ?, ?, ?, ?)""",
            entry,
        )

    conn.commit()
    return seeded_db


@pytest.fixture
def seeded_db_with_recurring(
    db_manager: DatabaseManager,
) -> DatabaseManager:
    """Database with recurring transaction patterns (v2 facts)."""
    from uuid import uuid4

    conn = db_manager.get_connection()

    # Create monthly recurring Netflix subscription as v2 facts
    today = date.today()
    for i in range(6):
        tx_date = today - timedelta(days=30 * (5 - i))
        cloud_id = str(uuid4())
        conn.execute(
            "INSERT INTO clouds (id, status) VALUES (?, ?)",
            (cloud_id, "collapsed"),
        )
        conn.execute(
            """INSERT INTO facts
               (id, cloud_id, fact_type, vendor, total_amount, currency,
                event_date)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                f"rec-{i}",
                cloud_id,
                "purchase",
                "Netflix",
                "15.99",
                "EUR",
                tx_date.isoformat(),
            ),
        )

    # Create biweekly gym payments (less regular)
    for i in range(4):
        tx_date = today - timedelta(days=14 * (3 - i))
        cloud_id = str(uuid4())
        conn.execute(
            "INSERT INTO clouds (id, status) VALUES (?, ?)",
            (cloud_id, "collapsed"),
        )
        conn.execute(
            """INSERT INTO facts
               (id, cloud_id, fact_type, vendor, total_amount, currency,
                event_date)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                f"gym-{i}",
                cloud_id,
                "purchase",
                "Fitness Club",
                "29.00",
                "EUR",
                tx_date.isoformat(),
            ),
        )

    conn.commit()
    return db_manager


# ===========================================================================
# Tool Tests
# ===========================================================================


class TestSearchTransactions:
    """Tests for the search_transactions tool."""

    def test_search_by_text(self, seeded_db):
        """Search finds facts matching vendor name."""
        result = search_transactions(seeded_db, query="Supermarket")
        assert result["total"] >= 1
        for tx in result["results"]:
            assert "supermarket" in (tx["vendor"] or "").lower()

    def test_search_by_vendor(self, seeded_db):
        """Search finds facts matching vendor name."""
        result = search_transactions(seeded_db, query="Restaurant")
        assert result["total"] >= 1

    def test_search_with_date_from(self, seeded_db):
        """Date from filter restricts results."""
        today = date.today()
        recent = today - timedelta(days=7)
        result = search_transactions(
            seeded_db, query="Supermarket", date_from=recent.isoformat()
        )
        # Should only find the most recent transaction
        for tx in result["results"]:
            tx_date = tx["event_date"]
            if isinstance(tx_date, str):
                tx_date = date.fromisoformat(tx_date)
            assert tx_date >= recent

    def test_search_with_date_range(self, seeded_db):
        """Date range filter restricts results to window."""
        today = date.today()
        dt_from = today - timedelta(days=40)
        dt_to = today - timedelta(days=30)
        result = search_transactions(
            seeded_db,
            query="Supermarket",
            date_from=dt_from.isoformat(),
            date_to=dt_to.isoformat(),
        )
        for tx in result["results"]:
            tx_date = tx["event_date"]
            if isinstance(tx_date, str):
                tx_date = date.fromisoformat(tx_date)
            assert tx_date >= dt_from
            assert tx_date <= dt_to

    def test_search_finds_by_vendor_name(self, seeded_db):
        """Specific vendor search returns matching facts."""
        result = search_transactions(seeded_db, query="Supermarket A")
        assert result["total"] >= 1
        for tx in result["results"]:
            assert "supermarket" in (tx.get("vendor") or "").lower()

    def test_search_with_limit(self, seeded_db):
        """Limit parameter restricts result count."""
        result = search_transactions(seeded_db, query="e", limit=2)
        assert len(result["results"]) <= 2

    def test_search_no_results(self, seeded_db):
        """Search with no matches returns empty results."""
        result = search_transactions(seeded_db, query="NonexistentVendor12345")
        assert result["total"] == 0
        assert result["results"] == []

    def test_search_result_structure(self, seeded_db):
        """Search results have expected fields."""
        result = search_transactions(seeded_db, query="Supermarket")
        assert "query" in result
        assert "total" in result
        assert "results" in result
        if result["results"]:
            tx = result["results"][0]
            assert "id" in tx
            assert "vendor" in tx
            assert "total_amount" in tx
            assert "event_date" in tx


class TestSpendingSummary:
    """Tests for the get_spending_summary tool."""

    def test_monthly_grouping(self, seeded_db):
        """Monthly grouping returns aggregated data."""
        result = get_spending_summary(seeded_db, period="month")
        assert result["group_by"] == "month"
        assert len(result["data"]) >= 1
        for entry in result["data"]:
            assert "period" in entry
            assert "count" in entry
            assert "total" in entry

    def test_daily_grouping(self, seeded_db):
        """Daily grouping returns per-day data."""
        result = get_spending_summary(seeded_db, period="day")
        assert result["group_by"] == "day"
        # Daily periods should look like YYYY-MM-DD
        for entry in result["data"]:
            assert len(entry["period"]) == 10

    def test_weekly_grouping(self, seeded_db):
        """Weekly grouping returns per-week data."""
        result = get_spending_summary(seeded_db, period="week")
        assert result["group_by"] == "week"

    def test_date_range_filter(self, seeded_db):
        """Date range filters spending summary."""
        today = date.today()
        date_from = (today - timedelta(days=14)).isoformat()
        result = get_spending_summary(seeded_db, period="month", date_from=date_from)
        # Should only include recent data
        assert result["group_by"] == "month"

    def test_summary_has_statistics(self, seeded_db):
        """Summary entries include statistical fields."""
        result = get_spending_summary(seeded_db, period="month")
        if result["data"]:
            entry = result["data"][0]
            assert "average" in entry
            assert "min_amount" in entry
            assert "max_amount" in entry


class TestLineItems:
    """Tests for the get_line_items tool."""

    def test_get_all_line_items(self, seeded_db):
        """Get all line items without filters."""
        result = get_line_items(seeded_db)
        assert result["total"] == 4
        assert len(result["results"]) == 4

    def test_filter_by_category(self, seeded_db):
        """Category filter returns only matching items."""
        result = get_line_items(seeded_db, category="dairy")
        assert result["total"] >= 1
        for item in result["results"]:
            assert item["category"] == "dairy"

    def test_filter_by_date(self, seeded_db):
        """Date filter restricts line items by transaction date."""
        today = date.today()
        recent = today - timedelta(days=7)
        result = get_line_items(seeded_db, date_from=recent.isoformat())
        # Should only include line items from recent transactions
        for item in result["results"]:
            tx_date = item["event_date"]
            if isinstance(tx_date, str):
                tx_date = date.fromisoformat(tx_date)
            assert tx_date >= recent

    def test_limit(self, seeded_db):
        """Limit parameter restricts result count."""
        result = get_line_items(seeded_db, limit=2)
        assert len(result["results"]) <= 2

    def test_line_item_structure(self, seeded_db):
        """Line items have expected fields."""
        result = get_line_items(seeded_db)
        if result["results"]:
            item = result["results"][0]
            assert "name" in item
            assert "quantity" in item
            assert "unit_price" in item
            assert "total_price" in item
            assert "category" in item


class TestSpendingPatterns:
    """Tests for the analyze_spending_patterns tool."""

    def test_basic_analysis(self, seeded_db):
        """Basic spending analysis returns expected structure."""
        result = analyze_spending_patterns(seeded_db, months=3)
        assert "period_months" in result
        assert "period_start" in result
        assert "top_vendors" in result
        assert "top_categories" in result
        assert "summary" in result

    def test_top_vendors(self, seeded_db):
        """Top vendors are sorted by total spending."""
        result = analyze_spending_patterns(seeded_db, months=3)
        vendors = result["top_vendors"]
        if len(vendors) >= 2:
            # Should be sorted by total descending
            assert vendors[0]["total"] >= vendors[1]["total"]

    def test_summary_stats(self, seeded_db):
        """Summary includes transaction count and totals."""
        result = analyze_spending_patterns(seeded_db, months=3)
        summary = result["summary"]
        assert "transaction_count" in summary
        assert "total_spending" in summary
        assert "average_transaction" in summary
        assert summary["transaction_count"] > 0

    def test_period_parameter(self, seeded_db):
        """Period months parameter affects the analysis window."""
        result_3m = analyze_spending_patterns(seeded_db, months=3)
        result_1m = analyze_spending_patterns(seeded_db, months=1)
        assert result_3m["period_months"] == 3
        assert result_1m["period_months"] == 1


class TestBudgetComparison:
    """Tests for the get_budget_comparison tool."""

    def test_valid_scenario(self, seeded_db_with_budgets):
        """Budget comparison with valid scenario returns data."""
        result = get_budget_comparison(seeded_db_with_budgets, scenario_id="budget-1")
        assert "scenario" in result
        assert result["scenario"]["id"] == "budget-1"
        assert result["scenario"]["name"] == "Monthly Budget"
        assert "entries" in result
        assert len(result["entries"]) == 4

    def test_nonexistent_scenario(self, seeded_db):
        """Budget comparison with invalid scenario returns error."""
        result = get_budget_comparison(seeded_db, scenario_id="nonexistent")
        assert "error" in result

    def test_comparison_structure(self, seeded_db_with_budgets):
        """Comparison entries have expected fields."""
        result = get_budget_comparison(seeded_db_with_budgets, scenario_id="budget-1")
        if result.get("comparisons"):
            comp = result["comparisons"][0]
            assert "period" in comp
            assert "category" in comp
            assert "budgeted" in comp
            assert "actual" in comp
            assert "variance" in comp
            assert "over_budget" in comp

    def test_budget_entries_content(self, seeded_db_with_budgets):
        """Budget entries match the seeded data."""
        result = get_budget_comparison(seeded_db_with_budgets, scenario_id="budget-1")
        categories = {e["category"] for e in result["entries"]}
        assert "groceries" in categories
        assert "dining" in categories
        assert "transport" in categories
        assert "utilities" in categories


class TestRecurringExpenses:
    """Tests for the get_recurring_expenses tool."""

    def test_detect_recurring(self, seeded_db_with_recurring):
        """Detects recurring transaction patterns."""
        result = get_recurring_expenses(seeded_db_with_recurring, min_occurrences=3)
        assert "patterns" in result
        assert "total" in result
        # Netflix has 6 monthly payments, should be detected
        assert result["total"] >= 1

    def test_recurring_pattern_structure(self, seeded_db_with_recurring):
        """Recurring patterns have expected fields."""
        result = get_recurring_expenses(seeded_db_with_recurring, min_occurrences=3)
        if result["patterns"]:
            pattern = result["patterns"][0]
            assert "vendor" in pattern
            assert "avg_amount" in pattern
            assert "frequency_days" in pattern
            assert "period_type" in pattern
            assert "confidence" in pattern
            assert "next_expected" in pattern
            assert "occurrences" in pattern
            assert "fact_ids" in pattern

    def test_min_occurrences_filter(self, seeded_db_with_recurring):
        """Higher min_occurrences filters out less frequent patterns."""
        result_low = get_recurring_expenses(seeded_db_with_recurring, min_occurrences=3)
        result_high = get_recurring_expenses(
            seeded_db_with_recurring, min_occurrences=5
        )
        assert result_high["total"] <= result_low["total"]

    def test_no_recurring_on_empty(self, db_manager):
        """Empty database returns no recurring patterns."""
        result = get_recurring_expenses(db_manager, min_occurrences=3)
        assert result["total"] == 0
        assert result["patterns"] == []


# ===========================================================================
# Resource Tests
# ===========================================================================


class TestResources:
    """Tests for MCP resource providers."""

    def test_recent_transactions(self, seeded_db):
        """Recent transactions resource returns last 30 days."""
        result = get_recent_transactions(seeded_db)
        assert "date_from" in result
        assert "date_to" in result
        assert "total" in result
        assert "transactions" in result
        # tx-1 through tx-4 are within 30 days
        assert result["total"] >= 3

    def test_recent_transactions_date_range(self, seeded_db):
        """Recent transactions are within the 30-day window."""
        result = get_recent_transactions(seeded_db)
        today = date.today()
        cutoff = today - timedelta(days=30)
        for tx in result["transactions"]:
            tx_date = tx["event_date"]
            if isinstance(tx_date, str):
                tx_date = date.fromisoformat(tx_date)
            assert tx_date >= cutoff

    def test_monthly_report(self, seeded_db):
        """Monthly report returns structured spending data."""
        today = date.today()
        result = get_monthly_report(seeded_db, year=today.year, month=today.month)
        assert "period" in result
        assert result["period"]["year"] == today.year
        assert result["period"]["month"] == today.month
        assert "expenses" in result
        assert "income" in result
        assert "top_vendors" in result
        assert "artifacts_processed" in result

    def test_monthly_report_expenses(self, seeded_db):
        """Monthly report correctly aggregates expenses."""
        today = date.today()
        result = get_monthly_report(seeded_db, year=today.year, month=today.month)
        assert result["expenses"]["count"] >= 0
        assert isinstance(result["expenses"]["total"], (int, float))

    def test_monthly_report_empty_month(self, seeded_db):
        """Monthly report for empty month returns zeros."""
        result = get_monthly_report(seeded_db, year=2020, month=1)
        assert result["expenses"]["count"] == 0
        assert result["expenses"]["total"] == 0
        assert result["income"]["count"] == 0

    def test_expiring_warranties(self, seeded_db):
        """Expiring warranties returns items within 90-day window."""
        result = get_expiring_warranties(seeded_db)
        assert "as_of" in result
        assert "cutoff_date" in result
        assert "total" in result
        assert "items" in result
        # item-1 expires in 30 days, item-2 in 60 days
        assert result["total"] == 2

    def test_expiring_warranties_excludes_past(self, seeded_db):
        """Expiring warranties does not include already expired items."""
        result = get_expiring_warranties(seeded_db)
        today = date.today()
        for item in result["items"]:
            warranty_date = item["warranty_expires"]
            if isinstance(warranty_date, str):
                warranty_date = date.fromisoformat(warranty_date)
            assert warranty_date >= today

    def test_expiring_warranty_fields(self, seeded_db):
        """Expiring warranty items have expected fields."""
        result = get_expiring_warranties(seeded_db)
        if result["items"]:
            item = result["items"][0]
            assert "name" in item
            assert "category" in item
            assert "warranty_expires" in item
            assert "warranty_type" in item
            assert "purchase_price" in item


# ===========================================================================
# New Tool Tests
# ===========================================================================


class TestGetFactDetail:
    """Tests for the get_fact_detail tool."""

    def test_get_existing_fact(self, seeded_db):
        """get_fact_detail returns fact metadata for a known fact ID."""
        result = get_fact_detail(seeded_db, "fact-1")
        assert "error" not in result
        assert result["vendor"] == "Supermarket A"
        # total_amount may be stored as numeric by SQLite DECIMAL affinity
        assert float(result["total_amount"]) == 150.0
        assert "items" in result

    def test_get_nonexistent_fact(self, seeded_db):
        """get_fact_detail returns an error dict for an unknown fact ID."""
        result = get_fact_detail(seeded_db, "nonexistent")
        assert "error" in result
        assert "nonexistent" in result["error"]

    def test_fact_has_items(self, seeded_db):
        """fact-1 has three line items: Milk 1L, Bread, Chicken breast."""
        result = get_fact_detail(seeded_db, "fact-1")
        assert "items" in result
        item_names = [item["name"] for item in result["items"]]
        assert "Milk 1L" in item_names
        assert "Bread" in item_names
        assert "Chicken breast" in item_names

    def test_fact_items_count(self, seeded_db):
        """fact-1 has exactly 3 line items."""
        result = get_fact_detail(seeded_db, "fact-1")
        assert len(result["items"]) == 3

    def test_fact_without_items(self, seeded_db):
        """fact-2 has no line items but still returns an items key."""
        result = get_fact_detail(seeded_db, "fact-2")
        assert "error" not in result
        assert "items" in result
        assert result["items"] == []


class TestInspectFactDetail:
    """Tests for the inspect_fact_detail tool."""

    def test_inspect_existing_fact(self, seeded_db):
        """inspect_fact_detail returns nested drill-down for a known fact ID."""
        result = inspect_fact_detail(seeded_db, "fact-1")
        assert "error" not in result
        assert "fact" in result
        assert "cloud" in result
        assert "bundles" in result
        assert "items" in result

    def test_inspect_fact_field(self, seeded_db):
        """The nested fact field contains correct vendor data."""
        result = inspect_fact_detail(seeded_db, "fact-1")
        assert result["fact"]["vendor"] == "Supermarket A"

    def test_inspect_nonexistent(self, seeded_db):
        """inspect_fact_detail returns an error dict for unknown fact ID."""
        result = inspect_fact_detail(seeded_db, "nonexistent")
        assert "error" in result
        assert "nonexistent" in result["error"]

    def test_inspect_items_present(self, seeded_db):
        """inspect_fact_detail includes the line items for fact-1."""
        result = inspect_fact_detail(seeded_db, "fact-1")
        assert len(result["items"]) == 3


class TestListUnassigned:
    """Tests for the list_unassigned_bundles tool."""

    def test_no_unassigned(self, seeded_db):
        """seeded_db has no unassigned bundles."""
        result = list_unassigned_bundles(seeded_db)
        assert result["total"] == 0
        assert result["bundles"] == []

    def test_result_structure(self, seeded_db):
        """Result always has 'total' and 'bundles' keys."""
        result = list_unassigned_bundles(seeded_db)
        assert "total" in result
        assert "bundles" in result
        assert isinstance(result["total"], int)
        assert isinstance(result["bundles"], list)

    def test_empty_db_no_unassigned(self, db_manager):
        """Empty database also returns the correct structure."""
        result = list_unassigned_bundles(db_manager)
        assert result["total"] == 0
        assert result["bundles"] == []


class TestIngestDocument:
    """Tests for the ingest_document tool."""

    def test_file_not_found(self, seeded_db):
        """ingest_document returns an error dict when file does not exist."""
        result = ingest_document(seeded_db, "/nonexistent/path/receipt.jpg")
        assert "error" in result
        assert "File not found" in result["error"]

    def test_file_not_found_path_in_error(self, seeded_db):
        """Error message includes the original path."""
        path = "/nonexistent/path/receipt.jpg"
        result = ingest_document(seeded_db, path)
        assert path in result["error"]

    def test_missing_file_no_database_side_effects(self, seeded_db):
        """A missing-file error leaves the document count unchanged."""
        count_before = seeded_db.fetchone("SELECT COUNT(*) AS cnt FROM documents", ())[
            "cnt"
        ]
        ingest_document(seeded_db, "/no/such/file.png")
        count_after = seeded_db.fetchone("SELECT COUNT(*) AS cnt FROM documents", ())[
            "cnt"
        ]
        assert count_after == count_before

    def test_invalid_doc_type(self, seeded_db):
        """ingest_document rejects an invalid doc_type."""
        result = ingest_document(seeded_db, "/some/file.jpg", doc_type="bogus")
        assert "error" in result
        assert "bogus" in result["error"].lower()

    def test_valid_doc_type_still_checks_file(self, seeded_db):
        """doc_type is valid but file doesn't exist — file error takes precedence."""
        result = ingest_document(seeded_db, "/nonexistent/file.jpg", doc_type="receipt")
        assert "error" in result
        assert "File not found" in result["error"]


class TestIngestBytes:
    """Tests for the ingest_bytes tool."""

    def test_invalid_doc_type(self, seeded_db):
        """ingest_bytes rejects an invalid doc_type."""
        from alibi.mcp.tools import ingest_bytes

        result = ingest_bytes(seeded_db, b"data", "test.jpg", doc_type="bogus")
        assert "error" in result
        assert "bogus" in result["error"].lower()


class TestCorrectFactVendor:
    """Tests for the correct_fact_vendor tool."""

    def test_correct_existing_fact(self, seeded_db):
        """correct_fact_vendor succeeds for a known fact ID."""
        result = correct_fact_vendor(seeded_db, "fact-1", "New Market")
        assert result["success"] is True
        assert result["fact_id"] == "fact-1"
        assert "vendor" in result

    def test_correct_nonexistent_fact(self, seeded_db):
        """correct_fact_vendor returns failure for an unknown fact ID."""
        result = correct_fact_vendor(seeded_db, "nonexistent", "New Market")
        assert result["success"] is False
        assert "error" in result

    def test_vendor_is_updated(self, seeded_db):
        """After correction, get_fact_detail reflects the new vendor name."""
        correct_fact_vendor(seeded_db, "fact-1", "New Market")
        detail = get_fact_detail(seeded_db, "fact-1")
        # normalize_vendor applies title-case, so compare case-insensitively
        assert "market" in detail["vendor"].lower()

    def test_vendor_name_normalised(self, seeded_db):
        """correct_fact_vendor normalises the vendor name before storing."""
        correct_fact_vendor(seeded_db, "fact-2", "fresh foods ltd")
        detail = get_fact_detail(seeded_db, "fact-2")
        # normalizer returns title-case, should not be all lower
        assert detail["vendor"] != "fresh foods ltd"

    def test_result_contains_fact_id(self, seeded_db):
        """Successful result includes the correct fact_id."""
        result = correct_fact_vendor(seeded_db, "fact-3", "Metro Plus")
        assert result["fact_id"] == "fact-3"


class TestMoveFactBundle:
    """Tests for the move_fact_bundle tool.

    The seeded_db creates facts and clouds directly without bundles in the
    cloud_bundles table, so move_bundle will return success=False for any
    bundle_id because the bundle cannot be found. These tests verify the
    error-path contract of the MCP wrapper.
    """

    def test_nonexistent_bundle_returns_failure(self, seeded_db):
        """move_fact_bundle with an unknown bundle_id returns success=False."""
        result = move_fact_bundle(seeded_db, "nonexistent-bundle-id")
        assert result["success"] is False

    def test_result_has_required_keys(self, seeded_db):
        """move_fact_bundle result always contains the expected keys."""
        result = move_fact_bundle(seeded_db, "nonexistent-bundle-id")
        assert "success" in result
        assert "error" in result
        assert "source_cloud_id" in result
        assert "target_cloud_id" in result
        assert "source_fact_id" in result
        assert "target_fact_id" in result

    def test_nonexistent_bundle_with_target_cloud(self, seeded_db):
        """move_fact_bundle with unknown bundle and target cloud returns failure."""
        result = move_fact_bundle(
            seeded_db,
            "nonexistent-bundle-id",
            target_cloud_id="nonexistent-cloud-id",
        )
        assert result["success"] is False


class TestAnnotateEntity:
    """Tests for the annotate_entity tool."""

    def test_annotate_fact(self, seeded_db):
        """annotate_entity adds a person annotation to a fact."""
        result = annotate_entity(
            seeded_db, "fact", "fact-1", "person", "bought_for", "Maria"
        )
        assert result.get("success") is True
        assert "annotation_id" in result

    def test_annotation_id_is_string(self, seeded_db):
        """The returned annotation_id is a non-empty string."""
        result = annotate_entity(
            seeded_db, "fact", "fact-1", "project", "project", "Kitchen"
        )
        assert isinstance(result["annotation_id"], str)
        assert len(result["annotation_id"]) > 0

    def test_invalid_target_type(self, seeded_db):
        """annotate_entity rejects an invalid target_type."""
        result = annotate_entity(
            seeded_db, "invalid", "fact-1", "person", "bought_for", "Maria"
        )
        assert "error" in result
        assert "invalid" in result["error"].lower()

    def test_annotation_structure(self, seeded_db):
        """Successful result contains target_type and target_id fields."""
        result = annotate_entity(
            seeded_db, "fact", "fact-1", "category", "custom_cat", "office"
        )
        assert result["target_type"] == "fact"
        assert result["target_id"] == "fact-1"

    def test_annotate_fact_item_type(self, seeded_db):
        """annotate_entity accepts 'fact_item' as target_type."""
        result = annotate_entity(
            seeded_db, "fact_item", "some-id", "note", "note", "allergic"
        )
        assert result.get("success") is True

    def test_annotate_vendor_type(self, seeded_db):
        """annotate_entity accepts 'vendor' as target_type."""
        result = annotate_entity(
            seeded_db, "vendor", "Supermarket A", "tag", "tag", "local"
        )
        assert result.get("success") is True

    def test_annotate_identity_type(self, seeded_db):
        """annotate_entity accepts 'identity' as target_type."""
        result = annotate_entity(
            seeded_db, "identity", "identity-1", "note", "memo", "primary"
        )
        assert result.get("success") is True

    def test_invalid_type_lists_valid_options(self, seeded_db):
        """Error message for invalid target_type mentions the valid options."""
        result = annotate_entity(seeded_db, "transaction", "id", "type", "key", "val")
        error = result["error"]
        assert "fact" in error

    def test_multiple_annotations_unique_ids(self, seeded_db):
        """Each annotation gets a distinct ID."""
        r1 = annotate_entity(
            seeded_db, "fact", "fact-1", "person", "bought_for", "Alice"
        )
        r2 = annotate_entity(seeded_db, "fact", "fact-1", "person", "bought_for", "Bob")
        assert r1["annotation_id"] != r2["annotation_id"]


class TestSearchTransactionsItemSearch:
    """Tests verifying search_transactions searches item names via service layer."""

    def test_search_finds_items_by_name(self, seeded_db):
        """Search for an item name returns the fact containing that item."""
        result = search_transactions(seeded_db, query="Milk")
        assert result["total"] >= 1
        fact_ids = [tx["id"] for tx in result["results"]]
        assert "fact-1" in fact_ids

    def test_search_finds_fact_via_item_olive(self, seeded_db):
        """Search for 'olive' finds fact-5 via the Olive oil item."""
        result = search_transactions(seeded_db, query="olive")
        assert result["total"] >= 1
        fact_ids = [tx["id"] for tx in result["results"]]
        assert "fact-5" in fact_ids

    def test_search_finds_fact_via_item_bread(self, seeded_db):
        """Search for 'Bread' finds fact-1 via the Bread item."""
        result = search_transactions(seeded_db, query="Bread")
        assert result["total"] >= 1
        fact_ids = [tx["id"] for tx in result["results"]]
        assert "fact-1" in fact_ids

    def test_item_search_does_not_duplicate_facts(self, seeded_db):
        """A fact with multiple matching items appears only once."""
        # fact-1 has Milk 1L, Bread, Chicken breast - query matching multiple
        result = search_transactions(seeded_db, query="Milk")
        count = sum(1 for tx in result["results"] if tx["id"] == "fact-1")
        assert count == 1

    def test_item_search_combined_with_vendor_search(self, seeded_db):
        """Search on a vendor term still returns vendor-matched facts."""
        result = search_transactions(seeded_db, query="Supermarket")
        assert result["total"] >= 1
        for tx in result["results"]:
            assert "supermarket" in (tx.get("vendor") or "").lower()

    def test_item_search_no_results_for_unknown_term(self, seeded_db):
        """Search for a term matching neither vendor nor item returns empty."""
        result = search_transactions(seeded_db, query="xyznonexistent999")
        assert result["total"] == 0
        assert result["results"] == []
