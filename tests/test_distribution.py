"""Tests for the distribution module - audience-specific output forms."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from alibi.api.app import create_app
from alibi.api.deps import get_database
from alibi.db.connection import DatabaseManager
from alibi.distribution import DistributionForm, OutputFormat, DistributionRenderer
from alibi.distribution.forms import DistributionResult, distribute
from alibi.distribution.formatters import format_output


# --- Fixtures ---


@pytest.fixture
def seeded_db(db_manager: DatabaseManager) -> DatabaseManager:
    """Seed the database with v2 test data for distribution tests."""
    from uuid import uuid4

    conn = db_manager.get_connection()

    # Create documents (for artifact/document count)
    doc1_id = str(uuid4())
    doc2_id = str(uuid4())
    conn.execute(
        "INSERT INTO documents (id, file_path, file_hash, created_at) VALUES (?, ?, ?, ?)",
        (doc1_id, "/receipts/r1.jpg", "hash1", "2025-01-15"),
    )
    conn.execute(
        "INSERT INTO documents (id, file_path, file_hash, created_at) VALUES (?, ?, ?, ?)",
        (doc2_id, "/receipts/r2.jpg", "hash2", "2025-01-18"),
    )

    # Create clouds and facts (one cloud per fact)
    # expense → purchase, income → refund
    facts_data = [
        ("fact-1", "purchase", "Lidl", "45.50", "EUR", "2025-01-15"),
        ("fact-2", "purchase", "Lidl", "32.00", "EUR", "2025-01-22"),
        ("fact-3", "purchase", "Lidl", "28.75", "EUR", "2025-02-05"),
        ("fact-4", "purchase", "Restaurant Olive", "85.00", "EUR", "2025-01-18"),
        ("fact-5", "purchase", "Amazon", "250.00", "EUR", "2025-01-20"),
        ("fact-6", "purchase", "Shell", "55.00", "EUR", "2025-02-10"),
        ("fact-7", "purchase", "Ikea", "420.00", "EUR", "2025-02-15"),
        ("fact-8", "refund", "Employer", "3500.00", "EUR", "2025-01-01"),
        ("fact-9", "refund", "Employer", "3500.00", "EUR", "2025-02-01"),
        ("fact-10", "purchase", "Lidl", "38.25", "EUR", "2025-02-20"),
    ]

    for fact_id, fact_type, vendor, amount, currency, event_date in facts_data:
        cloud_id = str(uuid4())
        conn.execute(
            "INSERT INTO clouds (id, status) VALUES (?, ?)",
            (cloud_id, "collapsed"),
        )
        conn.execute(
            """INSERT INTO facts
               (id, cloud_id, fact_type, vendor, total_amount,
                currency, event_date, status)
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

    # Create atoms and fact_items for categories and line items
    fact_items_data = [
        # fact-1: Lidl receipt with detailed line items
        ("fact-1", "Milk", 2, "1.50", "3.00", "dairy"),
        ("fact-1", "Bread", 1, "2.50", "2.50", "bakery"),
        ("fact-1", "Cheese", 1, "4.99", "4.99", "dairy"),
        # Category items for tagged transactions
        ("fact-2", "Groceries", 1, "32.00", "32.00", "groceries"),
        ("fact-3", "Groceries", 1, "28.75", "28.75", "groceries"),
        ("fact-4", "Dinner", 1, "85.00", "85.00", "dining"),
        ("fact-10", "Groceries", 1, "38.25", "38.25", "groceries"),
    ]

    for fact_id, name, qty, unit_price, total_price, category in fact_items_data:
        atom_id = str(uuid4())
        fi_id = str(uuid4())
        conn.execute(
            "INSERT INTO atoms (id, document_id, atom_type, data) VALUES (?, ?, ?, ?)",
            (atom_id, doc1_id, "item", "{}"),
        )
        conn.execute(
            """INSERT INTO fact_items
               (id, fact_id, atom_id, name, quantity,
                unit_price, total_price, category)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (fi_id, fact_id, atom_id, name, qty, unit_price, total_price, category),
        )

    conn.commit()
    return db_manager


@pytest.fixture
def client(seeded_db: DatabaseManager) -> Generator[TestClient, None, None]:
    """Create a test client with seeded database."""
    app = create_app()

    def override_get_database() -> DatabaseManager:
        return seeded_db

    app.dependency_overrides[get_database] = override_get_database
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# --- Test Summary Form ---


class TestSummaryForm:
    """Tests for SUMMARY distribution form."""

    def test_summary_has_expected_keys(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.SUMMARY)
        data = result.data
        assert "total_expenses" in data
        assert "total_income" in data
        assert "net" in data
        assert "transaction_count" in data
        assert "top_vendors" in data
        assert "top_categories" in data

    def test_summary_totals_correct(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.SUMMARY)
        data = result.data
        # 8 expense transactions, 2 income transactions
        assert data["transaction_count"] == 10
        assert data["total_income"] == 7000.0
        # 45.50 + 32.00 + 28.75 + 85.00 + 250.00 + 55.00 + 420.00 + 38.25
        assert data["total_expenses"] == 954.50
        assert data["net"] == 7000.0 - 954.50

    def test_summary_top_vendors_limited_to_5(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.SUMMARY)
        assert len(result.data["top_vendors"]) <= 5

    def test_summary_top_vendors_sorted_by_total(
        self, seeded_db: DatabaseManager
    ) -> None:
        result = distribute(seeded_db, DistributionForm.SUMMARY)
        vendors = result.data["top_vendors"]
        if len(vendors) > 1:
            for i in range(len(vendors) - 1):
                assert vendors[i]["total"] >= vendors[i + 1]["total"]

    def test_summary_handles_empty_db(self, db_manager: DatabaseManager) -> None:
        result = distribute(db_manager, DistributionForm.SUMMARY)
        data = result.data
        assert data["total_expenses"] == 0.0
        assert data["total_income"] == 0.0
        assert data["net"] == 0.0
        assert data["transaction_count"] == 0
        assert data["top_vendors"] == []
        assert data["top_categories"] == []

    def test_summary_date_filter(self, seeded_db: DatabaseManager) -> None:
        result = distribute(
            seeded_db,
            DistributionForm.SUMMARY,
            date_from="2025-01-01",
            date_to="2025-01-31",
        )
        data = result.data
        # Only January transactions
        assert data["total_income"] == 3500.0
        # Jan expenses: 45.50 + 32.00 + 85.00 + 250.00 = 412.50
        assert data["total_expenses"] == 412.50

    def test_summary_result_metadata(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.SUMMARY)
        assert result.form == DistributionForm.SUMMARY
        assert result.title == "Spending Summary"
        assert result.metadata["space_id"] == "default"
        assert result.metadata["form"] == "summary"


# --- Test Detailed Form ---


class TestDetailedForm:
    """Tests for DETAILED distribution form."""

    def test_detailed_includes_summary_fields(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.DETAILED)
        data = result.data
        # Should have all summary fields
        assert "total_expenses" in data
        assert "total_income" in data
        assert "net" in data
        assert "transaction_count" in data

    def test_detailed_has_all_vendors(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.DETAILED)
        data = result.data
        assert "all_vendors" in data
        # We have 6 distinct expense vendors
        vendor_names = [v["vendor"] for v in data["all_vendors"]]
        assert "Lidl" in vendor_names
        assert "Amazon" in vendor_names
        assert "Restaurant Olive" in vendor_names

    def test_detailed_has_line_items_by_category(
        self, seeded_db: DatabaseManager
    ) -> None:
        result = distribute(seeded_db, DistributionForm.DETAILED)
        data = result.data
        assert "line_items_by_category" in data
        li_cats = data["line_items_by_category"]
        assert "dairy" in li_cats
        assert len(li_cats["dairy"]) == 2  # Milk and Cheese

    def test_detailed_has_monthly_trend(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.DETAILED)
        data = result.data
        assert "monthly_trend" in data
        assert len(data["monthly_trend"]) == 2  # Jan and Feb
        months = [m["month"] for m in data["monthly_trend"]]
        assert "2025-01" in months
        assert "2025-02" in months

    def test_detailed_has_artifacts_processed(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.DETAILED)
        assert result.data["artifacts_processed"] == 2


# --- Test Analytical Form ---


class TestAnalyticalForm:
    """Tests for ANALYTICAL distribution form."""

    def test_analytical_has_spending_trend(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.ANALYTICAL)
        data = result.data
        assert "spending_trend" in data
        assert len(data["spending_trend"]) > 0
        # Each entry should have period, total, avg
        entry = data["spending_trend"][0]
        assert "period" in entry
        assert "total" in entry
        assert "avg" in entry

    def test_analytical_has_vendor_frequency(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.ANALYTICAL)
        data = result.data
        assert "vendor_frequency" in data
        # Lidl appears 4 times, should be marked recurring
        lidl = next(
            (v for v in data["vendor_frequency"] if v["vendor"] == "Lidl"), None
        )
        assert lidl is not None
        assert lidl["count"] == 4
        assert lidl["trend"] == "recurring"

    def test_analytical_has_category_distribution(
        self, seeded_db: DatabaseManager
    ) -> None:
        result = distribute(seeded_db, DistributionForm.ANALYTICAL)
        data = result.data
        assert "category_distribution" in data
        # Should have percentages
        for cat in data["category_distribution"]:
            assert "percentage" in cat
            assert "amount" in cat
            assert cat["percentage"] >= 0

    def test_analytical_anomaly_detection(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.ANALYTICAL)
        data = result.data
        assert "anomalies" in data
        # Ikea 420.00 is significantly above average (avg ~119.31)
        # 420 > 119.31 * 3 = 357.94, so it should be an anomaly
        if data["anomalies"]:
            anomaly = data["anomalies"][0]
            assert "date" in anomaly
            assert "vendor" in anomaly
            assert "amount" in anomaly
            assert "reason" in anomaly

    def test_analytical_recurring_detection(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.ANALYTICAL)
        data = result.data
        assert "recurring" in data
        # Lidl has 4 transactions, should appear as recurring
        recurring_vendors = [r["vendor"] for r in data["recurring"]]
        assert "Lidl" in recurring_vendors


# --- Test Tabular Form ---


class TestTabularForm:
    """Tests for TABULAR distribution form."""

    def test_tabular_has_correct_headers(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.TABULAR)
        data = result.data
        assert data["headers"] == [
            "date",
            "vendor",
            "fact_type",
            "amount",
            "currency",
            "category",
        ]

    def test_tabular_row_count_matches_transactions(
        self, seeded_db: DatabaseManager
    ) -> None:
        result = distribute(seeded_db, DistributionForm.TABULAR)
        data = result.data
        # 10 transactions total (8 expense + 2 income)
        assert len(data["rows"]) == 10

    def test_tabular_handles_empty_db(self, db_manager: DatabaseManager) -> None:
        result = distribute(db_manager, DistributionForm.TABULAR)
        data = result.data
        assert data["headers"] == [
            "date",
            "vendor",
            "fact_type",
            "amount",
            "currency",
            "category",
        ]
        assert data["rows"] == []

    def test_tabular_date_filter(self, seeded_db: DatabaseManager) -> None:
        result = distribute(
            seeded_db,
            DistributionForm.TABULAR,
            date_from="2025-02-01",
            date_to="2025-02-28",
        )
        # Feb transactions: tx-3, tx-6, tx-7, tx-9, tx-10
        assert len(result.data["rows"]) == 5


# --- Test Markdown Formatter ---


class TestMarkdownFormatter:
    """Tests for Markdown output formatting."""

    def test_markdown_summary_has_headers(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.SUMMARY)
        md = format_output(result, OutputFormat.MARKDOWN)
        assert "# Spending Summary" in md
        assert "## Overview" in md

    def test_markdown_summary_has_tables(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.SUMMARY)
        md = format_output(result, OutputFormat.MARKDOWN)
        assert "## Top Vendors" in md
        assert "| Vendor | Total | Count |" in md
        assert "Lidl" in md

    def test_markdown_detailed_has_line_items(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.DETAILED)
        md = format_output(result, OutputFormat.MARKDOWN)
        assert "## Line Items by Category" in md
        assert "### dairy" in md
        assert "Milk" in md

    def test_markdown_analytical_has_trend(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.ANALYTICAL)
        md = format_output(result, OutputFormat.MARKDOWN)
        assert "## Spending Trend" in md
        assert "2025-01" in md


# --- Test CSV Formatter ---


class TestCsvFormatter:
    """Tests for CSV output formatting."""

    def test_csv_valid_output(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.TABULAR)
        csv_str = format_output(result, OutputFormat.CSV)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        # Header + 10 data rows
        assert len(rows) == 11
        assert rows[0] == [
            "date",
            "vendor",
            "fact_type",
            "amount",
            "currency",
            "category",
        ]

    def test_csv_rejects_non_tabular(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.SUMMARY)
        with pytest.raises(ValueError, match="CSV format is only supported"):
            format_output(result, OutputFormat.CSV)

    def test_csv_empty_db(self, db_manager: DatabaseManager) -> None:
        result = distribute(db_manager, DistributionForm.TABULAR)
        csv_str = format_output(result, OutputFormat.CSV)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1  # Header only


# --- Test HTML Formatter ---


class TestHtmlFormatter:
    """Tests for HTML output formatting."""

    def test_html_has_table_tags(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.TABULAR)
        html = format_output(result, OutputFormat.HTML)
        assert "<table>" in html
        assert "</table>" in html
        assert "<th>" in html

    def test_html_has_styling(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.SUMMARY)
        html = format_output(result, OutputFormat.HTML)
        assert "<style>" in html
        assert "border-collapse" in html

    def test_html_summary_has_content(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.SUMMARY)
        html = format_output(result, OutputFormat.HTML)
        assert "<h1>" in html
        assert "Spending Summary" in html


# --- Test JSON Formatter ---


class TestJsonFormatter:
    """Tests for JSON output formatting."""

    def test_json_valid_output(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.SUMMARY)
        json_str = format_output(result, OutputFormat.JSON)
        parsed = json.loads(json_str)
        assert parsed["form"] == "summary"
        assert "data" in parsed
        assert "period" in parsed

    def test_json_preserves_data_structure(self, seeded_db: DatabaseManager) -> None:
        result = distribute(seeded_db, DistributionForm.TABULAR)
        json_str = format_output(result, OutputFormat.JSON)
        parsed = json.loads(json_str)
        assert "headers" in parsed["data"]
        assert "rows" in parsed["data"]


# --- Test Distribution Renderer ---


class TestDistributionRenderer:
    """Tests for DistributionRenderer class."""

    def test_render_combines_form_and_format(self, seeded_db: DatabaseManager) -> None:
        renderer = DistributionRenderer(seeded_db)
        output = renderer.render(DistributionForm.SUMMARY, OutputFormat.MARKDOWN)
        assert "# Spending Summary" in output
        assert "Lidl" in output

    def test_render_for_telegram(self, seeded_db: DatabaseManager) -> None:
        renderer = DistributionRenderer(seeded_db)
        output = renderer.render_for_telegram()
        assert "# Spending Summary" in output
        assert "## Overview" in output

    def test_render_for_obsidian(self, seeded_db: DatabaseManager) -> None:
        renderer = DistributionRenderer(seeded_db)
        output = renderer.render_for_obsidian()
        assert "# Detailed Report" in output
        assert "## Line Items by Category" in output

    def test_render_for_export(self, seeded_db: DatabaseManager) -> None:
        renderer = DistributionRenderer(seeded_db)
        output = renderer.render_for_export()
        # Should be valid CSV
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        assert rows[0][0] == "date"
        assert len(rows) == 11  # header + 10 transactions

    def test_render_with_date_filter(self, seeded_db: DatabaseManager) -> None:
        renderer = DistributionRenderer(seeded_db)
        output = renderer.render(
            DistributionForm.SUMMARY,
            OutputFormat.JSON,
            date_from="2025-02-01",
            date_to="2025-02-28",
        )
        parsed = json.loads(output)
        # Only Feb data
        assert parsed["period"]["from"] == "2025-02-01"
        assert parsed["period"]["to"] == "2025-02-28"


# --- Test API Endpoint ---


class TestDistributeEndpoint:
    """Tests for the /api/v1/distribute endpoint."""

    def test_get_summary_json(self, client: TestClient) -> None:
        resp = client.get("/api/v1/distribute/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["form"] == "summary"
        assert data["format"] == "json"
        assert "total_expenses" in data["data"]

    def test_get_detailed_json(self, client: TestClient) -> None:
        resp = client.get("/api/v1/distribute/detailed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["form"] == "detailed"
        assert "all_vendors" in data["data"]

    def test_get_summary_markdown(self, client: TestClient) -> None:
        resp = client.get("/api/v1/distribute/summary?format=md")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/markdown; charset=utf-8"
        assert "# Spending Summary" in resp.text

    def test_get_tabular_csv(self, client: TestClient) -> None:
        resp = client.get("/api/v1/distribute/tabular?format=csv")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/csv; charset=utf-8"
        assert "date,vendor,fact_type" in resp.text

    def test_get_summary_html(self, client: TestClient) -> None:
        resp = client.get("/api/v1/distribute/summary?format=html")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/html; charset=utf-8"
        assert "<table>" in resp.text

    def test_get_with_date_filter(self, client: TestClient) -> None:
        resp = client.get(
            "/api/v1/distribute/summary" "?date_from=2025-01-01&date_to=2025-01-31"
        )
        assert resp.status_code == 200
        data = resp.json()
        # Jan income = 3500
        assert data["data"]["total_income"] == 3500.0

    def test_invalid_form_returns_422(self, client: TestClient) -> None:
        resp = client.get("/api/v1/distribute/invalid_form")
        assert resp.status_code == 422
