"""Tests for Telegram bot handlers."""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from alibi.db.connection import DatabaseManager, reset_db
from alibi.db.models import (
    Item,
    ItemStatus,
    Space,
    SpaceType,
    User,
)


@pytest.fixture(scope="function")
def db():
    """Create a test database for each test."""
    from pathlib import Path
    import tempfile

    # Use a temporary database file
    temp_dir = tempfile.mkdtemp()
    test_db_path = Path(temp_dir) / "test_alibi.db"

    # Create a test-specific database manager
    import os

    os.environ["ALIBI_DB_PATH"] = str(test_db_path)
    reset_db()

    from alibi.config import Config

    config = Config(db_path=test_db_path)
    db_manager = DatabaseManager(config=config)
    db_manager.initialize()

    yield db_manager

    db_manager.close()
    reset_db()

    # Cleanup
    if test_db_path.exists():
        test_db_path.unlink()


@pytest.fixture
def test_user(db):
    """Create a test user."""
    user = User(id="test-user", name="Test User")
    # Use INSERT OR IGNORE to avoid conflicts
    db.execute(
        "INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)",
        (user.id, user.name),
    )
    db.get_connection().commit()
    return user


@pytest.fixture
def test_space(db, test_user):
    """Create a test space."""
    space = Space(
        id="test-space",
        name="Test Space",
        type=SpaceType.PRIVATE,
        owner_id=test_user.id,
    )
    # Use INSERT OR IGNORE to avoid conflicts
    db.execute(
        "INSERT OR IGNORE INTO spaces (id, name, type, owner_id) VALUES (?, ?, ?, ?)",
        (space.id, space.name, space.type.value, space.owner_id),
    )
    db.get_connection().commit()
    return space


def _create_fact(db, fact_id, vendor, amount, event_date, currency="EUR"):
    """Helper to create a cloud + fact for testing."""
    cloud_id = f"cloud-{fact_id}"
    db.execute(
        "INSERT INTO clouds (id, status) VALUES (?, 'collapsed')",
        (cloud_id,),
    )
    db.execute(
        """INSERT INTO facts
           (id, cloud_id, fact_type, vendor, total_amount, currency, event_date, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            fact_id,
            cloud_id,
            "purchase",
            vendor,
            str(amount),
            currency,
            event_date.isoformat(),
            "confirmed",
        ),
    )


class TestExpensesHandler:
    """Tests for /expenses command."""

    def test_no_expenses(self, db, test_space):
        """Test when there are no expenses."""
        sql = """
            SELECT vendor, event_date, total_amount, currency
            FROM facts
            WHERE event_date >= ?
            AND fact_type = 'purchase'
            ORDER BY event_date DESC
            LIMIT 50
        """
        since_date = date.today() - timedelta(days=7)
        rows = db.fetchall(sql, (since_date.isoformat(),))
        assert len(rows) == 0

    def test_with_expenses(self, db, test_space):
        """Test with some expense facts."""
        today = date.today()
        _create_fact(db, "fact-1", "Amazon", Decimal("29.99"), today)
        db.get_connection().commit()

        sql = """
            SELECT vendor, event_date, total_amount, currency
            FROM facts
            WHERE event_date >= ?
            AND fact_type = 'purchase'
            ORDER BY event_date DESC
            LIMIT 50
        """
        since_date = today - timedelta(days=7)
        rows = db.fetchall(sql, (since_date.isoformat(),))

        assert len(rows) == 1
        assert rows[0][0] == "Amazon"
        assert float(rows[0][2]) == 29.99


class TestWarrantyHandler:
    """Tests for /warranty command."""

    def test_no_items(self, db, test_space):
        """Test when there are no items with warranty."""
        today = date.today()
        warning_date = today + timedelta(days=90)

        sql = """
            SELECT name, category, warranty_expires, warranty_type, purchase_price, currency
            FROM items
            WHERE warranty_expires IS NOT NULL
            AND warranty_expires <= ?
            AND warranty_expires >= ?
            AND status = 'active'
            ORDER BY warranty_expires ASC
        """

        rows = db.fetchall(sql, (warning_date.isoformat(), today.isoformat()))
        assert len(rows) == 0

    def test_with_expiring_warranty(self, db, test_space):
        """Test with items with expiring warranty."""
        today = date.today()
        expires_soon = today + timedelta(days=30)

        item = Item(
            id="item-1",
            space_id=test_space.id,
            name="Samsung TV",
            category="electronics",
            purchase_date=today - timedelta(days=335),
            purchase_price=Decimal("599.99"),
            currency="EUR",
            status=ItemStatus.ACTIVE,
            warranty_expires=expires_soon,
            warranty_type="manufacturer",
        )

        db.execute(
            """
            INSERT INTO items
            (id, space_id, name, category, purchase_date, purchase_price,
             currency, status, warranty_expires, warranty_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.space_id,
                item.name,
                item.category,
                item.purchase_date.isoformat() if item.purchase_date else None,
                str(item.purchase_price) if item.purchase_price else None,
                item.currency,
                item.status.value,
                item.warranty_expires.isoformat() if item.warranty_expires else None,
                item.warranty_type,
            ),
        )
        db.get_connection().commit()

        # Query should return the item
        warning_date = today + timedelta(days=90)
        sql = """
            SELECT name, category, warranty_expires, warranty_type, purchase_price, currency
            FROM items
            WHERE warranty_expires IS NOT NULL
            AND warranty_expires <= ?
            AND warranty_expires >= ?
            AND status = 'active'
            ORDER BY warranty_expires ASC
        """

        rows = db.fetchall(sql, (warning_date.isoformat(), today.isoformat()))
        assert len(rows) == 1
        assert rows[0][0] == "Samsung TV"
        assert rows[0][1] == "electronics"


class TestFindHandler:
    """Tests for /find command."""

    def test_find_fact(self, db, test_space):
        """Test finding a fact by vendor."""
        today = date.today()
        _create_fact(db, "fact-1", "Amazon", Decimal("29.99"), today)
        db.get_connection().commit()

        sql = """
            SELECT vendor, event_date, total_amount, currency
            FROM facts
            WHERE LOWER(vendor) LIKE ?
            ORDER BY event_date DESC
            LIMIT 10
        """

        rows = db.fetchall(sql, ("%amazon%",))
        assert len(rows) == 1
        assert rows[0][0] == "Amazon"


class TestSummaryHandler:
    """Tests for /summary command."""

    def test_monthly_summary(self, db, test_space):
        """Test monthly summary calculation."""
        today = date.today()
        month_start = date(today.year, today.month, 1)

        # Create some facts for current month
        for i in range(3):
            _create_fact(
                db,
                f"fact-{i}",
                f"Vendor {i}",
                Decimal("50.00"),
                month_start + timedelta(days=i),
            )
        db.get_connection().commit()

        sql = """
            SELECT fact_type, COUNT(*) as count, SUM(total_amount) as total, currency
            FROM facts
            WHERE event_date >= ?
            GROUP BY fact_type, currency
            ORDER BY fact_type
        """

        rows = db.fetchall(sql, (month_start.isoformat(),))
        assert len(rows) == 1
        assert rows[0][0] == "purchase"
        assert rows[0][1] == 3  # count
        assert float(rows[0][2]) == 150.00  # total
