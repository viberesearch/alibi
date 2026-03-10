"""Monthly financial reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager


@dataclass
class CategorySummary:
    """Summary of spending in a category."""

    category: str
    total: Decimal
    count: int
    percentage: float = 0.0

    def __post_init__(self) -> None:
        """Ensure percentage is clamped."""
        self.percentage = max(0.0, min(100.0, self.percentage))


@dataclass
class VendorSummary:
    """Summary of spending at a vendor."""

    vendor: str
    total: Decimal
    count: int
    average: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        """Calculate average."""
        if self.count > 0:
            self.average = self.total / self.count


@dataclass
class MonthlyReport:
    """Monthly spending report."""

    year: int
    month: int
    space_id: str

    # Totals
    total_income: Decimal = Decimal("0")
    total_expenses: Decimal = Decimal("0")
    net_change: Decimal = Decimal("0")

    # Counts
    transaction_count: int = 0
    artifact_count: int = 0

    # Breakdowns
    by_category: list[CategorySummary] = field(default_factory=list)
    by_vendor: list[VendorSummary] = field(default_factory=list)
    top_expenses: list[dict[str, Any]] = field(default_factory=list)

    # Comparisons (vs previous month)
    expense_change_pct: float | None = None
    income_change_pct: float | None = None

    def __post_init__(self) -> None:
        """Calculate net change."""
        self.net_change = self.total_income - self.total_expenses


@dataclass
class WarrantyItem:
    """Item with warranty information."""

    id: str
    name: str
    category: str | None
    warranty_expires: date | None
    warranty_type: str | None
    days_remaining: int | None = None
    status: str = "active"

    def __post_init__(self) -> None:
        """Calculate days remaining."""
        if self.warranty_expires:
            delta = self.warranty_expires - date.today()
            self.days_remaining = delta.days


@dataclass
class InsuranceInventory:
    """Insurance inventory report."""

    space_id: str
    total_value: Decimal = Decimal("0")
    item_count: int = 0
    items: list[dict[str, Any]] = field(default_factory=list)
    by_category: dict[str, Decimal] = field(default_factory=dict)


class ReportGenerator:
    """Generates various financial reports."""

    def __init__(self, db: DatabaseManager) -> None:
        """Initialize report generator.

        Args:
            db: Database manager
        """
        self.db = db

    def generate_monthly_report(
        self,
        year: int,
        month: int,
        space_id: str = "default",
        include_previous: bool = True,
    ) -> MonthlyReport:
        """Generate monthly spending report.

        Args:
            year: Report year
            month: Report month (1-12)
            space_id: Space to report on
            include_previous: Include comparison to previous month

        Returns:
            MonthlyReport with all data
        """
        # Calculate date range
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1)
        else:
            end_date = date(year, month + 1, 1)

        report = MonthlyReport(year=year, month=month, space_id=space_id)

        # Get income and expenses
        totals = self._get_period_totals(space_id, start_date, end_date)
        report.total_income = totals["income"]
        report.total_expenses = totals["expenses"]
        report.transaction_count = totals["count"]
        report.net_change = report.total_income - report.total_expenses

        # Get artifact count
        report.artifact_count = self._get_artifact_count(space_id, start_date, end_date)

        # Get category breakdown
        report.by_category = self._get_category_breakdown(
            space_id, start_date, end_date, report.total_expenses
        )

        # Get vendor breakdown
        report.by_vendor = self._get_vendor_breakdown(space_id, start_date, end_date)

        # Get top expenses
        report.top_expenses = self._get_top_expenses(
            space_id, start_date, end_date, limit=10
        )

        # Compare to previous month
        if include_previous:
            prev_month = month - 1 if month > 1 else 12
            prev_year = year if month > 1 else year - 1
            prev_start = date(prev_year, prev_month, 1)
            prev_end = start_date

            prev_totals = self._get_period_totals(space_id, prev_start, prev_end)

            if prev_totals["expenses"] > 0:
                change = (
                    report.total_expenses - prev_totals["expenses"]
                ) / prev_totals["expenses"]
                report.expense_change_pct = float(change * 100)

            if prev_totals["income"] > 0:
                change = (report.total_income - prev_totals["income"]) / prev_totals[
                    "income"
                ]
                report.income_change_pct = float(change * 100)

        return report

    def _get_period_totals(
        self,
        space_id: str,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        """Get income/expense totals for a period."""
        row = self.db.fetchone(
            """
            SELECT
                COALESCE(SUM(CASE WHEN fact_type = 'refund'
                             THEN total_amount ELSE 0 END), 0) as income,
                COALESCE(SUM(CASE WHEN fact_type IN ('purchase', 'subscription_payment')
                             THEN total_amount ELSE 0 END), 0) as expenses,
                COUNT(*) as count
            FROM facts
            WHERE event_date >= ?
              AND event_date < ?
            """,
            (start_date.isoformat(), end_date.isoformat()),
        )

        if row:
            return {
                "income": Decimal(str(row[0])),
                "expenses": Decimal(str(row[1])),
                "count": int(row[2]),
            }
        return {"income": Decimal("0"), "expenses": Decimal("0"), "count": 0}

    def _get_artifact_count(
        self,
        space_id: str,
        start_date: date,
        end_date: date,
    ) -> int:
        """Get count of documents in period."""
        row = self.db.fetchone(
            """
            SELECT COUNT(*)
            FROM documents
            WHERE created_at >= ?
              AND created_at < ?
            """,
            (start_date.isoformat(), end_date.isoformat()),
        )
        return int(row[0]) if row else 0

    def _get_category_breakdown(
        self,
        space_id: str,
        start_date: date,
        end_date: date,
        total_expenses: Decimal,
    ) -> list[CategorySummary]:
        """Get spending by category."""
        rows = self.db.fetchall(
            """
            SELECT
                COALESCE(fi.category, 'uncategorized') as category,
                SUM(fi.total_price) as total,
                COUNT(*) as count
            FROM fact_items fi
            JOIN facts f ON fi.fact_id = f.id
            WHERE f.fact_type IN ('purchase', 'subscription_payment')
              AND f.event_date >= ?
              AND f.event_date < ?
            GROUP BY COALESCE(fi.category, 'uncategorized')
            ORDER BY total DESC
            """,
            (start_date.isoformat(), end_date.isoformat()),
        )

        result = []
        for row in rows:
            total = Decimal(str(row[1]))
            pct = float(total / total_expenses * 100) if total_expenses > 0 else 0.0
            result.append(
                CategorySummary(
                    category=row[0],
                    total=total,
                    count=int(row[2]),
                    percentage=pct,
                )
            )
        return result

    def _get_vendor_breakdown(
        self,
        space_id: str,
        start_date: date,
        end_date: date,
    ) -> list[VendorSummary]:
        """Get spending by vendor."""
        rows = self.db.fetchall(
            """
            SELECT
                COALESCE(vendor, 'Unknown') as vendor,
                SUM(total_amount) as total,
                COUNT(*) as count
            FROM facts
            WHERE fact_type IN ('purchase', 'subscription_payment')
              AND event_date >= ?
              AND event_date < ?
            GROUP BY COALESCE(vendor, 'Unknown')
            ORDER BY total DESC
            LIMIT 20
            """,
            (start_date.isoformat(), end_date.isoformat()),
        )

        return [
            VendorSummary(
                vendor=row[0],
                total=Decimal(str(row[1])),
                count=int(row[2]),
            )
            for row in rows
        ]

    def _get_top_expenses(
        self,
        space_id: str,
        start_date: date,
        end_date: date,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get top individual expenses."""
        rows = self.db.fetchall(
            """
            SELECT id, vendor, fact_type, total_amount, currency, event_date
            FROM facts
            WHERE fact_type IN ('purchase', 'subscription_payment')
              AND event_date >= ?
              AND event_date < ?
            ORDER BY total_amount DESC
            LIMIT ?
            """,
            (start_date.isoformat(), end_date.isoformat(), limit),
        )

        return [
            {
                "id": row[0],
                "vendor": row[1] or "Unknown",
                "description": row[2] or "",
                "amount": Decimal(str(row[3])),
                "currency": row[4] or "EUR",
                "date": row[5],
            }
            for row in rows
        ]

    def get_warranty_status(
        self,
        space_id: str = "default",
        include_expired: bool = False,
        days_warning: int = 90,
    ) -> list[WarrantyItem]:
        """Get warranty status for all items.

        Args:
            space_id: Space to check
            include_expired: Include items with expired warranties
            days_warning: Days before expiry to flag as warning

        Returns:
            List of items with warranty info
        """
        sql = """
            SELECT id, name, category, warranty_expires, warranty_type, status
            FROM items
            WHERE space_id = ?
              AND warranty_expires IS NOT NULL
        """
        params: list[Any] = [space_id]

        if not include_expired:
            sql += " AND warranty_expires >= date('now')"

        sql += " ORDER BY warranty_expires ASC"

        rows = self.db.fetchall(sql, tuple(params))

        result = []
        for row in rows:
            expires = None
            if row[3]:
                try:
                    expires = date.fromisoformat(row[3])
                except (ValueError, TypeError):
                    pass

            result.append(
                WarrantyItem(
                    id=row[0],
                    name=row[1],
                    category=row[2],
                    warranty_expires=expires,
                    warranty_type=row[4],
                    status=row[5] or "active",
                )
            )

        return result

    def get_insurance_inventory(
        self,
        space_id: str = "default",
    ) -> InsuranceInventory:
        """Get inventory of insured items.

        Args:
            space_id: Space to check

        Returns:
            InsuranceInventory with all insured items
        """
        rows = self.db.fetchall(
            """
            SELECT id, name, category, purchase_price, current_value, currency,
                   warranty_expires, warranty_type
            FROM items
            WHERE space_id = ?
              AND insurance_covered = TRUE
              AND status = 'active'
            ORDER BY COALESCE(current_value, purchase_price) DESC
            """,
            (space_id,),
        )

        inventory = InsuranceInventory(space_id=space_id)

        for row in rows:
            value = (
                Decimal(str(row[4]))
                if row[4]
                else Decimal(str(row[3])) if row[3] else Decimal("0")
            )
            category = row[2] or "uncategorized"

            inventory.items.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "category": category,
                    "value": value,
                    "currency": row[5] or "EUR",
                    "warranty_expires": row[6],
                    "warranty_type": row[7],
                }
            )

            inventory.total_value += value
            inventory.item_count += 1

            if category in inventory.by_category:
                inventory.by_category[category] += value
            else:
                inventory.by_category[category] = value

        return inventory


def format_report_text(report: MonthlyReport) -> str:
    """Format monthly report as text.

    Args:
        report: MonthlyReport to format

    Returns:
        Formatted text report
    """
    month_names = [
        "",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    lines = [
        f"# Monthly Report: {month_names[report.month]} {report.year}",
        "",
        "## Summary",
        "",
        f"- Total Income: {report.total_income:,.2f}",
        f"- Total Expenses: {report.total_expenses:,.2f}",
        f"- Net Change: {report.net_change:+,.2f}",
        f"- Transactions: {report.transaction_count}",
        f"- Documents: {report.artifact_count}",
        "",
    ]

    if report.expense_change_pct is not None:
        direction = "up" if report.expense_change_pct > 0 else "down"
        lines.append(
            f"Expenses {direction} {abs(report.expense_change_pct):.1f}% vs last month"
        )
        lines.append("")

    if report.by_category:
        lines.append("## By Category")
        lines.append("")
        lines.append("| Category | Amount | % |")
        lines.append("|----------|--------|---|")
        for cat in report.by_category[:10]:
            lines.append(
                f"| {cat.category} | {cat.total:,.2f} | {cat.percentage:.1f}% |"
            )
        lines.append("")

    if report.by_vendor:
        lines.append("## Top Vendors")
        lines.append("")
        lines.append("| Vendor | Amount | Count |")
        lines.append("|--------|--------|-------|")
        for v in report.by_vendor[:10]:
            lines.append(f"| {v.vendor} | {v.total:,.2f} | {v.count} |")
        lines.append("")

    if report.top_expenses:
        lines.append("## Top Expenses")
        lines.append("")
        lines.append("| Date | Vendor | Amount |")
        lines.append("|------|--------|--------|")
        for exp in report.top_expenses[:10]:
            lines.append(f"| {exp['date']} | {exp['vendor']} | {exp['amount']:,.2f} |")
        lines.append("")

    return "\n".join(lines)
