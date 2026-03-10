"""Analytics, report, nutrition, and predictions commands."""

from __future__ import annotations

from datetime import datetime

import click
from rich.table import Table

from alibi.commands.shared import console, format_amount
from alibi.config import get_config
from alibi.db.connection import get_db


# ---------------------------------------------------------------------------
# analytics group
# ---------------------------------------------------------------------------


@click.group()
def analytics() -> None:
    """Analytics stack integration."""
    pass


@analytics.command("export")
@click.option(
    "--url",
    envvar="ALIBI_ANALYTICS_STACK_URL",
    default="http://localhost:8070",
    help="Analytics-stack base URL",
)
@click.option("--dry-run", is_flag=True, default=False, help="Show payload stats only")
def analytics_export_cmd(url: str, dry_run: bool) -> None:
    """Push all facts to the analytics stack."""
    from alibi.services.export_analytics import (
        build_export_payload,
        push_to_analytics_stack,
    )

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    if dry_run:
        payload = build_export_payload(db_manager)
        console.print("[bold blue]Dry run -- export payload stats:[/bold blue]")
        console.print(f"  Facts:       {len(payload['facts'])}")
        console.print(f"  Fact items:  {len(payload['fact_items'])}")
        console.print(f"  Annotations: {len(payload['annotations'])}")
        return

    console.print(f"[bold blue]Pushing facts to {url}...[/bold blue]")
    try:
        result = push_to_analytics_stack(db_manager, url)
    except ConnectionError as exc:
        console.print(f"[red]Export failed:[/red] {exc}")
        return

    console.print("[green]Export complete:[/green]")
    console.print(f"  Facts:       {result['facts_count']}")
    console.print(f"  Fact items:  {result['items_count']}")
    console.print(f"  Annotations: {result['annotations_count']}")


@analytics.command("corrections")
@click.option("-l", "--limit", default=1000, help="Max events to analyze")
@click.option("--min-count", default=2, help="Min occurrences to show")
@click.option("--suggestions", is_flag=True, help="Show refinement suggestions")
def analytics_corrections(limit: int, min_count: int, suggestions: bool) -> None:
    """Correction confusion matrix and refinement suggestions."""
    db = get_db()

    if suggestions:
        from alibi.services import get_refinement_suggestions

        items = get_refinement_suggestions(db, limit=limit)
        if not items:
            console.print("[green]No refinement suggestions.[/green]")
            return
        for s in items:
            priority = (
                "[red]HIGH[/red]"
                if s["priority"] == "high"
                else "[yellow]MEDIUM[/yellow]"
            )
            console.print(f"{priority} [{s['type']}] {s['action']}")
        return

    from alibi.services import correction_confusion_matrix

    matrix = correction_confusion_matrix(db, limit=limit, min_count=min_count)
    console.print(f"Total corrections: [bold]{matrix.total_corrections}[/bold]\n")

    if matrix.top_corrected_fields:
        console.print("[bold]Top corrected fields:[/bold]")
        for field_name, count in matrix.top_corrected_fields.items():
            console.print(f"  {field_name}: {count}")
        console.print()

    if matrix.category_confusions:
        console.print("[bold]Category confusions:[/bold]")
        for c in matrix.category_confusions[:10]:
            console.print(f"  {c.original} <-> {c.corrected}: {c.count} occurrences")
        console.print()

    if matrix.vendor_stats:
        console.print("[bold]Vendors with most corrections:[/bold]")
        for v in matrix.vendor_stats[:10]:
            console.print(f"  {v.vendor_name}: {v.total_corrections} corrections")
        console.print()

    if matrix.refinement_candidates:
        console.print("[bold]Refinement candidates:[/bold]")
        for cat in matrix.refinement_candidates:
            console.print(f"  - {cat}")


@analytics.command("locations")
@click.option("-r", "--radius", default=100.0, help="Cluster radius in meters")
def analytics_locations(radius: float) -> None:
    """Spending by location (heatmap data)."""
    db = get_db()
    from alibi.services import location_spending

    results = location_spending(db, cluster_radius_m=radius)
    if not results:
        console.print("[yellow]No location data available.[/yellow]")
        return

    table = Table(title="Spending by Location")
    table.add_column("Place", style="bold")
    table.add_column("Visits", justify="right")
    table.add_column("Total", justify="right", style="green")
    table.add_column("Avg", justify="right")
    table.add_column("Vendors")

    for r in results:
        table.add_row(
            r.place_name or f"{r.lat:.4f},{r.lng:.4f}",
            str(r.visit_count),
            f"{r.total_amount:.2f}",
            f"{r.avg_amount:.2f}",
            ", ".join(r.vendors[:3]),
        )

    console.print(table)


@analytics.command("branches")
@click.option("-v", "--vendor-key", default=None, help="Filter to vendor")
def analytics_branches(vendor_key: str | None) -> None:
    """Compare vendor branches across locations."""
    db = get_db()
    from alibi.services import vendor_branches

    results = vendor_branches(db, vendor_key=vendor_key)
    if not results:
        console.print("[yellow]No multi-branch vendors found.[/yellow]")
        return

    for r in results:
        console.print(
            f"\n[bold]{r.vendor_name}[/bold] "
            f"({r.branch_count} branches, total: {r.total_spent:.2f})"
        )
        for b in r.branches:
            marker = ""
            if r.most_visited and b.map_url == r.most_visited.map_url:
                marker += " [blue]most-visited[/blue]"
            if r.highest_avg and b.map_url == r.highest_avg.map_url:
                marker += " [green]highest-avg[/green]"
            console.print(
                f"  {b.place_name or 'Unknown'}: "
                f"{b.visit_count} visits, avg {b.avg_basket:.2f}{marker}"
            )


@analytics.command("nearby")
@click.argument("lat", type=float)
@click.argument("lng", type=float)
@click.option("-r", "--radius", default=2000.0, help="Radius in meters")
@click.option("-l", "--limit", default=10, help="Max suggestions")
def analytics_nearby(lat: float, lng: float, radius: float, limit: int) -> None:
    """Suggest vendors near a location."""
    db = get_db()
    from alibi.services import nearby_vendors

    results = nearby_vendors(db, lat, lng, radius_m=radius, limit=limit)
    if not results:
        console.print("[yellow]No vendors found nearby.[/yellow]")
        return

    for s in results:
        console.print(
            f"[bold]{s.vendor_name}[/bold] ({s.distance_meters:.0f}m away) "
            f"- {s.visit_count} visits, avg {s.avg_basket:.2f}"
        )
        console.print(f"  {s.reason}")


@analytics.command("cloud-quality")
@click.option("--vendors", "-v", default=20, help="Max vendors in breakdown")
@click.option("--months", "-m", default=12, help="Max months in trend")
def analytics_cloud_quality(vendors: int, months: int) -> None:
    """Cloud formation quality metrics and accuracy report."""
    db = get_db()
    from alibi.services import cloud_quality_report

    report = cloud_quality_report(db, limit_vendors=vendors, limit_trends=months)

    console.print(f"[bold]Cloud Formation Quality Report[/bold]\n")
    console.print(f"  Total clouds: {report.total_clouds}")
    console.print(f"  Total facts: {report.total_facts}")
    console.print(f"  Total corrections: {report.total_corrections}")
    console.print(f"  Overall accuracy: {report.overall_accuracy:.1%}")
    console.print(f"  False positive rate: {report.false_positive_rate:.1%}")

    if report.size_distribution:
        sd = report.size_distribution
        console.print(f"\n[bold]Cloud Size Distribution[/bold]")
        console.print(f"  Single-bundle: {sd.single_bundle}")
        console.print(f"  Two-bundle: {sd.two_bundles}")
        console.print(f"  Three+: {sd.three_plus}")
        console.print(f"  Avg bundles/cloud: {sd.avg_bundles}")
        console.print(f"  Max bundles: {sd.max_bundles}")

    if report.match_type_stats:
        console.print(f"\n[bold]Match Type Effectiveness[/bold]")
        for mt in report.match_type_stats:
            console.print(
                f"  {mt.match_type}: {mt.total_uses} uses, "
                f"avg conf {mt.avg_confidence:.2f}, "
                f"correction rate {mt.correction_rate:.1%}"
            )

    if report.vendor_accuracy:
        console.print(f"\n[bold]Top Vendors by Corrections[/bold]")
        for va in report.vendor_accuracy[:10]:
            if va.correction_count == 0:
                continue
            name = va.vendor_name or va.vendor_key[:12]
            console.print(
                f"  {name}: {va.correction_count} corrections, "
                f"accuracy {va.accuracy_rate:.1%}, "
                f"FP {va.false_positives}"
            )

    if report.top_false_positive_pairs:
        console.print(f"\n[bold]Top False Positive Pairs[/bold]")
        for a, b, cnt in report.top_false_positive_pairs:
            console.print(f"  {a[:12]} <-> {b[:12]}: {cnt} times")

    if report.trends:
        console.print(f"\n[bold]Monthly Trends[/bold]")
        for t in report.trends[-6:]:
            console.print(
                f"  {t.period}: {t.total_corrections} corrections "
                f"({t.false_positives} FP), "
                f"{t.facts_created} facts, "
                f"rate {t.correction_rate:.1%}"
            )


# ---------------------------------------------------------------------------
# report group
# ---------------------------------------------------------------------------


@click.group()
def report() -> None:
    """Generate financial reports."""
    pass


@report.command("monthly")
@click.option("--year", "-y", type=int, help="Report year (default: current)")
@click.option("--month", "-m", type=int, help="Report month 1-12 (default: current)")
@click.option("--space", "-s", default="default", help="Space to report on")
@click.option("--output", "-o", type=click.Path(), help="Save report to file")
def report_monthly(
    year: int | None, month: int | None, space: str, output: str | None
) -> None:
    """Generate monthly spending report."""
    from alibi.reports.monthly import ReportGenerator, format_report_text

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    # Default to current month
    today = datetime.now()
    report_year = year or today.year
    report_month = month or today.month

    if report_month < 1 or report_month > 12:
        console.print("[red]Month must be between 1 and 12.[/red]")
        return

    console.print(
        f"[bold blue]Generating report for {report_month}/{report_year}...[/bold blue]"
    )

    generator = ReportGenerator(db_manager)
    monthly = generator.generate_monthly_report(report_year, report_month, space)

    # Format as text
    report_text = format_report_text(monthly)

    if output:
        from pathlib import Path

        Path(output).write_text(report_text)
        console.print(f"[green]Report saved to:[/green] {output}")
    else:
        console.print(report_text)


@report.command("warranty")
@click.option("--space", "-s", default="default", help="Space to check")
@click.option("--days", "-d", default=90, help="Warning threshold in days")
@click.option("--expired", is_flag=True, help="Include expired warranties")
def report_warranty(space: str, days: int, expired: bool) -> None:
    """Show warranty status for items."""
    from alibi.reports.monthly import ReportGenerator

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    generator = ReportGenerator(db_manager)
    items = generator.get_warranty_status(
        space, include_expired=expired, days_warning=days
    )

    if not items:
        console.print("[yellow]No items with warranty information found.[/yellow]")
        return

    table = Table(title="Warranty Status")
    table.add_column("Item", style="cyan")
    table.add_column("Category")
    table.add_column("Expires")
    table.add_column("Days Left", justify="right")
    table.add_column("Type")

    for item in items:
        expires_str = (
            item.warranty_expires.isoformat() if item.warranty_expires else "N/A"
        )

        if item.days_remaining is not None:
            if item.days_remaining < 0:
                days_str = f"[red]Expired[/red]"
            elif item.days_remaining <= days:
                days_str = f"[yellow]{item.days_remaining}[/yellow]"
            else:
                days_str = f"[green]{item.days_remaining}[/green]"
        else:
            days_str = "N/A"

        table.add_row(
            item.name[:30],
            item.category or "",
            expires_str,
            days_str,
            item.warranty_type or "",
        )

    console.print(table)


@report.command("insurance")
@click.option("--space", "-s", default="default", help="Space to check")
def report_insurance(space: str) -> None:
    """Show insurance inventory."""
    from alibi.reports.monthly import ReportGenerator

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    generator = ReportGenerator(db_manager)
    inventory = generator.get_insurance_inventory(space)

    if not inventory.items:
        console.print("[yellow]No insured items found.[/yellow]")
        return

    console.print(f"[bold]Insurance Inventory[/bold]")
    console.print(f"Total Value: [green]{inventory.total_value:,.2f}[/green]")
    console.print(f"Items: {inventory.item_count}")
    console.print()

    # By category
    if inventory.by_category:
        cat_table = Table(title="Value by Category")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Value", justify="right")

        for cat, value in sorted(
            inventory.by_category.items(), key=lambda x: x[1], reverse=True
        ):
            cat_table.add_row(cat, f"{value:,.2f}")

        console.print(cat_table)
        console.print()

    # Item list
    item_table = Table(title="Insured Items")
    item_table.add_column("Item", style="cyan")
    item_table.add_column("Category")
    item_table.add_column("Value", justify="right")
    item_table.add_column("Warranty")

    for item in inventory.items:
        warranty = item.get("warranty_expires", "")
        item_table.add_row(
            item["name"][:30],
            item.get("category", ""),
            f"{item['value']:,.2f}",
            str(warranty) if warranty else "",
        )

    console.print(item_table)


@report.command("recurring")
@click.option("--min-occurrences", "-n", default=3, help="Minimum occurrences")
@click.option("--min-confidence", "-c", default=0.5, help="Minimum confidence (0-1)")
@click.option("--days", "-d", default=30, help="Days to look ahead for upcoming")
def report_recurring(min_occurrences: int, min_confidence: float, days: int) -> None:
    """Detect and display recurring transactions.

    Analyzes fact history to find recurring patterns like
    subscriptions, rent, utilities, etc.
    """
    from alibi.services.analytics import (
        detect_subscriptions,
        get_upcoming_subscriptions,
    )

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    console.print("[bold blue]Detecting recurring transactions...[/bold blue]\n")

    patterns = detect_subscriptions(
        db_manager,
        min_occurrences=min_occurrences,
        min_confidence=min_confidence,
    )

    if not patterns:
        console.print("[yellow]No recurring patterns detected.[/yellow]")
        console.print("Try lowering --min-occurrences or --min-confidence.")
        return

    # Show detected patterns
    table = Table(title=f"Recurring Transactions ({len(patterns)} found)")
    table.add_column("Vendor", style="cyan")
    table.add_column("Amount", justify="right")
    table.add_column("Frequency", justify="right")
    table.add_column("Confidence", justify="right", style="green")
    table.add_column("Last", justify="right")
    table.add_column("Next", justify="right")

    for p in patterns:
        freq_str = p.period_type.replace("_", " ").title()
        if freq_str == "Irregular":
            freq_str = f"{p.frequency_days}d"

        conf_style = "green" if p.confidence >= 0.8 else "yellow"
        conf_str = f"[{conf_style}]{p.confidence:.0%}[/{conf_style}]"

        table.add_row(
            p.vendor[:30],
            f"{p.avg_amount:.2f}",
            freq_str,
            conf_str,
            p.last_date.isoformat(),
            p.next_expected.isoformat(),
        )

    console.print(table)

    # Show upcoming
    upcoming = get_upcoming_subscriptions(patterns, days_ahead=days)
    if upcoming:
        console.print()
        upcoming_table = Table(title=f"Upcoming in Next {days} Days")
        upcoming_table.add_column("Date", style="cyan")
        upcoming_table.add_column("Vendor")
        upcoming_table.add_column("Expected Amount", justify="right")

        for pattern, expected_date in upcoming[:10]:
            upcoming_table.add_row(
                expected_date.isoformat(),
                pattern.vendor[:30],
                f"{pattern.avg_amount:.2f}",
            )

        console.print(upcoming_table)

    # Summary
    monthly_total = sum(float(p.avg_amount) * (30 / p.frequency_days) for p in patterns)
    console.print()
    console.print(
        f"[bold]Estimated monthly recurring:[/bold] [green]{monthly_total:.2f}[/green]"
    )


@report.command("trends")
@click.option("--months", "-m", default=6, help="Number of months to analyze")
def report_trends(months: int) -> None:
    """Show spending trends over time.

    Analyzes spending patterns, category trends, and savings rate.
    """
    from alibi.services.analytics import analyze_spending_patterns

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    console.print(f"[bold blue]Analyzing {months} months of spending...[/bold blue]\n")

    insights = analyze_spending_patterns(db_manager, months=months)

    if not insights.monthly_trends:
        console.print("[yellow]No transaction data found.[/yellow]")
        return

    # Monthly overview
    monthly_table = Table(title="Monthly Overview")
    monthly_table.add_column("Month", style="cyan")
    monthly_table.add_column("Income", justify="right", style="green")
    monthly_table.add_column("Expenses", justify="right", style="red")
    monthly_table.add_column("Net", justify="right")
    monthly_table.add_column("Txns", justify="right", style="dim")

    for trend in insights.monthly_trends:
        net_style = "green" if trend.net >= 0 else "red"
        monthly_table.add_row(
            trend.month,
            f"{trend.total_income:.2f}",
            f"{trend.total_expenses:.2f}",
            f"[{net_style}]{trend.net:.2f}[/{net_style}]",
            str(trend.transaction_count),
        )

    console.print(monthly_table)

    # Savings rate
    savings_pct = insights.savings_rate * 100
    savings_style = "green" if savings_pct >= 0 else "red"
    console.print()
    console.print(
        f"[bold]Savings Rate:[/bold] [{savings_style}]{savings_pct:.1f}%[/{savings_style}]"
    )

    # Category trends
    if insights.category_trends:
        console.print()
        cat_table = Table(title="Top Categories (by avg monthly)")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Avg Monthly", justify="right")
        cat_table.add_column("Trend", justify="center")

        trend_icons = {
            "increasing": "[red]^[/red]",
            "decreasing": "[green]v[/green]",
            "stable": "[dim]-[/dim]",
        }

        for cat_trend in insights.category_trends[:10]:
            cat_table.add_row(
                cat_trend.category[:25],
                f"{cat_trend.avg_monthly:.2f}",
                trend_icons.get(cat_trend.trend_direction, "-"),
            )

        console.print(cat_table)

    # Notable changes
    if insights.biggest_increase_category or insights.biggest_decrease_category:
        console.print()
        console.print("[bold]Notable Changes:[/bold]")
        if insights.biggest_increase_category:
            console.print(
                f"  [red]Biggest increase:[/red] {insights.biggest_increase_category}"
            )
        if insights.biggest_decrease_category:
            console.print(
                f"  [green]Biggest decrease:[/green] {insights.biggest_decrease_category}"
            )


@report.command("anomalies")
@click.option("--days", "-d", default=90, help="Days of history for baseline")
@click.option(
    "--threshold", "-t", default=2.0, help="Std deviations to flag as anomaly"
)
def report_anomalies(days: int, threshold: float) -> None:
    """Detect unusual spending patterns.

    Finds facts that deviate significantly from normal patterns.
    """
    from alibi.services.analytics import detect_anomalies

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    console.print("[bold blue]Detecting spending anomalies...[/bold blue]\n")

    anomalies = detect_anomalies(
        db_manager,
        lookback_days=days,
        std_threshold=threshold,
    )

    if not anomalies:
        console.print("[green]No anomalies detected.[/green]")
        console.print("Try lowering --threshold for more sensitive detection.")
        return

    table = Table(title=f"Spending Anomalies ({len(anomalies)} found)")
    table.add_column("Date", style="cyan")
    table.add_column("Vendor")
    table.add_column("Amount", justify="right", style="red")
    table.add_column("Severity", justify="right")
    table.add_column("Type")
    table.add_column("Explanation", max_width=35)

    for a in anomalies:
        severity_style = "red" if a.severity > 0.6 else "yellow"
        table.add_row(
            a.date.isoformat() if a.date else "",
            (a.vendor or "Unknown")[:25],
            f"{a.amount:.2f}",
            f"[{severity_style}]{a.severity:.0%}[/{severity_style}]",
            a.anomaly_type,
            a.explanation[:35] if a.explanation else "",
        )

    console.print(table)

    # Summary by type
    console.print()
    type_counts: dict[str, int] = {}
    for a in anomalies:
        type_counts[a.anomaly_type] = type_counts.get(a.anomaly_type, 0) + 1

    console.print("[bold]By Type:[/bold]")
    for anomaly_type, count in sorted(
        type_counts.items(), key=lambda x: x[1], reverse=True
    ):
        console.print(f"  {anomaly_type}: {count}")


@report.command("compare")
@click.argument("month1")
@click.argument("month2")
def report_compare(month1: str, month2: str) -> None:
    """Compare spending between two months.

    MONTH1 and MONTH2 should be in YYYY-MM format.

    Example: lt report compare 2024-01 2024-02
    """
    from datetime import date

    from alibi.services.analytics import compare_periods

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    # Parse months
    try:
        year1, mon1 = map(int, month1.split("-"))
        year2, mon2 = map(int, month2.split("-"))
    except ValueError:
        console.print("[red]Invalid month format. Use YYYY-MM.[/red]")
        return

    # Get full month date ranges
    from calendar import monthrange

    p1_start = date(year1, mon1, 1)
    p1_end = date(year1, mon1, monthrange(year1, mon1)[1])
    p2_start = date(year2, mon2, 1)
    p2_end = date(year2, mon2, monthrange(year2, mon2)[1])

    console.print(f"[bold blue]Comparing {month1} vs {month2}[/bold blue]\n")

    comparison = compare_periods(
        db_manager,
        p1_start,
        p1_end,
        p2_start,
        p2_end,
    )

    # Overview table
    overview = Table(title="Period Comparison")
    overview.add_column("Metric", style="cyan")
    overview.add_column(month1, justify="right")
    overview.add_column(month2, justify="right")
    overview.add_column("Change", justify="right")

    p1 = comparison["period1"]
    p2 = comparison["period2"]
    changes = comparison["changes"]

    exp_style = "red" if changes["expense_change_pct"] > 0 else "green"
    inc_style = "green" if changes["income_change_pct"] > 0 else "red"

    overview.add_row(
        "Expenses",
        f"{p1['total_expenses']:.2f}",
        f"{p2['total_expenses']:.2f}",
        f"[{exp_style}]{changes['expense_change_pct']:+.1f}%[/{exp_style}]",
    )
    overview.add_row(
        "Income",
        f"{p1['total_income']:.2f}",
        f"{p2['total_income']:.2f}",
        f"[{inc_style}]{changes['income_change_pct']:+.1f}%[/{inc_style}]",
    )
    overview.add_row(
        "Transactions",
        str(p1["transaction_count"]),
        str(p2["transaction_count"]),
        "",
    )

    console.print(overview)

    # Category changes
    cat_changes = changes.get("category_changes", [])
    if cat_changes:
        console.print()
        cat_table = Table(title="Category Changes")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column(month1, justify="right")
        cat_table.add_column(month2, justify="right")
        cat_table.add_column("Change", justify="right")

        for cat in cat_changes:
            change_style = "red" if cat["change_pct"] > 0 else "green"
            cat_table.add_row(
                cat["category"][:20],
                f"{cat['period1']:.2f}",
                f"{cat['period2']:.2f}",
                f"[{change_style}]{cat['change_pct']:+.1f}%[/{change_style}]",
            )

        console.print(cat_table)


# ---------------------------------------------------------------------------
# nutrition group
# ---------------------------------------------------------------------------


@click.group()
def nutrition() -> None:
    """Nutritional analytics from purchased items."""
    pass


@nutrition.command("summary")
@click.option(
    "--period",
    "-p",
    default="month",
    show_default=True,
    type=click.Choice(["day", "week", "month"]),
    help="Grouping period",
)
@click.option("--start", "start_date", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", "end_date", default=None, help="End date (YYYY-MM-DD)")
def nutrition_summary_cmd(
    period: str, start_date: str | None, end_date: str | None
) -> None:
    """Show nutritional summary from purchased items with OFF data.

    Aggregates energy, macronutrients, and nutriscore distribution
    grouped by the selected period.  Only items with a barcode that
    has a cached Open Food Facts entry are included.
    """
    from alibi.analytics.nutrition import nutrition_summary

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    result = nutrition_summary(
        db, start_date=start_date, end_date=end_date, period=period
    )
    periods = result.get("periods", [])

    if not periods:
        console.print("[dim]No nutritional data found for the given filters.[/dim]")
        return

    table = Table(title=f"Nutrition Summary (by {period})")
    table.add_column("Period", style="bold")
    table.add_column("Items w/ Nutrition", justify="right")
    table.add_column("Total Items", justify="right")
    table.add_column("Coverage %", justify="right")
    table.add_column("Energy kcal", justify="right")
    table.add_column("Sugars g", justify="right")
    table.add_column("Proteins g", justify="right")
    table.add_column("Nutriscore (A-E)")

    for p in periods:
        totals = p.get("totals") or {}
        ns = p.get("nutriscore_distribution") or {}
        ns_str = " ".join(f"{g.upper()}:{c}" for g, c in sorted(ns.items())) or "-"
        energy = totals.get("energy_kcal", 0.0)
        energy_str = f"{energy:,.0f}" if energy else "-"
        sugars = totals.get("sugars_g", 0.0)
        sugars_str = f"{sugars:,.1f}" if sugars else "-"
        proteins = totals.get("proteins_g", 0.0)
        proteins_str = f"{proteins:,.1f}" if proteins else "-"
        table.add_row(
            p["period"],
            str(p["items_with_nutrition"]),
            str(p["items_total"]),
            f"{p['coverage_pct']:.1f}%",
            energy_str,
            sugars_str,
            proteins_str,
            ns_str,
        )

    console.print(table)

    top_sugar = result.get("top_sugar_items", [])
    top_calorie = result.get("top_calorie_items", [])

    if top_sugar:
        console.print("\n[bold]Top 10 by Sugar Content:[/bold]")
        for item in top_sugar:
            console.print(
                f"  {item['name'][:40]:<40} {item['sugars_g']:>8.1f} g sugars"
            )

    if top_calorie:
        console.print("\n[bold]Top 10 by Calories:[/bold]")
        for item in top_calorie:
            console.print(f"  {item['name'][:40]:<40} {item['energy_kcal']:>8.0f} kcal")


@nutrition.command("item")
@click.argument("fact_item_id")
def nutrition_item_cmd(fact_item_id: str) -> None:
    """Show nutritional data for a single fact item.

    Looks up the item's barcode in the product cache and prints
    per-100g nutriments plus computed totals when unit weight is known.
    """
    from alibi.analytics.nutrition import item_nutrition

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    result = item_nutrition(db, fact_item_id)
    if result is None:
        console.print(f"[yellow]Fact item '{fact_item_id}' not found.[/yellow]")
        return

    error = result.get("error")
    if error == "no_barcode":
        console.print(f"[yellow]Item '{result['item_name']}' has no barcode.[/yellow]")
        return
    if error == "not_cached":
        console.print(
            f"[yellow]Barcode {result['barcode']} not in product cache. "
            "Run 'lt enrich pending' first.[/yellow]"
        )
        return
    if error == "not_found_in_off":
        console.print(
            f"[yellow]Barcode {result['barcode']} was not found in Open Food Facts.[/yellow]"
        )
        return

    table = Table(title=f"Nutrition: {result.get('item_name', fact_item_id)}")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Item ID", result["item_id"])
    table.add_row("Item Name", result.get("item_name") or "-")
    table.add_row("Barcode", result.get("barcode") or "-")
    table.add_row("Brand", result.get("brand") or "-")
    table.add_row("Category", result.get("category") or "-")
    table.add_row("Product Name (OFF)", result.get("product_name") or "-")
    table.add_row("Product Quantity", result.get("product_quantity") or "-")
    grade = result.get("nutriscore_grade") or "-"
    table.add_row("Nutriscore", grade.upper())

    console.print(table)

    per100 = result.get("nutriments_per_100g") or {}
    totals = result.get("totals")

    if per100:
        nut_table = Table(title="Nutriments per 100 g")
        nut_table.add_column("Nutriment", style="bold")
        nut_table.add_column("Per 100g", justify="right")
        if totals:
            nut_table.add_column("Purchase Total", justify="right")

        label_map = {
            "energy_kcal": "Energy (kcal)",
            "fat_g": "Fat (g)",
            "saturated_fat_g": "Saturated Fat (g)",
            "carbohydrates_g": "Carbohydrates (g)",
            "sugars_g": "Sugars (g)",
            "fiber_g": "Fiber (g)",
            "proteins_g": "Proteins (g)",
            "salt_g": "Salt (g)",
        }

        for key, label in label_map.items():
            val = per100.get(key)
            if val is None:
                continue
            per100_str = f"{val:.2f}"
            if totals:
                total_val = totals.get(key, 0.0)
                nut_table.add_row(label, per100_str, f"{total_val:.2f}")
            else:
                nut_table.add_row(label, per100_str)

        console.print(nut_table)

        if not totals:
            console.print(
                "[dim]Note: purchase totals not available -- "
                "unit weight (unit_quantity) is missing for this item.[/dim]"
            )


# ---------------------------------------------------------------------------
# predictions group
# ---------------------------------------------------------------------------


@click.group()
def predictions() -> None:
    """MindsDB spending predictions and category inference."""
    pass


@predictions.command("status")
def predictions_status_cmd() -> None:
    """Show status of all predictor models."""
    from alibi.services.predictions import list_models

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    try:
        models = list_models(db)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    if not models:
        console.print("[dim]No models found.[/dim]")
        return

    table = Table(title="MindsDB Predictor Models")
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Predict")
    table.add_column("Engine")

    for m in models:
        status = m.get("status", "unknown")
        style = (
            "green"
            if status == "complete"
            else "yellow" if status == "training" else "red"
        )
        table.add_row(
            m.get("name", ""),
            f"[{style}]{status}[/{style}]",
            m.get("predict", ""),
            m.get("engine", ""),
        )

    console.print(table)


@predictions.command("train-forecast")
@click.option(
    "--window", default=6, show_default=True, help="Historical months per prediction"
)
@click.option(
    "--horizon", default=3, show_default=True, help="Future months to predict"
)
def predictions_train_forecast_cmd(window: int, horizon: int) -> None:
    """Train the spending forecast model."""
    from alibi.services.predictions import train_spending_forecast

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    console.print(
        f"[bold blue]Training spending forecast "
        f"(window={window}, horizon={horizon})...[/bold blue]"
    )

    try:
        result = train_spending_forecast(db, window=window, horizon=horizon)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        return

    console.print(
        f"[green]Training complete:[/green] {result['model']} -> {result['status']}"
    )


@predictions.command("train-category")
def predictions_train_category_cmd() -> None:
    """Train the category classifier model."""
    from alibi.services.predictions import train_category_classifier

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    console.print("[bold blue]Training category classifier...[/bold blue]")

    try:
        result = train_category_classifier(db)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        return

    console.print(
        f"[green]Training complete:[/green] {result['model']} -> {result['status']}"
    )


@predictions.command("forecast")
@click.option("--months", "-m", default=3, show_default=True, help="Months to forecast")
@click.option("--category", "-c", default=None, help="Filter by category")
def predictions_forecast_cmd(months: int, category: str | None) -> None:
    """Show spending forecast predictions."""
    from alibi.services.predictions import get_spending_forecast

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    try:
        results = get_spending_forecast(db, months=months, category=category)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    if not results:
        console.print("[dim]No forecast data available.[/dim]")
        return

    table = Table(title=f"Spending Forecast ({months} months)")
    table.add_column("Date", style="bold")
    table.add_column("Category")
    table.add_column("Forecast", justify="right")
    table.add_column("Confidence", justify="right")

    for r in results:
        conf = r.get("confidence")
        conf_str = f"{conf:.2f}" if conf is not None else "-"
        amt = r.get("forecast_amount")
        amt_str = f"{amt:.2f}" if amt is not None else "-"
        table.add_row(
            str(r.get("forecast_date", "")),
            r.get("category", ""),
            amt_str,
            conf_str,
        )

    console.print(table)


@predictions.command("classify")
@click.argument("vendor")
@click.argument("description")
@click.argument("amount", type=float)
def predictions_classify_cmd(vendor: str, description: str, amount: float) -> None:
    """Predict category for an item: lt predictions classify VENDOR DESCRIPTION AMOUNT."""
    from alibi.services.predictions import classify_category

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    try:
        result = classify_category(db, vendor, description, amount)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    console.print(f"[bold]Category:[/bold] {result.get('category', 'unknown')}")
    conf = result.get("category_confidence", 0.0)
    console.print(f"[bold]Confidence:[/bold] {conf:.2f}")


@predictions.command("classify-pending")
@click.option(
    "--limit", "-l", default=100, show_default=True, help="Max items to classify"
)
@click.option(
    "--min-confidence", default=0.5, show_default=True, help="Min confidence threshold"
)
@click.option(
    "--apply", is_flag=True, default=False, help="Apply predictions to database"
)
def predictions_classify_pending_cmd(
    limit: int, min_confidence: float, apply: bool
) -> None:
    """Classify uncategorized items using the trained model."""
    from alibi.services.predictions import classify_uncategorized

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    try:
        results = classify_uncategorized(db, limit=limit, min_confidence=min_confidence)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    if not results:
        console.print(
            "[dim]No items to classify or no predictions above threshold.[/dim]"
        )
        return

    table = Table(title=f"Category Predictions ({len(results)} items)")
    table.add_column("Item ID", style="dim")
    table.add_column("Vendor")
    table.add_column("Item")
    table.add_column("Category", style="bold")
    table.add_column("Confidence", justify="right")

    for r in results:
        table.add_row(
            r.get("item_id", "")[:8],
            r.get("vendor_name", ""),
            r.get("item_name", ""),
            r.get("category", ""),
            f"{r.get('confidence', 0):.2f}",
        )

    console.print(table)

    if apply:
        from alibi.services.correction import update_fact_item

        applied = 0
        for r in results:
            try:
                update_fact_item(db, r["item_id"], {"category": r["category"]})
                applied += 1
            except Exception as exc:
                console.print(f"[red]Failed to update {r['item_id']}: {exc}[/red]")

        console.print(
            f"[green]Applied {applied}/{len(results)} category predictions.[/green]"
        )
    else:
        console.print("[dim]Use --apply to save predictions to database.[/dim]")
