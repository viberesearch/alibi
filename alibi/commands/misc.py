"""Miscellaneous top-level commands: health, init, setup, status, version,
completion, query, transactions, match, search, reingest, template, review,
verify, export, import, daemon, mycelium, telegram, vectordb."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from typing import Any

import click
from rich.table import Table

from alibi.commands.shared import console, format_amount, is_quiet, is_verbose
from alibi.config import get_config
from alibi.db.connection import get_db
from alibi.errors import (
    NO_INBOX_CONFIGURED,
    UNSUPPORTED_FILE_TYPE,
    VAULT_NOT_FOUND,
    format_error,
)


# ---------------------------------------------------------------------------
# Standalone commands
# ---------------------------------------------------------------------------


@click.command()
def health() -> None:
    """Check health of all services."""
    from alibi.health import check_health, get_available_models

    console.print("[bold blue]Alibi Health Check[/bold blue]\n")

    health_status = check_health(check_model=True)

    # Ollama status
    if health_status.ollama_available:
        console.print(
            f"[green]✓[/green] Ollama: Connected ({health_status.ollama_url})"
        )
        if health_status.ollama_model_loaded:
            console.print(
                f"[green]✓[/green] Model: {health_status.ollama_model} available"
            )
        else:
            console.print(f"[red]✗[/red] Model: {health_status.ollama_model} not found")
            models = get_available_models()
            if models:
                console.print(f"    Available: {', '.join(models[:5])}")
            console.print(f"    Run: ollama pull {health_status.ollama_model}")
    else:
        console.print(
            f"[red]✗[/red] Ollama: Not available ({health_status.ollama_url})"
        )
        console.print("    Run: ollama serve")

    # Database status
    if health_status.database_accessible:
        console.print(f"[green]✓[/green] Database: {health_status.database_path}")
    else:
        console.print(f"[red]✗[/red] Database: Not initialized")
        console.print("    Run: lt init")

    # Vault status
    if health_status.vault_path and health_status.vault_path != "Not configured":
        if health_status.vault_exists:
            console.print(f"[green]✓[/green] Vault: {health_status.vault_path}")
        else:
            console.print(
                f"[yellow]⚠[/yellow] Vault: {health_status.vault_path} (not found)"
            )
    else:
        console.print(f"[yellow]⚠[/yellow] Vault: Not configured")

    # Summary
    console.print()
    if health_status.healthy:
        console.print("[bold green]All services healthy![/bold green]")
    else:
        console.print("[bold red]Some services need attention.[/bold red]")
        if health_status.errors:
            console.print("\n[bold]Issues:[/bold]")
            for error in health_status.errors:
                console.print(f"  - {error}")


@click.command()
def init() -> None:
    """Initialize the Alibi database and configuration."""
    from alibi.health import check_health

    config = get_config()
    db_instance = get_db()

    console.print("[bold blue]Initializing Alibi...[/bold blue]")

    # Check health first
    health_status = check_health(check_model=False)
    if not health_status.ollama_available:
        console.print(
            f"[yellow]Warning:[/yellow] Ollama not available at {config.ollama_url}"
        )
        console.print("  Vision extraction will not work until Ollama is running.")
        console.print("  Start with: ollama serve\n")

    # Validate configuration
    errors = config.validate_paths()
    if errors:
        for error in errors:
            console.print(f"[yellow]Warning:[/yellow] {error}")

    # Initialize database
    try:
        db_instance.initialize()
        console.print(f"[green]Database initialized at:[/green] {db_instance.db_path}")
    except Exception as e:
        console.print(f"[red]Error initializing database:[/red] {e}")
        raise click.Abort() from e

    # Show configuration
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Database: {db_instance.db_path}")
    console.print(f"  Vault: {config.vault_path or 'Not configured'}")
    console.print(f"  Ollama: {config.ollama_url}")
    console.print(f"  Model: {config.ollama_model}")

    console.print("\n[green]Alibi initialized successfully![/green]")


@click.command()
def setup() -> None:
    """Interactive first-time setup wizard."""
    from pathlib import Path

    from rich.panel import Panel

    from alibi.health import check_health

    console.print(Panel("[bold blue]Alibi Setup Wizard[/bold blue]"))

    # Step 1: Check/configure Ollama
    console.print("\n[bold]Step 1: Ollama Configuration[/bold]")
    health_result = check_health(check_model=False)

    if not health_result.ollama_available:
        console.print("[yellow]Ollama not available[/yellow]")
        console.print("  Install: brew install ollama")
        console.print("  Start: ollama serve")
        if not click.confirm("Continue anyway?", default=True):
            return
    else:
        console.print("[green]Ollama connected[/green]")

        # Check model
        health_full = check_health(check_model=True)
        if not health_full.ollama_model_loaded:
            model = click.prompt("Ollama model", default="qwen3-vl:30b")
            console.print(f"  Pull with: ollama pull {model}")
        else:
            console.print(f"  Model: {health_full.ollama_model}")

    # Step 2: Configure vault path
    console.print("\n[bold]Step 2: Vault Configuration[/bold]")
    default_vault = Path.home() / "Documents" / "alibi-vault"
    vault_path = click.prompt(
        "Vault path",
        default=str(default_vault),
        type=click.Path(),
    )
    vault_path = Path(vault_path)

    if not vault_path.exists():
        if click.confirm(f"Create {vault_path}?", default=True):
            vault_path.mkdir(parents=True, exist_ok=True)
            (vault_path / "inbox").mkdir(exist_ok=True)
            (vault_path / "inbox" / "documents").mkdir(exist_ok=True)
            console.print(f"[green]Created {vault_path}[/green]")

    # Step 3: Generate .env file
    console.print("\n[bold]Step 3: Configuration File[/bold]")
    env_path = Path(".env")
    config = get_config()
    env_content = f"""# Alibi Configuration
ALIBI_VAULT_PATH={vault_path}
ALIBI_OLLAMA_URL={config.ollama_url}
ALIBI_OLLAMA_MODEL={config.ollama_model}
ALIBI_DEFAULT_CURRENCY={config.default_currency}
"""

    if env_path.exists():
        if click.confirm(".env exists. Overwrite?", default=False):
            env_path.write_text(env_content)
            console.print("[green]Updated .env[/green]")
        else:
            console.print("[dim]Skipped .env update[/dim]")
    else:
        env_path.write_text(env_content)
        console.print("[green]Created .env[/green]")

    # Step 4: Initialize database
    console.print("\n[bold]Step 4: Database Initialization[/bold]")
    db_instance = get_db()
    try:
        db_instance.initialize()
        console.print(f"[green]Database initialized at:[/green] {db_instance.db_path}")
    except Exception as e:
        console.print(f"[red]Error initializing database:[/red] {e}")

    # Summary
    console.print("\n" + "=" * 50)
    console.print("[bold green]Setup complete![/bold green]")
    console.print("\nNext steps:")
    console.print(f"  1. Drop receipts in: {vault_path / 'inbox' / 'documents'}")
    console.print("  2. Process with: lt process")
    console.print("  3. Check status: lt status")


@click.command()
def status() -> None:
    """Show the current status of Alibi."""
    from alibi import __version__

    config = get_config()
    db_instance = get_db()

    console.print("[bold blue]Alibi Status[/bold blue]\n")

    # Database status
    if db_instance.is_initialized():
        console.print(f"[green]Database:[/green] Initialized at {db_instance.db_path}")
        console.print(f"  Schema version: {db_instance.get_schema_version()}")

        # Get stats
        stats = db_instance.get_stats()

        table = Table(title="Data Summary")
        table.add_column("Entity", style="cyan")
        table.add_column("Count", justify="right", style="green")

        for entity, count in stats.items():
            table.add_row(entity.capitalize(), str(count))

        console.print(table)
    else:
        console.print("[yellow]Database:[/yellow] Not initialized")
        console.print("  Run 'lt init' to initialize the database.")

    # Configuration status
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Vault path: {config.vault_path or 'Not set'}")

    if config.vault_path:
        inbox_path = config.get_inbox_path()
        if inbox_path and inbox_path.exists():
            console.print(f"  [green]Inbox:[/green] {inbox_path}")
        elif inbox_path:
            console.print(f"  [yellow]Inbox:[/yellow] {inbox_path} (not found)")

    console.print(f"  Ollama URL: {config.ollama_url}")
    console.print(f"  Model: {config.ollama_model}")
    console.print(f"  Currency: {config.default_currency}")


@click.command()
def version() -> None:
    """Show version information."""
    from alibi import __version__

    console.print(f"[bold]Alibi[/bold] v{__version__}")
    console.print("Life Tracker: Transaction-based life management system")


@click.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    required=True,
    help="Shell type to generate completion for",
)
def completion(shell: str) -> None:
    """Generate shell completion script.

    \\b
    Bash:
      lt completion --shell bash >> ~/.bashrc
      source ~/.bashrc

    \\b
    Zsh:
      lt completion --shell zsh >> ~/.zshrc
      source ~/.zshrc

    \\b
    Fish:
      lt completion --shell fish > ~/.config/fish/completions/lt.fish
    """
    # Click uses different env vars for each shell
    shell_complete_var = {
        "bash": "_LT_COMPLETE=bash_source",
        "zsh": "_LT_COMPLETE=zsh_source",
        "fish": "_LT_COMPLETE=fish_source",
    }

    # Get the completion script by invoking click's completion mechanism
    env = os.environ.copy()
    env_var = shell_complete_var[shell]
    key, value = env_var.split("=")
    env[key] = value

    # Output instructions and the completion script
    console.print(f"# Shell completion for Alibi CLI ({shell})")
    console.print(f"# Add this to your shell configuration file\n")

    # Use click's built-in shell completion
    result = subprocess.run(
        [sys.executable, "-m", "alibi.cli"],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    else:
        # Fallback: generate basic completion
        if shell == "bash":
            print('eval "$(_LT_COMPLETE=bash_source lt)"')
        elif shell == "zsh":
            print('eval "$(_LT_COMPLETE=zsh_source lt)"')
        elif shell == "fish":
            print("_LT_COMPLETE=fish_source lt | source")


@click.command()
@click.option("--vendor", "-v", help="Filter by vendor name")
@click.option(
    "--type", "-t", "artifact_type", help="Filter by type (receipt, invoice, etc.)"
)
@click.option("--limit", "-l", default=20, help="Maximum results to show")
def query(vendor: str | None, artifact_type: str | None, limit: int) -> None:
    """Query artifacts in the database."""
    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        console.print("Run 'lt init' to initialize.")
        return

    # Build query -- documents joined with facts for context
    sql = """
        SELECT d.id, d.file_path, d.ingested_at,
               f.fact_type, f.vendor, f.event_date, f.total_amount, f.currency
        FROM documents d
        LEFT JOIN bundles b ON b.document_id = d.id
        LEFT JOIN cloud_bundles cb ON cb.bundle_id = b.id
        LEFT JOIN facts f ON f.cloud_id = cb.cloud_id
        WHERE 1=1
    """
    params: list[str] = []

    if vendor:
        sql += " AND LOWER(f.vendor) LIKE ?"
        params.append(f"%{vendor.lower()}%")

    if artifact_type:
        sql += " AND f.fact_type = ?"
        params.append(artifact_type)

    sql += " GROUP BY d.id ORDER BY d.created_at DESC LIMIT ?"
    params.append(str(limit))

    rows = db_manager.fetchall(sql, tuple(params))

    if not rows:
        console.print("[yellow]No documents found.[/yellow]")
        return

    table = Table(title=f"Documents ({len(rows)} found)")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Type", style="cyan")
    table.add_column("Vendor", style="green")
    table.add_column("Date")
    table.add_column("Amount", justify="right")
    table.add_column("File")

    for row in rows:
        doc_id = row[0][:8] if row[0] else ""
        file_path = row[1] or ""
        fact_type = row[3] or ""
        doc_vendor = (row[4] or "Unknown")[:30]
        doc_date = str(row[5]) if row[5] else ""
        amount = float(row[6]) if row[6] else None
        currency = row[7] or ""
        amount_str = format_amount(amount, currency)
        file_name = file_path.rsplit("/", 1)[-1][:25] if file_path else ""

        table.add_row(doc_id, fact_type, doc_vendor, doc_date, amount_str, file_name)

    console.print(table)


@click.command()
@click.option("--vendor", "-v", help="Filter by vendor name")
@click.option("--account", "-a", help="Filter by account")
@click.option(
    "--unmatched", "-u", is_flag=True, help="Show only unmatched transactions"
)
@click.option("--limit", "-l", default=20, help="Maximum results to show")
def transactions(
    vendor: str | None, account: str | None, unmatched: bool, limit: int
) -> None:
    """List transactions and find matches."""
    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        console.print("Run 'lt init' to initialize.")
        return

    # Build query from v2 facts table
    sql = """
        SELECT f.id, f.vendor, f.fact_type, f.event_date, f.total_amount, f.currency,
               f.vendor_key,
               (SELECT COUNT(DISTINCT b.document_id) FROM cloud_bundles cb
                JOIN bundles b ON b.id = cb.bundle_id
                WHERE cb.cloud_id = f.cloud_id) as doc_count
        FROM facts f
        WHERE 1=1
    """
    params: list[str] = []

    if vendor:
        sql += " AND LOWER(f.vendor) LIKE ?"
        params.append(f"%{vendor.lower()}%")

    if account:
        sql += " AND f.vendor_key = ?"
        params.append(account)

    if unmatched:
        sql += " AND f.cloud_id IS NULL"

    sql += " ORDER BY f.event_date DESC LIMIT ?"
    params.append(str(limit))

    rows = db_manager.fetchall(sql, tuple(params))

    if not rows:
        console.print("[yellow]No facts found.[/yellow]")
        return

    table = Table(title=f"Facts ({len(rows)} found)")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Vendor", style="green")
    table.add_column("Type", max_width=20)
    table.add_column("Date")
    table.add_column("Amount", justify="right")
    table.add_column("Docs", justify="center")

    for row in rows:
        fact_id = str(row[0])[:8] if row[0] else ""
        fact_vendor = (row[1] or "")[:25]
        fact_type = (row[2] or "")[:20]
        fact_date = str(row[3]) if row[3] else ""
        amount = float(row[4]) if row[4] else None
        currency = row[5] or ""
        amount_str = format_amount(amount, currency)
        doc_count = row[7] if row[7] else 0
        doc_str = f"[green]{doc_count}[/green]" if doc_count > 0 else "[dim]0[/dim]"

        table.add_row(fact_id, fact_vendor, fact_type, fact_date, amount_str, doc_str)

    console.print(table)


@click.command()
@click.argument("document_id")
def match(document_id: str) -> None:
    """Show facts linked to a document via cloud formation."""
    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    # Find document (support partial ID)
    doc_row = db_manager.fetchone(
        "SELECT id, file_path FROM documents WHERE id LIKE ?",
        (f"{document_id}%",),
    )

    if not doc_row:
        console.print(f"[red]Document not found:[/red] {document_id}")
        return

    console.print(f"[bold]Document:[/bold] {doc_row['file_path']}")
    console.print(f"  ID: {doc_row['id']}")
    console.print()

    # Get linked facts via bundle/cloud chain
    fact_rows = db_manager.fetchall(
        """
        SELECT DISTINCT f.id, f.vendor, f.fact_type, f.event_date,
               f.total_amount, f.currency, f.status
        FROM facts f
        JOIN cloud_bundles cb ON f.cloud_id = cb.cloud_id
        JOIN bundles b ON cb.bundle_id = b.id
        WHERE b.document_id = ?
        ORDER BY f.event_date DESC
        """,
        (doc_row["id"],),
    )

    if not fact_rows:
        console.print("[yellow]No linked facts found.[/yellow]")
        return

    table = Table(title=f"Linked Facts ({len(fact_rows)} found)")
    table.add_column("Vendor", style="green")
    table.add_column("Type")
    table.add_column("Date")
    table.add_column("Amount", justify="right")
    table.add_column("Status")

    for row in fact_rows:
        amount = row["total_amount"]
        currency = row["currency"] or "EUR"
        amount_str = format_amount(float(amount), currency) if amount else ""
        table.add_row(
            (row["vendor"] or "")[:25],
            row["fact_type"] or "",
            str(row["event_date"]) if row["event_date"] else "",
            amount_str,
            row["status"] or "",
        )

    console.print(table)


@click.command("search")
@click.argument("query_text")
@click.option("--limit", "-l", default=10, help="Maximum results")
@click.option("--space", "-s", default="default", help="Space ID")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["unified", "sql", "vector"]),
    default="unified",
    help="Search mode",
)
def search_cmd(query_text: str, limit: int, space: str, mode: str) -> None:
    """Search transactions, items, and artifacts.

    Combines SQL text search with semantic vector search for best results.

    Examples:
        lt search "amazon"
        lt search "grocery shopping" --mode vector
        lt search "electronics" --limit 20
    """
    from alibi.vectordb.index import VectorIndex
    from alibi.vectordb.search import semantic_search, sql_search, unified_search

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        console.print("Run 'lt init' to initialize.")
        return

    index: VectorIndex | None = None
    use_vector = mode in ("unified", "vector")
    use_sql = mode in ("unified", "sql")

    if use_vector:
        index = _get_vector_index()
        if not index.is_initialized():
            if mode == "vector":
                console.print("[yellow]Vector index not initialized.[/yellow]")
                console.print("Run 'lt vectordb init' to create the index.")
                return
            else:
                console.print(
                    "[dim]Vector index not initialized, using SQL only.[/dim]"
                )
                use_vector = False

    console.print(f"[bold blue]Search:[/bold blue] {query_text}")
    console.print(f"  Mode: {mode}\n")

    try:
        if mode == "unified":
            results = unified_search(
                db=db_manager,
                index=index,
                query=query_text,
                limit=limit,
                space_id=space,
                use_vector=use_vector,
                use_sql=use_sql,
            )
        elif mode == "vector":
            results = semantic_search(index, query_text, limit=limit)  # type: ignore
        else:  # sql
            results = sql_search(db_manager, query_text, limit=limit, space_id=space)
    except Exception as e:
        console.print(f"[red]Search error:[/red] {e}")
        raise click.Abort() from e

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Results ({len(results)} found)")
    table.add_column("Score", justify="right", style="cyan")
    table.add_column("Source", style="dim")
    table.add_column("Type", style="green")
    table.add_column("Description", max_width=35)
    table.add_column("Amount", justify="right")
    table.add_column("Date")

    for r in results:
        score_str = f"{r.score:.2f}"
        desc = r.description[:35] if r.description else r.vendor or ""
        amount_str = f"{r.amount:.2f} {r.currency}" if r.amount else ""

        table.add_row(
            score_str,
            r.source,
            r.entity_type,
            desc,
            amount_str,
            r.date or "",
        )

    console.print(table)


@click.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=False),
    help="Re-ingest single document from its YAML",
)
@click.option("--scan", "do_scan", is_flag=True, help="Scan inbox for all edited YAMLs")
@click.option("--dry-run", is_flag=True, help="Show what would change without applying")
def reingest(path: str | None, do_scan: bool, dry_run: bool) -> None:
    """Re-ingest documents from edited .alibi.yaml files.

    Use --path to re-ingest a single document, or --scan to find all
    edited YAMLs in the inbox and re-ingest them.

    Examples:
        lt reingest --path /path/to/receipt.jpg
        lt reingest --scan
        lt reingest --scan --dry-run
    """
    from pathlib import Path as P

    from alibi.services.ingestion import (
        reingest_from_yaml,
        scan_yaml_corrections,
    )

    db_instance = get_db()
    if not db_instance.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        console.print("Run 'lt init' to initialize.")
        return

    if path:
        source = P(path)
        is_group = source.is_dir()

        if dry_run:
            console.print(f"[dim]Would re-ingest:[/dim] {source}")
            return

        result = reingest_from_yaml(db_instance, source, is_group=is_group)
        if result.success:
            console.print(
                f"[green]Re-ingested:[/green] {source.name} " f"-> {result.document_id}"
            )
        else:
            console.print(f"[red]Failed:[/red] {result.error}")
        return

    if do_scan:
        config = get_config()
        inbox = config.get_inbox_path()
        if inbox is None:
            console.print("[red]No inbox configured.[/red]")
            return

        changed = scan_yaml_corrections(db_instance)
        if not changed:
            console.print("[green]No YAML corrections found.[/green]")
            return

        console.print(f"Found [bold]{len(changed)}[/bold] edited YAML(s):\n")
        for src in changed:
            if dry_run:
                console.print(f"  [dim]Would re-ingest:[/dim] {src}")
            else:
                is_group = src.is_dir()
                result = reingest_from_yaml(db_instance, src, is_group=is_group)
                if result.success:
                    console.print(
                        f"  [green]OK[/green] {src.name} -> {result.document_id}"
                    )
                else:
                    console.print(f"  [red]FAIL[/red] {src.name}: {result.error}")
        return

    console.print("[yellow]Specify --path or --scan.[/yellow]")
    console.print("Run 'lt reingest --help' for usage.")


@click.command()
@click.argument("document_type")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Write template to file instead of stdout",
)
def template(document_type: str, output: str | None) -> None:
    """Generate a blank .alibi.yaml template for a document type.

    Supported types: receipt, payment_confirmation, invoice, statement,
    contract, warranty.

    Examples:
        lt template receipt
        lt template invoice --output invoice.alibi.yaml
    """
    import yaml

    from alibi.extraction.yaml_cache import (
        SUPPORTED_DOCUMENT_TYPES,
        generate_blank_template,
    )

    if document_type not in SUPPORTED_DOCUMENT_TYPES:
        console.print(
            f"[red]Unknown type:[/red] {document_type}\n"
            f"Supported: {', '.join(SUPPORTED_DOCUMENT_TYPES)}"
        )
        raise click.Abort()

    tmpl = generate_blank_template(document_type)
    yaml_text = yaml.dump(
        tmpl,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )

    if output:
        from pathlib import Path as P

        P(output).write_text(yaml_text, encoding="utf-8")
        console.print(f"[green]Template written to:[/green] {output}")
    else:
        console.print(yaml_text)


@click.command()
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.5,
    show_default=True,
    help="Confidence threshold below which a document needs review",
)
@click.option(
    "--edit",
    is_flag=True,
    help="Open each YAML in $EDITOR for manual correction",
)
def review(threshold: float, edit: bool) -> None:
    """List documents with low confidence extractions needing review.

    Scans the inbox for .alibi.yaml files where extraction confidence is
    below the threshold or where critical fields (vendor, date, total)
    could not be parsed.  Use --edit to open each YAML in $EDITOR.

    Examples:
        lt review
        lt review --threshold 0.6
        lt review --edit
    """
    from alibi.services.ingestion import scan_low_confidence_yamls

    config = get_config()
    inbox = config.get_inbox_path()
    if inbox is None:
        console.print("[red]No inbox configured.[/red]")
        return

    flagged = scan_low_confidence_yamls(threshold=threshold)
    if not flagged:
        console.print("[green]No documents need review.[/green]")
        return

    console.print(
        f"Found [bold]{len(flagged)}[/bold] document(s) needing review "
        f"(threshold={threshold}):\n"
    )

    from alibi.extraction.yaml_cache import get_yaml_path

    table = Table(show_header=True, header_style="bold")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Confidence", justify="right")
    table.add_column("Gaps", style="yellow")
    table.add_column("Flagged", justify="center")

    for source_path, meta in flagged:
        confidence = meta.get("confidence")
        conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
        gaps = meta.get("parser_gaps") or []
        gaps_str = ", ".join(gaps) if gaps else ""
        flagged_str = "[red]yes[/red]" if meta.get("needs_review") else ""
        table.add_row(source_path.name, conf_str, gaps_str, flagged_str)

    console.print(table)

    if edit:
        editor = os.environ.get("EDITOR", "vi")
        for source_path, _meta in flagged:
            yaml_path = get_yaml_path(source_path)
            if not yaml_path.exists():
                console.print(f"[yellow]No YAML for {source_path.name}[/yellow]")
                continue
            console.print(f"\nOpening [cyan]{yaml_path.name}[/cyan] ...")
            subprocess.run([editor, str(yaml_path)])


@click.command("verify")
@click.option("-n", "--limit", default=20, help="Max documents to verify")
@click.option("--doc-id", multiple=True, help="Specific document IDs")
def verify_cmd(limit: int, doc_id: tuple[str, ...]) -> None:
    """Cross-validate extracted receipts via Gemini."""
    db_instance = get_db()
    from alibi.services import verify_extractions

    doc_ids = list(doc_id) if doc_id else None
    results = verify_extractions(db_instance, doc_ids=doc_ids, limit=limit)

    ok = sum(1 for r in results if r.all_ok)
    issues = sum(1 for r in results if not r.all_ok)
    console.print(
        f"Verified {len(results)} documents: "
        f"[green]{ok} OK[/green], [red]{issues} with issues[/red]"
    )

    for r in results:
        if not r.all_ok:
            console.print(f"\n[bold]{r.doc_id}[/bold]:")
            for issue in r.issues:
                console.print(
                    f"  - {issue['field']}: "
                    f"{issue.get('note', issue.get('suggested', 'issue'))}"
                )


# ---------------------------------------------------------------------------
# export group
# ---------------------------------------------------------------------------


@click.group()
def export() -> None:
    """Export data to various formats.

    Supports exporting to Obsidian notes or CSV/JSON files.
    """
    pass


@export.command("transactions")
@click.option("--space", "-s", default="default", help="Space to export from")
@click.option(
    "--since", "-d", type=click.DateTime(formats=["%Y-%m-%d"]), help="Export since date"
)
@click.option("--overwrite", "-f", is_flag=True, help="Overwrite existing notes")
def export_transactions(space: str, since: datetime | None, overwrite: bool) -> None:
    """Export facts to Obsidian notes."""
    from alibi.obsidian.notes import NoteExporter

    db_manager = get_db()
    config = get_config()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    if config.vault_path is None:
        NO_INBOX_CONFIGURED.display(console)
        return

    try:
        exporter = NoteExporter(db_manager, config.vault_path)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    since_date = since.date() if since else None

    console.print("[bold blue]Exporting facts...[/bold blue]")
    if since_date:
        console.print(f"  Since: {since_date}")

    paths = exporter.export_all_facts(
        since=since_date,
        overwrite=overwrite,
    )

    if not paths:
        console.print("[yellow]No facts to export.[/yellow]")
        return

    console.print(f"[green]Exported {len(paths)} fact notes:[/green]")
    for p in paths[:5]:
        console.print(f"  {p.name}")
    if len(paths) > 5:
        console.print(f"  ... and {len(paths) - 5} more")


@export.command("templates")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
def export_templates(output: str | None) -> None:
    """Copy Obsidian templates to vault."""
    from pathlib import Path
    import shutil

    config = get_config()

    # Determine output path
    if output:
        out_path = Path(output)
    elif config.vault_path:
        out_path = config.vault_path / "_templates" / "alibi"
    else:
        console.print("[red]No output path specified and no vault configured.[/red]")
        return

    # Find template source
    template_src = Path(__file__).parent.parent / "obsidian" / "templates"
    if not template_src.exists():
        console.print(f"[red]Template directory not found:[/red] {template_src}")
        return

    # Copy templates
    out_path.mkdir(parents=True, exist_ok=True)

    copied = 0
    for tmpl in template_src.glob("*.md"):
        dest = out_path / tmpl.name
        shutil.copy(tmpl, dest)
        console.print(f"  Copied: {tmpl.name}")
        copied += 1

    console.print(f"[green]Copied {copied} templates to:[/green] {out_path}")


@export.command("csv")
@click.argument("output", type=click.Path())
@click.option(
    "--type",
    "-t",
    "data_type",
    type=click.Choice(["transactions", "items", "artifacts", "all"]),
    default="transactions",
    help="Data type to export",
)
@click.option("--format", "-f", type=click.Choice(["csv", "json"]), default="csv")
@click.option("--since", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--until", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--space", "-s", default="default")
def export_csv_cmd(
    output: str,
    data_type: str,
    format: str,
    since: datetime | None,
    until: datetime | None,
    space: str,
) -> None:
    """Export data to CSV or JSON file.

    Examples:

        lt export csv transactions.csv

        lt export csv data.json --format json --type all

        lt export csv expenses.csv --since 2024-01-01
    """
    from pathlib import Path

    from alibi.export import (
        export_all,
        export_artifacts,
        export_items,
        export_transactions as export_txn,
    )

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    output_path = Path(output)
    since_date = since.date() if since else None
    until_date = until.date() if until else None

    console.print(f"[bold blue]Exporting {data_type}...[/bold blue]")

    if data_type == "all":
        # Output is a directory for all
        results = export_all(
            db_manager,
            output_path,
            format=format,  # type: ignore
            space_id=space,
            since=since_date,
            until=until_date,
        )
        console.print(f"[green]Exported {len(results)} files to {output_path}[/green]")
        for r in results:
            console.print(f"  {r.path.name}: {r.record_count} records")
    else:
        if data_type == "transactions":
            result = export_txn(
                db_manager,
                output_path,
                format=format,  # type: ignore
                space_id=space,
                since=since_date,
                until=until_date,
            )
        elif data_type == "items":
            result = export_items(
                db_manager,
                output_path,
                format=format,  # type: ignore
                space_id=space,
            )
        else:  # artifacts
            result = export_artifacts(
                db_manager,
                output_path,
                format=format,  # type: ignore
                space_id=space,
            )

        console.print(
            f"[green]Exported {result.record_count} records to {result.path}[/green]"
        )
        console.print(f"  Size: {result.size_bytes / 1024:.1f} KB")


@export.command("masked")
@click.argument("output", type=click.Path())
@click.option(
    "--tier",
    "-t",
    type=click.IntRange(0, 4),
    default=2,
    help="Disclosure tier (0=hidden, 4=exact)",
)
@click.option("--format", "-f", type=click.Choice(["csv", "json"]), default="json")
@click.option("--since", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--until", type=click.DateTime(formats=["%Y-%m-%d"]))
def export_masked(
    output: str,
    tier: int,
    format: str,
    since: datetime | None,
    until: datetime | None,
) -> None:
    """Export transactions with tier-based disclosure masking.

    Tiers control what data is visible:

        0: amounts hidden, vendor=category, dates=month-year

        1: amounts rounded, vendor=category, dates=1st of month

        2: exact amounts/dates, vendor visible, no line items

        3: includes line items

        4: full provenance (unmasked)

    Examples:

        lt export masked public.json --tier 1

        lt export masked shared.csv --tier 2 --format csv
    """
    import json
    from pathlib import Path

    from alibi.db.models import Tier
    from alibi.masking.service import MaskingService
    from alibi.services import list_facts

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    filters: dict[str, Any] = {}
    if since:
        filters["date_from"] = since.date().isoformat()
    if until:
        filters["date_to"] = until.date().isoformat()

    result = list_facts(db_manager, filters=filters, offset=0, limit=100000)
    facts = result["facts"]

    masking_svc = MaskingService()
    tier_enum = Tier(str(tier))
    masked = masking_svc.mask_for_tier(facts, tier_enum)

    output_path = Path(output)
    if format == "json":
        output_path.write_text(json.dumps(masked, indent=2, default=str))
    else:
        import csv

        with output_path.open("w", newline="") as f:
            if masked:
                writer = csv.DictWriter(f, fieldnames=masked[0].keys())
                writer.writeheader()
                writer.writerows(masked)

    console.print(
        f"[green]Exported {len(masked)} masked records (tier {tier}) "
        f"to {output_path}[/green]"
    )


@export.command("anonymized")
@click.argument("output", type=click.Path())
@click.option(
    "--level",
    "-l",
    type=click.Choice(["categories_only", "pseudonymized", "statistical"]),
    default="pseudonymized",
    help="Anonymization level",
)
@click.option(
    "--key-file", "-k", type=click.Path(), help="Save restoration key to file"
)
@click.option("--since", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--until", type=click.DateTime(formats=["%Y-%m-%d"]))
def export_anonymized(
    output: str,
    level: str,
    key_file: str | None,
    since: datetime | None,
    until: datetime | None,
) -> None:
    """Export transactions with privacy-preserving anonymization.

    Levels:

        categories_only: Only categories, no names/amounts/dates.

        pseudonymized: Fake names, shifted amounts/dates (reversible).

        statistical: Only aggregates, no individual records.

    The restoration key (for pseudonymized) enables local reversal.

    Examples:

        lt export anonymized data.json --level pseudonymized --key-file key.json

        lt export anonymized stats.json --level statistical
    """
    import json
    from pathlib import Path

    from alibi.anonymization.exporter import AnonymizationLevel, anonymize_export
    from alibi.services import list_fact_items_with_fact, list_facts

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    filters: dict[str, Any] = {}
    if since:
        filters["date_from"] = since.date().isoformat()
    if until:
        filters["date_to"] = until.date().isoformat()

    result = list_facts(db_manager, filters=filters, offset=0, limit=100000)
    facts = result["facts"]

    items_by_fact: dict[str, list[Any]] = {}
    all_items = list_fact_items_with_fact(db_manager, filters=filters)
    for item in all_items:
        fid = item.get("fact_id", "")
        items_by_fact.setdefault(fid, []).append(item)

    anon_level = AnonymizationLevel(level)
    anonymized_data, key = anonymize_export(facts, items_by_fact, anon_level)

    output_path = Path(output)
    output_path.write_text(json.dumps(anonymized_data, indent=2, default=str))

    console.print(
        f"[green]Exported {len(anonymized_data)} anonymized records "
        f"({level}) to {output_path}[/green]"
    )

    if key_file and anon_level == AnonymizationLevel.PSEUDONYMIZED:
        key_path = Path(key_file)
        key_path.write_text(key.to_json())
        console.print(f"[green]Restoration key saved to {key_path}[/green]")
        console.print(
            "[yellow]Keep this key secure -- it enables de-anonymization.[/yellow]"
        )
    elif anon_level == AnonymizationLevel.PSEUDONYMIZED and not key_file:
        console.print(
            "[yellow]Tip: use --key-file to save the restoration key "
            "for later de-anonymization.[/yellow]"
        )


# ---------------------------------------------------------------------------
# budget group
# ---------------------------------------------------------------------------


@click.group()
def budget() -> None:
    """Manage budget scenarios and compare spending."""
    pass


@budget.command("list")
@click.option("--space", "-s", default="default", help="Space ID")
def budget_list(space: str) -> None:
    """List budget scenarios."""
    from alibi.budgets.service import BudgetService

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    svc = BudgetService(db_manager)
    scenarios = svc.list_scenarios(space)

    if not scenarios:
        console.print("[yellow]No budget scenarios found.[/yellow]")
        return

    table = Table(title="Budget Scenarios")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Period")

    for s in scenarios:
        period = ""
        if s.period_start and s.period_end:
            period = f"{s.period_start} to {s.period_end}"
        elif s.period_start:
            period = f"from {s.period_start}"
        table.add_row(s.id[:8], s.name, s.data_type.value, period)

    console.print(table)


@budget.command("create")
@click.argument("name")
@click.option(
    "--type",
    "-t",
    "data_type",
    type=click.Choice(["actual", "projected", "target"]),
    default="target",
)
@click.option("--space", "-s", default="default")
@click.option("--description", "-d", default=None)
def budget_create(
    name: str, data_type: str, space: str, description: str | None
) -> None:
    """Create a new budget scenario."""
    import uuid

    from alibi.budgets.models import BudgetScenario
    from alibi.budgets.service import BudgetService
    from alibi.db.models import DataType

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    scenario = BudgetScenario(
        id=str(uuid.uuid4()),
        space_id=space,
        name=name,
        description=description,
        data_type=DataType(data_type),
    )
    svc = BudgetService(db_manager)
    svc.create_scenario(scenario)
    console.print(f"[green]Created scenario:[/green] {scenario.id[:8]} ({name})")


@budget.command("add-entry")
@click.argument("scenario_id")
@click.argument("category")
@click.argument("amount", type=float)
@click.argument("period")
@click.option("--currency", "-c", default="EUR")
@click.option("--note", "-n", default=None)
def budget_add_entry(
    scenario_id: str,
    category: str,
    amount: float,
    period: str,
    currency: str,
    note: str | None,
) -> None:
    """Add a budget entry: lt budget add-entry <id> groceries 500 2025-01."""
    import uuid
    from decimal import Decimal

    from alibi.budgets.models import BudgetEntry
    from alibi.budgets.service import BudgetService

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    entry = BudgetEntry(
        id=str(uuid.uuid4()),
        scenario_id=scenario_id,
        category=category,
        amount=Decimal(str(amount)),
        currency=currency,
        period=period,
        note=note,
    )
    svc = BudgetService(db_manager)
    svc.add_entry(entry)
    console.print(
        f"[green]Added:[/green] {category} {format_amount(amount)} {currency} ({period})"
    )


@budget.command("compare")
@click.argument("base_id")
@click.argument("compare_id")
@click.option("--period", "-p", default=None, help="Filter by period (YYYY-MM)")
def budget_compare(base_id: str, compare_id: str, period: str | None) -> None:
    """Compare two budget scenarios."""
    from alibi.budgets.service import BudgetService

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    svc = BudgetService(db_manager)
    comparisons = svc.compare(base_id, compare_id, period)

    if not comparisons:
        console.print("[yellow]No comparison data found.[/yellow]")
        return

    table = Table(title="Budget Comparison")
    table.add_column("Category")
    table.add_column("Period")
    table.add_column("Base", justify="right")
    table.add_column("Compare", justify="right")
    table.add_column("Variance", justify="right")

    for c in comparisons:
        var_style = "red" if c.variance > 0 else "green"
        table.add_row(
            c.category,
            c.period,
            format_amount(float(c.base_amount)),
            format_amount(float(c.compare_amount)),
            f"[{var_style}]{format_amount(float(c.variance))}[/{var_style}]",
        )

    console.print(table)


@budget.command("actual")
@click.argument("period")
@click.option("--space", "-s", default="default")
def budget_actual(period: str, space: str) -> None:
    """Show actual spending for a period (YYYY-MM)."""
    from alibi.budgets.service import BudgetService

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    svc = BudgetService(db_manager)
    entries = svc.get_actual_spending(space, period)

    if not entries:
        console.print(f"[yellow]No spending data for {period}.[/yellow]")
        return

    table = Table(title=f"Actual Spending: {period}")
    table.add_column("Category")
    table.add_column("Amount", justify="right")
    table.add_column("Currency")

    total = 0.0
    for e in entries:
        table.add_row(e.category, format_amount(float(e.amount)), e.currency)
        total += float(e.amount)

    table.add_row("[bold]Total[/bold]", f"[bold]{format_amount(total)}[/bold]", "")
    console.print(table)


# ---------------------------------------------------------------------------
# import group
# ---------------------------------------------------------------------------


@click.group("import")
def import_cmd() -> None:
    """Import transactions from bank exports."""
    pass


@import_cmd.command("csv")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    "format_type",
    type=click.Choice(["n26", "revolut", "generic"]),
    help="CSV format (auto-detected if not specified)",
)
@click.option("--account", "-a", help="Account name for transactions")
@click.option("--space", "-s", default="default", help="Space ID")
def import_csv(
    file: str, format_type: str | None, account: str | None, space: str
) -> None:
    """Import transactions from a CSV file.

    Supports N26, Revolut, and generic CSV formats.
    Format is auto-detected if not specified.
    """
    from pathlib import Path

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from alibi.ingestion.csv_parser import CSVFormat
    from alibi.ingestion.importer import TransactionImporter

    db_manager = get_db()
    config = get_config()

    # Ensure database is initialized
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized. Initializing...[/yellow]")
        db_manager.initialize()

    file_path = Path(file)
    console.print(f"[bold blue]Importing from:[/bold blue] {file_path.name}")

    # Convert format string to enum
    csv_format: CSVFormat | None = None
    if format_type:
        csv_format = CSVFormat(format_type)

    importer = TransactionImporter(
        db=db_manager,
        space_id=space,
        default_currency=config.default_currency,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Processing CSV...", total=None)
        result = importer.import_csv(file_path, csv_format, account)

    if not result.success:
        console.print(f"[red]Import failed:[/red] {result.error_message}")
        raise click.Abort()

    # Show results
    console.print()
    table = Table(title="Import Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")

    table.add_row("Total rows", str(result.total_rows))
    table.add_row("Imported", str(result.imported))
    table.add_row("Duplicates skipped", str(result.duplicates))
    table.add_row("Errors", str(result.errors))

    console.print(table)

    if result.imported > 0:
        console.print(
            f"\n[green]Successfully imported {result.imported} transaction(s).[/green]"
        )
    elif result.duplicates > 0:
        console.print(
            f"\n[yellow]All {result.duplicates} transaction(s) were duplicates.[/yellow]"
        )


@import_cmd.command("ofx")
@click.argument("file", type=click.Path(exists=True))
@click.option("--account", "-a", help="Account name override")
@click.option("--space", "-s", default="default", help="Space ID")
def import_ofx(file: str, account: str | None, space: str) -> None:
    """Import transactions from an OFX/QFX file.

    OFX (Open Financial Exchange) is a standard bank export format.
    """
    from pathlib import Path

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from alibi.ingestion.importer import TransactionImporter

    db_manager = get_db()
    config = get_config()

    # Ensure database is initialized
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized. Initializing...[/yellow]")
        db_manager.initialize()

    file_path = Path(file)
    console.print(f"[bold blue]Importing from:[/bold blue] {file_path.name}")

    importer = TransactionImporter(
        db=db_manager,
        space_id=space,
        default_currency=config.default_currency,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Processing OFX...", total=None)
        result = importer.import_ofx(file_path, account)

    if not result.success:
        console.print(f"[red]Import failed:[/red] {result.error_message}")
        raise click.Abort()

    # Show results
    console.print()
    table = Table(title="Import Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")

    table.add_row("Total rows", str(result.total_rows))
    table.add_row("Imported", str(result.imported))
    table.add_row("Duplicates skipped", str(result.duplicates))
    table.add_row("Errors", str(result.errors))

    console.print(table)

    if result.imported > 0:
        console.print(
            f"\n[green]Successfully imported {result.imported} transaction(s).[/green]"
        )
    elif result.duplicates > 0:
        console.print(
            f"\n[yellow]All {result.duplicates} transaction(s) were duplicates.[/yellow]"
        )


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------


@click.command()
@click.option("--host", "-h", default="127.0.0.1", help="Bind address")
@click.option("--port", "-p", default=3100, type=int, help="Port number")
@click.option("--reload", "-r", is_flag=True, help="Auto-reload on code changes")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the API server (includes Web UI at /web)."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Error: uvicorn not installed[/red]")
        raise click.Abort()

    console.print(f"[bold blue]Starting Alibi API server...[/bold blue]")
    console.print(f"  API docs: http://{host}:{port}/docs")
    console.print(f"  Web UI:   http://{host}:{port}/web")
    console.print(f"  Health:   http://{host}:{port}/health")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    uvicorn.run(
        "alibi.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


# ---------------------------------------------------------------------------
# daemon group
# ---------------------------------------------------------------------------


@click.group()
def daemon() -> None:
    """Daemon management commands for background processing."""
    pass


@daemon.command("start")
@click.option(
    "--foreground",
    "-f",
    is_flag=True,
    default=True,
    help="Run in foreground (default for development)",
)
@click.option(
    "--debounce",
    "-d",
    type=float,
    default=2.0,
    help="Debounce time in seconds before processing",
)
def daemon_start(foreground: bool, debounce: float) -> None:
    """Start the watcher daemon.

    The daemon watches the inbox directory for new documents and
    automatically processes them through the extraction pipeline.
    """
    from alibi.daemon.watcher_service import WatcherDaemon

    config = get_config()

    # Check if already running
    daemon_status_result = WatcherDaemon.get_running_status()
    if daemon_status_result and daemon_status_result.running:
        console.print(
            f"[yellow]Daemon already running (pid={daemon_status_result.pid})[/yellow]"
        )
        console.print("Use 'lt daemon stop' to stop it first.")
        raise click.Abort()

    inbox_path = config.get_inbox_path()
    if inbox_path is None:
        NO_INBOX_CONFIGURED.display(console)
        raise click.Abort()

    if not inbox_path.exists():
        err = format_error(VAULT_NOT_FOUND, path=str(inbox_path))
        err.display(console)
        raise click.Abort()

    console.print(f"[bold blue]Starting Alibi Watcher Daemon[/bold blue]")
    console.print(f"  Inbox: {inbox_path}")
    console.print(f"  Debounce: {debounce}s")
    console.print(f"  Mode: {'Foreground' if foreground else 'Background'}")
    console.print()

    if foreground:
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        daemon_instance = WatcherDaemon(
            inbox_path=inbox_path,
            debounce_seconds=debounce,
        )
        daemon_instance.start(foreground=foreground)
    except KeyboardInterrupt:
        console.print("\n[yellow]Daemon stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise click.Abort() from e


@daemon.command("stop")
def daemon_stop() -> None:
    """Stop the watcher daemon."""
    from alibi.daemon.watcher_service import WatcherDaemon

    daemon_status_result = WatcherDaemon.get_running_status()
    if not daemon_status_result or not daemon_status_result.running:
        console.print("[yellow]Daemon is not running.[/yellow]")
        return

    console.print(f"Stopping daemon (pid={daemon_status_result.pid})...")
    if WatcherDaemon.stop_running():
        console.print("[green]Daemon stopped.[/green]")
    else:
        console.print("[red]Failed to stop daemon.[/red]")


@daemon.command("status")
def daemon_status() -> None:
    """Check watcher daemon status."""
    from alibi.daemon.watcher_service import LOG_FILE, PID_FILE, WatcherDaemon

    daemon_status_result = WatcherDaemon.get_running_status()

    if daemon_status_result and daemon_status_result.running:
        console.print("[green]Daemon Status: Running[/green]")
        console.print(f"  PID: {daemon_status_result.pid}")
        console.print(f"  PID File: {PID_FILE}")
        console.print(f"  Log File: {LOG_FILE}")
    else:
        console.print("[yellow]Daemon Status: Not running[/yellow]")

    # Show configuration
    config = get_config()
    inbox_path = config.get_inbox_path()
    console.print()
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Inbox Path: {inbox_path or 'Not configured'}")
    if inbox_path:
        console.print(f"  Inbox Exists: {inbox_path.exists()}")


@daemon.command("install")
@click.option(
    "--system",
    type=click.Choice(["auto", "systemd", "launchd"]),
    default="auto",
    help="Service system to use",
)
def daemon_install(system: str) -> None:
    """Install the daemon as a system service.

    For Linux: Installs a systemd user service.
    For macOS: Installs a launchd user agent.
    """
    import platform
    import shutil
    from pathlib import Path

    current_os = platform.system().lower()

    if system == "auto":
        if current_os == "darwin":
            system = "launchd"
        elif current_os == "linux":
            system = "systemd"
        else:
            console.print(f"[red]Unsupported OS: {current_os}[/red]")
            raise click.Abort()

    if system == "launchd":
        # macOS launchd installation
        plist_src = (
            Path(__file__).parent.parent
            / "daemon"
            / "launchd"
            / "com.alibi.watcher.plist"
        )
        plist_dest = (
            Path.home() / "Library" / "LaunchAgents" / "com.alibi.watcher.plist"
        )

        if not plist_src.exists():
            console.print(f"[red]Plist file not found:[/red] {plist_src}")
            raise click.Abort()

        plist_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(plist_src, plist_dest)

        console.print("[green]Launchd agent installed.[/green]")
        console.print(f"  Location: {plist_dest}")
        console.print()
        console.print("[bold]To manage the service:[/bold]")
        console.print(f"  Load:    launchctl load {plist_dest}")
        console.print(f"  Unload:  launchctl unload {plist_dest}")
        console.print(f"  Status:  launchctl list | grep alibi")

    elif system == "systemd":
        # Linux systemd installation
        service_src = (
            Path(__file__).parent.parent
            / "daemon"
            / "systemd"
            / "alibi-watcher.service"
        )
        service_dest = (
            Path.home() / ".config" / "systemd" / "user" / "alibi-watcher.service"
        )

        if not service_src.exists():
            console.print(f"[red]Service file not found:[/red] {service_src}")
            raise click.Abort()

        service_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(service_src, service_dest)

        console.print("[green]Systemd user service installed.[/green]")
        console.print(f"  Location: {service_dest}")
        console.print()
        console.print("[bold]To manage the service:[/bold]")
        console.print("  Enable:  systemctl --user enable alibi-watcher")
        console.print("  Start:   systemctl --user start alibi-watcher")
        console.print("  Status:  systemctl --user status alibi-watcher")
        console.print("  Logs:    journalctl --user -u alibi-watcher -f")


# ---------------------------------------------------------------------------
# mycelium group
# ---------------------------------------------------------------------------


@click.group()
def mycelium() -> None:
    """Mycelium vault integration for iOS sync.

    Commands for integrating with the Obsidian vault and processing
    documents synced from iOS via Working Copy.
    """
    pass


@mycelium.command("start")
@click.option(
    "--foreground",
    "-f",
    is_flag=True,
    default=True,
    help="Run in foreground (default)",
)
@click.option(
    "--vault",
    "-v",
    type=click.Path(exists=True),
    help="Path to Obsidian vault",
)
@click.option(
    "--debounce",
    "-d",
    type=float,
    default=3.0,
    help="Debounce time in seconds (higher for git sync)",
)
@click.option(
    "--notes/--no-notes",
    default=True,
    help="Generate Obsidian notes for processed documents",
)
@click.option(
    "--archive/--no-archive",
    default=False,
    help="Archive processed files after processing",
)
def mycelium_start(
    foreground: bool,
    vault: str | None,
    debounce: float,
    notes: bool,
    archive: bool,
) -> None:
    """Start the Mycelium vault watcher."""
    from pathlib import Path

    from alibi.mycelium.watcher import MyceliumWatcher

    # Check if already running
    watcher_status = MyceliumWatcher.get_running_status()
    if watcher_status and watcher_status.running:
        console.print(
            f"[yellow]Mycelium watcher already running (pid={watcher_status.pid})[/yellow]"
        )
        console.print("Use 'lt mycelium stop' to stop it first.")
        raise click.Abort()

    vault_path = Path(vault) if vault else None

    console.print(f"[bold blue]Starting Mycelium Vault Watcher[/bold blue]")

    try:
        watcher = MyceliumWatcher(
            vault_path=vault_path,
            generate_notes=notes,
            archive_processed=archive,
            debounce_seconds=debounce,
        )
        console.print(f"  Vault: {watcher.vault_path}")
        console.print(f"  Inbox: {watcher.inbox_path}")
        console.print(f"  Debounce: {debounce}s")
        console.print(f"  Generate Notes: {notes}")
        console.print(f"  Archive Processed: {archive}")
        console.print()

        if foreground:
            console.print("[dim]Press Ctrl+C to stop[/dim]\n")

        watcher.start(foreground=foreground)
    except KeyboardInterrupt:
        console.print("\n[yellow]Mycelium watcher stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise click.Abort() from e


@mycelium.command("stop")
def mycelium_stop() -> None:
    """Stop the Mycelium vault watcher."""
    from alibi.mycelium.watcher import MyceliumWatcher

    watcher_status = MyceliumWatcher.get_running_status()
    if not watcher_status or not watcher_status.running:
        console.print("[yellow]Mycelium watcher is not running.[/yellow]")
        return

    console.print(f"Stopping Mycelium watcher (pid={watcher_status.pid})...")
    if MyceliumWatcher.stop_running():
        console.print("[green]Mycelium watcher stopped.[/green]")
    else:
        console.print("[red]Failed to stop watcher.[/red]")


@mycelium.command("status")
def mycelium_status() -> None:
    """Check Mycelium watcher status."""
    from alibi.mycelium.watcher import (
        DEFAULT_VAULT_PATH,
        MYCELIUM_LOG_FILE,
        MYCELIUM_PID_FILE,
        MyceliumWatcher,
    )

    watcher_status = MyceliumWatcher.get_running_status()

    if watcher_status and watcher_status.running:
        console.print("[green]Mycelium Watcher: Running[/green]")
        console.print(f"  PID: {watcher_status.pid}")
        console.print(f"  PID File: {MYCELIUM_PID_FILE}")
        console.print(f"  Log File: {MYCELIUM_LOG_FILE}")
    else:
        console.print("[yellow]Mycelium Watcher: Not running[/yellow]")

    # Show configuration
    config = get_config()
    vault_path = config.vault_path or DEFAULT_VAULT_PATH
    inbox_path = vault_path / "inbox" / "documents"

    console.print()
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Vault Path: {vault_path}")
    console.print(f"  Vault Exists: {vault_path.exists()}")
    console.print(f"  Inbox Path: {inbox_path}")
    console.print(f"  Inbox Exists: {inbox_path.exists()}")

    # Show recent log entries
    if MYCELIUM_LOG_FILE.exists():
        console.print()
        console.print("[bold]Recent Log (last 5 lines):[/bold]")
        try:
            lines = MYCELIUM_LOG_FILE.read_text().strip().split("\n")
            for line in lines[-5:]:
                console.print(f"  [dim]{line}[/dim]")
        except Exception:
            pass


@mycelium.command("scan")
@click.option(
    "--vault",
    "-v",
    type=click.Path(exists=True),
    help="Path to Obsidian vault",
)
@click.option(
    "--notes/--no-notes",
    default=True,
    help="Generate Obsidian notes for processed documents",
)
@click.option(
    "--archive/--no-archive",
    default=False,
    help="Archive processed files after processing",
)
def mycelium_scan(vault: str | None, notes: bool, archive: bool) -> None:
    """Manually scan and process files in vault inbox."""
    from pathlib import Path

    from alibi.mycelium.watcher import MyceliumWatcher

    vault_path = Path(vault) if vault else None

    console.print(f"[bold blue]Scanning Mycelium Vault Inbox[/bold blue]")

    watcher = MyceliumWatcher(
        vault_path=vault_path,
        generate_notes=notes,
        archive_processed=archive,
    )

    console.print(f"  Inbox: {watcher.inbox_path}")
    console.print()

    if not watcher.inbox_path.exists():
        console.print("[yellow]Inbox directory does not exist.[/yellow]")
        return

    results = watcher.scan_inbox()

    if not results:
        console.print("[yellow]No files found to process.[/yellow]")
        return

    # Show results
    success_count = sum(1 for r in results if r.success and not r.is_duplicate)
    duplicate_count = sum(1 for r in results if r.is_duplicate)
    error_count = sum(1 for r in results if not r.success)

    console.print()
    for result in results:
        if result.success:
            if result.is_duplicate:
                console.print(f"[yellow]Duplicate:[/yellow] {result.file_path.name}")
            else:
                console.print(f"[green]Processed:[/green] {result.file_path.name}")
                if result.extracted_data:
                    vendor_name = result.extracted_data.get("vendor", "Unknown")
                    total = result.extracted_data.get("total", "N/A")
                    console.print(f"  Vendor: {vendor_name}, Total: {total}")
        else:
            console.print(f"[red]Error:[/red] {result.file_path.name}")
            if result.error:
                console.print(f"  {result.error}")

    console.print()
    console.print(f"[bold]Summary:[/bold]")
    console.print(f"  Processed: {success_count}")
    console.print(f"  Duplicates: {duplicate_count}")
    console.print(f"  Errors: {error_count}")


@mycelium.command("sync")
@click.option(
    "--vault",
    "-v",
    type=click.Path(exists=True),
    help="Path to Obsidian vault",
)
@click.option(
    "--notes/--no-notes",
    default=True,
    help="Generate Obsidian notes for processed documents",
)
def mycelium_sync(vault: str | None, notes: bool) -> None:
    """Pull from git and process new files."""
    from pathlib import Path

    from alibi.mycelium.sync import process_after_sync

    vault_path = Path(vault) if vault else None

    console.print(f"[bold blue]Syncing Mycelium Vault[/bold blue]")

    sync_result = process_after_sync(
        vault_path=vault_path,
        generate_notes=notes,
    )

    if not sync_result.success:
        console.print(f"[red]Sync failed:[/red] {sync_result.error}")
        raise click.Abort()

    console.print(
        f"  Commit: {sync_result.commit_hash[:8] if sync_result.commit_hash else 'N/A'}"
    )
    console.print(f"  Message: {sync_result.commit_message or 'N/A'}")
    console.print(f"  New Files: {len(sync_result.new_files)}")
    console.print(f"  Modified Files: {len(sync_result.modified_files)}")

    if sync_result.new_files:
        console.print()
        console.print("[bold]New files processed:[/bold]")
        for f in sync_result.new_files:
            console.print(f"  {f.name}")


@mycelium.command("install-hook")
@click.option(
    "--vault",
    "-v",
    type=click.Path(exists=True),
    help="Path to Obsidian vault",
)
def mycelium_install_hook(vault: str | None) -> None:
    """Install git post-merge hook in vault."""
    from pathlib import Path

    from alibi.mycelium.sync import install_post_pull_hook

    vault_path = Path(vault) if vault else None

    try:
        hook_path = install_post_pull_hook(vault_path)
        console.print(f"[green]Installed git post-merge hook:[/green]")
        console.print(f"  {hook_path}")
        console.print()
        console.print(
            "[dim]The hook will run 'lt mycelium scan' after each git pull.[/dim]"
        )
    except Exception as e:
        console.print(f"[red]Failed to install hook:[/red] {e}")
        raise click.Abort() from e


# ---------------------------------------------------------------------------
# telegram group
# ---------------------------------------------------------------------------


@click.group()
def telegram() -> None:
    """Telegram bot integration.

    Commands for managing the Alibi Telegram bot for quick queries.
    """
    pass


@telegram.command("start")
def telegram_start() -> None:
    """Start the Telegram bot.

    Requires TELEGRAM_BOT_TOKEN environment variable.
    """
    from alibi.telegram.main import run

    config = get_config()

    if not config.telegram_token:
        console.print("[red]Error: TELEGRAM_BOT_TOKEN not set[/red]")
        console.print("Please set TELEGRAM_BOT_TOKEN environment variable.")
        raise click.Abort()

    console.print("[bold blue]Starting Alibi Telegram Bot...[/bold blue]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Bot stopped[/yellow]")


# ---------------------------------------------------------------------------
# vectordb group
# ---------------------------------------------------------------------------


def _get_vector_index() -> Any:
    """Create VectorIndex using config lance_path if set."""
    from alibi.vectordb.index import VectorIndex

    config = get_config()
    lance_path = config.get_lance_path()
    return VectorIndex(db_path=lance_path)


@click.group()
def vectordb() -> None:
    """Vector database management for semantic search.

    Commands for managing the LanceDB vector index used for
    semantic search queries like "find receipts like this one".
    """
    pass


@vectordb.command("init")
@click.option("--space", "-s", default="default", help="Space ID to index")
@click.option("--rebuild", "-r", is_flag=True, help="Drop and rebuild entire index")
def vectordb_init(space: str, rebuild: bool) -> None:
    """Initialize or rebuild the vector index."""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        console.print("Run 'lt init' to initialize.")
        return

    index = _get_vector_index()

    if index.is_initialized() and not rebuild:
        stats = index.get_stats()
        console.print("[green]Vector index already initialized.[/green]")
        console.print(f"  Total vectors: {stats['total']}")
        console.print(f"  Transactions: {stats['transactions']}")
        console.print(f"  Artifacts: {stats['artifacts']}")
        console.print(f"  Items: {stats['items']}")
        console.print("\nUse --rebuild to rebuild from scratch.")
        return

    console.print("[bold blue]Building vector index...[/bold blue]")
    console.print(f"  Space: {space}")
    console.print()

    # Count totals for progress bar
    txn_row = db_manager.fetchone("SELECT COUNT(*) FROM facts")
    artifact_row = db_manager.fetchone("SELECT COUNT(*) FROM documents")
    item_row = db_manager.fetchone("SELECT COUNT(*) FROM fact_items")
    txn_count = txn_row[0] if txn_row else 0
    artifact_count = artifact_row[0] if artifact_row else 0
    item_count = item_row[0] if item_row else 0

    total = txn_count + artifact_count + item_count

    if total == 0:
        console.print("[yellow]No data to index.[/yellow]")
        index.initialize()
        console.print("[green]Empty index created.[/green]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing...", total=total)

        def progress_callback(
            entity_type: str, current_idx: int, total_idx: int
        ) -> None:
            progress.update(
                task,
                completed=current_idx
                + (
                    0
                    if entity_type == "transactions"
                    else (
                        txn_count
                        if entity_type == "artifacts"
                        else txn_count + artifact_count
                    )
                ),
            )
            progress.update(task, description=f"Indexing {entity_type}...")

        try:
            counts = index.rebuild_from_db(
                db_manager,
                space_id=space,
                progress_callback=progress_callback,
            )
        except Exception as e:
            console.print(f"\n[red]Error building index:[/red] {e}")
            console.print(
                "\nMake sure Ollama is running with the nomic-embed-text model:"
            )
            console.print("  ollama pull nomic-embed-text")
            console.print("  ollama serve")
            raise click.Abort() from e

    console.print()
    console.print("[green]Vector index built successfully![/green]")
    console.print(f"  Transactions: {counts['transactions']}")
    console.print(f"  Artifacts: {counts['artifacts']}")
    console.print(f"  Items: {counts['items']}")
    console.print(f"  Location: {index.db_path}")


@vectordb.command("status")
def vectordb_status() -> None:
    """Show vector index status."""
    index = _get_vector_index()

    console.print("[bold]Vector Index Status[/bold]\n")
    console.print(f"  Path: {index.db_path}")

    if not index.is_initialized():
        console.print("  [yellow]Status: Not initialized[/yellow]")
        console.print("\nRun 'lt vectordb init' to create the index.")
        return

    stats = index.get_stats()
    console.print("  [green]Status: Initialized[/green]")
    console.print(f"\n[bold]Index Contents:[/bold]")
    console.print(f"  Total vectors: {stats['total']}")
    console.print(f"  Transactions: {stats['transactions']}")
    console.print(f"  Artifacts: {stats['artifacts']}")
    console.print(f"  Items: {stats['items']}")


@vectordb.command("search")
@click.argument("query_text")
@click.option("--limit", "-l", default=10, help="Maximum results")
@click.option(
    "--type",
    "-t",
    "entity_type",
    type=click.Choice(["transaction", "artifact", "item", "all"]),
    default="all",
    help="Filter by entity type",
)
def vectordb_search(query_text: str, limit: int, entity_type: str) -> None:
    """Search using vector similarity.

    Example: lt vectordb search "grocery shopping"
    """
    from alibi.vectordb.index import IndexType
    from alibi.vectordb.search import semantic_search

    index = _get_vector_index()

    if not index.is_initialized():
        console.print("[yellow]Vector index not initialized.[/yellow]")
        console.print("Run 'lt vectordb init' to create the index.")
        return

    # Build type filter
    index_types = None
    if entity_type != "all":
        type_map = {
            "transaction": IndexType.TRANSACTION,
            "artifact": IndexType.ARTIFACT,
            "item": IndexType.ITEM,
        }
        index_types = [type_map[entity_type]]

    console.print(f"[bold blue]Semantic search:[/bold blue] {query_text}\n")

    try:
        results = semantic_search(
            index, query_text, limit=limit, index_types=index_types
        )
    except Exception as e:
        console.print(f"[red]Search error:[/red] {e}")
        console.print("\nMake sure Ollama is running with nomic-embed-text.")
        raise click.Abort() from e

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Results ({len(results)} found)")
    table.add_column("Score", justify="right", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Description", max_width=40)
    table.add_column("Amount", justify="right")
    table.add_column("Date")

    for r in results:
        score_str = f"{r.score:.2f}"
        desc = r.description[:40] if r.description else r.vendor or ""
        amount_str = f"{r.amount:.2f} {r.currency}" if r.amount else ""

        table.add_row(
            score_str,
            r.entity_type,
            desc,
            amount_str,
            r.date or "",
        )

    console.print(table)
