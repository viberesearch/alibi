"""Command-line interface for Alibi.

Thin entry point that imports and registers all commands from
alibi.commands.* modules.
"""

from __future__ import annotations

import click

from alibi import __version__
from alibi.commands.shared import set_verbosity
from alibi.config import get_config  # noqa: F401 — re-export for test patching
from alibi.db.connection import get_db  # noqa: F401 — re-export for test patching


@click.group()
@click.version_option(version=__version__, prog_name="alibi")
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-essential output")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, quiet: bool, verbose: bool) -> None:
    """Alibi - Life Tracker: Transaction-based life management system.

    Track financial transactions, documents, and assets locally with privacy.
    """
    ctx.ensure_object(dict)
    if quiet:
        set_verbosity(0)
    elif verbose:
        set_verbosity(2)
    else:
        set_verbosity(1)


# ---------------------------------------------------------------------------
# Register standalone commands from alibi.commands.process
# ---------------------------------------------------------------------------
from alibi.commands.process import gemini_batch, gemini_vision, process, scan

cli.add_command(scan)
cli.add_command(process)
cli.add_command(gemini_batch)
cli.add_command(gemini_vision)

# ---------------------------------------------------------------------------
# Register groups from alibi.commands.facts
# ---------------------------------------------------------------------------
from alibi.commands.facts import facts

cli.add_command(facts)

# ---------------------------------------------------------------------------
# Register groups from alibi.commands.enrich
# ---------------------------------------------------------------------------
from alibi.commands.enrich import enrich

cli.add_command(enrich)

# ---------------------------------------------------------------------------
# Register groups from alibi.commands.analytics
# ---------------------------------------------------------------------------
from alibi.commands.analytics import analytics, nutrition, predictions, report

cli.add_command(analytics)
cli.add_command(report)
cli.add_command(nutrition)
cli.add_command(predictions)

# ---------------------------------------------------------------------------
# Register groups from alibi.commands.corrections
# ---------------------------------------------------------------------------
from alibi.commands.corrections import corrections

cli.add_command(corrections)

# ---------------------------------------------------------------------------
# Register groups from alibi.commands.maintenance
# ---------------------------------------------------------------------------
from alibi.commands.maintenance import db, maintain, schedule

cli.add_command(db)
cli.add_command(maintain)
cli.add_command(schedule)

# ---------------------------------------------------------------------------
# Register groups from alibi.commands.users
# ---------------------------------------------------------------------------
from alibi.commands.users import user

cli.add_command(user)

# ---------------------------------------------------------------------------
# Register groups from alibi.commands.yaml_ops
# ---------------------------------------------------------------------------
from alibi.commands.yaml_ops import clouds, yaml

cli.add_command(yaml)
cli.add_command(clouds)

# ---------------------------------------------------------------------------
# Register commands and groups from alibi.commands.misc
# ---------------------------------------------------------------------------
from alibi.commands.misc import (
    budget,
    completion,
    daemon,
    export,
    health,
    import_cmd,
    init,
    match,
    mycelium,
    query,
    reingest,
    review,
    search_cmd,
    serve,
    setup,
    status,
    telegram,
    template,
    transactions,
    vectordb,
    verify_cmd,
    version,
)

cli.add_command(health)
cli.add_command(init)
cli.add_command(setup)
cli.add_command(status)
cli.add_command(version)
cli.add_command(completion)
cli.add_command(query)
cli.add_command(transactions)
cli.add_command(match)
cli.add_command(search_cmd)
cli.add_command(reingest)
cli.add_command(template)
cli.add_command(review)
cli.add_command(verify_cmd)
cli.add_command(export)
cli.add_command(import_cmd)
cli.add_command(serve)
cli.add_command(daemon)
cli.add_command(mycelium)
cli.add_command(telegram)
cli.add_command(vectordb)
cli.add_command(budget)


if __name__ == "__main__":
    cli()
