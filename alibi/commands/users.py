"""User management commands."""

from __future__ import annotations

import click
from rich.table import Table

from alibi.commands.shared import console
from alibi.db.connection import get_db


@click.group()
def user() -> None:
    """Manage users and API keys."""
    pass


@user.command("create")
@click.option("--name", "-n", default=None, help="Display name (optional)")
def user_create(name: str | None) -> None:
    """Create a new user."""
    from alibi.services.auth import create_user

    db = get_db()
    if not db.is_initialized():
        db.initialize()
    result = create_user(db, name)
    console.print(f"[green]Created user:[/green] {result['id']}")
    if result["name"]:
        console.print(f"  Name: {result['name']}")


@user.command("list")
def user_list() -> None:
    """List all users."""
    from alibi.services.auth import get_display_name, list_contacts, list_users

    db = get_db()
    if not db.is_initialized():
        db.initialize()
    users = list_users(db)
    if not users:
        console.print("[yellow]No users found.[/yellow]")
        return

    table = Table(title="Users")
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Name")
    table.add_column("Contacts")

    for u in users:
        contacts = list_contacts(db, u["id"])
        contact_strs = [f"{c['contact_type']}:{c['value']}" for c in contacts]
        table.add_row(
            u["id"][:12] + "...",
            get_display_name(u),
            ", ".join(contact_strs) if contact_strs else "",
        )
    console.print(table)


@user.command("key-create")
@click.argument("user_id")
@click.option("--label", "-l", default="default", help="Key label")
def user_key_create(user_id: str, label: str) -> None:
    """Generate a mnemonic API key for a user. Shows the key ONCE."""
    from alibi.services.auth import create_api_key, get_user

    db = get_db()
    if not db.is_initialized():
        db.initialize()

    u = get_user(db, user_id)
    if not u:
        console.print(f"[red]User not found: {user_id}[/red]")
        raise click.Abort()

    result = create_api_key(db, user_id, label=label)
    console.print("[green]API key created.[/green]")
    console.print(f"  Label: {result['label']}")
    console.print(f"  Prefix: {result['prefix']}")
    console.print()
    console.print(
        "[bold yellow]Save this key now -- it cannot be shown again:[/bold yellow]"
    )
    console.print(f"\n  {result['mnemonic']}\n")


@user.command("key-list")
@click.argument("user_id")
def user_key_list(user_id: str) -> None:
    """List API keys for a user (no plaintext shown)."""
    from alibi.services.auth import list_api_keys

    db = get_db()
    if not db.is_initialized():
        db.initialize()

    keys = list_api_keys(db, user_id)
    if not keys:
        console.print("[yellow]No API keys found.[/yellow]")
        return

    table = Table(title="API Keys")
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Prefix")
    table.add_column("Label")
    table.add_column("Last Used")
    table.add_column("Active")

    for k in keys:
        table.add_row(
            k["id"][:12] + "...",
            k["key_prefix"],
            k["label"],
            str(k["last_used_at"] or "never"),
            "yes" if k["is_active"] else "no",
        )
    console.print(table)


@user.command("key-revoke")
@click.argument("key_id")
def user_key_revoke(key_id: str) -> None:
    """Revoke an API key."""
    from alibi.services.auth import revoke_api_key

    db = get_db()
    if not db.is_initialized():
        db.initialize()

    if revoke_api_key(db, key_id):
        console.print(f"[green]Revoked key {key_id[:12]}...[/green]")
    else:
        console.print(f"[red]Key not found or already revoked: {key_id}[/red]")


@user.command("set-name")
@click.argument("user_id")
@click.argument("name")
def user_set_name(user_id: str, name: str) -> None:
    """Set display name for a user."""
    from alibi.services.auth import update_user

    db = get_db()
    if not db.is_initialized():
        db.initialize()

    if update_user(db, user_id, name=name):
        console.print(f"[green]Name set to: {name}[/green]")
    else:
        console.print(f"[red]User not found: {user_id}[/red]")


@user.command("add-contact")
@click.argument("user_id")
@click.argument("contact_type", type=click.Choice(["telegram", "email"]))
@click.argument("value")
@click.option("--label", "-l", default=None, help="Contact label")
def user_add_contact(
    user_id: str, contact_type: str, value: str, label: str | None
) -> None:
    """Add a contact (telegram/email) to a user."""
    from alibi.services.auth import add_contact

    db = get_db()
    if not db.is_initialized():
        db.initialize()

    result = add_contact(db, user_id, contact_type, value, label)
    if result:
        console.print(f"[green]Added {contact_type}: {value}[/green]")
    else:
        console.print(f"[red]User not found: {user_id}[/red]")


@user.command("remove-contact")
@click.argument("contact_id")
def user_remove_contact(contact_id: str) -> None:
    """Remove a contact by ID."""
    from alibi.services.auth import remove_contact

    db = get_db()
    if not db.is_initialized():
        db.initialize()

    if remove_contact(db, contact_id):
        console.print("[green]Contact removed.[/green]")
    else:
        console.print(f"[red]Contact not found: {contact_id}[/red]")
