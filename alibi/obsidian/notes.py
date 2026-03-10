"""Obsidian note generation for transactions, items, and facts."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from alibi.config import get_config
from alibi.db.models import Artifact, Item

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager


# Template for transaction notes
# Template for item notes
ITEM_NOTE_TEMPLATE = """---
type: item
id: "{id}"
name: "{name}"
category: "{category}"
purchase_date: {purchase_date}
purchase_price: {purchase_price}
current_value: {current_value}
warranty_expires: {warranty_expires}
status: {status}
tags: [{tags}]
created: {created}
---

# {name}

**Category**: {category}
**Status**: {status}

## Purchase Information

- **Date**: {purchase_date}
- **Price**: {formatted_price}
- **Location**: {location}

## Warranty

- **Expires**: {warranty_expires}
- **Type**: {warranty_type}
- **Insurance Covered**: {insurance_covered}

## Specifications

{specifications}

## Linked Documents

{artifact_links}

## Notes

{notes}
"""


# Template for contract notes
CONTRACT_NOTE_TEMPLATE = """---
type: contract
id: "{id}"
vendor: "{vendor}"
date: {date}
start_date: {start_date}
end_date: {end_date}
amount: {amount}
currency: {currency}
payment_terms: {payment_terms}
renewal: {renewal}
tags: [{tags}]
artifacts: [{artifacts}]
created: {created}
---

# Contract: {vendor}

| | |
|---|---|
| **Contract #** | {document_id} |
| **Effective Date** | {date} |
| **Start Date** | {start_date} |
| **End Date** | {end_date} |
| **Value** | {formatted_amount} |
| **Payment Terms** | {payment_terms} |
| **Renewal** | {renewal} |

## Vendor

{vendor_details}

## Services / Products

{line_items_table}

## Linked Documents

{artifact_links}

## Notes

{notes}
"""

# Template for warranty notes
WARRANTY_NOTE_TEMPLATE = """---
type: warranty
id: "{id}"
vendor: "{vendor}"
product: "{product}"
date: {date}
warranty_end: {warranty_end}
warranty_type: {warranty_type}
amount: {amount}
currency: {currency}
tags: [{tags}]
artifacts: [{artifacts}]
created: {created}
---

# Warranty: {product}

| | |
|---|---|
| **Product** | {product} |
| **Model** | {model} |
| **Serial Number** | {serial_number} |
| **Vendor** | {vendor} |
| **Purchase Date** | {date} |
| **Purchase Price** | {formatted_amount} |
| **Warranty Type** | {warranty_type} |
| **Warranty Expires** | {warranty_end} |
| **Certificate #** | {document_id} |

## Coverage

{coverage}

## Vendor

{vendor_details}

## Linked Documents

{artifact_links}

## Notes

{notes}
"""


# Template for fact notes (v2 atom-cloud-fact)
FACT_NOTE_TEMPLATE = """---
type: fact
id: "{id}"
fact_type: {fact_type}
date: {date}
vendor: "{vendor}"
vendor_key: "{vendor_key}"
amount: {amount}
currency: {currency}
status: {status}
documents: [{documents}]
tags: [{tags}]
created: {created}
---

# {vendor} - {formatted_amount}

| | |
|---|---|
| **Date** | {date} |
| **Amount** | {formatted_amount} |
| **Type** | {fact_type} |
| **Status** | {status} |
{payment_rows}

## Vendor

{vendor_details}

## Line Items

{line_items_table}

## Source Documents

{document_links}

## Notes

{notes}
"""


def format_currency(amount: Decimal | None, currency: str = "EUR") -> str:
    """Format amount with currency symbol."""
    if amount is None:
        return "N/A"

    symbols = {
        "EUR": "\u20ac",
        "USD": "$",
        "GBP": "\u00a3",
        "CHF": "CHF ",
    }
    symbol = symbols.get(currency, f"{currency} ")
    return f"{symbol}{amount:,.2f}"


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    # Replace problematic characters
    replacements = {
        "/": "-",
        "\\": "-",
        ":": "-",
        "*": "",
        "?": "",
        '"': "",
        "<": "",
        ">": "",
        "|": "",
    }
    result = name
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    return result.strip()


def _format_vendor_details(artifacts: list[Artifact]) -> str:
    """Format vendor details from linked artifacts."""
    if not artifacts:
        return "_No vendor details_"

    # Use first artifact with vendor info
    for artifact in artifacts:
        parts = []
        if artifact.vendor_address:
            parts.append(f"**Address**: {artifact.vendor_address}")
        if artifact.vendor_phone:
            parts.append(f"**Phone**: {artifact.vendor_phone}")
        if artifact.vendor_website:
            parts.append(f"**Website**: {artifact.vendor_website}")
        if artifact.vendor_vat:
            parts.append(f"**VAT**: {artifact.vendor_vat}")
        if artifact.vendor_tax_id:
            parts.append(f"**Tax ID**: {artifact.vendor_tax_id}")
        if parts:
            return "\n".join(parts)

    return "_No vendor details_"


def _format_line_items_table(line_items: list[dict[str, Any]]) -> str:
    """Format line items as a markdown table.

    Args:
        line_items: List of line item dicts from DB row or refiner output.

    Returns:
        Markdown table string
    """
    if not line_items:
        return "_No line items_"

    rows = [
        "| Item | Brand | Qty | Unit | Size | Unit Price | Total | EUR/std | Category |",
        "|------|-------|-----|------|------|-----------|-------|---------|----------|",
    ]

    for item in line_items:
        name = item.get("name", "")
        brand = item.get("brand") or ""
        qty = item.get("quantity", "1")
        unit = item.get("unit") or "pcs"
        unit_quantity = item.get("unit_quantity")
        unit_price = item.get("unit_price")
        total_price = item.get("total_price")
        category = item.get("category") or ""
        currency = item.get("currency", "EUR")
        comp_price = item.get("comparable_unit_price")
        comp_unit = item.get("comparable_unit")

        size_str = f"{_fmt_num(unit_quantity)}{unit}" if unit_quantity else ""
        up_str = f"{float(unit_price):.2f}" if unit_price else ""
        tp_str = f"{float(total_price):.2f} {currency}" if total_price else ""
        comp_str = (
            f"{float(comp_price):.2f}/{comp_unit}" if comp_price and comp_unit else ""
        )

        rows.append(
            f"| {name} | {brand} | {_fmt_num(qty)} | {unit} "
            f"| {size_str} | {up_str} | {tp_str} | {comp_str} | {category} |"
        )

    return "\n".join(rows)


def _fmt_num(val: Any) -> str:
    """Format a numeric value, dropping trailing zeros."""
    if val is None:
        return ""
    f = float(val)
    return f"{f:g}"


def _format_vendor_details_from_atom(vendor_data: dict[str, Any] | None) -> str:
    """Format vendor details from a VENDOR atom data dict."""
    if not vendor_data:
        return "_No vendor details_"

    parts = []
    if vendor_data.get("address"):
        parts.append(f"**Address**: {vendor_data['address']}")
    if vendor_data.get("phone"):
        parts.append(f"**Phone**: {vendor_data['phone']}")
    if vendor_data.get("website"):
        parts.append(f"**Website**: {vendor_data['website']}")
    if vendor_data.get("vat_number"):
        parts.append(f"**VAT Number**: {vendor_data['vat_number']}")
    if vendor_data.get("tax_id"):
        parts.append(f"**Tax ID**: {vendor_data['tax_id']}")
    return "\n".join(parts) if parts else "_No vendor details_"


def _format_payment_rows(payments: list[dict[str, Any]] | None) -> str:
    """Format payment details as markdown table rows."""
    if not payments:
        return ""

    rows = []
    for i, p in enumerate(payments):
        method = p.get("method", "")
        card = p.get("card_last4", "")
        amount = p.get("amount", "")
        label = f"Payment {i + 1}" if len(payments) > 1 else "Payment"

        parts = []
        if method:
            parts.append(method.title())
        if card:
            parts.append(f"*{card}")
        if amount:
            parts.append(str(amount))

        if parts:
            rows.append(f"| **{label}** | {' '.join(parts)} |")

    return "\n".join(rows)


def _format_fact_items_table(items: list[dict[str, Any]]) -> str:
    """Format fact items as a markdown table."""
    if not items:
        return "_No line items_"

    rows = [
        "| Item | Brand | Qty | Unit | Unit Price | Total | EUR/std | Category |",
        "|------|-------|-----|------|-----------|-------|---------|----------|",
    ]

    for item in items:
        name = item.get("name", "")
        brand = item.get("brand") or ""
        qty = item.get("quantity", "1")
        unit = item.get("unit") or "pcs"
        unit_qty = item.get("unit_quantity")
        unit_price = item.get("unit_price")
        total_price = item.get("total_price")
        category = item.get("category") or ""
        comp_price = item.get("comparable_unit_price")
        comp_unit = item.get("comparable_unit")

        # Show "4 x 0.355l" when unit_quantity present, else just "4"
        if unit_qty is not None:
            qty_str = f"{_fmt_num(qty)} x {_fmt_num(unit_qty)}{unit}"
        else:
            qty_str = _fmt_num(qty)

        up_str = f"{float(unit_price):.2f}" if unit_price else ""
        tp_str = f"{float(total_price):.2f}" if total_price else ""
        comp_str = (
            f"{float(comp_price):.2f}/{comp_unit}" if comp_price and comp_unit else ""
        )

        rows.append(
            f"| {name} | {brand} | {qty_str} | {unit} "
            f"| {up_str} | {tp_str} | {comp_str} | {category} |"
        )

    return "\n".join(rows)


def generate_fact_note(
    fact: dict[str, Any],
    items: list[dict[str, Any]] | None = None,
    documents: list[dict[str, Any]] | None = None,
    vendor_atom: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Generate Obsidian markdown note for a v2 fact.

    Args:
        fact: Fact dict from v2_store (id, vendor, total_amount, etc.)
        items: Fact item dicts from get_fact_items()
        documents: Source document dicts from get_fact_documents()
        vendor_atom: Vendor atom data dict (address, phone, etc.)
        tags: Optional list of tag names

    Returns:
        Markdown content for the note
    """
    items = items or []
    documents = documents or []
    tags = tags or []

    # Parse payments from JSON if string
    payments = fact.get("payments")
    if isinstance(payments, str):
        payments = json.loads(payments)

    # Format document links
    doc_links = []
    doc_ids = []
    for doc in documents:
        file_path = doc.get("file_path", "")
        if file_path:
            name = Path(file_path).name
            doc_links.append(f"- [[{name}]]")
        doc_ids.append(f'"{doc.get("id", "")}"')

    document_links = "\n".join(doc_links) if doc_links else "_No source documents_"

    # Format tags for YAML
    tags_yaml = ", ".join(f'"{t}"' for t in tags) if tags else ""

    # Parse amount
    amount_val = fact.get("total_amount")
    if amount_val is not None:
        amount_decimal = Decimal(str(amount_val))
    else:
        amount_decimal = None

    event_date = fact.get("event_date") or ""
    if isinstance(event_date, date):
        event_date = event_date.isoformat()

    currency = fact.get("currency") or "EUR"

    return FACT_NOTE_TEMPLATE.format(
        id=fact.get("id", ""),
        fact_type=fact.get("fact_type", "purchase"),
        date=event_date,
        vendor=fact.get("vendor") or "Unknown",
        vendor_key=fact.get("vendor_key") or "",
        amount=f"{amount_decimal:.2f}" if amount_decimal is not None else "0",
        currency=currency,
        status=fact.get("status", "confirmed"),
        documents=", ".join(doc_ids),
        tags=tags_yaml,
        created=datetime.now().isoformat(),
        formatted_amount=format_currency(amount_decimal, currency),
        payment_rows=_format_payment_rows(payments),
        vendor_details=_format_vendor_details_from_atom(vendor_atom),
        line_items_table=_format_fact_items_table(items),
        document_links=document_links,
        notes="",
    )


def generate_item_note(
    item: Item,
    artifacts: list[Artifact] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Generate Obsidian markdown note for an item.

    Args:
        item: Item to generate note for
        artifacts: Optional list of linked artifacts
        tags: Optional list of tag names

    Returns:
        Markdown content for the note
    """
    artifacts = artifacts or []
    tags = tags or []

    # Format artifacts as links
    artifact_links = ""
    artifact_ids: list[str] = []
    for artifact in artifacts:
        artifact_ids.append(f'"{artifact.id}"')
        if artifact.file_path:
            file_path = Path(artifact.file_path)
            link_type = artifact.type.value
            artifact_links += f"- [[{file_path.name}]] ({link_type})\n"
        else:
            artifact_links += f"- {artifact.type.value}: {artifact.id[:8]}\n"

    if not artifact_links:
        artifact_links = "_No linked documents_"

    # Format tags for YAML
    tags_yaml = ", ".join(f'"{t}"' for t in tags) if tags else ""

    # Format specifications if available
    specifications = ""
    if item.model:
        specifications += f"- **Model**: {item.model}\n"
    if item.serial_number:
        specifications += f"- **Serial Number**: {item.serial_number}\n"
    if not specifications:
        specifications = "_No specifications recorded_"

    return ITEM_NOTE_TEMPLATE.format(
        id=item.id,
        name=item.name,
        category=item.category or "uncategorized",
        purchase_date=(
            item.purchase_date.isoformat() if item.purchase_date else "unknown"
        ),
        purchase_price=f"{item.purchase_price:.2f}" if item.purchase_price else "0",
        current_value=f"{item.current_value:.2f}" if item.current_value else "0",
        warranty_expires=(
            item.warranty_expires.isoformat() if item.warranty_expires else "none"
        ),
        status=item.status.value if item.status else "active",
        tags=tags_yaml,
        created=datetime.now().isoformat(),
        formatted_price=format_currency(item.purchase_price, "EUR"),
        location="Unknown",  # Item model doesn't have location field
        warranty_type=item.warranty_type or "standard",
        insurance_covered="Yes" if item.insurance_covered else "No",
        specifications=specifications,
        artifact_links=artifact_links,
        notes="",
    )


def _format_artifact_links(artifacts: list[Artifact]) -> str:
    """Format artifact links for note templates."""
    if not artifacts:
        return "_No linked documents_"

    lines = []
    for artifact in artifacts:
        if artifact.file_path:
            file_path = Path(artifact.file_path)
            lines.append(f"- [[{file_path.name}]] ({artifact.type.value})")
        else:
            lines.append(f"- {artifact.type.value}: {artifact.id[:8]}")

    return "\n".join(lines) if lines else "_No linked documents_"


def _format_artifact_ids(artifacts: list[Artifact]) -> str:
    """Format artifact IDs for YAML frontmatter."""
    return ", ".join(f'"{a.id}"' for a in artifacts)


def generate_contract_note(
    artifact: Artifact,
    extracted_data: dict[str, Any] | None = None,
    artifacts: list[Artifact] | None = None,
    tags: list[str] | None = None,
    line_items: list[dict[str, Any]] | None = None,
) -> str:
    """Generate Obsidian markdown note for a contract.

    Args:
        artifact: Contract artifact
        extracted_data: Extracted/refined data from LLM
        artifacts: All linked artifacts (including self)
        tags: Optional tag names
        line_items: Optional line item dicts

    Returns:
        Markdown content for the note
    """
    artifacts = artifacts or [artifact]
    tags = tags or []
    data = extracted_data or artifact.extracted_data or {}

    tags_yaml = ", ".join(f'"{t}"' for t in tags) if tags else ""
    date_str = artifact.document_date.isoformat() if artifact.document_date else ""

    return CONTRACT_NOTE_TEMPLATE.format(
        id=artifact.id,
        vendor=artifact.vendor or data.get("vendor") or "Unknown",
        date=date_str,
        start_date=data.get("start_date") or date_str or "",
        end_date=data.get("end_date") or "",
        amount=f"{artifact.amount:.2f}" if artifact.amount else "0",
        currency=artifact.currency or "EUR",
        payment_terms=data.get("payment_terms") or "",
        renewal=data.get("renewal") or "",
        document_id=artifact.document_id or data.get("document_id") or "",
        tags=tags_yaml,
        artifacts=_format_artifact_ids(artifacts),
        created=datetime.now().isoformat(),
        formatted_amount=format_currency(artifact.amount, artifact.currency or "EUR"),
        vendor_details=_format_vendor_details(artifacts),
        line_items_table=_format_line_items_table(line_items or []),
        artifact_links=_format_artifact_links(artifacts),
        notes="",
    )


def generate_warranty_note(
    artifact: Artifact,
    extracted_data: dict[str, Any] | None = None,
    artifacts: list[Artifact] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Generate Obsidian markdown note for a warranty.

    Args:
        artifact: Warranty artifact
        extracted_data: Extracted/refined data from LLM
        artifacts: All linked artifacts (including self)
        tags: Optional tag names

    Returns:
        Markdown content for the note
    """
    artifacts = artifacts or [artifact]
    tags = tags or []
    data = extracted_data or artifact.extracted_data or {}

    tags_yaml = ", ".join(f'"{t}"' for t in tags) if tags else ""
    date_str = artifact.document_date.isoformat() if artifact.document_date else ""

    product = (
        data.get("product") or data.get("product_name") or artifact.vendor or "Unknown"
    )
    coverage = data.get("coverage") or "_No coverage details_"

    return WARRANTY_NOTE_TEMPLATE.format(
        id=artifact.id,
        vendor=artifact.vendor or data.get("vendor") or "Unknown",
        product=product,
        date=date_str,
        warranty_end=data.get("warranty_end") or data.get("warranty_expires") or "",
        warranty_type=data.get("warranty_type") or "standard",
        amount=f"{artifact.amount:.2f}" if artifact.amount else "0",
        currency=artifact.currency or "EUR",
        model=data.get("model") or data.get("product_model") or "",
        serial_number=data.get("serial_number") or "",
        document_id=artifact.document_id or data.get("document_id") or "",
        tags=tags_yaml,
        artifacts=_format_artifact_ids(artifacts),
        created=datetime.now().isoformat(),
        formatted_amount=format_currency(artifact.amount, artifact.currency or "EUR"),
        vendor_details=_format_vendor_details(artifacts),
        coverage=coverage,
        artifact_links=_format_artifact_links(artifacts),
        notes="",
    )


def get_note_filename(
    entity_type: str,
    entity_date: date | None,
    entity_name: str,
) -> str:
    """Generate filename for a note.

    Args:
        entity_type: Type of entity (transaction, item)
        entity_date: Date associated with entity
        entity_name: Name/vendor of entity

    Returns:
        Sanitized filename without extension
    """
    date_str = entity_date.strftime("%Y-%m-%d") if entity_date else "unknown"
    name_sanitized = sanitize_filename(entity_name)[:50]
    return f"{date_str}_{entity_type}_{name_sanitized}"


class NoteExporter:
    """Exports transactions and items to Obsidian notes."""

    def __init__(
        self,
        db: DatabaseManager,
        vault_path: Path | None = None,
    ) -> None:
        """Initialize the exporter.

        Args:
            db: Database manager for loading data
            vault_path: Path to Obsidian vault (defaults to config)
        """
        self.db = db
        config = get_config()
        self.vault_path = vault_path or config.vault_path
        if self.vault_path is None:
            raise ValueError("No vault path configured. Set ALIBI_VAULT_PATH.")

    def _get_note_directory(self, space_id: str, entity_type: str) -> Path:
        """Get the directory for notes of a given type.

        Args:
            space_id: Space the entity belongs to
            entity_type: Type of entity (transactions, items)

        Returns:
            Path to the note directory
        """
        if self.vault_path is None:
            raise ValueError("Vault path not configured")

        # Use space_id as subdirectory, with shared vs private
        if space_id == "default" or space_id.startswith("shared"):
            base = self.vault_path / "shared" / "finances" / entity_type
        else:
            base = self.vault_path / space_id / "finances" / entity_type

        return base

    def _get_line_items(self, fact_id: str) -> list[dict[str, Any]]:
        """Query fact items for a fact.

        Args:
            fact_id: Fact ID

        Returns:
            List of fact item dicts
        """
        rows = self.db.fetchall(
            """
            SELECT name, quantity, unit_price, total_price, category, brand
            FROM fact_items
            WHERE fact_id = ?
            ORDER BY name
            """,
            (fact_id,),
        )
        return [dict(r) for r in rows]

    def export_item(
        self,
        item: Item,
        artifacts: list[Artifact] | None = None,
        tags: list[str] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Export an item to an Obsidian note.

        Args:
            item: Item to export
            artifacts: Linked artifacts
            tags: Tag names
            overwrite: Whether to overwrite existing file

        Returns:
            Path to the created note
        """
        note_dir = self._get_note_directory(item.space_id, "items")
        note_dir.mkdir(parents=True, exist_ok=True)

        filename = get_note_filename(
            "item",
            item.purchase_date,
            item.name,
        )
        note_path = note_dir / f"{filename}.md"

        if note_path.exists() and not overwrite:
            counter = 1
            while note_path.exists():
                note_path = note_dir / f"{filename}_{counter}.md"
                counter += 1

        content = generate_item_note(item, artifacts, tags)
        note_path.write_text(content)

        return note_path

    def export_contract(
        self,
        artifact: Artifact,
        extracted_data: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        line_items: list[dict[str, Any]] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Export a contract artifact to an Obsidian note.

        Args:
            artifact: Contract artifact
            extracted_data: Extracted/refined data
            tags: Tag names
            line_items: Line item dicts
            overwrite: Whether to overwrite existing file

        Returns:
            Path to the created note
        """
        note_dir = self._get_note_directory(artifact.space_id, "contracts")
        note_dir.mkdir(parents=True, exist_ok=True)

        filename = get_note_filename(
            "contract",
            artifact.document_date,
            artifact.vendor or "unknown",
        )
        note_path = note_dir / f"{filename}.md"

        if note_path.exists() and not overwrite:
            counter = 1
            while note_path.exists():
                note_path = note_dir / f"{filename}_{counter}.md"
                counter += 1

        content = generate_contract_note(
            artifact, extracted_data, [artifact], tags, line_items
        )
        note_path.write_text(content)

        return note_path

    def export_warranty(
        self,
        artifact: Artifact,
        extracted_data: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Export a warranty artifact to an Obsidian note.

        Args:
            artifact: Warranty artifact
            extracted_data: Extracted/refined data
            tags: Tag names
            overwrite: Whether to overwrite existing file

        Returns:
            Path to the created note
        """
        note_dir = self._get_note_directory(artifact.space_id, "warranties")
        note_dir.mkdir(parents=True, exist_ok=True)

        filename = get_note_filename(
            "warranty",
            artifact.document_date,
            artifact.vendor or "unknown",
        )
        note_path = note_dir / f"{filename}.md"

        if note_path.exists() and not overwrite:
            counter = 1
            while note_path.exists():
                note_path = note_dir / f"{filename}_{counter}.md"
                counter += 1

        content = generate_warranty_note(artifact, extracted_data, [artifact], tags)
        note_path.write_text(content)

        return note_path

    def export_fact(
        self,
        fact: dict[str, Any],
        tags: list[str] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Export a v2 fact to an Obsidian note.

        Automatically loads items, documents, and vendor atom from DB.

        Args:
            fact: Fact dict from v2_store
            tags: Tag names
            overwrite: Whether to overwrite existing file

        Returns:
            Path to the created note
        """
        from alibi.db.v2_store import (
            get_fact_documents,
            get_fact_items,
            get_fact_vendor_atom,
        )

        note_dir = self._get_note_directory("default", "facts")
        note_dir.mkdir(parents=True, exist_ok=True)

        event_date = fact.get("event_date")
        if isinstance(event_date, str):
            event_date = date.fromisoformat(event_date)

        filename = get_note_filename(
            "fact",
            event_date,
            fact.get("vendor") or "unknown",
        )
        note_path = note_dir / f"{filename}.md"

        if note_path.exists() and not overwrite:
            counter = 1
            while note_path.exists():
                note_path = note_dir / f"{filename}_{counter}.md"
                counter += 1

        items = get_fact_items(self.db, fact["id"])
        documents = get_fact_documents(self.db, fact["id"])
        vendor_atom = get_fact_vendor_atom(self.db, fact["id"])

        content = generate_fact_note(fact, items, documents, vendor_atom, tags)
        note_path.write_text(content)

        return note_path

    def export_all_facts(
        self,
        since: date | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        """Export all facts to Obsidian notes.

        Args:
            since: Only export facts on or after this date
            overwrite: Whether to overwrite existing files

        Returns:
            List of paths to created notes
        """
        from alibi.db.v2_store import list_facts

        facts = list_facts(self.db, date_from=since)
        paths: list[Path] = []

        for fact in facts:
            path = self.export_fact(fact, overwrite=overwrite)
            paths.append(path)

        return paths
