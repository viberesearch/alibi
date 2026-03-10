"""Obsidian note generation for processed artifacts.

This module generates Obsidian-compatible markdown notes
for artifacts processed by the Alibi pipeline.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from alibi.config import get_config
from alibi.db.models import Artifact, DocumentStatus, DocumentType
from alibi.obsidian.notes import format_currency, sanitize_filename

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager


# Template for artifact/document notes
ARTIFACT_NOTE_TEMPLATE = """---
type: document
artifact_id: "{artifact_id}"
document_type: {document_type}
vendor: "{vendor}"
date: {date}
amount: {amount}
currency: {currency}
file: "{file_name}"
status: {status}
tags: [{tags}]
created: {created}
---

# {title}

**Type**: {document_type}
**Date**: {date}
**Vendor**: {vendor}

## Financial Details

| Field | Value |
|-------|-------|
| Amount | {formatted_amount} |
| Document ID | {document_id} |

## Source Document

![[{file_name}]]

## Extracted Data

{extracted_details}

## Notes

{notes}
"""


def generate_artifact_note(
    artifact: Artifact,
    tags: list[str] | None = None,
) -> str:
    """Generate Obsidian markdown note for an artifact.

    Args:
        artifact: Artifact to generate note for
        tags: Optional list of tag names

    Returns:
        Markdown content for the note
    """
    tags = tags or []

    # Build title from vendor and type
    vendor = artifact.vendor or "Unknown"
    doc_type = artifact.type.value if artifact.type else "document"
    date_str = artifact.document_date.isoformat() if artifact.document_date else "N/A"
    title = f"{vendor} - {doc_type.title()}"

    # Format extracted data details
    extracted_details = ""
    if artifact.extracted_data:
        for key, value in artifact.extracted_data.items():
            if key not in ("vendor", "total", "amount", "document_date", "raw_text"):
                extracted_details += f"- **{key.replace('_', ' ').title()}**: {value}\n"
    if not extracted_details:
        extracted_details = "_No additional details extracted_"

    # Get file name from path
    file_name = ""
    if artifact.file_path:
        file_name = Path(artifact.file_path).name

    # Format tags for YAML
    tags_yaml = ", ".join(f'"{t}"' for t in tags) if tags else ""

    return ARTIFACT_NOTE_TEMPLATE.format(
        artifact_id=artifact.id,
        document_type=doc_type,
        vendor=vendor,
        date=date_str,
        amount=f"{artifact.amount:.2f}" if artifact.amount else "0",
        currency=artifact.currency or "EUR",
        file_name=file_name,
        status=artifact.status.value if artifact.status else "processed",
        tags=tags_yaml,
        created=datetime.now().isoformat(),
        title=title,
        formatted_amount=format_currency(artifact.amount, artifact.currency or "EUR"),
        document_id=artifact.document_id or "N/A",
        extracted_details=extracted_details,
        notes="",
    )


def get_artifact_note_filename(artifact: Artifact) -> str:
    """Generate filename for an artifact note.

    Args:
        artifact: Artifact to generate filename for

    Returns:
        Sanitized filename without extension
    """
    date_str = (
        artifact.document_date.strftime("%Y-%m-%d")
        if artifact.document_date
        else datetime.now().strftime("%Y-%m-%d")
    )
    vendor = sanitize_filename(artifact.vendor or "unknown")[:30]
    doc_type = artifact.type.value if artifact.type else "doc"

    return f"{date_str}_{doc_type}_{vendor}"


class ArtifactNoteExporter:
    """Exports artifacts to Obsidian notes in the vault."""

    def __init__(
        self,
        db: DatabaseManager,
        vault_path: Path | None = None,
    ) -> None:
        """Initialize the exporter.

        Args:
            db: Database manager for loading artifacts
            vault_path: Path to Obsidian vault (defaults to config)
        """
        self.db = db
        config = get_config()
        self.vault_path = vault_path or config.vault_path
        if self.vault_path is None:
            raise ValueError("No vault path configured. Set ALIBI_VAULT_PATH.")

    def _get_note_directory(self, artifact_type: DocumentType) -> Path:
        """Get the directory for notes of a given artifact type.

        Args:
            artifact_type: Type of artifact

        Returns:
            Path to the note directory
        """
        if self.vault_path is None:
            raise ValueError("Vault path not configured")

        # Map artifact types to vault directories
        type_dirs = {
            DocumentType.RECEIPT: "receipts",
            DocumentType.INVOICE: "invoices",
            DocumentType.STATEMENT: "statements",
            DocumentType.WARRANTY: "warranties",
            DocumentType.CONTRACT: "contracts",
            DocumentType.POLICY: "policies",
            DocumentType.OTHER: "other",
        }

        subdir = type_dirs.get(artifact_type, "other")
        return self.vault_path / "vault" / "documents" / subdir

    def export_artifact(
        self,
        artifact: Artifact,
        tags: list[str] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Export an artifact to an Obsidian note.

        Args:
            artifact: Artifact to export
            tags: Optional tag names
            overwrite: Whether to overwrite existing file

        Returns:
            Path to the created note
        """
        note_dir = self._get_note_directory(artifact.type)
        note_dir.mkdir(parents=True, exist_ok=True)

        filename = get_artifact_note_filename(artifact)
        note_path = note_dir / f"{filename}.md"

        if note_path.exists() and not overwrite:
            counter = 1
            while note_path.exists():
                note_path = note_dir / f"{filename}_{counter}.md"
                counter += 1

        content = generate_artifact_note(artifact, tags)
        note_path.write_text(content)

        return note_path

    def export_artifact_by_id(
        self,
        document_id: str,
        tags: list[str] | None = None,
        overwrite: bool = False,
    ) -> Optional[Path]:
        """Export a document by its ID.

        Args:
            document_id: ID of document to export
            tags: Optional tag names
            overwrite: Whether to overwrite existing file

        Returns:
            Path to created note, or None if document not found
        """
        row = self.db.fetchone(
            "SELECT id, file_path, file_hash, raw_extraction FROM documents WHERE id = ?",
            (document_id,),
        )

        if not row:
            return None

        # Parse raw_extraction if it's a string
        extracted_data: dict[str, Any] = {}
        raw = row["raw_extraction"]
        if isinstance(raw, str):
            try:
                import json

                extracted_data = json.loads(raw)
            except (ValueError, TypeError):
                extracted_data = {}

        # Get linked fact data via bundle/cloud chain
        fact_row = self.db.fetchone(
            """
            SELECT f.vendor, f.total_amount, f.currency, f.event_date, f.fact_type
            FROM facts f
            JOIN cloud_bundles cb ON f.cloud_id = cb.cloud_id
            JOIN bundles b ON cb.bundle_id = b.id
            WHERE b.document_id = ?
            LIMIT 1
            """,
            (document_id,),
        )

        vendor = fact_row["vendor"] if fact_row else None
        amount = (
            Decimal(str(fact_row["total_amount"]))
            if fact_row and fact_row["total_amount"]
            else None
        )
        currency = fact_row["currency"] if fact_row else "EUR"
        event_date = fact_row["event_date"] if fact_row else None
        fact_type = fact_row["fact_type"] if fact_row else "other"

        # Map fact_type to DocumentType for directory routing
        type_map = {
            "purchase": DocumentType.RECEIPT,
            "invoice": DocumentType.INVOICE,
            "contract": DocumentType.CONTRACT,
            "warranty": DocumentType.WARRANTY,
            "subscription_payment": DocumentType.RECEIPT,
            "refund": DocumentType.RECEIPT,
        }
        artifact_type = type_map.get(fact_type, DocumentType.OTHER)

        artifact = Artifact(
            id=row["id"],
            space_id="default",
            type=artifact_type,
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            vendor=vendor,
            document_date=event_date,
            amount=amount,
            currency=currency,
            extracted_data=extracted_data,
            status=DocumentStatus.PROCESSED,
        )

        return self.export_artifact(artifact, tags, overwrite)

    def export_recent_artifacts(
        self,
        limit: int = 50,
        overwrite: bool = False,
    ) -> list[Path]:
        """Export recent documents to Obsidian notes.

        Args:
            limit: Maximum number of documents to export
            overwrite: Whether to overwrite existing files

        Returns:
            List of paths to created notes
        """
        rows = self.db.fetchall(
            """
            SELECT id FROM documents
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )

        paths: list[Path] = []
        for row in rows:
            path = self.export_artifact_by_id(row[0], overwrite=overwrite)
            if path:
                paths.append(path)

        return paths
