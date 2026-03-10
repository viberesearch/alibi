"""Export functionality for Alibi data."""

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager


@dataclass
class ExportResult:
    """Result of an export operation."""

    path: Path
    format: str
    record_count: int
    size_bytes: int


def _serialize_value(value: Any) -> Any:
    """Serialize value for export."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def export_transactions(
    db: "DatabaseManager",
    output_path: Path,
    format: Literal["csv", "json"] = "csv",
    space_id: str = "default",
    since: date | None = None,
    until: date | None = None,
) -> ExportResult:
    """Export transactions to CSV or JSON.

    Args:
        db: Database manager instance
        output_path: Path to write export file
        format: Output format (csv or json)
        space_id: Space to export from
        since: Only export transactions on or after this date
        until: Only export transactions on or before this date

    Returns:
        ExportResult with path and metadata
    """
    # Build query with filters
    query = """
        SELECT id, vendor, fact_type, total_amount, currency,
               event_date, status
        FROM facts
        WHERE 1=1
    """
    params: list[Any] = []

    if since:
        query += " AND event_date >= ?"
        params.append(since.isoformat())
    if until:
        query += " AND event_date <= ?"
        params.append(until.isoformat())

    query += " ORDER BY event_date DESC"

    rows = db.fetchall(query, tuple(params))

    headers = [
        "id",
        "vendor",
        "type",
        "amount",
        "currency",
        "date",
        "status",
    ]

    if format == "csv":
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in rows:
                writer.writerow([_serialize_value(v) for v in row])
    else:  # json
        records = [
            dict(zip(headers, [_serialize_value(v) for v in row])) for row in rows
        ]
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)

    return ExportResult(
        path=output_path,
        format=format,
        record_count=len(rows),
        size_bytes=output_path.stat().st_size,
    )


def export_items(
    db: "DatabaseManager",
    output_path: Path,
    format: Literal["csv", "json"] = "csv",
    space_id: str = "default",
) -> ExportResult:
    """Export items/assets to CSV or JSON.

    Args:
        db: Database manager instance
        output_path: Path to write export file
        format: Output format (csv or json)
        space_id: Space to export from

    Returns:
        ExportResult with path and metadata
    """
    query = """
        SELECT id, name, category, description, purchase_date,
               purchase_price, currency, warranty_expires, location
        FROM items
        WHERE space_id = ?
        ORDER BY name
    """

    rows = db.fetchall(query, (space_id,))

    headers = [
        "id",
        "name",
        "category",
        "description",
        "purchase_date",
        "purchase_price",
        "currency",
        "warranty_expires",
        "location",
    ]

    if format == "csv":
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in rows:
                writer.writerow([_serialize_value(v) for v in row])
    else:  # json
        records = [
            dict(zip(headers, [_serialize_value(v) for v in row])) for row in rows
        ]
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)

    return ExportResult(
        path=output_path,
        format=format,
        record_count=len(rows),
        size_bytes=output_path.stat().st_size,
    )


def export_artifacts(
    db: "DatabaseManager",
    output_path: Path,
    format: Literal["csv", "json"] = "csv",
    space_id: str = "default",
) -> ExportResult:
    """Export documents to CSV or JSON.

    Args:
        db: Database manager instance
        output_path: Path to write export file
        format: Output format (csv or json)
        space_id: Space to export from (unused, kept for API compat)

    Returns:
        ExportResult with path and metadata
    """
    query = """
        SELECT d.id, d.file_path, d.ingested_at,
               f.fact_type, f.vendor, f.event_date, f.total_amount, f.currency
        FROM documents d
        LEFT JOIN bundles b ON b.document_id = d.id
        LEFT JOIN cloud_bundles cb ON cb.bundle_id = b.id
        LEFT JOIN facts f ON f.cloud_id = cb.cloud_id
        GROUP BY d.id
        ORDER BY d.created_at DESC
    """

    rows = db.fetchall(query)

    headers = [
        "id",
        "file_path",
        "ingested_at",
        "type",
        "vendor",
        "date",
        "amount",
        "currency",
    ]

    if format == "csv":
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in rows:
                writer.writerow([_serialize_value(v) for v in row])
    else:  # json
        records = [
            dict(zip(headers, [_serialize_value(v) for v in row])) for row in rows
        ]
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)

    return ExportResult(
        path=output_path,
        format=format,
        record_count=len(rows),
        size_bytes=output_path.stat().st_size,
    )


def export_all(
    db: "DatabaseManager",
    output_dir: Path,
    format: Literal["csv", "json"] = "csv",
    space_id: str = "default",
    since: date | None = None,
    until: date | None = None,
) -> list[ExportResult]:
    """Export all data to a directory.

    Args:
        db: Database manager instance
        output_dir: Directory to write export files
        format: Output format (csv or json)
        space_id: Space to export from
        since: Only export transactions on or after this date
        until: Only export transactions on or before this date

    Returns:
        List of ExportResult for each exported file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = format

    results = []

    # Export facts (transactions)
    txn_path = output_dir / f"facts.{extension}"
    results.append(
        export_transactions(
            db, txn_path, format=format, space_id=space_id, since=since, until=until
        )
    )

    # Export items (v1 inventory, kept for backward compat)
    items_path = output_dir / f"items.{extension}"
    results.append(export_items(db, items_path, format=format, space_id=space_id))

    # Export documents
    artifacts_path = output_dir / f"documents.{extension}"
    results.append(
        export_artifacts(db, artifacts_path, format=format, space_id=space_id)
    )

    return results
