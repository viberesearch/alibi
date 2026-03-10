"""Analytics stack export — pushes facts to the fleet analytics platform."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)


def _get_fact_provenance(
    db: DatabaseManager,
) -> dict[str, tuple[str | None, str | None]]:
    """Map fact IDs to (source, user_id) from their linked documents.

    Traverses the fact → cloud → cloud_bundles → bundles → bundle_atoms →
    atoms → documents join chain. Only facts linked to documents with a
    non-NULL source are included; the rest resolve to (None, None) at call
    site.
    """
    conn = db.get_connection()
    rows = conn.execute(
        """
        SELECT DISTINCT f.id, d.source, d.user_id
        FROM facts f
        JOIN clouds c ON c.id = f.cloud_id
        JOIN cloud_bundles cb ON cb.cloud_id = c.id
        JOIN bundles b ON b.id = cb.bundle_id
        JOIN bundle_atoms ba ON ba.bundle_id = b.id
        JOIN atoms a ON a.id = ba.atom_id
        JOIN documents d ON d.id = a.document_id
        WHERE d.source IS NOT NULL
        """
    ).fetchall()
    result: dict[str, tuple[str | None, str | None]] = {}
    for row in rows:
        fact_id = row[0]
        if fact_id not in result:
            result[fact_id] = (row[1], row[2])
    return result


def build_export_payload(db: DatabaseManager) -> dict[str, Any]:
    """Build the full export payload: facts + fact_items + annotations + documents.

    Returns a dict with four top-level keys matching the analytics-stack
    ``alibi.*`` PostgreSQL schema::

        {
            "facts": [...],       # each fact enriched with source + user_id
            "fact_items": [...],
            "annotations": [...],
            "documents": [...]    # id, source, user_id, created_at
        }

    Provenance fields (``source``, ``user_id``) on each fact come from the
    document(s) linked to that fact via the cloud → bundle → atom chain.
    Facts with no linked document resolve to ``null`` for both fields.
    """
    from alibi.annotations.store import get_annotations
    from alibi.db.v2_store import get_fact_items, list_facts

    facts = list_facts(db, limit=100_000)
    all_items: list[dict[str, Any]] = []
    for fact in facts:
        items = get_fact_items(db, fact["id"])
        all_items.extend(items)

    # Sanitize numeric fields: non-numeric values → None (defense-in-depth).
    # SQLite may store "" or text strings for columns that PostgreSQL expects
    # as DOUBLE PRECISION.
    _NUMERIC_ITEM_FIELDS = {
        "quantity",
        "unit_price",
        "total_price",
        "unit_quantity",
        "tax_rate",
        "discount",
        "comparable_unit_price",
    }
    for item in all_items:
        for field in _NUMERIC_ITEM_FIELDS:
            v = item.get(field)
            if v is None:
                continue
            if isinstance(v, (int, float)):
                continue
            try:
                item[field] = float(v)
            except (ValueError, TypeError):
                item[field] = None

    annotations = get_annotations(db)

    provenance = _get_fact_provenance(db)
    for fact in facts:
        source, user_id = provenance.get(fact["id"], (None, None))
        fact["source"] = source or "cli"
        fact["user_id"] = user_id or "system"

    conn = db.get_connection()
    doc_rows = conn.execute(
        "SELECT id, source, user_id, created_at FROM documents"
    ).fetchall()
    documents = []
    for row in doc_rows:
        doc = dict(row)
        doc["source"] = doc.get("source") or "cli"
        doc["user_id"] = doc.get("user_id") or "system"
        documents.append(doc)

    return {
        "facts": facts,
        "fact_items": all_items,
        "annotations": annotations,
        "documents": documents,
    }


def push_to_analytics_stack(
    db: DatabaseManager,
    url: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Export all facts and push to the analytics-stack ingest endpoint.

    Args:
        db: Database manager.
        url: Base URL of the analytics-stack (e.g. ``http://localhost:8070``).
        timeout: HTTP request timeout in seconds.

    Returns:
        Dict with ``status``, ``facts_count``, ``items_count``,
        ``annotations_count``, ``documents_count``, and optionally
        ``response``.

    Raises:
        ConnectionError: If the analytics-stack endpoint is unreachable.
    """
    payload = build_export_payload(db)

    result: dict[str, Any] = {
        "facts_count": len(payload["facts"]),
        "items_count": len(payload["fact_items"]),
        "annotations_count": len(payload["annotations"]),
        "documents_count": len(payload["documents"]),
    }

    endpoint = f"{url.rstrip('/')}/v1/ingest/alibi"
    body = json.dumps(payload, default=str).encode("utf-8")

    logger.info(
        "Pushing %d facts, %d items, %d annotations, %d documents to %s",
        result["facts_count"],
        result["items_count"],
        result["annotations_count"],
        result["documents_count"],
        endpoint,
    )

    try:
        req = Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read().decode("utf-8")
            result["status"] = "ok"
            result["http_status"] = resp.status
            try:
                result["response"] = json.loads(resp_body)
            except (json.JSONDecodeError, ValueError):
                result["response"] = resp_body
    except (URLError, OSError) as exc:
        logger.warning("Analytics push failed: %s", exc)
        raise ConnectionError(f"Analytics stack unreachable: {exc}") from exc

    logger.info("Analytics push complete: %s", result["status"])
    return result
