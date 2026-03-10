"""V2 atom-cloud-fact persistence layer.

Stores and queries atoms, bundles, clouds, and facts in the v2 SQLite tables.
Sole storage path — v1 tables are no longer populated.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Any

from alibi.db.connection import DatabaseManager
from alibi.db.models import (
    Atom,
    AtomType,
    Bundle,
    BundleAtom,
    BundleType,
    Cloud,
    CloudBundle,
    CloudMatchType,
    CloudStatus,
    Document,
    Fact,
    FactItem,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Store operations
# ---------------------------------------------------------------------------


def store_document(db: DatabaseManager, doc: Document) -> None:
    """Store a v2 document record."""
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT OR IGNORE INTO documents "
            "(id, file_path, file_hash, perceptual_hash, raw_extraction, "
            "source, user_id, yaml_hash, yaml_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc.id,
                doc.file_path,
                doc.file_hash,
                doc.perceptual_hash,
                json.dumps(doc.raw_extraction) if doc.raw_extraction else None,
                doc.source,
                doc.user_id,
                doc.yaml_hash,
                doc.yaml_path,
            ),
        )


def store_atoms(db: DatabaseManager, atoms: list[Atom]) -> None:
    """Store atoms in bulk."""
    if not atoms:
        return
    with db.transaction() as cursor:
        cursor.executemany(
            "INSERT OR IGNORE INTO atoms "
            "(id, document_id, atom_type, data, confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (
                    a.id,
                    a.document_id,
                    a.atom_type.value,
                    json.dumps(a.data),
                    float(a.confidence),
                )
                for a in atoms
            ],
        )


def store_bundle(
    db: DatabaseManager,
    bundle: Bundle,
    bundle_atoms: list[BundleAtom],
) -> None:
    """Store a bundle and its atom links."""
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT OR IGNORE INTO bundles "
            "(id, document_id, bundle_type, cloud_id) VALUES (?, ?, ?, ?)",
            (bundle.id, bundle.document_id, bundle.bundle_type.value, bundle.cloud_id),
        )
        for ba in bundle_atoms:
            cursor.execute(
                "INSERT OR IGNORE INTO bundle_atoms "
                "(bundle_id, atom_id, role) VALUES (?, ?, ?)",
                (ba.bundle_id, ba.atom_id, ba.role.value),
            )


def store_cloud(
    db: DatabaseManager,
    cloud: Cloud,
    cloud_bundle: CloudBundle,
) -> None:
    """Store a new cloud with its first bundle link."""
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT OR IGNORE INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
            (cloud.id, cloud.status.value, float(cloud.confidence)),
        )
        cursor.execute(
            "INSERT OR IGNORE INTO cloud_bundles "
            "(cloud_id, bundle_id, match_type, match_confidence) "
            "VALUES (?, ?, ?, ?)",
            (
                cloud_bundle.cloud_id,
                cloud_bundle.bundle_id,
                cloud_bundle.match_type.value,
                float(cloud_bundle.match_confidence),
            ),
        )
        # Set authoritative cloud_id on the bundle
        cursor.execute(
            "UPDATE bundles SET cloud_id = ? WHERE id = ?",
            (cloud_bundle.cloud_id, cloud_bundle.bundle_id),
        )


def add_cloud_bundle(db: DatabaseManager, cloud_bundle: CloudBundle) -> None:
    """Add a bundle to an existing cloud."""
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT OR IGNORE INTO cloud_bundles "
            "(cloud_id, bundle_id, match_type, match_confidence) "
            "VALUES (?, ?, ?, ?)",
            (
                cloud_bundle.cloud_id,
                cloud_bundle.bundle_id,
                cloud_bundle.match_type.value,
                float(cloud_bundle.match_confidence),
            ),
        )
        # Update authoritative cloud_id on the bundle
        cursor.execute(
            "UPDATE bundles SET cloud_id = ? WHERE id = ?",
            (cloud_bundle.cloud_id, cloud_bundle.bundle_id),
        )


def update_cloud_status(
    db: DatabaseManager,
    cloud_id: str,
    status: CloudStatus,
    confidence: Decimal | None = None,
) -> None:
    """Update cloud status and optionally confidence."""
    with db.transaction() as cursor:
        if confidence is not None:
            cursor.execute(
                "UPDATE clouds SET status = ?, confidence = ? WHERE id = ?",
                (status.value, float(confidence), cloud_id),
            )
        else:
            cursor.execute(
                "UPDATE clouds SET status = ? WHERE id = ?",
                (status.value, cloud_id),
            )


def store_fact(
    db: DatabaseManager,
    fact: Fact,
    items: list[FactItem],
) -> None:
    """Store a collapsed fact and its denormalized items."""
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT OR IGNORE INTO facts "
            "(id, cloud_id, fact_type, vendor, vendor_key, total_amount, currency, "
            "event_date, payments, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact.id,
                fact.cloud_id,
                fact.fact_type.value,
                fact.vendor,
                fact.vendor_key,
                float(fact.total_amount) if fact.total_amount is not None else None,
                fact.currency,
                fact.event_date.isoformat() if fact.event_date else None,
                json.dumps(fact.payments) if fact.payments else None,
                fact.status.value,
            ),
        )
        for item in items:
            cursor.execute(
                "INSERT OR IGNORE INTO fact_items "
                "(id, fact_id, atom_id, name, name_normalized, "
                "quantity, unit, unit_price, total_price, "
                "brand, category, comparable_unit_price, comparable_unit, "
                "barcode, unit_quantity, tax_rate, tax_type, "
                "enrichment_source, enrichment_confidence, product_variant) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.id,
                    item.fact_id,
                    item.atom_id,
                    item.name,
                    item.name_normalized,
                    float(item.quantity),
                    item.unit.value,
                    float(item.unit_price) if item.unit_price is not None else None,
                    float(item.total_price) if item.total_price is not None else None,
                    item.brand,
                    item.category,
                    (
                        float(item.comparable_unit_price)
                        if item.comparable_unit_price is not None
                        else None
                    ),
                    item.comparable_unit.value if item.comparable_unit else None,
                    item.barcode,
                    (
                        float(item.unit_quantity)
                        if item.unit_quantity is not None
                        else None
                    ),
                    float(item.tax_rate) if item.tax_rate is not None else None,
                    item.tax_type.value,
                    item.enrichment_source,
                    item.enrichment_confidence,
                    item.product_variant,
                ),
            )
        # Update cloud status to collapsed
        cursor.execute(
            "UPDATE clouds SET status = ?, confidence = ? WHERE id = ?",
            (CloudStatus.COLLAPSED.value, 1.0, fact.cloud_id),
        )


# ---------------------------------------------------------------------------
# Query operations (for cloud formation matching)
# ---------------------------------------------------------------------------


def get_bundle_summaries(db: DatabaseManager) -> list[dict[str, Any]]:
    """Load existing bundle summaries with their atoms for cloud matching.

    Uses a single JOIN query instead of N+1 per-bundle atom fetches.

    Returns list of dicts with:
        - bundle_id, bundle_type, cloud_id
        - atoms: list of atom dicts (atom_type, data)
    """
    rows = db.fetchall(
        "SELECT b.id AS bundle_id, b.bundle_type, cb.cloud_id, "
        "a.atom_type, a.data "
        "FROM bundles b "
        "LEFT JOIN cloud_bundles cb ON b.id = cb.bundle_id "
        "LEFT JOIN bundle_atoms ba ON b.id = ba.bundle_id "
        "LEFT JOIN atoms a ON ba.atom_id = a.id "
        "ORDER BY b.created_at, b.id",
        (),
    )

    bundles_map: dict[str, dict[str, Any]] = {}
    bundle_order: list[str] = []

    for row in rows:
        bid = row["bundle_id"]
        if bid not in bundles_map:
            bundles_map[bid] = {
                "bundle_id": bid,
                "bundle_type": row["bundle_type"],
                "cloud_id": row["cloud_id"],
                "atoms": [],
            }
            bundle_order.append(bid)

        if row["atom_type"] is not None:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            bundles_map[bid]["atoms"].append(
                {"atom_type": row["atom_type"], "data": data}
            )

    return [bundles_map[bid] for bid in bundle_order]


def get_bundle_summaries_for_vendor(
    db: DatabaseManager,
    vendor_key: str | None = None,
    vendor_name: str | None = None,
) -> list[dict[str, Any]]:
    """Load bundle summaries pre-filtered by vendor key.

    When a vendor_key (VAT number / registration ID) is available,
    returns only bundles whose vendor atom contains that key — a
    reliable and significant reduction of the working set.

    For name-only lookups the pre-filter is skipped because vendor
    name normalization (legal-suffix stripping, case folding) makes
    LIKE matching against raw atom JSON unreliable.

    Args:
        db: Database manager.
        vendor_key: Vendor registration key (VAT number etc.).
        vendor_name: Kept for API compatibility; ignored (falls back
            to full query when vendor_key is absent).

    Returns:
        Same format as get_bundle_summaries().
    """
    if not vendor_key:
        return get_bundle_summaries(db)

    # Match vendor_key in the JSON data (vat_number, tax_id)
    key_upper = vendor_key.upper().replace(" ", "")
    matching_sql = (
        "SELECT DISTINCT ba.bundle_id "
        "FROM bundle_atoms ba "
        "JOIN atoms a ON ba.atom_id = a.id "
        "WHERE a.atom_type = 'vendor' "
        "AND (a.data LIKE ? OR a.data LIKE ?)"
    )
    params = (
        f'%"vat_number": "{key_upper}"%',
        f'%"tax_id": "{key_upper}"%',
    )
    matching_rows = db.fetchall(matching_sql, params)
    matching_ids = {row["bundle_id"] for row in matching_rows}

    if not matching_ids:
        # No existing bundles share this vendor_key — caller will create
        # a new cloud anyway, but fall back to full set so that other
        # matching signals (name, amount, date) still get a chance.
        return get_bundle_summaries(db)

    # Now fetch full summaries only for matching bundles
    placeholders = ",".join("?" * len(matching_ids))
    rows = db.fetchall(
        "SELECT b.id AS bundle_id, b.bundle_type, cb.cloud_id, "
        "a.atom_type, a.data "
        "FROM bundles b "
        "LEFT JOIN cloud_bundles cb ON b.id = cb.bundle_id "
        "LEFT JOIN bundle_atoms ba ON b.id = ba.bundle_id "
        "LEFT JOIN atoms a ON ba.atom_id = a.id "
        f"WHERE b.id IN ({placeholders}) "
        "ORDER BY b.created_at, b.id",
        tuple(matching_ids),
    )

    bundles_map: dict[str, dict[str, Any]] = {}
    bundle_order: list[str] = []

    for row in rows:
        bid = row["bundle_id"]
        if bid not in bundles_map:
            bundles_map[bid] = {
                "bundle_id": bid,
                "bundle_type": row["bundle_type"],
                "cloud_id": row["cloud_id"],
                "atoms": [],
            }
            bundle_order.append(bid)

        if row["atom_type"] is not None:
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            bundles_map[bid]["atoms"].append(
                {"atom_type": row["atom_type"], "data": data}
            )

    return [bundles_map[bid] for bid in bundle_order]


def get_cloud_bundle_data(db: DatabaseManager, cloud_id: str) -> list[dict[str, Any]]:
    """Load all bundles and their atoms for a cloud (for collapse).

    Returns list of dicts with:
        - bundle_id, bundle_type
        - atoms: list of atom dicts (id, atom_type, data)
    """
    # Single JOIN query instead of per-bundle atom fetches (fixes N+1)
    rows = db.fetchall(
        "SELECT b.id AS bundle_id, b.bundle_type, "
        "a.id AS atom_id, a.atom_type, a.data "
        "FROM bundles b "
        "JOIN cloud_bundles cb ON b.id = cb.bundle_id "
        "JOIN bundle_atoms ba ON b.id = ba.bundle_id "
        "JOIN atoms a ON ba.atom_id = a.id "
        "WHERE cb.cloud_id = ? "
        "ORDER BY b.id",
        (cloud_id,),
    )

    bundles: dict[str, dict[str, Any]] = {}
    for r in rows:
        bid = r["bundle_id"]
        if bid not in bundles:
            bundles[bid] = {
                "bundle_id": bid,
                "bundle_type": r["bundle_type"],
                "atoms": [],
            }
        data = r["data"]
        if isinstance(data, str):
            data = json.loads(data)
        bundles[bid]["atoms"].append(
            {"id": r["atom_id"], "atom_type": r["atom_type"], "data": data}
        )

    return list(bundles.values())


def get_cloud_for_bundle(db: DatabaseManager, bundle_id: str) -> str | None:
    """Get the cloud ID for a bundle, if assigned.

    Reads from the authoritative bundles.cloud_id field.
    """
    row = db.fetchone(
        "SELECT cloud_id FROM bundles WHERE id = ?",
        (bundle_id,),
    )
    return row["cloud_id"] if row else None


def get_fact_for_cloud(db: DatabaseManager, cloud_id: str) -> dict[str, Any] | None:
    """Get the fact for a cloud, if collapsed."""
    row = db.fetchone(
        "SELECT * FROM facts WHERE cloud_id = ?",
        (cloud_id,),
    )
    if not row:
        return None
    return dict(row)


def get_cloud_locations(
    db: DatabaseManager, cloud_ids: set[str]
) -> dict[str, tuple[float, float]]:
    """Batch-lookup location annotations for clouds via their facts.

    Returns a mapping of cloud_id -> (lat, lng) for clouds whose
    collapsed facts have a location annotation.
    """
    if not cloud_ids:
        return {}

    placeholders = ",".join("?" * len(cloud_ids))
    rows = db.fetchall(
        f"SELECT f.cloud_id, a.metadata "
        f"FROM facts f "
        f"JOIN annotations a ON a.target_id = f.id "
        f"WHERE f.cloud_id IN ({placeholders}) "
        f"AND a.annotation_type = 'location' "
        f"AND a.target_type = 'fact' "
        f"AND a.key = 'map_url'",
        tuple(cloud_ids),
    )

    result: dict[str, tuple[float, float]] = {}
    for row in rows:
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        if meta and meta.get("lat") is not None and meta.get("lng") is not None:
            result[row["cloud_id"]] = (meta["lat"], meta["lng"])

    return result


def get_fact_items(db: DatabaseManager, fact_id: str) -> list[dict[str, Any]]:
    """Get all items for a fact."""
    rows = db.fetchall(
        "SELECT * FROM fact_items WHERE fact_id = ? ORDER BY name",
        (fact_id,),
    )
    return [dict(r) for r in rows]


def list_fact_items_uncategorized(
    db: DatabaseManager, limit: int = 100
) -> list[dict[str, Any]]:
    """Get fact_items that have no category assigned."""
    rows = db.fetchall(
        """SELECT fi.* FROM fact_items fi
        WHERE (fi.category IS NULL OR fi.category = '')
        ORDER BY fi.id
        LIMIT ?""",
        (limit,),
    )
    return [dict(r) for r in rows]


def get_document_by_hash(db: DatabaseManager, file_hash: str) -> dict[str, Any] | None:
    """Find a v2 document by file hash."""
    row = db.fetchone(
        "SELECT * FROM documents WHERE file_hash = ?",
        (file_hash,),
    )
    return dict(row) if row else None


def get_document_by_path(db: DatabaseManager, file_path: str) -> dict[str, Any] | None:
    """Find a v2 document by file path (resolved to string)."""
    row = db.fetchone(
        "SELECT * FROM documents WHERE file_path = ?",
        (file_path,),
    )
    return dict(row) if row else None


def update_yaml_hash(db: DatabaseManager, doc_id: str, yaml_hash: str | None) -> None:
    """Update the yaml_hash on an existing document."""
    db.execute(
        "UPDATE documents SET yaml_hash = ? WHERE id = ?",
        (yaml_hash, doc_id),
    )
    db.get_connection().commit()


def update_yaml_path(db: DatabaseManager, doc_id: str, yaml_path: str | None) -> None:
    """Update the yaml_path on an existing document."""
    db.execute(
        "UPDATE documents SET yaml_path = ? WHERE id = ?",
        (yaml_path, doc_id),
    )
    db.get_connection().commit()


# ---------------------------------------------------------------------------
# Fact query operations (for notes, analytics, subscriptions)
# ---------------------------------------------------------------------------


def get_fact_by_id(db: DatabaseManager, fact_id: str) -> dict[str, Any] | None:
    """Get a fact by its ID."""
    row = db.fetchone("SELECT * FROM facts WHERE id = ?", (fact_id,))
    return dict(row) if row else None


def list_facts(
    db: DatabaseManager,
    date_from: date | None = None,
    date_to: date | None = None,
    vendor: str | None = None,
    fact_type: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """List facts with optional filters.

    Args:
        db: Database manager
        date_from: Only facts on or after this date
        date_to: Only facts on or before this date
        vendor: Filter by vendor name (case-insensitive substring)
        fact_type: Filter by fact_type value
        limit: Maximum results

    Returns:
        List of fact dicts ordered by event_date DESC
    """
    sql = "SELECT * FROM facts WHERE 1=1"
    params: list[Any] = []

    if date_from:
        sql += " AND event_date >= ?"
        params.append(date_from.isoformat())
    if date_to:
        sql += " AND event_date <= ?"
        params.append(date_to.isoformat())
    if vendor:
        sql += " AND LOWER(vendor) LIKE ?"
        params.append(f"%{vendor.lower()}%")
    if fact_type:
        sql += " AND fact_type = ?"
        params.append(fact_type)

    sql += " ORDER BY event_date DESC LIMIT ?"
    params.append(limit)

    rows = db.fetchall(sql, tuple(params))
    return [dict(r) for r in rows]


def get_facts_for_document(
    db: DatabaseManager, document_id: str
) -> list[dict[str, Any]]:
    """Get facts associated with a document (document → bundles → clouds → facts).

    Returns list of fact dicts, or empty list if no facts are linked.
    """
    rows = db.fetchall(
        """
        SELECT DISTINCT f.*
        FROM documents d
        JOIN bundles b ON b.document_id = d.id
        JOIN cloud_bundles cb ON cb.bundle_id = b.id
        JOIN facts f ON f.cloud_id = cb.cloud_id
        WHERE d.id = ?
        """,
        (document_id,),
    )
    return [dict(r) for r in rows]


def get_fact_documents(db: DatabaseManager, fact_id: str) -> list[dict[str, Any]]:
    """Get source documents for a fact (fact → cloud → bundles → documents).

    Returns list of document dicts with file_path, file_hash, etc.
    """
    rows = db.fetchall(
        """
        SELECT DISTINCT d.*
        FROM facts f
        JOIN cloud_bundles cb ON f.cloud_id = cb.cloud_id
        JOIN bundles b ON cb.bundle_id = b.id
        JOIN documents d ON b.document_id = d.id
        WHERE f.id = ?
        """,
        (fact_id,),
    )
    return [dict(r) for r in rows]


def get_fact_vendor_atom(db: DatabaseManager, fact_id: str) -> dict[str, Any] | None:
    """Get the vendor atom data for a fact (address, phone, registration, etc.).

    Traverses fact → cloud → bundles → atoms to find the first VENDOR atom.
    """
    row = db.fetchone(
        """
        SELECT a.data
        FROM facts f
        JOIN cloud_bundles cb ON f.cloud_id = cb.cloud_id
        JOIN bundles bun ON cb.bundle_id = bun.id
        JOIN bundle_atoms ba ON bun.id = ba.bundle_id
        JOIN atoms a ON ba.atom_id = a.id
        WHERE f.id = ? AND a.atom_type = 'vendor'
        LIMIT 1
        """,
        (fact_id,),
    )
    if not row:
        return None
    data = row["data"]
    if isinstance(data, str):
        data = json.loads(data)
    result: dict[str, Any] = data
    return result


def get_facts_grouped_by_vendor(
    db: DatabaseManager,
    fact_type: str = "purchase",
) -> dict[str, list[dict[str, Any]]]:
    """Get facts grouped by vendor_key, for subscription detection.

    Returns dict mapping vendor_key to list of fact dicts, sorted by event_date.
    Falls back to vendor name for facts without a vendor_key.
    """
    rows = db.fetchall(
        """
        SELECT * FROM facts
        WHERE fact_type = ? AND vendor IS NOT NULL AND event_date IS NOT NULL
        ORDER BY vendor_key, event_date
        """,
        (fact_type,),
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = row["vendor_key"] or row["vendor"]
        grouped[key].append(dict(row))

    return dict(grouped)


def update_fact_type(db: DatabaseManager, fact_id: str, fact_type: str) -> None:
    """Update the fact_type for a fact."""
    db.execute(
        "UPDATE facts SET fact_type = ? WHERE id = ?",
        (fact_type, fact_id),
    )
    db.get_connection().commit()


# ---------------------------------------------------------------------------
# Fact inspection (deep drill-down for user review)
# ---------------------------------------------------------------------------


def inspect_fact(db: DatabaseManager, fact_id: str) -> dict[str, Any] | None:
    """Full drill-down of a fact: cloud, bundles, atoms, source documents.

    Returns a nested dict:
        fact: {id, vendor, total_amount, event_date, status, ...}
        cloud: {id, status, confidence}
        bundles: [{id, bundle_type, match_type, match_confidence,
                   document: {id, file_path, file_hash},
                   atoms: [{id, atom_type, data, confidence}]}]
        items: [{id, name, quantity, unit, total_price, atom_id}]
    """
    fact = get_fact_by_id(db, fact_id)
    if not fact:
        return None

    cloud_id = fact["cloud_id"]

    # Cloud metadata
    cloud_row = db.fetchone("SELECT * FROM clouds WHERE id = ?", (cloud_id,))
    cloud_data = dict(cloud_row) if cloud_row else {}

    # Bundles in this cloud with match metadata
    bundle_rows = db.fetchall(
        """
        SELECT b.id AS bundle_id, b.bundle_type, b.document_id,
               cb.match_type, cb.match_confidence
        FROM bundles b
        JOIN cloud_bundles cb ON b.id = cb.bundle_id
        WHERE cb.cloud_id = ?
        """,
        (cloud_id,),
    )

    bundles = []
    for br in bundle_rows:
        # Source document
        doc_row = db.fetchone(
            "SELECT id, file_path, file_hash FROM documents WHERE id = ?",
            (br["document_id"],),
        )
        doc_data = dict(doc_row) if doc_row else {}

        # Atoms in this bundle
        atom_rows = db.fetchall(
            """
            SELECT a.id, a.atom_type, a.data, a.confidence, ba.role
            FROM atoms a
            JOIN bundle_atoms ba ON a.id = ba.atom_id
            WHERE ba.bundle_id = ?
            """,
            (br["bundle_id"],),
        )

        atoms = []
        for ar in atom_rows:
            data = ar["data"]
            if isinstance(data, str):
                data = json.loads(data)
            atoms.append(
                {
                    "id": ar["id"],
                    "atom_type": ar["atom_type"],
                    "role": ar["role"],
                    "data": data,
                    "confidence": ar["confidence"],
                }
            )

        bundles.append(
            {
                "id": br["bundle_id"],
                "bundle_type": br["bundle_type"],
                "match_type": br["match_type"],
                "match_confidence": br["match_confidence"],
                "document": doc_data,
                "atoms": atoms,
            }
        )

    # Fact items with atom provenance
    items = get_fact_items(db, fact_id)

    return {
        "fact": fact,
        "cloud": cloud_data,
        "bundles": bundles,
        "items": items,
    }


def list_clouds(
    db: DatabaseManager,
    status: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List clouds with summary info (bundle count, fact status).

    Args:
        status: Filter by cloud status (forming, collapsed, disputed).
        limit: Maximum results.

    Returns:
        List of cloud summary dicts with bundle_count and fact info.
    """
    sql = """
        SELECT c.id, c.status, c.confidence, c.created_at,
               COUNT(cb.bundle_id) AS bundle_count,
               f.id AS fact_id, f.vendor AS fact_vendor,
               f.total_amount, f.event_date, f.status AS fact_status
        FROM clouds c
        LEFT JOIN cloud_bundles cb ON c.id = cb.cloud_id
        LEFT JOIN facts f ON c.id = f.cloud_id
    """
    params: list[Any] = []
    if status:
        sql += " WHERE c.status = ?"
        params.append(status)
    sql += " GROUP BY c.id ORDER BY c.created_at DESC LIMIT ?"
    params.append(limit)

    rows = db.fetchall(sql, tuple(params))
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Fact correction operations
# ---------------------------------------------------------------------------


def move_bundle_to_cloud(
    db: DatabaseManager,
    bundle_id: str,
    target_cloud_id: str,
) -> bool:
    """Move a bundle from its current cloud to a different cloud.

    Updates both the authoritative bundles.cloud_id and the cloud_bundles
    junction table. Does NOT delete facts or re-collapse — caller handles that.

    Returns True if the move succeeded.
    """
    # Verify bundle and target cloud exist
    bundle_row = db.fetchone("SELECT id FROM bundles WHERE id = ?", (bundle_id,))
    if not bundle_row:
        return False
    cloud_row = db.fetchone("SELECT id FROM clouds WHERE id = ?", (target_cloud_id,))
    if not cloud_row:
        return False

    with db.transaction() as cursor:
        # Update authoritative cloud_id on the bundle
        cursor.execute(
            "UPDATE bundles SET cloud_id = ? WHERE id = ?",
            (target_cloud_id, bundle_id),
        )
        # Update junction table (metadata)
        cursor.execute(
            "DELETE FROM cloud_bundles WHERE bundle_id = ?",
            (bundle_id,),
        )
        cursor.execute(
            "INSERT INTO cloud_bundles (cloud_id, bundle_id, match_type, "
            "match_confidence) VALUES (?, ?, ?, ?)",
            (target_cloud_id, bundle_id, CloudMatchType.MANUAL.value, 1.0),
        )

    return True


def move_bundle_to_new_cloud(
    db: DatabaseManager,
    bundle_id: str,
) -> str | None:
    """Move a bundle out of its current cloud into a brand new cloud.

    Creates a new cloud in FORMING status. Updates bundles.cloud_id.
    Does NOT delete facts or re-collapse — caller handles that.

    Returns the new cloud ID, or None on failure.
    """
    from uuid import uuid4

    bundle_row = db.fetchone("SELECT id FROM bundles WHERE id = ?", (bundle_id,))
    if not bundle_row:
        return None

    new_cloud_id = str(uuid4())
    with db.transaction() as cursor:
        # Create new cloud
        cursor.execute(
            "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
            (new_cloud_id, CloudStatus.FORMING.value, 0.0),
        )
        # Update authoritative cloud_id on the bundle
        cursor.execute(
            "UPDATE bundles SET cloud_id = ? WHERE id = ?",
            (new_cloud_id, bundle_id),
        )
        # Update junction table
        cursor.execute(
            "DELETE FROM cloud_bundles WHERE bundle_id = ?",
            (bundle_id,),
        )
        cursor.execute(
            "INSERT INTO cloud_bundles (cloud_id, bundle_id, match_type, "
            "match_confidence) VALUES (?, ?, ?, ?)",
            (new_cloud_id, bundle_id, CloudMatchType.MANUAL.value, 1.0),
        )

    return new_cloud_id


def delete_fact(db: DatabaseManager, fact_id: str) -> bool:
    """Delete a fact and its items. Resets the cloud to FORMING status.

    The cloud and its bundles are preserved — only the collapsed fact is
    removed, so the cloud can be re-collapsed after corrections.

    Returns True if the fact existed and was deleted.
    """
    fact = get_fact_by_id(db, fact_id)
    if not fact:
        return False

    cloud_id = fact["cloud_id"]
    with db.transaction() as cursor:
        cursor.execute("DELETE FROM fact_items WHERE fact_id = ?", (fact_id,))
        cursor.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        cursor.execute(
            "UPDATE clouds SET status = ? WHERE id = ?",
            (CloudStatus.FORMING.value, cloud_id),
        )

    return True


def delete_fact_items(db: DatabaseManager, item_ids: list[str]) -> int:
    """Delete specific fact items by ID. Returns count of deleted rows."""
    if not item_ids:
        return 0
    with db.transaction() as cursor:
        placeholders = ",".join("?" for _ in item_ids)
        cursor.execute(
            f"DELETE FROM fact_items WHERE id IN ({placeholders})",
            item_ids,
        )
        return cursor.rowcount


def delete_empty_clouds(db: DatabaseManager) -> int:
    """Delete clouds with no bundles (orphaned after bundle moves).

    Uses bundles.cloud_id as the authoritative source.
    Returns the count of deleted clouds.
    """
    # Find orphaned clouds (no bundles point to them)
    rows = db.fetchall(
        """
        SELECT c.id FROM clouds c
        LEFT JOIN bundles b ON c.id = b.cloud_id
        WHERE b.id IS NULL
        """,
        (),
    )
    if not rows:
        return 0

    orphan_ids = [r["id"] for r in rows]
    with db.transaction() as cursor:
        for cid in orphan_ids:
            # Delete any facts for these clouds first
            cursor.execute(
                "DELETE FROM fact_items WHERE fact_id IN "
                "(SELECT id FROM facts WHERE cloud_id = ?)",
                (cid,),
            )
            cursor.execute("DELETE FROM facts WHERE cloud_id = ?", (cid,))
            cursor.execute("DELETE FROM clouds WHERE id = ?", (cid,))

    return len(orphan_ids)


def cleanup_document(db: DatabaseManager, document_id: str) -> dict[str, Any]:
    """Remove a document and all its dependent data (compensating cleanup).

    Deletes in order: fact_items → facts → cloud_bundles → clouds (if empty)
    → bundle_atoms → bundles → atoms → document.

    For surviving clouds (still have bundles from other documents), the stale
    fact is deleted and the cloud is re-collapsed with remaining bundles.
    Annotations on doomed facts/items are collected before deletion so they
    can be migrated to new facts after re-ingestion.

    Returns:
        Dict with keys:
        - cleaned (bool): True if document existed and was cleaned up
        - saved_annotations (list[dict]): annotations from deleted facts/items
        - surviving_cloud_ids (list[str]): cloud IDs that survived with
          remaining bundles and were re-collapsed
    """
    from alibi.annotations.store import collect_annotations_for_cleanup
    from alibi.clouds.correction import recollapse_cloud as _recollapse

    result: dict[str, Any] = {
        "cleaned": False,
        "saved_annotations": [],
        "surviving_cloud_ids": [],
    }

    doc = db.fetchone("SELECT id FROM documents WHERE id = ?", (document_id,))
    if not doc:
        return result

    # Find bundles for this document
    bundle_rows = db.fetchall(
        "SELECT id, cloud_id FROM bundles WHERE document_id = ?",
        (document_id,),
    )
    bundle_ids = [r["id"] for r in bundle_rows]
    cloud_ids = {r["cloud_id"] for r in bundle_rows if r["cloud_id"]}

    # Collect ALL fact_ids and fact_item_ids that will be deleted
    all_fact_ids: list[str] = []
    all_fact_item_ids: list[str] = []

    for cid in cloud_ids:
        fact_rows = db.fetchall("SELECT id FROM facts WHERE cloud_id = ?", (cid,))
        for fr in fact_rows:
            all_fact_ids.append(fr["id"])
            item_rows = db.fetchall(
                "SELECT id FROM fact_items WHERE fact_id = ?", (fr["id"],)
            )
            all_fact_item_ids.extend(ir["id"] for ir in item_rows)

    # Collect annotations before cascade delete
    saved_annotations = collect_annotations_for_cleanup(
        db, all_fact_ids, all_fact_item_ids
    )
    result["saved_annotations"] = saved_annotations

    with db.transaction() as cursor:
        for bid in bundle_ids:
            cursor.execute("DELETE FROM cloud_bundles WHERE bundle_id = ?", (bid,))
            cursor.execute("DELETE FROM bundle_atoms WHERE bundle_id = ?", (bid,))

        cursor.execute("DELETE FROM bundles WHERE document_id = ?", (document_id,))

        for cid in cloud_ids:
            remaining = db.fetchone(
                "SELECT COUNT(*) as cnt FROM bundles WHERE cloud_id = ?",
                (cid,),
            )
            if remaining and remaining["cnt"] == 0:
                # Empty cloud — delete fact, items, and cloud
                cursor.execute(
                    "DELETE FROM annotations WHERE target_type = 'fact_item' "
                    "AND target_id IN "
                    "(SELECT id FROM fact_items WHERE fact_id IN "
                    "(SELECT id FROM facts WHERE cloud_id = ?))",
                    (cid,),
                )
                cursor.execute(
                    "DELETE FROM annotations WHERE target_type = 'fact' "
                    "AND target_id IN "
                    "(SELECT id FROM facts WHERE cloud_id = ?)",
                    (cid,),
                )
                cursor.execute(
                    "DELETE FROM fact_items WHERE fact_id IN "
                    "(SELECT id FROM facts WHERE cloud_id = ?)",
                    (cid,),
                )
                cursor.execute("DELETE FROM facts WHERE cloud_id = ?", (cid,))
                cursor.execute("DELETE FROM clouds WHERE id = ?", (cid,))
            else:
                # Surviving cloud — delete stale fact, re-collapse later
                cursor.execute(
                    "DELETE FROM annotations WHERE target_type = 'fact_item' "
                    "AND target_id IN "
                    "(SELECT id FROM fact_items WHERE fact_id IN "
                    "(SELECT id FROM facts WHERE cloud_id = ?))",
                    (cid,),
                )
                cursor.execute(
                    "DELETE FROM annotations WHERE target_type = 'fact' "
                    "AND target_id IN "
                    "(SELECT id FROM facts WHERE cloud_id = ?)",
                    (cid,),
                )
                cursor.execute(
                    "DELETE FROM fact_items WHERE fact_id IN "
                    "(SELECT id FROM facts WHERE cloud_id = ?)",
                    (cid,),
                )
                cursor.execute("DELETE FROM facts WHERE cloud_id = ?", (cid,))
                result["surviving_cloud_ids"].append(cid)

        cursor.execute("DELETE FROM atoms WHERE document_id = ?", (document_id,))
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    # Re-collapse surviving clouds outside the main transaction
    for cid in result["surviving_cloud_ids"]:
        try:
            _recollapse(db, cid)
        except Exception as e:
            logger.warning(f"Re-collapse failed for cloud {cid[:8]}: {e}")

    result["cleaned"] = True
    logger.info(f"Cleaned up document {document_id[:8]} and dependent data")
    return result


def cleanup_orphaned_atoms(db: DatabaseManager) -> int:
    """Delete atoms not linked to any bundle via bundle_atoms.

    Returns the count of deleted atoms.
    """
    rows = db.fetchall(
        """
        SELECT a.id FROM atoms a
        LEFT JOIN bundle_atoms ba ON a.id = ba.atom_id
        WHERE ba.atom_id IS NULL
        """,
        (),
    )
    if not rows:
        return 0

    orphan_ids = [r["id"] for r in rows]
    with db.transaction() as cursor:
        for aid in orphan_ids:
            cursor.execute("DELETE FROM atoms WHERE id = ?", (aid,))

    logger.info(f"Cleaned up {len(orphan_ids)} orphaned atoms")
    return len(orphan_ids)


def cleanup_orphaned_bundles(db: DatabaseManager) -> int:
    """Delete bundles with NULL cloud_id that have no active processing.

    These are bundles that were never assigned to a cloud, likely
    from interrupted pipeline runs.

    Returns the count of deleted bundles.
    """
    rows = db.fetchall(
        """
        SELECT b.id FROM bundles b
        WHERE b.cloud_id IS NULL
        """,
        (),
    )
    if not rows:
        return 0

    orphan_ids = [r["id"] for r in rows]
    with db.transaction() as cursor:
        for bid in orphan_ids:
            cursor.execute("DELETE FROM bundle_atoms WHERE bundle_id = ?", (bid,))
            cursor.execute("DELETE FROM bundles WHERE id = ?", (bid,))

    logger.info(f"Cleaned up {len(orphan_ids)} orphaned bundles")
    return len(orphan_ids)


def run_maintenance(db: DatabaseManager) -> dict[str, int]:
    """Run all cleanup operations. Returns counts of cleaned items."""
    return {
        "orphaned_atoms": cleanup_orphaned_atoms(db),
        "orphaned_bundles": cleanup_orphaned_bundles(db),
        "empty_clouds": delete_empty_clouds(db),
    }


def get_cloud_id_for_bundle(db: DatabaseManager, bundle_id: str) -> str | None:
    """Get the cloud ID for a bundle. Alias for get_cloud_for_bundle."""
    return get_cloud_for_bundle(db, bundle_id)


def get_bundles_in_cloud(db: DatabaseManager, cloud_id: str) -> list[str]:
    """Get all bundle IDs in a cloud (from authoritative bundles.cloud_id)."""
    rows = db.fetchall(
        "SELECT id FROM bundles WHERE cloud_id = ?",
        (cloud_id,),
    )
    return [r["id"] for r in rows]


def set_bundle_cloud(
    db: DatabaseManager,
    bundle_id: str,
    cloud_id: str | None,
) -> bool:
    """Set the cloud_id on a bundle — the user-facing reassignment field.

    This is the simplest correction interface: the user edits the
    cloud_id field on a bundle record to reassign it.

    - Set to a cloud ID: bundle moves to that cloud
    - Set to None: bundle is unassigned (detached from all clouds)

    Also updates the cloud_bundles junction table to stay in sync.
    Does NOT handle fact deletion or re-collapse — caller must do that
    (or use correction.move_bundle() for the full workflow).

    Returns True if the bundle exists and was updated.
    """
    bundle_row = db.fetchone("SELECT id FROM bundles WHERE id = ?", (bundle_id,))
    if not bundle_row:
        return False

    if cloud_id is not None:
        cloud_row = db.fetchone("SELECT id FROM clouds WHERE id = ?", (cloud_id,))
        if not cloud_row:
            return False

    with db.transaction() as cursor:
        # Update the authoritative field
        cursor.execute(
            "UPDATE bundles SET cloud_id = ? WHERE id = ?",
            (cloud_id, bundle_id),
        )
        # Keep junction table in sync
        cursor.execute(
            "DELETE FROM cloud_bundles WHERE bundle_id = ?",
            (bundle_id,),
        )
        if cloud_id is not None:
            cursor.execute(
                "INSERT INTO cloud_bundles (cloud_id, bundle_id, match_type, "
                "match_confidence) VALUES (?, ?, ?, ?)",
                (cloud_id, bundle_id, CloudMatchType.MANUAL.value, 1.0),
            )

    return True


def get_unassigned_bundles(db: DatabaseManager) -> list[dict[str, Any]]:
    """Get bundles with cloud_id = NULL (detached, need re-matching).

    Returns list of bundle dicts with document info.
    """
    rows = db.fetchall(
        """
        SELECT b.id, b.document_id, b.bundle_type, d.file_path
        FROM bundles b
        JOIN documents d ON b.document_id = d.id
        WHERE b.cloud_id IS NULL
        ORDER BY b.created_at
        """,
        (),
    )
    return [dict(r) for r in rows]


def set_cloud_status(
    db: DatabaseManager,
    cloud_id: str,
    status: str,
) -> None:
    """Set a cloud's status (forming, collapsed, disputed)."""
    db.execute(
        "UPDATE clouds SET status = ? WHERE id = ?",
        (status, cloud_id),
    )
    db.get_connection().commit()


# ---------------------------------------------------------------------------
# Historical lookup operations (for extraction consistency checks)
# ---------------------------------------------------------------------------


def find_vendors_by_registration(
    db: DatabaseManager,
    registration: str,
) -> list[dict[str, Any]]:
    """Find vendor atoms matching a registration ID (VAT number).

    Registration is the strongest vendor identifier — same across all
    stores in a chain. Returns all matching vendor atom data dicts.
    """
    rows = db.fetchall(
        """
        SELECT a.data
        FROM atoms a
        WHERE a.atom_type = 'vendor'
        """,
        (),
    )

    matches: list[dict[str, Any]] = []
    reg_normalized = registration.strip().upper().replace(" ", "")

    for row in rows:
        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        stored_reg = (
            (data.get("vat_number") or data.get("tax_id") or "")
            .strip()
            .upper()
            .replace(" ", "")
        )
        if stored_reg and stored_reg == reg_normalized:
            matches.append(data)

    return matches


def get_known_vendor_names(
    db: DatabaseManager,
    registration: str,
) -> list[str]:
    """Get all known vendor names for a registration ID.

    Returns distinct vendor names associated with this registration,
    ordered by frequency (most common first).
    """
    vendors = find_vendors_by_registration(db, registration)
    if not vendors:
        return []

    # Count name frequencies
    name_counts: dict[str, int] = defaultdict(int)
    for v in vendors:
        name = v.get("name", "").strip()
        if name:
            name_counts[name] += 1

    return sorted(name_counts, key=lambda n: name_counts[n], reverse=True)


def find_matching_fact_vendors(
    db: DatabaseManager,
    vendor_name: str,
) -> list[str]:
    """Find known vendor names that fuzzy-match the given name.

    Fallback for when no registration ID is available. Searches the
    facts table for vendors with similar names (case-insensitive substring).

    Returns distinct vendor names ordered by frequency (most common first).
    """
    # Use LIKE for substring matching
    rows = db.fetchall(
        """
        SELECT vendor, COUNT(*) AS cnt
        FROM facts
        WHERE vendor IS NOT NULL
          AND LOWER(vendor) LIKE ?
        GROUP BY vendor
        ORDER BY cnt DESC
        """,
        (f"%{vendor_name.lower()}%",),
    )
    return [r["vendor"] for r in rows]


def get_known_product_names_for_vendor(
    db: DatabaseManager,
    vendor_name: str,
) -> list[str]:
    """Get historical product names purchased from a vendor.

    Looks up fact_items for facts matching the vendor (case-insensitive
    substring). Returns distinct normalized product names.
    """
    rows = db.fetchall(
        """
        SELECT DISTINCT fi.name_normalized
        FROM fact_items fi
        JOIN facts f ON fi.fact_id = f.id
        WHERE LOWER(f.vendor) LIKE ?
          AND fi.name_normalized IS NOT NULL
          AND fi.name_normalized != ''
        ORDER BY fi.name_normalized
        """,
        (f"%{vendor_name.lower()}%",),
    )
    return [r["name_normalized"] for r in rows]


def get_vendor_details_history(
    db: DatabaseManager,
    vendor_name: str,
) -> dict[str, list[str]]:
    """Get historical vendor details (address, phone, website) for a vendor.

    Returns dict mapping field name to list of known values (most common first).
    Matches vendor by name (case-insensitive substring on facts table).
    """
    # Find vendor atoms via facts for this vendor
    rows = db.fetchall(
        """
        SELECT a.data
        FROM atoms a
        JOIN bundle_atoms ba ON a.id = ba.atom_id
        JOIN bundles b ON ba.bundle_id = b.id
        JOIN cloud_bundles cb ON b.id = cb.bundle_id
        JOIN facts f ON cb.cloud_id = f.cloud_id
        WHERE a.atom_type = 'vendor'
          AND LOWER(f.vendor) LIKE ?
        """,
        (f"%{vendor_name.lower()}%",),
    )

    details: dict[str, dict[str, int]] = {
        "address": defaultdict(int),
        "phone": defaultdict(int),
        "website": defaultdict(int),
        "vat_number": defaultdict(int),
        "tax_id": defaultdict(int),
    }

    for row in rows:
        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        for field_name in details:
            val = (data.get(field_name) or "").strip()
            if val:
                details[field_name][val] += 1

    result: dict[str, list[str]] = {}
    for field_name, counts in details.items():
        if counts:
            result[field_name] = sorted(counts, key=lambda v: counts[v], reverse=True)
        else:
            result[field_name] = []

    return result


def get_facts_by_vendor_key(
    db: DatabaseManager,
    vendor_key: str,
) -> list[dict[str, Any]]:
    """Get all facts matching a stable vendor key."""
    rows = db.fetchall(
        "SELECT * FROM facts WHERE vendor_key = ? ORDER BY event_date DESC",
        (vendor_key,),
    )
    return [dict(r) for r in rows]


def backfill_vendor_keys(
    db: DatabaseManager,
    old_key: str,
    new_key: str,
) -> int:
    """Update all facts with old_key to new_key.

    Used when a registration ID is discovered for a vendor that previously
    only had a name-hash key.

    Returns the number of facts updated.
    """
    cursor = db.execute(
        "UPDATE facts SET vendor_key = ? WHERE vendor_key = ?",
        (new_key, old_key),
    )
    db.get_connection().commit()
    count: int = cursor.rowcount
    return count


def get_canonical_unit_quantity(
    db: DatabaseManager,
    item_name: str,
    barcode: str | None = None,
    vendor_key: str | None = None,
    brand: str | None = None,
) -> dict[str, Any] | None:
    """Look up canonical unit_quantity from historical data.

    Cascade (priority order):
    1. Same item + same vendor (barcode or name match) -> highest confidence
    2. Same item + same brand (across vendors) -> high confidence
    3. Same item identity (any vendor) -> medium confidence

    Returns dict with keys: unit_quantity (float), unit (str), confidence (float), source (str)
    or None if no historical data found.
    """
    _BASE_WHERE = (
        "fi.unit_quantity IS NOT NULL AND fi.unit_quantity > 0"
        " AND ((fi.barcode = ? AND fi.barcode != '') OR LOWER(fi.name) = LOWER(?))"
    )
    _BASE_PARAMS: tuple[Any, ...] = (barcode or "", item_name)

    def _run(
        extra_where: str, params: tuple[Any, ...], confidence: float, source: str
    ) -> dict[str, Any] | None:
        sql = (
            "SELECT fi.unit_quantity, fi.unit, COUNT(*) as freq"
            " FROM fact_items fi"
            " JOIN facts f ON fi.fact_id = f.id"
            f" WHERE {_BASE_WHERE} AND {extra_where}"
            " GROUP BY fi.unit_quantity, fi.unit"
            " ORDER BY freq DESC"
            " LIMIT 1"
        )
        row = db.fetchone(sql, _BASE_PARAMS + params)
        if row is None:
            return None
        uq = row["unit_quantity"]
        if uq is None:
            return None
        return {
            "unit_quantity": float(uq),
            "unit": row["unit"],
            "confidence": confidence,
            "source": source,
        }

    # Level 1: same vendor
    if vendor_key:
        result = _run("f.vendor_key = ?", (vendor_key,), 0.95, "vendor_history")
        if result:
            return result

    # Level 2: same brand
    if brand and brand.strip():
        result = _run(
            "fi.brand IS NOT NULL AND fi.brand != '' AND LOWER(fi.brand) = LOWER(?)",
            (brand,),
            0.85,
            "brand_history",
        )
        if result:
            return result

    # Level 3: any vendor, name/barcode match only
    sql = (
        "SELECT fi.unit_quantity, fi.unit, COUNT(*) as freq"
        " FROM fact_items fi"
        " JOIN facts f ON fi.fact_id = f.id"
        f" WHERE {_BASE_WHERE}"
        " GROUP BY fi.unit_quantity, fi.unit"
        " ORDER BY freq DESC"
        " LIMIT 1"
    )
    row = db.fetchone(sql, _BASE_PARAMS)
    if row is None:
        return None
    uq = row["unit_quantity"]
    if uq is None:
        return None
    return {
        "unit_quantity": float(uq),
        "unit": row["unit"],
        "confidence": 0.75,
        "source": "name_history",
    }


def get_canonical_comparable_name(
    db: DatabaseManager,
    item_name: str,
    barcode: str | None = None,
    vendor_key: str | None = None,
    brand: str | None = None,
) -> dict[str, Any] | None:
    """Look up canonical comparable_name from historical data.

    Cascade (priority order):
    1. Same item + same vendor (barcode or name match) -> highest confidence
    2. Same item + same brand (across vendors) -> high confidence
    3. Same item identity (any vendor) -> medium confidence

    Returns dict with keys: comparable_name (str), confidence (float), source (str)
    or None if no historical data found.
    """
    _BASE_WHERE = (
        "fi.comparable_name IS NOT NULL AND fi.comparable_name != ''"
        " AND ((fi.barcode = ? AND fi.barcode != '') OR LOWER(fi.name) = LOWER(?))"
    )
    _BASE_PARAMS: tuple[Any, ...] = (barcode or "", item_name)

    def _run(
        extra_where: str, params: tuple[Any, ...], confidence: float, source: str
    ) -> dict[str, Any] | None:
        sql = (
            "SELECT fi.comparable_name, COUNT(*) as freq"
            " FROM fact_items fi"
            " JOIN facts f ON fi.fact_id = f.id"
            f" WHERE {_BASE_WHERE} AND {extra_where}"
            " GROUP BY fi.comparable_name"
            " ORDER BY freq DESC"
            " LIMIT 1"
        )
        row = db.fetchone(sql, _BASE_PARAMS + params)
        if row is None:
            return None
        cn = row["comparable_name"]
        if cn is None:
            return None
        return {
            "comparable_name": cn,
            "confidence": confidence,
            "source": source,
        }

    # Level 1: same vendor
    if vendor_key:
        result = _run("f.vendor_key = ?", (vendor_key,), 0.95, "vendor_history")
        if result:
            return result

    # Level 2: same brand
    if brand and brand.strip():
        result = _run(
            "fi.brand IS NOT NULL AND fi.brand != '' AND LOWER(fi.brand) = LOWER(?)",
            (brand,),
            0.85,
            "brand_history",
        )
        if result:
            return result

    # Level 3: any vendor, name/barcode match only
    sql = (
        "SELECT fi.comparable_name, COUNT(*) as freq"
        " FROM fact_items fi"
        " JOIN facts f ON fi.fact_id = f.id"
        f" WHERE {_BASE_WHERE}"
        " GROUP BY fi.comparable_name"
        " ORDER BY freq DESC"
        " LIMIT 1"
    )
    row = db.fetchone(sql, _BASE_PARAMS)
    if row is None:
        return None
    cn = row["comparable_name"]
    if cn is None:
        return None
    return {
        "comparable_name": cn,
        "confidence": 0.75,
        "source": "name_history",
    }


# ---------------------------------------------------------------------------
# Correction event persistence
# ---------------------------------------------------------------------------


def record_correction_event(
    db: DatabaseManager,
    entity_type: str,
    entity_id: str,
    field: str,
    old_value: Any,
    new_value: Any,
    source: str,
    user_id: str | None = None,
) -> str:
    """Insert a correction event row and return its ID."""
    from uuid import uuid4

    event_id = str(uuid4())
    old_str = str(old_value) if old_value is not None else None
    new_str = str(new_value) if new_value is not None else None
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO correction_events "
            "(id, entity_type, entity_id, field, old_value, new_value, source, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                entity_type,
                entity_id,
                field,
                old_str,
                new_str,
                source,
                user_id,
            ),
        )
    return event_id


def get_corrections_by_entity(
    db: DatabaseManager,
    entity_type: str,
    entity_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return corrections for a specific entity, newest first."""
    rows = db.fetchall(
        "SELECT * FROM correction_events "
        "WHERE entity_type = ? AND entity_id = ? "
        "ORDER BY created_at DESC "
        "LIMIT ?",
        (entity_type, entity_id, limit),
    )
    return [dict(row) for row in rows]


def get_corrections_by_field(
    db: DatabaseManager,
    field: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return corrections for a specific field across all entities, newest first."""
    rows = db.fetchall(
        "SELECT * FROM correction_events "
        "WHERE field = ? "
        "ORDER BY created_at DESC "
        "LIMIT ?",
        (field, limit),
    )
    return [dict(row) for row in rows]


def get_correction_rate(
    db: DatabaseManager,
    vendor_key: str,
    window_days: int = 90,
) -> dict[str, Any]:
    """Return correction rate for a vendor within the time window.

    Returns {"total_facts": int, "corrected_facts": int, "rate": float}.
    """
    row = db.fetchone(
        "SELECT COUNT(DISTINCT f.id) AS total_facts "
        "FROM facts f "
        "WHERE f.vendor_key = ? "
        "AND f.created_at >= strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ? || ' days')",
        (vendor_key, f"-{window_days}"),
    )
    total = row["total_facts"] if row else 0

    corrected_row = db.fetchone(
        "SELECT COUNT(DISTINCT ce.entity_id) AS corrected_facts "
        "FROM correction_events ce "
        "JOIN facts f ON ce.entity_id = f.id "
        "WHERE ce.entity_type = 'fact' "
        "AND f.vendor_key = ? "
        "AND ce.created_at >= strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ? || ' days')",
        (vendor_key, f"-{window_days}"),
    )
    corrected = corrected_row["corrected_facts"] if corrected_row else 0

    rate = corrected / total if total > 0 else 0.0
    return {"total_facts": total, "corrected_facts": corrected, "rate": rate}


def get_corrections_by_vendor(
    db: DatabaseManager,
    vendor_key: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return corrections for all facts/fact_items belonging to a vendor, newest first."""
    rows = db.fetchall(
        "SELECT ce.* FROM correction_events ce "
        "JOIN facts f ON ce.entity_id = f.id OR ce.entity_id IN ("
        "    SELECT fi.id FROM fact_items fi WHERE fi.fact_id = f.id"
        ") "
        "WHERE ce.entity_type IN ('fact', 'fact_item') "
        "AND f.vendor_key = ? "
        "ORDER BY ce.created_at DESC "
        "LIMIT ?",
        (vendor_key, limit),
    )
    return [dict(row) for row in rows]
