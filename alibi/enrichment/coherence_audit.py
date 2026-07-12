"""LLM audit of enrichment coherence, human-review-gated.

Enrichment passes occasionally hallucinate: a mineral water classified as
wine, a cocktail labelled cheese, an espresso with a 450 g "package size".
Deterministic checks cannot catch these — the fields are filled and
well-formed, just semantically wrong for the item.

This module is the scalable version of fixing such rows by hand:

1. ``audit_coherence`` — batches non-user-confirmed fact items through the
   local LLM, which judges whether (comparable_name, category_path, unit)
   are coherent with the item's name, and proposes corrections. Suggested
   category paths are constrained to the taxonomy already present in the DB
   so the audit cannot invent new branches. Read-only.
2. ``write_findings_yaml`` — writes a review file; every finding defaults to
   ``approved: false``.
3. ``apply_coherence_fixes`` — applies ONLY approved findings, stamps them
   ``enrichment_source='user_confirmed'`` (so later passes leave them alone),
   and rebuilds the item_stars analytics mirror.

Never applies anything without explicit human approval, mirroring the
``propose-name-merges`` / ``apply-name-merges`` pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from alibi.db.connection import DatabaseManager
from alibi.enrichment._batch import call_enrichment_llm

logger = logging.getLogger(__name__)

_BATCH_SIZE = 10

_PROMPT_TEMPLATE = """\
You are auditing a purchase-item database for ENRICHMENT ERRORS.
Vendor: {vendor}

Each line shows: idx. item name | comparable_name | category_path | size

Items:
{items_block}

For EVERY item decide whether comparable_name and category_path are
SEMANTICALLY COHERENT with what the item name describes. Typical errors:
mineral water classified as wine, a drink labelled cheese, a restaurant
dish with a weight parsed from its menu price.

Allowed category paths (choose suggestions ONLY from this list):
{allowed_paths}

Return JSON only:
{{"items": [
  {{"idx": 1, "coherent": true, "reason": null,
    "suggested_comparable_name": null, "suggested_category_path": null}},
  {{"idx": 2, "coherent": false, "reason": "Lauretana is mineral water, not wine",
    "suggested_comparable_name": "mineral water",
    "suggested_category_path": "food > beverages > water"}}
]}}

Rules:
- Return one entry for EVERY item (idx 1..N). Do not omit any.
- coherent=false ONLY when the classification is clearly wrong for the
  named product. When unsure, return coherent=true. Do NOT flag stylistic
  or granularity preferences.
- suggested_category_path MUST be copied verbatim from the allowed list,
  or null if none fits.
- Only the JSON object, no explanation."""

_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "coherent": {"type": "boolean"},
                    "reason": {"type": ["string", "null"]},
                    "suggested_comparable_name": {"type": ["string", "null"]},
                    "suggested_category_path": {"type": ["string", "null"]},
                },
                "required": ["idx", "coherent"],
            },
        },
    },
    "required": ["items"],
}


@dataclass
class CoherenceFinding:
    """One item the auditor judged incoherently enriched."""

    item_id: str
    name: str
    vendor: str | None
    comparable_name: str | None
    category_path: str | None
    suggested_comparable_name: str | None
    suggested_category_path: str | None
    reason: str | None
    approved: bool = False


@dataclass
class ApplyCoherenceResult:
    """Outcome of applying approved coherence fixes."""

    applied: list[CoherenceFinding] = field(default_factory=list)
    rebuilt_stars: int = 0


def _candidate_rows(
    db: DatabaseManager, limit: int, vendor: str | None = None
) -> list[dict[str, Any]]:
    """Fact items eligible for audit: enriched, not human-confirmed."""
    conn = db.get_connection()
    sql = (
        "SELECT fi.id, fi.name, fi.comparable_name, fi.category_path, "
        "       fi.unit, fi.unit_quantity, f.vendor "
        "FROM fact_items fi JOIN facts f ON f.id = fi.fact_id "
        "WHERE (fi.enrichment_source IS NULL "
        "       OR fi.enrichment_source != 'user_confirmed') "
        "  AND (fi.comparable_name IS NOT NULL OR fi.category_path IS NOT NULL) "
    )
    params: list[Any] = []
    if vendor:
        sql += "AND f.vendor = ? "
        params.append(vendor)
    sql += "ORDER BY f.event_date DESC, fi.id LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _allowed_category_paths(db: DatabaseManager) -> list[str]:
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT DISTINCT category_path FROM fact_items "
        "WHERE category_path IS NOT NULL AND category_path != '' "
        "ORDER BY category_path"
    ).fetchall()
    return [r["category_path"] for r in rows]


def _items_block(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, r in enumerate(rows):
        size = ""
        if r.get("unit_quantity"):
            size = f"{r['unit_quantity']}{r.get('unit') or ''}"
        lines.append(
            f"{i + 1}. {r['name']} | {r.get('comparable_name') or '-'} | "
            f"{r.get('category_path') or '-'} | {size or '-'}"
        )
    return "\n".join(lines)


def audit_coherence(
    db: DatabaseManager,
    limit: int = 200,
    vendor: str | None = None,
    model: str | None = None,
) -> list[CoherenceFinding]:
    """Audit enriched items for name/category coherence. Read-only.

    Returns findings for rows the model judged incoherent, with suggestions
    constrained to category paths already present in the DB.
    """
    rows = _candidate_rows(db, limit, vendor)
    if not rows:
        return []
    allowed = _allowed_category_paths(db)
    allowed_set = set(allowed)
    allowed_block = "\n".join(f"- {p}" for p in allowed)

    findings: list[CoherenceFinding] = []
    for start in range(0, len(rows), _BATCH_SIZE):
        batch = rows[start : start + _BATCH_SIZE]
        prompt = _PROMPT_TEMPLATE.format(
            vendor=batch[0].get("vendor") or "unknown",
            items_block=_items_block(batch),
            allowed_paths=allowed_block,
        )
        answers = call_enrichment_llm(
            prompt,
            model=model,
            label="coherence-audit",
            response_format=_RESPONSE_FORMAT,
        )
        for entry in answers:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("idx")
            if not isinstance(idx, int) or not (1 <= idx <= len(batch)):
                continue
            if entry.get("coherent", True):
                continue
            row = batch[idx - 1]
            suggested_path = entry.get("suggested_category_path")
            if suggested_path is not None and suggested_path not in allowed_set:
                # The model invented a path despite instructions — drop it,
                # keep the finding (the reason still points a human at it).
                suggested_path = None
            findings.append(
                CoherenceFinding(
                    item_id=row["id"],
                    name=row["name"],
                    vendor=row.get("vendor"),
                    comparable_name=row.get("comparable_name"),
                    category_path=row.get("category_path"),
                    suggested_comparable_name=entry.get("suggested_comparable_name"),
                    suggested_category_path=suggested_path,
                    reason=entry.get("reason"),
                )
            )
    return findings


_HEADER = """\
# Enrichment coherence audit — REVIEW FILE
#
# Each finding is an item whose comparable_name/category_path the local LLM
# judged inconsistent with the item's name. NOTHING has been changed.
#
# To accept a fix: set `approved: true` on that finding (edit the suggested_*
# values first if you want different ones), then run
#   lt enrich apply-coherence-fixes --file <this file>
# Unapproved findings are ignored.
"""


def _finding_to_dict(f: CoherenceFinding) -> dict[str, Any]:
    return {
        "item_id": f.item_id,
        "name": f.name,
        "vendor": f.vendor,
        "current": {
            "comparable_name": f.comparable_name,
            "category_path": f.category_path,
        },
        "suggested_comparable_name": f.suggested_comparable_name,
        "suggested_category_path": f.suggested_category_path,
        "reason": f.reason,
        "approved": f.approved,
    }


def write_findings_yaml(
    findings: list[CoherenceFinding],
    path: Path,
    *,
    generated: str,
) -> None:
    """Write the review file (all findings default to approved: false)."""
    doc = {
        "generated": generated,
        "findings": [_finding_to_dict(f) for f in findings],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
    path.write_text(_HEADER + body, encoding="utf-8")


def load_approved_findings(path: Path) -> list[CoherenceFinding]:
    """Load findings marked ``approved: true`` from a reviewed file."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("findings"), list):
        raise ValueError("expected a mapping with a 'findings' list")
    approved: list[CoherenceFinding] = []
    for raw in doc["findings"]:
        if not isinstance(raw, dict) or not raw.get("approved"):
            continue
        if not raw.get("item_id"):
            raise ValueError("approved finding without item_id")
        current = raw.get("current") or {}
        approved.append(
            CoherenceFinding(
                item_id=str(raw["item_id"]),
                name=str(raw.get("name") or ""),
                vendor=raw.get("vendor"),
                comparable_name=current.get("comparable_name"),
                category_path=current.get("category_path"),
                suggested_comparable_name=raw.get("suggested_comparable_name"),
                suggested_category_path=raw.get("suggested_category_path"),
                reason=raw.get("reason"),
                approved=True,
            )
        )
    return approved


def apply_coherence_fixes(
    db: DatabaseManager, findings: list[CoherenceFinding]
) -> ApplyCoherenceResult:
    """Apply approved findings and rebuild item_stars.

    Each approved finding updates only the suggested fields that are set,
    and stamps the row ``user_confirmed`` so future enrichment passes and
    audits skip it. Findings whose item no longer exists are skipped.
    """
    from alibi.services.item_stars import rebuild_item_stars

    result = ApplyCoherenceResult()
    conn = db.get_connection()
    for f in findings:
        if f.suggested_comparable_name is None and f.suggested_category_path is None:
            continue
        sets = [
            "enrichment_source = 'user_confirmed'",
            "enrichment_confidence = 1.0",
        ]
        params: list[Any] = []
        if f.suggested_comparable_name is not None:
            sets.append("comparable_name = ?")
            params.append(f.suggested_comparable_name)
        if f.suggested_category_path is not None:
            sets.append("category_path = ?")
            params.append(f.suggested_category_path)
        params.append(f.item_id)
        cur = conn.execute(
            f"UPDATE fact_items SET {', '.join(sets)} WHERE id = ?",  # nosec B608
            params,
        )
        if cur.rowcount:
            result.applied.append(f)
        else:
            logger.warning("Coherence fix skipped, item gone: %s", f.item_id[:8])
    conn.commit()
    if result.applied:
        result.rebuilt_stars = rebuild_item_stars(db)
    return result
