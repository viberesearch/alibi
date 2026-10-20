"""Hosted Jev review of assigned categories, off by default.

The local combined/categorize passes assign every ``category_path``. This pass
asks Typesafe's Jev decision model (``/v1/systemone``) to re-read items that
already carry a category, and compares its answer with the stored one:

* **agree** -- same path, or Jev's answer is an ancestor of it: nothing to do.
* **deepen** -- Jev picks a descendant of the stored path (``food > dairy`` ->
  ``food > dairy > cheese``) at or above ``jev_refine_threshold``. Applied
  directly when ``jev_auto_apply`` is on, otherwise written to the review file.
* **flag** -- Jev picks a conflicting path at or above ``jev_flag_threshold``.
  Always written to the review file; nothing changes until a human approves it
  and runs ``lt enrich apply-coherence-fixes``.
* **ignore** -- a conflicting answer below the threshold.

Efficiency, measured 2026-10-20 on 150 corpus items: one product per call took
0.47 s per item; 32 products per call took 0.026 s per item with about the same
answers (134/150 identical). So rows are grouped by product (comparable name,
falling back to the normalised and raw name) and stored path, one row per group
is sent, and groups are packed ``jev_batch_size`` to a call. The 57-path
taxonomy exceeds Jev's 26-option ``choice`` limit, so each batch is two calls:
the branch (food sub-branches plus the non-food tops), then the leaf for the
items whose branch has one.

Privacy: only the receipt line, comparable name, brand and store name leave the
machine -- no amounts, dates, barcodes or images.

Rows are stamped ``category_reviewed_by = '<served model>@t<TAXONOMY_VERSION>'``
once answered, so a later run only sends new products, and a model or taxonomy
change makes every row eligible again. A response whose ``model`` field differs
from the pinned model aborts the run before anything in that batch is written.
"""

from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from alibi.config import Config, get_config
from alibi.db.connection import DatabaseManager
from alibi.enrichment import taxonomy
from alibi.enrichment._batch import mark_processed
from alibi.enrichment.categorize import _apply_category
from alibi.enrichment.coherence_audit import CoherenceFinding, write_findings_yaml

logger = logging.getLogger(__name__)

_SOURCE = "jev_refine"
_MARKER_COLUMN = "category_reviewed_by"
# Never re-reviewed by this pass: a human or a deterministic rule decided them,
# or they inherited the category from another row (review that row instead).
_SKIP_SOURCES = ("user_confirmed", "keyword_category", "sibling_propagation")
_RETRY_STATUSES = (429, 529)


_REVIEW_HEADER = """\
# Jev category review -- REVIEW FILE
#
# Each finding is an item whose stored category_path the hosted Jev model
# answered differently: "conflicts" (a different category) or "more specific"
# (a deeper path; listed here while ALIBI_JEV_AUTO_APPLY is off). NOTHING has
# been changed.
#
# To accept a fix: set `approved: true` on that finding (edit
# suggested_category_path first if you want a different one), then run
#   lt enrich apply-coherence-fixes --file <this file>
# Unapproved findings are ignored.
"""


class JevError(RuntimeError):
    """The Jev API failed, or served a different model than the pinned one."""


@dataclass
class ReviewGroup:
    """Rows that share a product key and a stored category path."""

    category_path: str
    item_ids: list[str]
    representative: dict[str, Any]


@dataclass
class JevReviewResult:
    """Outcome of one review run."""

    groups_reviewed: int = 0
    items_marked: int = 0
    calls: int = 0
    refined_items: int = 0
    findings: list[CoherenceFinding] = field(default_factory=list)


def reviewed_marker(model: str) -> str:
    """Marker stored on reviewed rows: model plus taxonomy version."""
    return f"{model}@t{taxonomy.TAXONOMY_VERSION}"


def taxonomy_levels() -> tuple[list[str], dict[str, list[str]]]:
    """Split the taxonomy into branch options and the leaf options per branch."""
    branch_of: dict[str, str] = {}
    for path in taxonomy.VALID_PATHS:
        parts = path.split(taxonomy.PATH_SEPARATOR)
        if parts[0] == "food":
            branch_of[path] = taxonomy.PATH_SEPARATOR.join(parts[:2])
        else:
            branch_of[path] = parts[0]
    branches = sorted(set(branch_of.values()))
    children = {
        b: sorted(p for p, pb in branch_of.items() if pb == b) for b in branches
    }
    return branches, children


def decide(stored: str, answer: str, confidence: float, cfg: Config) -> str:
    """Classify Jev's answer against the stored path (see module docstring)."""
    sep = taxonomy.PATH_SEPARATOR
    if answer == stored or stored.startswith(answer + sep):
        return "agree"
    if answer.startswith(stored + sep):
        return "deepen" if confidence >= cfg.jev_refine_threshold else "agree"
    return "flag" if confidence >= cfg.jev_flag_threshold else "ignore"


def candidate_groups(db: DatabaseManager, limit: int, marker: str) -> list[ReviewGroup]:
    """Unreviewed categorised rows, grouped by product, newest first."""
    skip = ", ".join("?" for _ in _SKIP_SOURCES)
    rows = (
        db.get_connection()
        .execute(
            "SELECT fi.id, fi.name, fi.name_normalized, fi.comparable_name, "  # nosec B608
            "       fi.brand, fi.category_path, f.vendor "
            "FROM fact_items fi JOIN facts f ON f.id = fi.fact_id "
            "WHERE fi.category_path IS NOT NULL AND fi.category_path != '' "
            f"  AND COALESCE(fi.enrichment_source, '') NOT IN ({skip}) "
            f"  AND COALESCE(fi.{_MARKER_COLUMN}, '') != ? "
            "ORDER BY f.event_date DESC, fi.id",
            (*_SKIP_SOURCES, marker),
        )
        .fetchall()
    )
    groups: dict[tuple[str, str], ReviewGroup] = {}
    for raw in rows:
        row = dict(raw)
        key = (
            (
                row.get("comparable_name")
                or row.get("name_normalized")
                or row["name"]
                or ""
            )
            .strip()
            .lower()
        )
        gkey = (key, row["category_path"])
        group = groups.get(gkey)
        if group is None:
            if len(groups) >= limit:
                continue
            groups[gkey] = ReviewGroup(row["category_path"], [row["id"]], row)
        else:
            group.item_ids.append(row["id"])
    return list(groups.values())


def _state_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "receipt_line": row.get("name"),
        "comparable_name": row.get("comparable_name"),
        "brand": row.get("brand"),
        "store": row.get("vendor"),
    }


def call_systemone(
    state: dict[str, Any],
    questions: dict[str, Any],
    cfg: Config,
    client: httpx.Client,
) -> dict[str, Any]:
    """One ``/v1/systemone`` call. Retries 429/529 with backoff."""
    payload = {"model": cfg.jev_model, "state": state, "questions": questions}
    headers = {"Authorization": f"Bearer {cfg.jev_api_key}"}
    resp: httpx.Response | None = None
    for attempt in range(4):
        resp = client.post(cfg.jev_url, json=payload, headers=headers)
        if resp.status_code not in _RETRY_STATUSES:
            break
        time.sleep(2**attempt)
    assert resp is not None
    if resp.status_code != 200:
        raise JevError(f"Jev HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    served = data.get("model")
    if served != cfg.jev_model:
        raise JevError(f"Jev served {served!r}, pinned {cfg.jev_model!r}")
    answers = data.get("answers")
    if not isinstance(answers, dict):
        raise JevError("Jev response has no answers")
    return answers


def _ask_batch(
    batch: list[ReviewGroup], cfg: Config, client: httpx.Client
) -> tuple[list[tuple[str, float]], int]:
    """Return ``(path, confidence)`` per group, and the number of calls made."""
    branches, children = taxonomy_levels()
    state = {f"item_{k}": _state_entry(g.representative) for k, g in enumerate(batch)}
    q1 = {
        f"item_{k}": {
            "type": "choice",
            "instructions": f"Which spending category does item_{k} belong to? "
            f"Judge item_{k} only. Lines in Greek, Russian or Turkish are common.",
            "criteria": {b: None for b in branches},
        }
        for k in range(len(batch))
    }
    a1 = call_systemone(state, q1, cfg, client)
    picks: list[tuple[str, float]] = []
    q2: dict[str, Any] = {}
    for k in range(len(batch)):
        ans = a1[f"item_{k}"]
        branch, conf = ans["choice"], float(ans["confidence"])
        leaves = children[branch]
        picks.append((leaves[0], conf))
        if len(leaves) > 1:
            q2[f"item_{k}"] = {
                "type": "choice",
                "instructions": f"Within '{branch}', which subcategory fits "
                f"item_{k}? Judge item_{k} only.",
                "criteria": {leaf: None for leaf in leaves},
            }
    if not q2:
        return picks, 1
    a2 = call_systemone(state, q2, cfg, client)
    for key, ans in a2.items():
        k = int(key.removeprefix("item_"))
        picks[k] = (ans["choice"], min(picks[k][1], float(ans["confidence"])))
    return picks, 2


def review_categories(
    db: DatabaseManager,
    limit: int = 500,
    cfg: Config | None = None,
    client: httpx.Client | None = None,
) -> JevReviewResult:
    """Review up to ``limit`` product groups with Jev (see module docstring).

    Raises:
        JevError: on an API failure or a model mismatch. Batches finished
            before the failure keep their writes and markers.
    """
    cfg = cfg or get_config()
    if not cfg.jev_api_key:
        raise JevError("JEV_API_KEY is not set")
    marker = reviewed_marker(cfg.jev_model)
    groups = candidate_groups(db, limit, marker)
    result = JevReviewResult()
    if not groups:
        return result

    own_client = client is None
    http = client or httpx.Client(timeout=cfg.jev_timeout)
    try:
        size = max(1, min(cfg.jev_batch_size, 64))
        for start in range(0, len(groups), size):
            batch = groups[start : start + size]
            picks, calls = _ask_batch(batch, cfg, http)
            result.calls += calls
            for group, (path, conf) in zip(batch, picks):
                _record(db, group, path, conf, cfg, result)
            ids = [i for g in batch for i in g.item_ids]
            mark_processed(db, _MARKER_COLUMN, ids, marker)
            result.groups_reviewed += len(batch)
            result.items_marked += len(ids)
    finally:
        if own_client:
            http.close()

    if result.refined_items:
        from alibi.services.item_stars import rebuild_item_stars

        rebuild_item_stars(db)
    return result


def _record(
    db: DatabaseManager,
    group: ReviewGroup,
    path: str,
    conf: float,
    cfg: Config,
    result: JevReviewResult,
) -> None:
    verdict = decide(group.category_path, path, conf, cfg)
    if verdict in ("agree", "ignore"):
        return
    if verdict == "deepen" and cfg.jev_auto_apply:
        for item_id in group.item_ids:
            _apply_category(db, item_id, path, source=_SOURCE, confidence=conf)
        result.refined_items += len(group.item_ids)
        return
    row = group.representative
    kind = "more specific" if verdict == "deepen" else "conflicts"
    result.findings.append(
        CoherenceFinding(
            item_id=row["id"],
            name=row["name"],
            vendor=row.get("vendor"),
            comparable_name=row.get("comparable_name"),
            category_path=group.category_path,
            suggested_comparable_name=None,
            suggested_category_path=path,
            reason=f"{cfg.jev_model}: {kind}, confidence {conf:.2f}, "
            f"{len(group.item_ids)} row(s) share this product",
        )
    )


def default_review_dir(cfg: Config | None = None) -> Path:
    """``jev_review/`` next to the database."""
    cfg = cfg or get_config()
    return cfg.get_absolute_db_path().parent / "jev_review"


def write_review_file(findings: list[CoherenceFinding], directory: Path) -> Path:
    """Write one timestamped review file per run, so earlier files are kept."""
    now = datetime.datetime.now()
    path = directory / f"jev_review_{now:%Y%m%d_%H%M%S}.yaml"
    write_findings_yaml(
        findings,
        path,
        generated=now.isoformat(timespec="seconds"),
        header=_REVIEW_HEADER,
    )
    return path
