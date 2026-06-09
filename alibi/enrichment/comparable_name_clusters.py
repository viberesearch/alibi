"""Embedding-based comparable_name canonicalization (F2) -- human-review-gated.

The deterministic tidy (:func:`alibi.enrichment.comparable_names.retidy_comparable_names`)
collapses size / pack / percentage token noise, but it cannot reach the
*semantic* fragmentation that leaves the same product on different strings:
synonyms, singular/plural, translations and OCR garble ("fr. goat milk" vs
"goat milk", "artichoke" vs "artichokes"). Those never share a substring the
regex can strip, so they stay in separate analytics buckets.

This pass embeds each distinct ``comparable_name`` with the local nomic model,
clusters near-duplicates *within the same comparable_unit* above a high cosine
threshold, and emits a PROPOSAL file. Nothing is rewritten until a human flips
``approved: true`` on a cluster and runs the apply step -- a wrong merge fuses
two genuinely different products, so it is never automatic.

Two phases:
  1. :func:`propose_name_merges` -> cluster, then :func:`write_proposal_yaml`
     writes a human-editable review file (mutates no data).
  2. :func:`load_approved_clusters` + :func:`apply_name_merges` -> rewrite only
     the approved clusters, then rebuild the ``item_stars`` analytics mirror.

Grouping invariant: clusters never cross ``comparable_unit`` (EUR/L vs EUR/pcs
are different products), mirroring the analytics group-by-unit rule. NULL and
empty units are folded into one "unitless" key via ``COALESCE(comparable_unit,'')``.

Embedding failures fall back to a zero vector, which is similar to nothing, so a
name whose embedding failed is never merged -- it simply stays its own bucket.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

# A deliberately HIGH default: this pass proposes merges that a human then
# reviews, but the cost of a noisy proposal is reviewer fatigue, so we keep the
# candidate set tight. 0.92 cosine on nomic-embed reliably pairs singular/plural
# and close synonyms while leaving distinct products (e.g. "apple" vs "apricot")
# apart.
DEFAULT_THRESHOLD = 0.92

# How many example raw item names to show per variant in the proposal.
_DEFAULT_EXAMPLES = 3

EmbedFn = Callable[[str], list[float]]


@dataclass
class NameStat:
    """Aggregate of one distinct (comparable_name, comparable_unit) bucket."""

    name: str
    unit: str  # COALESCE(comparable_unit, '') -- '' is the unitless key
    count: int
    examples: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)


@dataclass
class MergeCluster:
    """A proposed group of comparable_names to collapse onto one canonical."""

    canonical: str
    comparable_unit: str
    members: list[NameStat]

    def variant_names(self) -> list[str]:
        """Member names other than the canonical (the ones that get rewritten)."""
        return [m.name for m in self.members if m.name != self.canonical]


@dataclass
class RewriteRecord:
    """One comparable_name -> canonical rewrite that apply performed."""

    old_name: str
    new_name: str
    comparable_unit: str
    rows: int


@dataclass
class ApplyResult:
    """Outcome of applying a set of approved merge clusters."""

    rewrites: list[RewriteRecord]
    rebuilt_stars: int

    @property
    def rewritten_rows(self) -> int:
        return sum(r.rows for r in self.rewrites)


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0.0 if either vector is zero-length/empty."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _default_embed_fn(text: str) -> list[float]:
    """Embed via the local nomic model, falling back to a zero vector on error.

    A zero vector is similar to nothing (cosine 0), so an item whose embedding
    failed can never be merged -- it stays in its own bucket rather than risk a
    wrong merge.
    """
    from alibi.vectordb.embeddings import EMBEDDING_DIM, EmbeddingError, get_embedding

    try:
        return get_embedding(text)
    except EmbeddingError as exc:  # pragma: no cover - network path
        logger.warning("Embedding failed for %r: %s (zero-vector fallback)", text, exc)
        return [0.0] * EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Phase 1: propose
# ---------------------------------------------------------------------------


def _gather_name_stats(db: DatabaseManager, examples: int) -> list[NameStat]:
    """Aggregate every distinct (comparable_name, unit) bucket with examples."""
    rows = db.fetchall(
        "SELECT comparable_name AS name, "
        "       COALESCE(comparable_unit, '') AS unit, "
        "       COUNT(*) AS cnt, "
        "       GROUP_CONCAT(DISTINCT name) AS raw_names, "
        "       GROUP_CONCAT(DISTINCT category) AS cats "
        "FROM fact_items "
        "WHERE comparable_name IS NOT NULL AND comparable_name != '' "
        "GROUP BY comparable_name, COALESCE(comparable_unit, '')"
    )
    stats: list[NameStat] = []
    for row in rows:
        raw = row["raw_names"] or ""
        ex = [s for s in (p.strip() for p in raw.split(",")) if s][:examples]
        cats_raw = row["cats"] or ""
        cats = sorted({s.strip() for s in cats_raw.split(",") if s.strip()})
        stats.append(
            NameStat(
                name=row["name"],
                unit=row["unit"],
                count=int(row["cnt"]),
                examples=ex,
                categories=cats,
            )
        )
    return stats


def _pick_canonical(members: list[NameStat]) -> str:
    """Deterministic canonical: most rows, then shortest, then alphabetical."""
    return min(members, key=lambda s: (-s.count, len(s.name), s.name)).name


def _cluster_unit_group(
    members: list[NameStat],
    embeddings: dict[str, list[float]],
    threshold: float,
) -> list[list[NameStat]]:
    """Union-find cluster of one unit's names by cosine >= threshold."""
    n = len(members)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(n):
        vi = embeddings[members[i].name]
        for j in range(i + 1, n):
            if _cosine(vi, embeddings[members[j].name]) >= threshold:
                union(i, j)

    groups: dict[int, list[NameStat]] = {}
    for idx, member in enumerate(members):
        groups.setdefault(find(idx), []).append(member)
    return list(groups.values())


def propose_name_merges(
    db: DatabaseManager,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    embed_fn: EmbedFn | None = None,
    examples: int = _DEFAULT_EXAMPLES,
) -> list[MergeCluster]:
    """Cluster near-duplicate comparable_names within each unit (read-only).

    Args:
        db: Database manager.
        threshold: Cosine similarity above which two names are clustered.
        embed_fn: Embedding function (defaults to the local nomic model). Tests
            inject a deterministic fake.
        examples: Example raw item names to attach per variant.

    Returns:
        Clusters with >= 2 distinct names, each with a deterministic canonical,
        sorted by unit then canonical for a stable, reviewable file.
    """
    embed = embed_fn or _default_embed_fn
    stats = _gather_name_stats(db, examples)
    if not stats:
        return []

    # Embed each distinct name once (names are unique per unit but may repeat
    # across units; the vector is identical, so cache by name).
    embeddings: dict[str, list[float]] = {}
    for s in stats:
        if s.name not in embeddings:
            embeddings[s.name] = embed(s.name)

    by_unit: dict[str, list[NameStat]] = {}
    for s in stats:
        by_unit.setdefault(s.unit, []).append(s)

    clusters: list[MergeCluster] = []
    for unit, members in by_unit.items():
        if len(members) < 2:
            continue
        for group in _cluster_unit_group(members, embeddings, threshold):
            if len(group) < 2:
                continue
            ordered = sorted(group, key=lambda s: (-s.count, s.name))
            clusters.append(
                MergeCluster(
                    canonical=_pick_canonical(group),
                    comparable_unit=unit,
                    members=ordered,
                )
            )

    clusters.sort(key=lambda c: (c.comparable_unit, c.canonical))
    return clusters


# ---------------------------------------------------------------------------
# Proposal file (YAML, human-edited)
# ---------------------------------------------------------------------------

_HEADER = """\
# Alibi F2 -- comparable_name merge proposals (REVIEW before applying).
#
# Each cluster groups comparable_names whose embeddings are near-duplicates
# WITHIN the same comparable_unit (EUR/L and EUR/pcs are never mixed). Nothing
# here is applied automatically.
#
# To MERGE a cluster: set `approved: true`. Every member `name` is then rewritten
# to `canonical` (edit `canonical` first if you prefer a different target word).
# To REJECT a cluster: leave `approved: false` or delete the whole entry.
#
# Then run:  lt enrich apply-name-merges --file {path}
# Only approved clusters are touched; a DB backup is taken first.
#
# generated: {generated}
# threshold: {threshold}
"""


def _cluster_to_dict(cluster: MergeCluster) -> dict[str, Any]:
    """Serialise a cluster to the ordered mapping written into the YAML file."""
    return {
        "canonical": cluster.canonical,
        "comparable_unit": cluster.comparable_unit,
        "approved": False,
        "members": [
            {
                "name": m.name,
                "count": m.count,
                "examples": m.examples,
                "categories": m.categories,
            }
            for m in cluster.members
        ],
    }


def write_proposal_yaml(
    clusters: list[MergeCluster],
    path: Path,
    *,
    threshold: float,
    generated: str,
) -> None:
    """Write the review file: a comment header + one entry per cluster.

    ``generated`` is passed in (not read from the clock) so the output is
    deterministic and testable.
    """
    body = yaml.safe_dump(
        {"clusters": [_cluster_to_dict(c) for c in clusters]},
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    header = _HEADER.format(path=path, generated=generated, threshold=threshold)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + body, encoding="utf-8")


def load_approved_clusters(path: Path) -> list[MergeCluster]:
    """Load a review file and return only the clusters marked ``approved: true``.

    Raises:
        ValueError: If the file is malformed (so a typo never silently applies
            nothing or, worse, the wrong thing).
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return []
    if not isinstance(data, dict) or "clusters" not in data:
        raise ValueError("Proposal file must be a mapping with a 'clusters' key.")
    raw_clusters = data["clusters"] or []
    if not isinstance(raw_clusters, list):
        raise ValueError("'clusters' must be a list.")

    approved: list[MergeCluster] = []
    for i, raw in enumerate(raw_clusters):
        if not isinstance(raw, dict):
            raise ValueError(f"Cluster #{i} is not a mapping.")
        if raw.get("approved") is not True:
            continue
        canonical = raw.get("canonical")
        unit = raw.get("comparable_unit")
        members_raw = raw.get("members")
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError(f"Cluster #{i}: 'canonical' must be a non-empty string.")
        if not isinstance(unit, str):
            raise ValueError(f"Cluster #{i}: 'comparable_unit' must be a string.")
        if not isinstance(members_raw, list) or not members_raw:
            raise ValueError(f"Cluster #{i}: 'members' must be a non-empty list.")
        members: list[NameStat] = []
        for m in members_raw:
            if not isinstance(m, dict) or not isinstance(m.get("name"), str):
                raise ValueError(f"Cluster #{i}: each member needs a string 'name'.")
            members.append(
                NameStat(
                    name=m["name"],
                    unit=unit,
                    count=int(m.get("count", 0) or 0),
                )
            )
        approved.append(
            MergeCluster(
                canonical=canonical.strip(),
                comparable_unit=unit,
                members=members,
            )
        )
    return approved


# ---------------------------------------------------------------------------
# Phase 2: apply
# ---------------------------------------------------------------------------


def apply_name_merges(db: DatabaseManager, clusters: list[MergeCluster]) -> ApplyResult:
    """Rewrite each approved cluster's members onto its canonical, then rebuild.

    For every member name != canonical, all matching fact_items (same name AND
    same unit) get ``comparable_name = canonical``. Matching on unit too keeps
    the group-by-unit invariant even though the human edited the file. The
    ``item_stars`` analytics mirror is rebuilt at the end so the surfaces never
    read stale buckets. Caller is responsible for backing up the DB first.
    """
    from alibi.services.item_stars import rebuild_item_stars

    rewrites: list[RewriteRecord] = []
    for cluster in clusters:
        for member in cluster.members:
            if member.name == cluster.canonical:
                continue
            with db.transaction() as cur:
                cur.execute(
                    "UPDATE fact_items SET comparable_name = ? "
                    "WHERE comparable_name = ? "
                    "AND COALESCE(comparable_unit, '') = ?",
                    (cluster.canonical, member.name, cluster.comparable_unit),
                )
                rows = cur.rowcount
            if rows:
                rewrites.append(
                    RewriteRecord(
                        old_name=member.name,
                        new_name=cluster.canonical,
                        comparable_unit=cluster.comparable_unit,
                        rows=rows,
                    )
                )
                logger.info(
                    "Merged %r -> %r (unit=%r, %d rows)",
                    member.name,
                    cluster.canonical,
                    cluster.comparable_unit,
                    rows,
                )

    rebuilt = rebuild_item_stars(db)
    return ApplyResult(rewrites=rewrites, rebuilt_stars=rebuilt)
