"""Backfill the `country` (jurisdiction) field on existing facts.

Facts created before migration 036 have `country = NULL`. This script
re-runs jurisdiction inference (the same `apply_jurisdiction` used by the live
pipeline) against each fact's source-document extraction, then writes back the
inferred country and the canonical currency.

Non-destructive: only touches `facts.country` (always) and `facts.currency`
(only when inference produces a different, confident value). Idempotent.

Usage:
    uv run python scripts/backfill_jurisdiction.py --dry-run
    uv run python scripts/backfill_jurisdiction.py            # apply
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from alibi.config import Config
from alibi.db.connection import DatabaseManager
from alibi.normalizers.jurisdiction import _has_try_cue, _haystack, apply_jurisdiction

_TEXT_FIELDS = (
    "vendor",
    "vendor_address",
    "vendor_legal_name",
    "vendor_vat",
    "raw_text",
)


def _doc_extraction_for_fact(db: DatabaseManager, fact_id: str) -> dict | None:
    """Merge the extractions of ALL documents feeding this fact's cloud.

    A cloud can hold several documents (e.g. receipt + card slip); only one may
    carry the address. Concatenating their text fields means jurisdiction
    inference sees every available signal instead of an arbitrary single doc.
    """
    rows = db.fetchall(
        "SELECT d.raw_extraction AS rx "
        "FROM documents d "
        "JOIN bundles b ON b.document_id = d.id "
        "JOIN facts f ON f.cloud_id = b.cloud_id "
        "WHERE f.id = ?",
        (fact_id,),
    )
    merged: dict[str, str] = {}
    found = False
    for r in rows:
        if not r["rx"]:
            continue
        try:
            data = json.loads(r["rx"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        found = True
        for k in _TEXT_FIELDS:
            if data.get(k):
                merged[k] = (merged.get(k, "") + " " + str(data[k])).strip()
        if "currency" not in merged and data.get("currency"):
            merged["currency"] = str(data["currency"])
    return merged if found else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/alibi.db")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Re-derive jurisdiction for EVERY fact (fixes prior misdetections), "
        "not just rows where country IS NULL.",
    )
    args = ap.parse_args()

    db = DatabaseManager(Config(db_path=args.db))
    where = "" if args.all else "WHERE country IS NULL"
    facts = db.fetchall(f"SELECT id, vendor, currency, country FROM facts {where}")
    print(f"Facts to (re)derive: {len(facts)}")

    by_country: Counter[str] = Counter()
    cur_changes = 0
    updated = 0
    unresolved = 0
    writes: list[tuple] = []

    for f in facts:
        ext = _doc_extraction_for_fact(db, f["id"])
        if not ext:
            unresolved += 1
            continue
        # Seed with the fact's stored currency so resolve_currency can compare.
        ext.setdefault("currency", f["currency"])
        apply_jurisdiction(ext)  # no default_country: only set on real signal
        country = ext.get("country")
        currency = ext.get("currency")
        # De-corruption guard: a TRY value with neither a Turkish jurisdiction
        # nor a real lira cue is a leftover from the earlier "turkey"/"TL"
        # mis-tagging on a badly-OCR'd doc -> reset to EUR (corpus default).
        if (
            currency == "TRY"
            and country not in ("TR", "CY-NORTH")
            and not _has_try_cue(_haystack(ext))
        ):
            currency = "EUR"
        new_cur = currency if currency and currency != f["currency"] else f["currency"]

        if not country and not args.all:
            unresolved += 1
            continue  # default mode: leave untouched when no signal

        if country:
            by_country[country] += 1
        else:
            unresolved += 1
        if new_cur != f["currency"]:
            cur_changes += 1
        if country != f["country"] or new_cur != f["currency"]:
            print(
                f"  {f['vendor']}: "
                f"{f['country']}/{f['currency']} -> {country}/{new_cur}"
            )
        writes.append((country, new_cur, f["id"]))
        updated += 1

    if not args.dry_run and writes:
        with db.transaction() as cur:
            cur.executemany(
                "UPDATE facts SET country = ?, currency = ? WHERE id = ?",
                writes,
            )
    db.close()
    print("\n--- summary ---")
    print(f"resolved/updated: {updated}  | unresolved (no signal): {unresolved}")
    print(f"currency corrections: {cur_changes}")
    print("by country:", dict(by_country))
    if args.dry_run:
        print("(dry-run: no writes)")


if __name__ == "__main__":
    main()
