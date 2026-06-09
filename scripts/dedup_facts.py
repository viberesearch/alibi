#!/usr/bin/env python3
"""Detect and resolve duplicate facts (same transaction ingested more than once).

A transaction can be ingested twice (an archive copy plus a later reprocess, or
two photos of one receipt). Cloud formation does not always merge them — most
often because OCR variance in the vendor registration ID keeps the twin out of
the candidate set — leaving two facts for one transaction.

This is a thin CLI over :func:`alibi.services.deduplicate_facts`; the detection
and safety logic live in :mod:`alibi.clouds.dedup`. Equivalent to ``lt facts
dedup``.

Safety: a candidate pair is auto-resolved (the redundant, poorer twin deleted,
the richest twin kept) ONLY when an independent signal confirms they are the
same physical transaction — a zero-item twin, matching document perceptual
hashes, or overlapping item *prices* (robust to OCR name garbling). Grouping on
extracted ``(vendor, date, total)`` alone is NOT sufficient: a wrong extraction
can collide on that signature and falsely merge two real receipts (this once
deleted a genuine receipt). Anything that does not clear the gate is reported as
REVIEW and left untouched.

Dry-run by default. Pass --apply to mutate. Back up the DB first.

Usage:
    uv run python scripts/dedup_facts.py            # dry-run, prints plan
    uv run python scripts/dedup_facts.py --apply    # delete redundant twins
"""

from __future__ import annotations

import argparse

from alibi.db.connection import get_db
from alibi.services import deduplicate_facts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Delete redundant twins (default: dry-run)",
    )
    args = ap.parse_args()

    db = get_db()
    report = deduplicate_facts(db, apply=args.apply)

    for action in report.resolved:
        k, r = action.keeper, action.redundant
        verb = "RESOLVED" if args.apply else "WOULD RESOLVE"
        label = (
            f"{(k.vendor or '?')[:24]:24} {k.event_date} {k.total_amount}{k.currency}"
        )
        print(f"{verb}  {label}")
        print(f"    KEEP   fact={k.fact_id[:8]} items={k.n_items}")
        print(f"    DROP   fact={r.fact_id[:8]} items={r.n_items}  ({action.reason})")

    for action in report.review:
        k, r = action.keeper, action.redundant
        label = (
            f"{(k.vendor or '?')[:24]:24} {k.event_date} {k.total_amount}{k.currency}"
        )
        print(f"REVIEW    {label}")
        print(
            f"    {k.fact_id[:8]} (items={k.n_items}) ~ "
            f"{r.fact_id[:8]} (items={r.n_items})  ({action.reason})"
        )

    verb = "Resolved" if args.apply else "Would resolve"
    print(
        f"\n{verb} {report.resolved_count} duplicate(s); "
        f"{report.review_count} flagged for review."
    )
    if not args.apply and report.resolved_count:
        print("Dry-run only. Re-run with --apply to mutate (back up the DB first).")


if __name__ == "__main__":
    main()
