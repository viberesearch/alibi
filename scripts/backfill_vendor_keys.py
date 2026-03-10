#!/usr/bin/env python3
"""One-time backfill: compute vendor_key for all facts where it is NULL.

For each fact without a vendor_key:
1. Look up its vendor atom (registration ID, vendor name)
2. Compute make_vendor_key(registration, vendor_name)
3. Update the fact row

Usage:
    uv run python scripts/backfill_vendor_keys.py          # dry run
    uv run python scripts/backfill_vendor_keys.py --apply  # apply changes
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alibi.db.connection import DatabaseManager
from alibi.db import v2_store
from alibi.extraction.historical import make_vendor_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill vendor_key on facts")
    parser.add_argument(
        "--apply", action="store_true", help="Apply changes (default is dry run)"
    )
    args = parser.parse_args()

    db = DatabaseManager()
    db.initialize()

    # Find all facts with NULL vendor_key
    rows = db.fetchall("SELECT id, vendor FROM facts WHERE vendor_key IS NULL")

    if not rows:
        print("No facts with NULL vendor_key found. Nothing to do.")
        return

    print(f"Found {len(rows)} facts with NULL vendor_key")

    updated = 0
    skipped = 0

    for row in rows:
        fact_id = row["id"]
        vendor_name = row["vendor"]

        # Look up vendor atom for registration ID
        vendor_atom = v2_store.get_fact_vendor_atom(db, fact_id)
        registration = None
        if vendor_atom:
            registration = vendor_atom.get("registration")

        # Compute vendor_key
        key = make_vendor_key(registration, vendor_name)
        if key is None:
            print(f"  SKIP {fact_id}: no vendor name or registration")
            skipped += 1
            continue

        source = "registration" if registration else "name-hash"
        print(f"  {fact_id}: {vendor_name!r} -> {key} ({source})")

        if args.apply:
            db.execute(
                "UPDATE facts SET vendor_key = ? WHERE id = ?",
                (key, fact_id),
            )
            updated += 1

    if args.apply:
        db.get_connection().commit()
        print(f"\nDone: {updated} updated, {skipped} skipped")
    else:
        print(
            f"\nDry run: {updated + len(rows) - skipped} would be updated, {skipped} skipped"
        )
        print("Run with --apply to apply changes")


if __name__ == "__main__":
    main()
