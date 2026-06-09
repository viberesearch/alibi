#!/usr/bin/env python3
"""Backfill comparable_unit_price on fact_items that are missing it.

comparable_unit_price (normalized EUR/L, EUR/kg, EUR/pcs) is computed during
extraction from total_price / quantity / unit / unit_quantity. Surgical fact
edits and older ingests can leave it NULL even though the raw inputs are
present, which blocks cross-vendor item analytics.

This recomputes it for every fact_item with a NULL comparable_unit_price and the
inputs needed, reusing the canonical formula in
``alibi.atoms.parser._calculate_comparable_price`` (no second implementation to
drift). Items without the inputs are left untouched and reported.

Dry-run by default. Pass --apply to write. Back up the DB first.

Usage:
    uv run python scripts/backfill_comparable_price.py            # dry-run
    uv run python scripts/backfill_comparable_price.py --apply    # write
"""

from __future__ import annotations

import argparse

from alibi.atoms.parser import _calculate_comparable_price
from alibi.db.connection import get_db


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply", action="store_true", help="Write changes (default: dry-run)"
    )
    args = ap.parse_args()

    db = get_db()
    rows = db.fetchall(
        "SELECT id, total_price, quantity, unit, unit_quantity "
        "FROM fact_items WHERE comparable_unit_price IS NULL",
        (),
    )

    computed = 0
    skipped = 0
    for row in rows:
        data: dict[str, object] = {}
        if row["total_price"] is not None:
            data["total_price"] = str(row["total_price"])
        if row["quantity"] is not None:
            data["quantity"] = str(row["quantity"])
        if row["unit"] is not None:
            data["unit"] = row["unit"]
        if row["unit_quantity"] is not None:
            data["unit_quantity"] = str(row["unit_quantity"])

        _calculate_comparable_price(data)
        price = data.get("comparable_unit_price")
        unit = data.get("comparable_unit")
        if price is None or unit is None:
            skipped += 1
            continue

        computed += 1
        if args.apply:
            with db.transaction() as cur:
                cur.execute(
                    "UPDATE fact_items SET comparable_unit_price = ?, "
                    "comparable_unit = ? WHERE id = ?",
                    (str(price), str(unit), row["id"]),
                )

    verb = "Updated" if args.apply else "Would update"
    print(
        f"{len(rows)} fact_item(s) missing comparable_unit_price.\n"
        f"{verb} {computed}; {skipped} left NULL (insufficient inputs)."
    )
    if not args.apply and computed:
        print("Dry-run only. Re-run with --apply to write (back up the DB first).")


if __name__ == "__main__":
    main()
