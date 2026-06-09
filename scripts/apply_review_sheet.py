"""Apply a reviewed CSV sheet back into the YAML store and the database.

Reads the sheet produced by make_review_sheet.py. For each row whose ``action``
is set:

  * ok     -> mark the fact confirmed (clears the needs_review flag), no edits
  * fix    -> write the non-empty ``fix_*`` values into the .alibi.yaml AND the
              linked fact, then mark it confirmed
  * delete -> remove the document and its dependent facts/items from the DB
  * blank  -> skip

Run with --dry-run first to preview. Non-destructive to anything you don't mark.

Usage:
    uv run python scripts/apply_review_sheet.py review/2026-06-04_review.csv --dry-run
    uv run python scripts/apply_review_sheet.py review/2026-06-04_review.csv
"""

from __future__ import annotations

import argparse
import csv
import glob

import yaml

from alibi.config import Config
from alibi.db.connection import DatabaseManager
from alibi.db import v2_store

# YAML field -> (yaml key, fact column, caster)
FIELD_MAP = {
    "fix_vendor": ("vendor", "vendor", str),
    "fix_date": ("date", "event_date", str),
    "fix_total": ("total", "total_amount", float),
    "fix_currency": ("currency", "currency", str),
    "fix_country": ("country", "country", str),
}


def _find_yaml(name: str) -> str | None:
    hits = glob.glob(f"data/yaml_store/**/{name}.alibi.yaml", recursive=True)
    return hits[0] if hits else None


def _doc_and_facts(db: DatabaseManager, file_hash: str):
    doc = db.fetchone("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
    facts = db.fetchall(
        "SELECT DISTINCT f.id AS id FROM facts f "
        "JOIN bundles b ON b.cloud_id = f.cloud_id "
        "JOIN documents d ON d.id = b.document_id "
        "WHERE d.file_hash = ?",
        (file_hash,),
    )
    return (doc["id"] if doc else None), [f["id"] for f in facts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = DatabaseManager(Config(db_path="data/alibi.db"))
    n_ok = n_fix = n_del = n_skip = 0

    with open(args.sheet, newline="") as fh:
        for row in csv.DictReader(fh):
            action = (row.get("action") or "").strip().lower()
            name = row["file"]
            # Implicit fix: a row with populated fix_* but no explicit action is
            # treated as a fix (so you can just fill the values and apply).
            has_fixes = any((row.get(c) or "").strip() for c in FIELD_MAP)
            if not action and has_fixes:
                action = "fix"
            if action not in ("ok", "fix", "delete"):
                n_skip += 1
                continue

            ypath = _find_yaml(name)
            if not ypath:
                print(f"  ! {name}: yaml not found, skipping")
                continue
            d = yaml.safe_load(open(ypath))
            file_hash = (d.get("_meta") or {}).get("file_hash")
            doc_id, fact_ids = (
                _doc_and_facts(db, file_hash) if file_hash else (None, [])
            )

            if action == "delete":
                print(f"  delete {name} (doc={str(doc_id)[:8]})")
                if not args.dry_run and doc_id:
                    v2_store.cleanup_document(db, doc_id)
                n_del += 1
                continue

            # collect field edits for 'fix'
            yaml_updates: dict = {}
            fact_updates: dict = {}
            if action == "fix":
                for col, (ykey, fcol, cast) in FIELD_MAP.items():
                    val = (row.get(col) or "").strip()
                    if not val:
                        continue
                    try:
                        casted = cast(val)
                    except ValueError:
                        print(
                            f"  ! {name}: bad value for {col}={val!r}, skipping field"
                        )
                        continue
                    yaml_updates[ykey] = casted
                    fact_updates[fcol] = casted

            tag = "fix " if action == "fix" else "ok  "
            print(f"  {tag} {name}: {yaml_updates or '(confirm only)'}")

            if not args.dry_run:
                # 1. YAML (the SSOT)
                if yaml_updates:
                    d.update(yaml_updates)
                    with open(ypath, "w") as out:
                        yaml.safe_dump(d, out, allow_unicode=True, sort_keys=False)
                # 2. DB facts
                with db.transaction() as cur:
                    for fid in fact_ids:
                        for fcol, val in fact_updates.items():
                            cur.execute(
                                f"UPDATE facts SET {fcol} = ? WHERE id = ?", (val, fid)
                            )
                        cur.execute(
                            "UPDATE facts SET status = 'confirmed' WHERE id = ?", (fid,)
                        )

            if action == "fix":
                n_fix += 1
            else:
                n_ok += 1

    db.close()
    verb = "would " if args.dry_run else ""
    print(f"\n{verb}applied: ok={n_ok} fix={n_fix} delete={n_del} (skipped {n_skip})")


if __name__ == "__main__":
    main()
