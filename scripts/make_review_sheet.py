"""Generate a CSV review sheet for documents that need manual verification.

Each row is a flagged document with its current values, the path to the original
photo (open it to check), and empty ``fix_*`` columns plus an ``action`` column
for you to fill in. Feed the edited sheet back with apply_review_sheet.py.

A document is flagged when ANY of:
  * its fact is cross-validation ``needs_review`` (item sum != total),
  * a required field (vendor/date/total/currency) is missing,
  * verify confidence < threshold (default 0.6; use --conf to widen),
  * the OCR was truncated (very long receipt).

Usage:
    uv run python scripts/make_review_sheet.py            # -> review/<tag>_review.csv
    uv run python scripts/make_review_sheet.py --conf 0.7 # widen the net
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import yaml

from alibi.config import Config
from alibi.db.connection import DatabaseManager

CORPUS_TAG = "2026-06-04"
IMAGE_DIR = f"/path/to/corpus/{CORPUS_TAG}"
TRUNCATED = {"IMG_0859", "IMG_0987", "IMG_1057", "IMG_1058"}
REQUIRED = ("vendor", "date", "total", "currency")

COLUMNS = [
    "file",
    "image",
    "type",
    "conf",
    "flags",
    "n_items",
    "vendor",
    "date",
    "total",
    "currency",
    "country",
    "fix_vendor",
    "fix_date",
    "fix_total",
    "fix_currency",
    "fix_country",
    "action",
    "notes",
]


def _needs_review_hashes(db: DatabaseManager) -> set[str]:
    rows = db.fetchall(
        "SELECT DISTINCT d.file_hash AS h "
        "FROM documents d "
        "JOIN bundles b ON b.document_id = d.id "
        "JOIN facts f ON f.cloud_id = b.cloud_id "
        "WHERE f.status = 'needs_review'"
    )
    return {r["h"] for r in rows if r["h"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--out", default=f"review/{CORPUS_TAG}_review.csv")
    args = ap.parse_args()

    db = DatabaseManager(Config(db_path="data/alibi.db"))
    nr_hashes = _needs_review_hashes(db)
    db.close()

    rows = []
    for path in glob.glob("data/yaml_store/**/*.alibi.yaml", recursive=True):
        try:
            d = yaml.safe_load(open(path))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        meta = d.get("_meta") or {}
        if CORPUS_TAG not in str(meta.get("source") or ""):
            continue
        name = os.path.basename(path).replace(".alibi.yaml", "")
        conf = meta.get("confidence")
        gaps = [f for f in REQUIRED if not d.get(f)]
        flags = []
        if meta.get("file_hash") in nr_hashes:
            flags.append("needs_review")
        if isinstance(conf, (int, float)) and conf < args.conf:
            flags.append(f"conf<{args.conf}")
        for g in gaps:
            flags.append(f"gap:{g}")
        if name in TRUNCATED:
            flags.append("truncated")
        if not flags:
            continue
        rows.append(
            {
                "file": name,
                "image": f"{IMAGE_DIR}/{name}.jpeg",
                "type": d.get("document_type") or "",
                "conf": f"{conf:.2f}" if isinstance(conf, (int, float)) else "",
                "flags": ";".join(flags),
                "n_items": len(d.get("line_items") or []),
                "vendor": d.get("vendor") or "",
                "date": d.get("date") or "",
                "total": d.get("total") if d.get("total") is not None else "",
                "currency": d.get("currency") or "",
                "country": d.get("country") or "",
                "fix_vendor": "",
                "fix_date": "",
                "fix_total": "",
                "fix_currency": "",
                "fix_country": "",
                "action": "",
                "notes": "",
            }
        )

    # Worst first: needs_review, then lowest confidence.
    rows.sort(
        key=lambda r: (
            0 if "needs_review" in r["flags"] else 1,
            float(r["conf"]) if r["conf"] else 1.0,
        )
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} flagged rows -> {args.out}")
    print("Fill in: fix_* columns (only the fields you change) and 'action'")
    print("  action = ok | fix | delete   (blank = skip for now)")


if __name__ == "__main__":
    main()
