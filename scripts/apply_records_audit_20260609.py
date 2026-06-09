"""Apply the 2026-06-09 records-audit corrections (buckets A/B/C/D).

Reads the structured audit findings (produced by the read-only audit workflow,
see docs/RECORDS_AUDIT_2026-06-09.md) and applies the user-approved high-
confidence corrections, routed through the services layer where possible:

  A. DELETE non-product lines (totals/tax/bank/discount/header fragments).
  B. comparable_name normalization:
       B1 (global, safe)   - lowercase every comparable_name (casing-merge).
       B2 (targeted)       - apply strict-clean parsed comparable_name for
                             high-confidence comparable_name / garbled_name
                             findings (garbled only fills a NULL/blank one).
  C. Null corrupt comparable_unit_price (+ comparable_unit) so they stop
     polluting analytics; the deterministic recompute pass can refill later.

Everything is conservative: any proposed value that is not a clean short
lowercase product phrase is SKIPPED and logged, never written. A timestamped DB
backup is taken before any write. The item_stars analytics mirror is rebuilt at
the end.

Usage:
    uv run python scripts/apply_records_audit_20260609.py [findings.json] [--dry-run]
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from alibi.config import get_config
from alibi.db import v2_store
from alibi.db.connection import DatabaseManager
from alibi.services.correction import update_fact_item
from alibi.services.item_stars import rebuild_item_stars

FINDINGS_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/audit_findings.json"
DRY_RUN = "--dry-run" in sys.argv

# A clean comparable_name: starts with a letter, only lowercase letters/digits/
# space/%/./hyphen, 2-40 chars, at most 4 words. Anything else is an explanation
# fragment and is skipped.
_CLEAN = re.compile(r"^[a-z][a-z0-9%.\- ]{1,38}$")
_CUTS = (" (", " — ", " - ", ":", ",", ";", " = ", "=")
_REJECT_PREFIX = ("review", "delete", "likely", "verify", "n/a", "unknown")


def clean_value(proposed: str) -> str | None:
    """Extract a clean comparable_name from a (possibly explanatory) proposal."""
    # Proposals shaped "GREEK = ΕΛΛ = english" put the real answer LAST; a
    # first-clause grab would write the transliteration. Too ambiguous to parse
    # safely -- skip the whole proposal.
    if "=" in proposed:
        return None
    v = proposed
    for cut in _CUTS:
        if cut in v:
            v = v.split(cut)[0]
    v = v.strip().strip("'\"").lower()
    if not v or v.startswith(_REJECT_PREFIX):
        return None
    if len(v.split()) > 4 or not _CLEAN.match(v):
        return None
    return v


def main() -> None:
    findings = json.load(open(FINDINGS_PATH))
    cfg = get_config()
    db = DatabaseManager(cfg)

    db_path = cfg.get_absolute_db_path()
    backup = db_path.with_suffix(db_path.suffix + ".bak_preauditapply")
    if not DRY_RUN:
        shutil.copy2(db_path, backup)
        print(f"[backup] {backup}")

    def hc(x: dict) -> bool:
        return x["confidence"] == "high"

    log: dict[str, list] = {
        "deleted": [],
        "cname_targeted": [],
        "price_nulled": [],
        "skipped": [],
    }

    # ---- A: delete high-confidence non-product lines -------------------------
    del_ids = sorted(
        {x["item_id"] for x in findings if x["issue_type"] == "not_a_product" and hc(x)}
    )
    del_set = set(del_ids)
    if not DRY_RUN and del_ids:
        n = v2_store.delete_fact_items(db, del_ids)
        log["deleted"] = del_ids
        print(f"[A] deleted {n} non-product fact_items")
    else:
        print(f"[A] would delete {len(del_ids)} non-product fact_items")

    # ---- C: null corrupt comparable prices -----------------------------------
    price_ids = sorted(
        {
            x["item_id"]
            for x in findings
            if x["issue_type"] == "comparable_price"
            and hc(x)
            and x["item_id"] not in del_set
        }
    )
    if not DRY_RUN and price_ids:
        with db.transaction() as cur:
            placeholders = ",".join("?" for _ in price_ids)
            cur.execute(
                "UPDATE fact_items SET comparable_unit_price = NULL, "
                f"comparable_unit = NULL WHERE id IN ({placeholders})",  # noqa: S608
                price_ids,
            )
        log["price_nulled"] = price_ids
    print(f"[C] nulled comparable price on {len(price_ids)} items")

    # ---- B2: targeted comparable_name fixes ----------------------------------
    cand = [
        x
        for x in findings
        if x["issue_type"] in ("comparable_name", "garbled_name")
        and hc(x)
        and x["item_id"] not in del_set
    ]
    for x in cand:
        val = clean_value(x["proposed_fix"])
        if val is None:
            log["skipped"].append(
                [x["item_id"], x["issue_type"], x["proposed_fix"][:60]]
            )
            continue
        # garbled_name proposals only fill a NULL/blank comparable_name (never
        # overwrite an already-clean one); comparable_name proposals always apply.
        if x["issue_type"] == "garbled_name":
            row = db.fetchone(
                "SELECT comparable_name FROM fact_items WHERE id = ?", (x["item_id"],)
            )
            if row is None or (row["comparable_name"] or "").strip():
                log["skipped"].append([x["item_id"], "garbled_has_cname", val])
                continue
        if not DRY_RUN:
            update_fact_item(db, x["item_id"], {"comparable_name": val})
        log["cname_targeted"].append([x["item_id"], val])
    print(
        f"[B2] set comparable_name on {len(log['cname_targeted'])} items "
        f"({len(log['skipped'])} skipped as unparseable/guarded)"
    )

    # ---- B1: global lowercase of comparable_name -----------------------------
    lowered = 0
    if not DRY_RUN:
        with db.transaction() as cur:
            cur.execute(
                "UPDATE fact_items SET comparable_name = lower(comparable_name) "
                "WHERE comparable_name IS NOT NULL "
                "AND comparable_name != lower(comparable_name)"
            )
            lowered = cur.rowcount
    print(f"[B1] lowercased {lowered} comparable_names")

    # ---- rebuild the analytics mirror ----------------------------------------
    if not DRY_RUN:
        stars = rebuild_item_stars(db)
        print(f"[stars] rebuilt item_stars: {stars} rows")

    log_path = Path("data/_audit_apply_log_20260609.json")
    log_path.write_text(json.dumps(log, indent=1))
    print(f"[log] {log_path}")


if __name__ == "__main__":
    main()
