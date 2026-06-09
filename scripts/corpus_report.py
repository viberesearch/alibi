"""Triage report for the 2026-06-04 corpus: counts, confidence, latency, and
the flagged-for-review list (verify confidence < 0.7 or required-field gaps)."""

from __future__ import annotations

import glob
import statistics as st

import yaml

CORPUS_TAG = "2026-06-04"
REQUIRED = ("vendor", "date", "total", "currency")


def _is_corpus(meta: dict) -> bool:
    return CORPUS_TAG in str(meta.get("source") or "")


def main() -> None:
    docs = []
    for path in glob.glob("data/yaml_store/**/*.alibi.yaml", recursive=True):
        try:
            d = yaml.safe_load(open(path))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        meta = d.get("_meta") or {}
        if not _is_corpus(meta):
            continue
        items = d.get("line_items") or []
        gaps = [f for f in REQUIRED if not d.get(f)]
        # payment/atm slips legitimately have no line items
        dt = d.get("document_type") or "?"
        docs.append(
            {
                "name": path.split("/")[-1].replace(".alibi.yaml", ""),
                "type": dt,
                "conf": meta.get("confidence"),
                "pipeline": meta.get("pipeline"),
                "n_items": len(items),
                "currency": d.get("currency"),
                "country": d.get("country"),
                "gaps": gaps,
                "timing": meta.get("timing") or {},
                "vendor": str(d.get("vendor") or "")[:30],
            }
        )

    print(f"CORPUS DOCS (yaml): {len(docs)}")
    by_type: dict[str, int] = {}
    for d in docs:
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
    print("by type:", by_type)

    receipts = [d for d in docs if d["type"] in ("receipt", "invoice")]
    confs = [d["conf"] for d in docs if isinstance(d["conf"], (int, float))]
    ritems = [d["n_items"] for d in receipts]
    if confs:
        print(
            f"confidence: mean={st.mean(confs):.3f} median={st.median(confs):.3f} "
            f"min={min(confs):.2f} max={max(confs):.2f}"
        )
    if ritems:
        print(
            f"items/receipt: mean={st.mean(ritems):.1f} "
            f"median={st.median(ritems)} max={max(ritems)} "
            f"(receipts={len(receipts)})"
        )

    def conf_lt(d, t):
        return isinstance(d["conf"], (int, float)) and d["conf"] < t

    # User's literal threshold.
    sub07 = [d for d in docs if conf_lt(d, 0.7) or d["gaps"]]
    print(f"\nverify-confidence < 0.7 OR field-gap: {len(sub07)} / {len(docs)}")

    # --- needs_review facts (cross-validation item-sum vs total mismatch) ---
    from alibi.config import Config
    from alibi.db.connection import DatabaseManager

    db = DatabaseManager(Config(db_path="data/alibi.db"))
    nr = db.fetchall(
        "SELECT vendor, currency, country FROM facts WHERE status='needs_review'"
    )
    db.close()
    print(f"cross-validation needs_review facts: {len(nr)}")

    # --- High-priority tier: conf<0.5 OR required-field gap OR truncated ---
    truncated = {"IMG_0859", "IMG_0987", "IMG_1057", "IMG_1058"}
    tierA = [d for d in docs if conf_lt(d, 0.5) or d["gaps"] or d["name"] in truncated]
    print(
        f"\n=== HIGH-PRIORITY REVIEW (conf<0.5, field-gap, or truncated): "
        f"{len(tierA)} ==="
    )
    for d in sorted(tierA, key=lambda x: (x["conf"] if x["conf"] else 0)):
        c = f"{d['conf']:.2f}" if isinstance(d["conf"], (int, float)) else " -  "
        trunc = " [TRUNCATED]" if d["name"] in truncated else ""
        print(
            f"  {d['name']:12} {d['type']:13} conf={c} items={d['n_items']:2} "
            f"{str(d['currency']):4} {str(d['country']):8} "
            f"gaps={','.join(d['gaps']) or '-':14} {d['vendor']}{trunc}"
        )

    # field-gap breakdown
    gapcount: dict[str, int] = {}
    for d in docs:
        for g in d["gaps"]:
            gapcount[g] = gapcount.get(g, 0) + 1
    print("\nrequired-field gaps:", gapcount)

    # biggest receipts
    big = sorted(receipts, key=lambda x: -x["n_items"])[:6]
    print("largest receipts:", [(d["name"], d["n_items"], d["vendor"]) for d in big])


if __name__ == "__main__":
    main()
