#!/usr/bin/env python3
"""Head-to-head benchmark for local structuring models (OCR text -> schema JSON).

Built to evaluate a new candidate model (e.g. ``gemma4:12b`` when it lands on
Ollama) against the incumbent structuring model (``qwen3.5:9b``) on the exact
production path: schema-enforced output (Ollama ``format``) + reasoning disabled
(``think=false``). It reuses the production internals, so the numbers reflect
what the pipeline would actually do, not an approximation.

It also optionally tests SINGLE-PASS vision extraction (image -> schema JSON) for
a multimodal candidate, which is the real consolidation question for a unified
model: can one model replace the OCR + text-structure two-stage?

Metrics (objective, no human judgement):
  json_ok   parseable JSON conforming enough to score
  items     avg line items extracted
  fill      avg fraction of core line-item fields populated (non-null)
  verify    avg arithmetic-verification confidence (alibi verification module)
  tokens    avg output tokens (Ollama eval_count) -- efficiency
  latency   avg seconds per document -- efficiency

Go/no-go: a candidate replaces the incumbent only if it is better on quality
(items/fill/verify) OR equal-quality and cheaper (tokens/latency).

USAGE
  # Compare incumbent vs candidate on the text-structuring path:
  uv run python scripts/benchmark_structure_model.py \
      --models qwen3.5:9b,gemma4:12b \
      --images /path/to/images --limit 10

  # Also test the candidate as a single-pass vision extractor:
  uv run python scripts/benchmark_structure_model.py \
      --models qwen3.5:9b,gemma4:12b --vision gemma4:12b \
      --images /path/to/images --limit 10

WHEN gemma4:12b LANDS ON OLLAMA
  1. Confirm the tag is live:
       curl -s -o /dev/null -w '%{http_code}' \
         https://registry.ollama.ai/v2/library/gemma4/manifests/12b
     (404 = not yet; 200/412 = published)
  2. Pull it:  ollama pull gemma4:12b
  3. Fire:     uv run python scripts/benchmark_structure_model.py \
                 --models qwen3.5:9b,gemma4:12b --vision gemma4:12b \
                 --images <batch dir> --limit 12
  Models not present locally are skipped with a notice (so the script is safe to
  run before the candidate exists -- it just benchmarks what is installed).
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

from alibi.config import get_config  # noqa: E402
from alibi.extraction.ocr import _prepare_image_for_ocr, ocr_image  # noqa: E402
from alibi.extraction.prompts import get_text_extraction_prompt  # noqa: E402
from alibi.extraction.structurer import (  # noqa: E402
    _call_ollama_text,
    get_extraction_json_schema,
)
from alibi.extraction.schemas import validate_extraction  # noqa: E402
from alibi.extraction.verification import verify_extraction  # noqa: E402
from alibi.extraction.vision import extract_json_from_response  # noqa: E402

IMAGE_EXTS = (".jpeg", ".jpg", ".png", ".webp", ".heic")
CORE_ITEM_FIELDS = [
    "name",
    "quantity",
    "unit_price",
    "total_price",
    "tax_rate",
    "unit_raw",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def installed_models(ollama_url: str) -> set[str]:
    """Return the set of model names present in the local Ollama."""
    try:
        r = httpx.get(f"{ollama_url}/api/tags", timeout=10)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except Exception as e:  # pragma: no cover - diagnostic
        print(f"  ! could not list Ollama models: {e}")
        return set()


def model_present(model: str, installed: set[str]) -> bool:
    """True only if the EXACT tag is installed.

    A tagless name (``gemma4``) defaults to ``gemma4:latest``. The tag must
    match exactly otherwise: ``gemma4:12b`` is NOT satisfied by an installed
    ``gemma4:e4b`` — that is the whole point of the availability guard.
    """
    if model in installed:
        return True
    if ":" not in model:
        return f"{model}:latest" in installed
    return False


def fill_rate(data: dict[str, Any]) -> float:
    items = data.get("line_items") or []
    filled = total = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        for k in CORE_ITEM_FIELDS:
            total += 1
            if it.get(k) not in (None, "", []):
                filled += 1
    return filled / total if total else 0.0


def ocr_cached(img: Path, cache_dir: Path, ocr_model: str | None) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cf = cache_dir / (img.stem + ".txt")
    if cf.exists():
        return cf.read_text()
    text = ocr_image(img, model=ocr_model)
    cf.write_text(text)
    return text


def _score(data: dict[str, Any], doc_type: str) -> dict[str, Any]:
    return {
        "json_ok": True,
        "req_errors": len(validate_extraction(data, doc_type)),
        "n_items": len(data.get("line_items") or []),
        "fill": round(fill_rate(data), 3),
        "verify": verify_extraction(data, None).confidence,
    }


def run_text(
    model: str, ocr: str, doc_type: str, url: str, schema: dict[str, Any]
) -> dict[str, Any]:
    """Structure cached OCR text via the production Ollama path (think=false + schema)."""
    prompt = get_text_extraction_prompt(ocr, doc_type, version=2, mode="specialized")
    t0 = time.time()
    out: dict[str, Any] = {"json_ok": False}
    try:
        res = _call_ollama_text(url, model, prompt, 180.0, response_format=schema)
        text = re.sub(
            r"<think>[\s\S]*?</think>", "", res.get("response", "") or ""
        ).strip()
        out.update(_score(extract_json_from_response(text), doc_type))
        out["eval"] = res.get("eval_count")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["latency"] = round(time.time() - t0, 1)
    return out


def run_vision(
    model: str, img: Path, doc_type: str, url: str, schema: dict[str, Any]
) -> dict[str, Any]:
    """Single-pass image -> schema JSON via a multimodal model (think=false + format)."""
    cfg = get_config()
    prompt = get_text_extraction_prompt(
        "(see attached image)", doc_type, version=2, mode="specialized"
    )
    b64 = base64.b64encode(_prepare_image_for_ocr(img)).decode()
    body = {
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "keep_alive": cfg.ollama_keep_alive,
        "think": False,
        "format": schema,
        "options": {"temperature": 0.1, "num_predict": cfg.ollama_num_predict},
    }
    t0 = time.time()
    out: dict[str, Any] = {"json_ok": False}
    try:
        d = httpx.post(f"{url}/api/generate", json=body, timeout=240).json()
        text = re.sub(
            r"<think>[\s\S]*?</think>", "", d.get("response", "") or ""
        ).strip()
        out.update(_score(extract_json_from_response(text), doc_type))
        out["eval"] = d.get("eval_count")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["latency"] = round(time.time() - t0, 1)
    return out


def aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    sides = [r[key] for r in rows if key in r]
    oks = [s for s in sides if s.get("json_ok")]
    n = max(len(sides), 1)
    d = max(len(oks), 1)
    return {
        "json_ok_rate": round(sum(bool(s.get("json_ok")) for s in sides) / n, 3),
        "avg_items": round(sum(s.get("n_items", 0) for s in oks) / d, 2),
        "avg_fill": round(sum(s.get("fill", 0) for s in oks) / d, 3),
        "avg_verify": round(sum(s.get("verify", 0) for s in oks) / d, 3),
        "avg_eval": round(sum((s.get("eval") or 0) for s in oks) / d, 0),
        "avg_latency": round(sum(s.get("latency", 0) for s in sides) / n, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = get_config()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--models",
        default=f"{cfg.ollama_structure_model},gemma4:12b",
        help="comma-separated structuring models; first is the incumbent baseline",
    )
    ap.add_argument(
        "--vision",
        default=None,
        help="optional multimodal model for single-pass image->JSON",
    )
    ap.add_argument("--images", required=True, help="directory of document images")
    ap.add_argument("--doc-type", default="receipt")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument(
        "--ocr-model", default=None, help="OCR model (default: config.ollama_ocr_model)"
    )
    ap.add_argument(
        "--ocr-cache", default=str(Path.home() / ".cache" / "alibi-bench" / "ocr")
    )
    ap.add_argument(
        "--out", default=str(Path.home() / ".cache" / "alibi-bench" / "results.json")
    )
    args = ap.parse_args()

    url = cfg.ollama_url
    schema = get_extraction_json_schema(args.doc_type)
    cache_dir = Path(args.ocr_cache)
    installed = installed_models(url)

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    models = [m for m in requested if model_present(m, installed)]
    for m in requested:
        if m not in models:
            print(f"  - SKIP {m}: not installed (pull it, then re-run)")
    if args.vision and not model_present(args.vision, installed):
        print(f"  - SKIP vision {args.vision}: not installed")
        args.vision = None
    if not models and not args.vision:
        print("No requested models are installed. Nothing to benchmark.")
        return

    src = Path(args.images)
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)[
        : args.limit
    ]
    if not imgs:
        print(f"No images found in {src}")
        return
    print(f"Benchmarking on {len(imgs)} images from {src}")
    print(
        f"  text models: {models}"
        + (f" | vision: {args.vision}" if args.vision else "")
    )

    rows: list[dict[str, Any]] = []
    for img in imgs:
        ocr = ocr_cached(img, cache_dir, args.ocr_model)
        row: dict[str, Any] = {"img": img.name}
        for m in models:
            row[f"text:{m}"] = run_text(m, ocr, args.doc_type, url, schema)
        if args.vision:
            row[f"vision:{args.vision}"] = run_vision(
                args.vision, img, args.doc_type, url, schema
            )
        rows.append(row)
        print(f"\n{img.name}")
        for k in list(row.keys())[1:]:
            r = row[k]
            if r.get("json_ok"):
                print(
                    f"  {k:24s} items={r['n_items']} fill={r['fill']} verify={r['verify']} "
                    f"tok={r.get('eval', '-')} {r['latency']}s"
                )
            else:
                print(
                    f"  {k:24s} FAILED: {r.get('error', 'no json')} ({r['latency']}s)"
                )

    # Aggregate + verdict
    keys = [k for k in rows[0].keys() if k != "img"]
    summary = {k: aggregate(rows, k) for k in keys}
    print("\n=== AGGREGATE ===")
    for k, a in summary.items():
        print(
            f"{k:24s} jsonOK={a['json_ok_rate']} items={a['avg_items']} fill={a['avg_fill']} "
            f"verify={a['avg_verify']} tok={a['avg_eval']} lat={a['avg_latency']}s"
        )

    if len(models) >= 2:
        base = f"text:{models[0]}"
        b = summary[base]
        print(f"\n=== VERDICT (vs incumbent {models[0]}) ===")
        for m in models[1:]:
            c = summary[f"text:{m}"]
            quality_better = (
                c["avg_items"] >= b["avg_items"]
                and c["avg_fill"] >= b["avg_fill"]
                and c["avg_verify"] >= b["avg_verify"]
                and c["json_ok_rate"] >= b["json_ok_rate"]
            )
            cheaper = (
                c["avg_eval"] <= b["avg_eval"] and c["avg_latency"] <= b["avg_latency"]
            )
            quality_worse = (
                c["avg_verify"] < b["avg_verify"]
                or c["json_ok_rate"] < b["json_ok_rate"]
            )
            if quality_better and cheaper:
                v = "REPLACE: better/equal quality AND cheaper"
            elif quality_better:
                v = "REPLACE candidate: better/equal quality (costs more -- weigh it)"
            elif not quality_worse and cheaper:
                v = "CONSIDER: equal-ish quality and cheaper"
            else:
                v = "KEEP incumbent: candidate not clearly better"
            print(f"  {m}: {v}")
            print(
                f"     items {b['avg_items']}->{c['avg_items']} | fill {b['avg_fill']}->{c['avg_fill']} | "
                f"verify {b['avg_verify']}->{c['avg_verify']} | tok {b['avg_eval']}->{c['avg_eval']} | "
                f"lat {b['avg_latency']}->{c['avg_latency']}s"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
