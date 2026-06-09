"""Process a directory of independent receipt photos ONE FILE AT A TIME.

`lt process --path <folder>` groups a whole folder into a single multi-page
document. For a corpus of independent receipts we want one document per file,
so this loops process_file() over each image in a single warm process (models
stay loaded, no per-file CLI startup cost). Sequential -> Ollama is never asked
to run two models at once.

Prints one concise line per file and a final summary. Idempotent: already
ingested files are reported as duplicates and skipped cheaply.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from alibi.db.connection import DatabaseManager
from alibi.config import Config
from alibi.processing.watcher import is_supported_file
from alibi.services.ingestion import process_file

CORPUS = Path(sys.argv[1] if len(sys.argv) > 1 else "/path/to/corpus/2026-06-04")


def main() -> None:
    db = DatabaseManager(Config(db_path="data/alibi.db"))
    if not db.is_initialized():
        db.initialize()

    files = sorted(f for f in CORPUS.iterdir() if f.is_file() and is_supported_file(f))
    print(f"corpus: {len(files)} files in {CORPUS}", flush=True)

    ok = dup = err = 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        ts = time.time()
        try:
            r = process_file(db, f)
            dt = time.time() - ts
            if getattr(r, "is_duplicate", False):
                dup += 1
                tag = "DUP "
            elif getattr(r, "success", False):
                ok += 1
                tag = "OK  "
            else:
                err += 1
                tag = "ERR "
            ext = getattr(r, "extracted_data", None) or {}
            vendor = str(ext.get("vendor") or "")[:28]
            cur = ext.get("currency") or ""
            country = ext.get("country") or ""
            conf = (ext.get("_meta") or {}).get("confidence")
            nitems = len(ext.get("line_items") or [])
            conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "  - "
            print(
                f"[{i:3}/{len(files)}] {tag} {f.name:18} {dt:5.1f}s "
                f"conf={conf_s} items={nitems:2} {cur:3} {str(country):7} {vendor}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 — keep the batch going
            err += 1
            print(f"[{i:3}/{len(files)}] ERR {f.name:18} -> {e!r}", flush=True)

    db.close()
    el = time.time() - t0
    print(
        f"\nDONE in {el/60:.1f} min | ok={ok} dup={dup} err={err} "
        f"avg={el / max(len(files), 1):.1f}s/file",
        flush=True,
    )


if __name__ == "__main__":
    main()
