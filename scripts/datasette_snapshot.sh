#!/usr/bin/env bash
# Produce a read-only snapshot of the live Alibi DB for the Datasette explorer.
#
# Why a snapshot and not the live file: the API owns alibi.db as a single WAL
# writer. Pointing Datasette (especially with --immutable) at the live file
# would either contend with that writer or serve a torn read. `sqlite3 .backup`
# is an online, WAL-safe copy; we write it to a temp file and atomically rename
# it into place so the explorer always opens a complete, consistent database.
#
# Run periodically (see the launchd refresh job) to keep the explorer fresh.
set -euo pipefail

cd "$(dirname "$0")/.."

SRC="${ALIBI_DB_PATH:-data/alibi.db}"
DEST_DIR="data/explore"
DEST="$DEST_DIR/alibi-explore.db"

if [ ! -f "$SRC" ]; then
  echo "Source DB not found: $SRC" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
TMP="$(mktemp "$DEST_DIR/.snapshot.XXXXXX")"
# Online backup (safe while the API holds the WAL); then atomic swap.
sqlite3 "$SRC" ".backup '$TMP'"
mv -f "$TMP" "$DEST"
chmod 0644 "$DEST"

echo "Snapshot written: $DEST ($(du -h "$DEST" | cut -f1))"
