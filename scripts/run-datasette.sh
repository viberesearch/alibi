#!/usr/bin/env bash
# Launch the Alibi Datasette record explorer on the host (read-only snapshot).
#
# Binds 127.0.0.1 by default: Datasette has no auth and this is the full
# financial DB, so it is NOT exposed on the LAN. Reach it from another machine
# via an SSH tunnel (`ssh -L 8001:localhost:8001 <host>`) or Tailscale. To expose
# it deliberately, set ALIBI_DATASETTE_HOST=0.0.0.0 and put auth in front.
#
# Datasette + plugins run via `uv tool run` (no project-dependency coupling).
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${ALIBI_DATASETTE_HOST:-127.0.0.1}"
PORT="${ALIBI_DATASETTE_PORT:-8001}"
DB="data/explore/alibi-explore.db"

# Ensure a snapshot exists before serving.
[ -f "$DB" ] || scripts/datasette_snapshot.sh

exec uv tool run --with datasette-vega datasette -- \
  --immutable "$DB" \
  --metadata datasette/metadata.yaml \
  --host "$HOST" \
  --port "$PORT" \
  --setting sql_time_limit_ms 8000 \
  --setting default_page_size 50 \
  --setting max_returned_rows 2000
