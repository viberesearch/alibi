#!/usr/bin/env bash
# Refresh the Datasette snapshot and reload the explorer so it serves the new
# data. The service runs with --immutable (which holds the snapshot file open
# for the life of the process), so a fresh snapshot is only picked up after a
# restart -- hence the kickstart. Driven by the launchd refresh timer.
set -euo pipefail

cd "$(dirname "$0")/.."

scripts/datasette_snapshot.sh

# Reload the running explorer if it is loaded (no-op otherwise).
launchctl kickstart -k "gui/$(id -u)/com.alibi.datasette" 2>/dev/null \
  || echo "datasette service not loaded; snapshot refreshed only"
