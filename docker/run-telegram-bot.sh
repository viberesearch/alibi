#!/usr/bin/env bash
# Build and run the thin Telegram bot container on the host (OrbStack).
#
# Prereqs on the host:
#   1. The Alibi API is running:  uv run lt serve   (listens on :3100)
#   2. A bot token + allowlist are exported or present in .env:
#        TELEGRAM_BOT_TOKEN=...
#        ALIBI_TELEGRAM_ALLOWED_USERS=<telegram_id>[,<telegram_id>...]
#   3. Each Telegram user has linked an API key once via /link <mnemonic>
#      (mint one with: uv run lt users keys create <user_id>).
#
# Single instance only: do NOT also run `lt telegram start` on the host while
# this container is up -- two pollers on one token cause Telegram 409 Conflict.
set -euo pipefail

cd "$(dirname "$0")/.."

COMPOSE_FILE="docker-compose.telegram.yml"
cmd="${1:-up}"

case "$cmd" in
  up)
    echo "Building and starting the thin Telegram bot..."
    docker compose -f "$COMPOSE_FILE" up -d --build
    echo "Started. Tail logs with: $0 logs"
    ;;
  down)
    docker compose -f "$COMPOSE_FILE" down
    ;;
  logs)
    docker compose -f "$COMPOSE_FILE" logs -f
    ;;
  restart)
    docker compose -f "$COMPOSE_FILE" restart
    ;;
  *)
    echo "Usage: $0 {up|down|logs|restart}" >&2
    exit 1
    ;;
esac
