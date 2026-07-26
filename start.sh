#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

HOST="${PROXY_HOST:-127.0.0.1}"
PORT="${PROXY_PORT:-8082}"

exec uvicorn main:app --host "$HOST" --port "$PORT"
