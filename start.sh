#!/bin/bash
set -euo pipefail
export $(grep -v '^#' .env | grep -v '^\s*$' | xargs)
uvicorn main:app --host 127.0.0.1 --port 8082
