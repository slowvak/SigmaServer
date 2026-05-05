#!/usr/bin/env bash
# restart.sh — Kill any existing SigmaServer process and restart it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8050

# Kill anything on the port
pids=$(lsof -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true)
if [ -n "$pids" ]; then
  echo "$pids" | xargs kill 2>/dev/null || true
  sleep 1
fi

echo "Starting SigmaServer on port $PORT …"
cd "$SCRIPT_DIR"
"/Users/bje/repos/Sigma/server/.venv/bin/python" -u server.py --port "$PORT" &
echo "SigmaServer started (PID $!)"
