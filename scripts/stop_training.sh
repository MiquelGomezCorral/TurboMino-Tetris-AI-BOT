#!/bin/bash
# Stop a running PPO training session.
# Usage:  bash scripts/stop_training.sh <config_name>
# Example: bash scripts/stop_training.sh config_curr_ppo.yaml

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_NAME="${1:-}"
if [ -z "$CONFIG_NAME" ]; then
    echo "ERROR: pass config name, e.g.: bash scripts/stop_training.sh config_curr_ppo.yaml"
    exit 1
fi

PIDFILE="$REPO_ROOT/logs/$CONFIG_NAME.pid"

if [ ! -f "$PIDFILE" ]; then
    echo "No PID file found at $PIDFILE — is training running?"
    exit 1
fi

PID=$(cat "$PIDFILE")

if ! kill -0 "$PID" 2>/dev/null; then
    echo "PID $PID is not running (stale PID file). Removing."
    rm -f "$PIDFILE"
    exit 0
fi

echo "Killing training process (PID $PID)..."
kill "$PID"

# wait a moment for graceful shutdown
sleep 1
if kill -0 "$PID" 2>/dev/null; then
    echo "Process did not exit, force-killing..."
    kill -9 "$PID" 2>/dev/null || true
fi

rm -f "$PIDFILE"
echo "Training stopped."
