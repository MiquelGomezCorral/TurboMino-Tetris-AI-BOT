#!/bin/bash
# Launch PPO training in background via nohup.
# Usage:  bash scripts/train_ppo.sh <config_name>
# Example: bash scripts/train_ppo.sh config_curr_ppo.yaml

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_NAME="${1:-}"
if [ -z "$CONFIG_NAME" ]; then
    echo "ERROR: pass config name, e.g.: bash scripts/train_ppo.sh config_curr_ppo.yaml"
    exit 1
fi

CONFIG_PATH="$REPO_ROOT/configs/$CONFIG_NAME"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: config not found at $CONFIG_PATH"
    echo "Available configs:"
    ls "$REPO_ROOT/configs"/*.yaml 2>/dev/null || true
    exit 1
fi

LOGDIR="$REPO_ROOT/logs"
mkdir -p "$LOGDIR"

LOGFILE="$LOGDIR/$CONFIG_NAME.log"
PIDFILE="$LOGDIR/$CONFIG_NAME.pid"

echo "Launching: python main.py --config $CONFIG_NAME train-ppo"
echo "  log:  $LOGFILE"
echo "  pid:  $PIDFILE"

cd "$REPO_ROOT/app"
nohup python main.py --config "$CONFIG_NAME" train-ppo > "$LOGFILE" 2>&1 &
PID=$!

echo "$PID" > "$PIDFILE"
echo
echo "Training started — PID $PID"
echo "Stop it with:  bash scripts/stop_training.sh $CONFIG_NAME"
echo "Tail log:      tail -f logs/$CONFIG_NAME.log"
