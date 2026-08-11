#!/bin/bash
# Launch Tetrio preprocessing in background via nohup.
# Usage:  bash scripts/process_tetrio.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

LOGDIR="$REPO_ROOT/logs"
mkdir -p "$LOGDIR"

JOB_NAME="process_tetrio"
LOGFILE="$LOGDIR/$JOB_NAME.log"
PIDFILE="$LOGDIR/$JOB_NAME.pid"

echo "Launching: python main.py precompute-tetrio"
echo "  log:  $LOGFILE"
echo "  pid:  $PIDFILE"

cd "$REPO_ROOT/app"
nohup python main.py precompute-tetrio > "$LOGFILE" 2>&1 &
PID=$!

echo "$PID" > "$PIDFILE"
echo
echo "Preprocessing started — PID $PID"
echo "Stop it with:  kill $PID"
echo "Tail log:      tail -f logs/$JOB_NAME.log"
