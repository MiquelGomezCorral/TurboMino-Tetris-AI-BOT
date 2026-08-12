#!/bin/bash
# Launch Tetrio pretraining in background via nohup.
# Usage:  bash scripts/train_tetrio.sh [config_name] [resume_checkpoint]
# Example: bash scripts/train_tetrio.sh config_tetrio.yaml models/pretrain_model/config_tetrio/last.ckpt

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_NAME="${1:-config_tetrio.yaml}"
RESUME_CHECKPOINT="${2:-}"

CONFIG_PATH="$REPO_ROOT/configs/$CONFIG_NAME"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: config not found at $CONFIG_PATH"
    echo "Available configs:"
    ls "$REPO_ROOT/configs"/*.yaml 2>/dev/null || true
    exit 1
fi

if [ -n "$RESUME_CHECKPOINT" ]; then
    if [ ! -f "$RESUME_CHECKPOINT" ]; then
        echo "ERROR: resume checkpoint not found at $RESUME_CHECKPOINT"
        exit 1
    fi
    RESUME_CHECKPOINT="$(realpath "$RESUME_CHECKPOINT")"
fi

LOGDIR="$REPO_ROOT/logs"
mkdir -p "$LOGDIR"

LOGFILE="$LOGDIR/$CONFIG_NAME.log"
PIDFILE="$LOGDIR/$CONFIG_NAME.pid"

echo "Launching: python main.py --config $CONFIG_NAME train-tetrio"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "  resume: $RESUME_CHECKPOINT"
fi
echo "  log:  $LOGFILE"
echo "  pid:  $PIDFILE"

cd "$REPO_ROOT/app"
if [ -n "$RESUME_CHECKPOINT" ]; then
    nohup python main.py --config "$CONFIG_NAME" train-tetrio --resume_model_path "$RESUME_CHECKPOINT" > "$LOGFILE" 2>&1 &
else
    nohup python main.py --config "$CONFIG_NAME" train-tetrio > "$LOGFILE" 2>&1 &
fi
PID=$!

echo "$PID" > "$PIDFILE"
echo
echo "Training started — PID $PID"
echo "Stop it with:  bash scripts/stop_training.sh $CONFIG_NAME"
echo "Tail log:      tail -f logs/$CONFIG_NAME.log"
