#!/bin/bash
# Monitor training progress

LOG_FILE="outputs/training_long_full.log"

echo "=== Training Progress Monitor ==="
echo ""

# Check if training is running
if [ -f outputs/train_pid.txt ]; then
    PID=$(cat outputs/train_pid.txt)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Training is running (PID: $PID)"
    else
        echo "✗ Training process not found"
    fi
else
    echo "? No PID file found"
fi

echo ""
echo "--- Latest Progress ---"
if [ -f "$LOG_FILE" ]; then
    # Show last few epoch summaries
    grep "INFO: Epoch" "$LOG_FILE" | tail -5
    echo ""

    # Show last metrics
    echo "--- Latest Metrics ---"
    grep "METRICS:" "$LOG_FILE" | tail -3
    echo ""

    # Show eval scores
    echo "--- Latest Eval Scores ---"
    grep "eval_acc" "$LOG_FILE" | tail -3
else
    echo "Log file not found yet"
fi
