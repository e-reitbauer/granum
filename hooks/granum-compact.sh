#!/usr/bin/env bash
# Stop hook — increment tool_call_count; trigger /compact at threshold
set -euo pipefail

GRANUM_DIR="${GRANUM_CWD:-.}/.granum"
COUNTER_FILE="$GRANUM_DIR/tool_call_count"
CONFIG_FILE="$GRANUM_DIR/config.json"

mkdir -p "$GRANUM_DIR"

# Read threshold from config, default 50
THRESHOLD=50
if [[ -f "$CONFIG_FILE" ]]; then
    val=$(python3 -c "import json,sys; d=json.load(open('$CONFIG_FILE')); print(d.get('compaction_threshold', 50))" 2>/dev/null || echo "50")
    THRESHOLD="${val:-50}"
fi

# Read and increment counter
COUNT=0
if [[ -f "$COUNTER_FILE" ]]; then
    COUNT=$(cat "$COUNTER_FILE" 2>/dev/null || echo "0")
fi
COUNT=$((COUNT + 1))

if (( COUNT >= THRESHOLD )); then
    echo "0" > "$COUNTER_FILE"
    echo "/compact"
else
    echo "$COUNT" > "$COUNTER_FILE"
fi

exit 0
