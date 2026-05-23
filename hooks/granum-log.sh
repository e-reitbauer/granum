#!/usr/bin/env bash
# UserPromptSubmit — append prompt to session.log, keep last 20 entries
set -euo pipefail

GRANUM_DIR="${GRANUM_CWD:-.}/.granum"
LOG_FILE="$GRANUM_DIR/session.log"

mkdir -p "$GRANUM_DIR"

PROMPT="${CLAUDE_PROMPT:-}"
if [[ -z "$PROMPT" ]]; then
    exit 0
fi

echo "$PROMPT" >> "$LOG_FILE"

# Keep last 20 entries
if [[ -f "$LOG_FILE" ]]; then
    tail -n 20 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
fi

exit 0
