#!/usr/bin/env bash
set -euo pipefail

DATA="data/test.jsonl"
if [ ! -f "$DATA" ]; then
    echo "ERROR: $DATA not found. Run: bash prepare.sh" >&2
    exit 1
fi

TOTAL=$(wc -l < "$DATA")
echo "Evaluating $TOTAL puzzles..." >&2
python3 eval/run_all.py "$DATA"
