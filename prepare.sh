#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
echo "Downloading ARC-AGI-2 (tiny: 30 problems, seed=42)..."
python3 << 'PY'
from datasets import load_dataset
import json, pathlib, random

random.seed(42)
ds = load_dataset('arc-agi-community/arc-agi-2', split='test')

problems = []
for row in ds:
    for q in row['question']:
        problems.append({
            "fewshots": row["fewshots"],
            "test_input": q["input"],
            "expected_output": q["output"],
        })

sampled = random.sample(problems, 30)

out = pathlib.Path('data/test.jsonl')
with out.open('w') as f:
    for p in sampled:
        f.write(json.dumps(p) + '\n')
print(f'Wrote {len(sampled)} problems to {out}')
PY
echo "Done. $(wc -l < data/test.jsonl) puzzles in data/test.jsonl"
