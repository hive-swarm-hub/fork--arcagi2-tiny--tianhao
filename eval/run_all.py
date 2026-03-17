"""Evaluate agent.py on ARC-AGI-2 puzzles. Exact grid match."""
import json
import subprocess
import sys

data_path = sys.argv[1]

with open(data_path) as f:
    tasks = [json.loads(line) for line in f]

total = len(tasks)
correct = 0

for task in tasks:
    try:
        proc = subprocess.run(
            ["python3", "agent.py"],
            input=json.dumps(task), capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            continue
        predicted = json.loads(proc.stdout.strip())
        expected = task["expected_output"]
        if predicted == expected:
            correct += 1
    except Exception:
        pass

print("---")
print(f"accuracy:         {correct / total:.6f}" if total > 0 else "accuracy:         0.000000")
print(f"correct:          {correct}")
print(f"total:            {total}")
