"""Evaluate agent.py on ARC-AGI-2 puzzles. Concurrent. Saves trajectories."""
import json
import subprocess
import sys
import os
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

data_path = sys.argv[1]
max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16

with open(data_path) as f:
    tasks = [json.loads(line) for line in f]

total = len(tasks)
correct = 0
completed = 0
results = []

# Set up trajectory dir
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
traj_dir = f"eval_results/{timestamp}/trajectories"
os.makedirs(traj_dir, exist_ok=True)


def eval_one(idx, task):
    try:
        env = {**os.environ, "EVAL_TRAJECTORY_DIR": traj_dir, "EVAL_INDEX": str(idx)}
        proc = subprocess.run(
            ["python3", "agent.py"],
            input=json.dumps(task), capture_output=True, text=True, timeout=1800, env=env,
        )
        predicted = None
        passed = False
        if proc.returncode == 0:
            try:
                predicted = json.loads(proc.stdout.strip())
                passed = predicted == task["expected_output"]
            except json.JSONDecodeError:
                pass

        return {
            "index": idx,
            "passed": passed,
            "predicted": predicted,
            "expected": task["expected_output"],
            "input": task["test_input"],
            "fewshots": task["fewshots"],
            "exit_code": proc.returncode,
        }
    except Exception as e:
        return {"index": idx, "passed": False, "error": str(e)}


print(f"Evaluating {total} puzzles ({max_workers} concurrent)...", file=sys.stderr)
print(f"Trajectories: {traj_dir}", file=sys.stderr)

with ThreadPoolExecutor(max_workers=max_workers) as pool:
    futures = {pool.submit(eval_one, i, t): i for i, t in enumerate(tasks)}
    for future in as_completed(futures):
        completed += 1
        result = future.result()
        results.append(result)
        if result.get("passed"):
            correct += 1
        print(f"  {completed}/{total} done, {correct} correct", file=sys.stderr, end="\r")

print(file=sys.stderr)

# Save results summary
results.sort(key=lambda r: r["index"])
summary_path = f"eval_results/{timestamp}/results.jsonl"
with open(summary_path, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"Results saved to {summary_path}", file=sys.stderr)

print("---")
print(f"accuracy:         {correct / total:.6f}" if total > 0 else "accuracy:         0.000000")
print(f"correct:          {correct}")
print(f"total:            {total}")
