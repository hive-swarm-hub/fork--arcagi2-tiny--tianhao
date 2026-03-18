# ARC-AGI-2 Tiny Solver

Improve a solver for ARC-AGI-2 abstract reasoning puzzles (30-problem subset).

## Setup

1. **Read the in-scope files**:
   - `agent.py` — the file you modify. The puzzle solver.
   - `eval/eval.sh` — runs evaluation. Do not modify.
   - `eval/run_all.py` — evaluation runner. Do not modify.
   - `prepare.sh` — downloads and samples the dataset. Do not modify.
2. **Run prepare**: `bash prepare.sh` to download the dataset.
3. **Verify data exists**: Check that `data/` contains `test.jsonl`.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.
5. **Run baseline**: `bash eval/eval.sh` to establish the starting accuracy.

## The benchmark

ARC-AGI-2 Tiny is a 30-problem subset (seed=42) of ARC-AGI-2. Each puzzle provides:
- A few input-output grid examples demonstrating a transformation
- A test input grid — the agent must predict the correct output grid

Grids contain integers 0-9 representing colors. Puzzles range from simple rotations to complex abstract transformations. A puzzle is correct only if every cell matches.

## Experimentation

**What you CAN do:**
- Modify `agent.py` — this is the only file you edit. Everything is fair game: prompting strategy, pattern description, grid analysis, chain-of-thought, multi-step reasoning, code generation for transformations.

**What you CANNOT do:**
- Modify `eval/`, `prepare.sh`, or test data.
- Change the model (set via `SOLVER_MODEL` env var).
- Install new packages beyond what's in `requirements.txt`.

**The goal: maximize accuracy.** Exact grid match — every cell must be correct. Accuracy = fraction of 30 puzzles solved.

**Cost** is a soft constraint.

**Simplicity criterion**: All else being equal, simpler is better.

## Output format

```
---
accuracy:         0.1000
correct:          3
total:            30
```

## Logging results

Log each experiment to `results.tsv` (tab-separated):

```
commit	accuracy	cost_usd	status	description
a1b2c3d	0.100000	1.50	keep	baseline
b2c3d4e	0.200000	3.00	keep	describe pattern in text first, then generate grid
```

## The experiment loop

LOOP FOREVER:

1. **THINK** — decide what to try next. Review results.tsv. ARC puzzles require understanding abstract transformations — consider describing the pattern in natural language first, then generating the output.
2. Modify `agent.py` with your experimental idea.
3. git commit
4. Run the experiment: `bash eval/eval.sh > run.log 2>&1`
5. Read the accuracy: `grep "^accuracy:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` for the stack trace and attempt a fix.
7. **Review artifacts**: Search `eval_results/` for per-problem results and full LLM trajectories. Study the failures before forming your next hypothesis.
8. Record the results in results.tsv (do not commit results.tsv).
9. If accuracy improved (higher), keep the git commit. If equal or worse, `git reset --hard HEAD~1`.

**Timeout**: If a run exceeds 30 minutes, kill it and treat it as a failure.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. The loop runs until interrupted.
