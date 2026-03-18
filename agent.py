"""ARC-AGI-2 solver — predicts output grids from input-output examples.

Takes a JSON task on stdin (fewshots + test input), prints the output grid as JSON on stdout.
Saves full LLM trajectory to eval_results/trajectories/<index>.json if EVAL_TRAJECTORY_DIR is set.
"""

import sys
import os
import json
import re

from openai import OpenAI


def call_model(client, model, messages, effort="medium", max_tokens=16384):
    return client.responses.create(
        model=model,
        reasoning={"effort": effort},
        input=messages,
        max_output_tokens=max_tokens,
    )


def parse_grid(raw_output):
    match = re.search(r'```json\s*(.*?)\s*```', raw_output, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r'\[.*\]', raw_output, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(raw_output)


def solve(fewshots: list, test_input: list) -> list:
    """Given few-shot examples and a test input grid, predict the output grid."""
    client = OpenAI()
    model = os.environ.get("SOLVER_MODEL", "gpt-5.4-mini")

    examples = ""
    for i, ex in enumerate(fewshots):
        examples += f"Example {i+1}:\nInput:\n{json.dumps(ex['input'])}\nOutput:\n{json.dumps(ex['output'])}\n\n"

    messages = [
        {"role": "system", "content": "You are solving an abstract reasoning puzzle. Given input-output grid examples, find the pattern and predict the output for the test input.\n\nThink step by step:\n1. Analyze each example pair to find the transformation pattern\n2. Describe the pattern in words\n3. Determine the output grid dimensions\n4. Apply the pattern to the test input\n\nAfter your reasoning, output the final answer as a JSON 2D array inside a ```json code block."},
        {"role": "user", "content": f"{examples}Now predict the output for:\nInput:\n{json.dumps(test_input)}"},
    ]

    all_raw = []

    # Best of 2: try medium reasoning twice, collect valid results
    for attempt in range(2):
        response = call_model(client, model, messages, effort="medium")
        raw = response.output_text.strip()
        if raw:
            try:
                grid = parse_grid(raw)
                all_raw.append(raw)
                # Save trajectory on first valid result
                _save_trajectory(response, model, messages, raw)
                return grid
            except (json.JSONDecodeError, ValueError):
                all_raw.append(raw)

    # All medium attempts truncated/failed — retry with low reasoning + more tokens
    response = call_model(client, model, messages, effort="low", max_tokens=32768)
    raw = response.output_text.strip()
    all_raw.append(raw)
    _save_trajectory(response, model, messages, "\n---\n".join(all_raw))
    return parse_grid(raw)


def _save_trajectory(response, model, messages, raw_output):
    traj_dir = os.environ.get("EVAL_TRAJECTORY_DIR")
    idx = os.environ.get("EVAL_INDEX")
    if traj_dir and idx is not None:
        os.makedirs(traj_dir, exist_ok=True)
        trajectory = {
            "index": int(idx),
            "model": model,
            "messages": messages,
            "raw_response": raw_output,
            "usage": {
                "input_tokens": response.usage.input_tokens if response.usage else None,
                "output_tokens": response.usage.output_tokens if response.usage else None,
            },
        }
        with open(os.path.join(traj_dir, f"{idx}.json"), "w") as f:
            json.dump(trajectory, f, indent=2)


if __name__ == "__main__":
    data = json.loads(sys.stdin.read().strip())
    result = solve(data["fewshots"], data["test_input"])
    print(json.dumps(result))
