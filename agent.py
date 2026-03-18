"""ARC-AGI-2 solver — predicts output grids from input-output examples.

Takes a JSON task on stdin (fewshots + test input), prints the output grid as JSON on stdout.
Saves full LLM trajectory to eval_results/trajectories/<index>.json if EVAL_TRAJECTORY_DIR is set.
"""

import sys
import os
import json
import re

from openai import OpenAI


def solve(fewshots: list, test_input: list) -> list:
    """Given few-shot examples and a test input grid, predict the output grid."""
    client = OpenAI()

    examples = ""
    for i, ex in enumerate(fewshots):
        examples += f"Example {i+1}:\nInput:\n{json.dumps(ex['input'])}\nOutput:\n{json.dumps(ex['output'])}\n\n"

    messages = [
        {"role": "system", "content": "You are solving an abstract reasoning puzzle. Given input-output grid examples, find the pattern and predict the output for the test input. Output ONLY a valid JSON 2D array (list of lists of integers). No explanation."},
        {"role": "user", "content": f"{examples}Now predict the output for:\nInput:\n{json.dumps(test_input)}\n\nOutput:"},
    ]

    model = os.environ.get("SOLVER_MODEL", "gpt-4.1-nano")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=4096,
    )

    raw_output = response.choices[0].message.content.strip()

    # Save trajectory if requested
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
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
            },
        }
        with open(os.path.join(traj_dir, f"{idx}.json"), "w") as f:
            json.dump(trajectory, f, indent=2)

    # Extract JSON array
    match = re.search(r'\[.*\]', raw_output, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(raw_output)


if __name__ == "__main__":
    data = json.loads(sys.stdin.read().strip())
    result = solve(data["fewshots"], data["test_input"])
    print(json.dumps(result))
