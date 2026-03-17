"""ARC-AGI-2 solver — predicts output grids from input-output examples.

Takes a JSON task on stdin (fewshots + test input), prints the output grid as JSON on stdout.
"""

import sys
import os
import json
import re

from openai import OpenAI


def solve(fewshots: list, test_input: list) -> list:
    """Given few-shot examples and a test input grid, predict the output grid."""
    client = OpenAI()

    # Format examples
    examples = ""
    for i, ex in enumerate(fewshots):
        examples += f"Example {i+1}:\nInput:\n{json.dumps(ex['input'])}\nOutput:\n{json.dumps(ex['output'])}\n\n"

    response = client.chat.completions.create(
        model=os.environ.get("SOLVER_MODEL", "gpt-4.1-nano"),
        messages=[
            {"role": "system", "content": "You are solving an abstract reasoning puzzle. Given input-output grid examples, find the pattern and predict the output for the test input. Output ONLY a valid JSON 2D array (list of lists of integers). No explanation."},
            {"role": "user", "content": f"{examples}Now predict the output for:\nInput:\n{json.dumps(test_input)}\n\nOutput:"},
        ],
        temperature=0,
        max_tokens=4096,
    )

    text = response.choices[0].message.content.strip()
    # Extract JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


if __name__ == "__main__":
    data = json.loads(sys.stdin.read().strip())
    result = solve(data["fewshots"], data["test_input"])
    print(json.dumps(result))
