"""Build aligned PyTorch -> JAX source-code pairs from local checkouts."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-root", required=True, type=Path)
    parser.add_argument("--jax-root", required=True, type=Path)
    parser.add_argument("--output", default="data/pairs.jsonl", type=Path)
    args = parser.parse_args()

    records = []

    for torch_file in args.torch_root.glob("*/baseline.py"):
        benchmark = torch_file.parent.name
        matches = list(args.jax_root.glob(f"level*/{benchmark}/baseline.py"))

        if len(matches) != 1:
            continue

        jax_file = matches[0]
        records.append(
            {
                "benchmark": benchmark,
                "source_framework": "pytorch",
                "target_framework": "jax",
                "input": torch_file.read_text(),
                "target": jax_file.read_text(),
                "torch_path": str(torch_file),
                "jax_path": str(jax_file),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(f"wrote {len(records)} aligned pairs to {args.output}")


if __name__ == "__main__":
    main()
