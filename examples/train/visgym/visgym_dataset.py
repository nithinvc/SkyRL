"""
Create a stub parquet dataset for VisGym RL training.

VisGymEnv generates its own multimodal prompts dynamically (images from
gymnasium.Env.render()), so the dataset rows are just "tickets" that tell
the system which environment to run. The prompt column is a placeholder
that passes PromptDataset's tokenizer length filter.

Usage:
    uv run examples/train/visgym/visgym_dataset.py --output_dir ~/data/visgym
    uv run examples/train/visgym/visgym_dataset.py --env_id counting/easy --num_rows 128
"""

import argparse
import os

import datasets


def make_rows(env_id: str, num_rows: int) -> list[dict]:
    rows = []
    for i in range(num_rows):
        rows.append(
            {
                "prompt": [{"role": "user", "content": "placeholder"}],
                "env_class": "visgym",
                "visgym_env_id": env_id,
            }
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create VisGym stub dataset")
    parser.add_argument("--output_dir", default="~/data/visgym")
    parser.add_argument("--env_id", default="maze_2d/easy", help="VisGym environment ID")
    parser.add_argument("--num_rows", type=int, default=64)
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    rows = make_rows(args.env_id, args.num_rows)
    ds = datasets.Dataset.from_list(rows)

    path = os.path.join(output_dir, "train.parquet")
    ds.to_parquet(path)
    print(f"Saved {len(ds)} rows to {path}")
    print(f"  env_id: {args.env_id}")
    print(f"  columns: {ds.column_names}")
