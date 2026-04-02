#!/usr/bin/env python3
"""Inspect dumped TrainingInputBatch pickles (vision fields: image_grid_thw, pixel_values).

Run from SkyRL repo root (unpickling needs skyrl-train optional deps):

  uv run --extra skyrl-train python examples/train/visgym/examine-batch.py

Optional: pass a path or glob as the first argument (default: visgym post_convert dumps).
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

# Run from SkyRL root; see module docstring for uv invocation.
_SKYRL_ROOT = Path(__file__).resolve().parents[2]
if str(_SKYRL_ROOT) not in sys.path:
    sys.path.insert(0, str(_SKYRL_ROOT))


def _describe_tensor(name: str, t, idx: int | None = None) -> None:
    prefix = f"  [{idx}] " if idx is not None else "  "
    if t is None:
        print(f"{prefix}{name}: None")
        return
    import torch

    if not isinstance(t, torch.Tensor):
        print(f"{prefix}{name}: type={type(t).__name__} repr={repr(t)[:200]}")
        return
    print(
        f"{prefix}{name}: shape={tuple(t.shape)} ndim={t.ndim} dtype={t.dtype} "
        f"device={t.device} min={t.min().item() if t.numel() else 'n/a'} max={t.max().item() if t.numel() else 'n/a'}"
    )
    if t.numel() <= 24:
        print(f"{prefix}  values: {t.flatten().tolist()}")


def _describe_tensor_list(key: str, tl) -> None:
    from skyrl.backends.skyrl_train.training_batch import TensorList

    if tl is None:
        print(f"{key}: None")
        return
    if not isinstance(tl, TensorList):
        print(f"{key}: unexpected type {type(tl)}")
        return
    n = len(tl.tensors)
    print(f"{key}: TensorList len={n}")
    bad = []
    for i, t in enumerate(tl.tensors):
        _describe_tensor("tensor", t, idx=i)
        # Qwen3-VL expects per-batch row in cat'd tensor [num_images, 3]; each TensorList slot should be [k, 3].
        if t.ndim == 1:
            n3 = t.numel() // 3 if t.numel() % 3 == 0 else None
            hint = f" (flat length {t.numel()} → {n3} triples if reshape (-1,3))" if n3 else ""
            bad.append((i, f"ndim==1{hint}; Qwen expects [num_images, 3]"))
        elif t.ndim == 2 and t.shape[1] != 3:
            bad.append((i, f"shape {tuple(t.shape)} (expected [*, 3])"))
    if bad:
        print(f"  *** PROBLEMS for {key}:")
        for i, msg in bad:
            print(f"    batch slot {i}: {msg}")
    else:
        print(f"  {key}: all slots have ndim>=2 and last dim 3 (or empty)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        nargs="?",
        default="/home/ray/exports/visgym_maze2d/dumped_data/global_step*_training_input_post_convert.pkl",
        help="Pickle path or glob (default: visgym post_convert dumps)",
    )
    p.add_argument("--pick", choices=["latest", "first"], default="latest", help="When glob matches many files")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.expanduser(args.path)))
    if not paths:
        print(f"No files matched: {args.path}")
        raise SystemExit(1)
    if len(paths) > 1:
        paths = sorted(paths)
        path = paths[-1] if args.pick == "latest" else paths[0]
        print(f"Matched {len(paths)} files; using {args.pick}: {path}\n")
    else:
        path = paths[0]
        print(f"Loading: {path}\n")

    with open(path, "rb") as f:
        batch = pickle.load(f)

    print(f"type: {type(batch).__name__}")
    if hasattr(batch, "metadata") and batch.metadata:
        print(f"metadata keys: {list(batch.metadata.keys())}")
    keys = list(batch.keys()) if hasattr(batch, "keys") else []
    print(f"batch keys ({len(keys)}): {keys}\n")

    if hasattr(batch, "__len__"):
        print(f"batch __len__ (batch size): {len(batch)}\n")

    for k in ("input_ids", "attention_mask", "labels", "position_ids"):
        if k in batch:
            v = batch[k]
            if hasattr(v, "shape"):
                print(f"{k}: shape={tuple(v.shape)} dtype={getattr(v, 'dtype', '')}")
            else:
                print(f"{k}: {type(v)}")

    print()
    if "image_grid_thw" in batch:
        _describe_tensor_list("image_grid_thw", batch["image_grid_thw"])
    else:
        print("image_grid_thw: <missing>")

    print()
    if "pixel_values" in batch:
        pv = batch["pixel_values"]
        from skyrl.backends.skyrl_train.training_batch import TensorList

        if isinstance(pv, TensorList):
            print(f"pixel_values: TensorList len={len(pv.tensors)}")
            for i, t in enumerate(pv.tensors):
                _describe_tensor("tensor", t, idx=i)
        else:
            _describe_tensor("pixel_values", pv)
    else:
        print("pixel_values: <missing>")


if __name__ == "__main__":
    main()
