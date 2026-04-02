# VisGym Environment Evaluation Report

**Model:** `/home/ray/models/visgym_model/mixed_qwen3vl`
**Config:** 32 rollouts, 20 max turns, temperature=0.7, max_tokens=4096
**Date:** 2026-04-02

## Summary Table

| Environment | Success Rate | Avg Steps | Avg Parse Fail | Avg Latency/Turn | Notes |
|---|---|---|---|---|---|
| matchstick_rotation/easy | **21/32 (65.6%)** | 3.91 | 0.03 | 1786 ms | Best candidate for RL training |
| matchstick_rotation/hard | 24/32 (75.0%) | 4.31 | 0.00 | 1438 ms | Surprisingly higher than easy |
| maze_2d/hard | 15/32 (46.9%) | 18.88 | 0.03 | 222 ms | Decent signal, long episodes |
| matchstick_equation/easy | 11/32 (34.4%) | 2.00 | 0.00 | 266 ms | Too hard, only 1 move attempt |
| matchstick_equation/hard | 0/32 (0.0%) | 2.00 | 0.00 | 355 ms | Completely fails with 2 break_moves |
| patch_reassembly/easy | 0/32 (0.0%) | 20.00 | 0.00 | 447 ms | Action parsing broken (truncated tuples) |
| patch_reassembly/hard | 0/32 (0.0%) | 19.09 | 0.12 | 426 ms | Same parsing issue as easy |
| colorization/easy | N/A | N/A | N/A | N/A | Missing dataset (needs HF download) |
| colorization/hard | N/A | N/A | N/A | N/A | Missing dataset (needs HF download) |
| zoom_in_puzzle/easy | N/A | N/A | N/A | N/A | Missing dataset (needs HF download) |
| zoom_in_puzzle/hard | N/A | N/A | N/A | N/A | Missing dataset (needs HF download) |

## Recommendation

**`matchstick_rotation/easy` at 65.6% is the best candidate for RL training.**

It hits the ~60% sweet spot where the model succeeds often enough for positive reward signal but fails enough to learn from. Key properties:
- Short episodes (avg 3.91 steps) = fast training iterations
- Clean action parsing (0.03 parse failures/ep)
- All 32 rollouts terminate properly (no max_turns exhaustion)
- Multi-turn: model moves matchsticks then submits, providing a real decision sequence

Runner-up: **maze_2d/hard at 46.9%** is also viable but has longer episodes (avg 18.88 steps, many hitting the 20-turn cap), which means more tokens per training sample and slower iteration.

## Detailed Observations

### matchstick_rotation (easy: 65.6%, hard: 75.0%)
The model adjusts matchstick positions (x, y offsets) and rotations. Hard mode has tighter tolerances (pos=5, ang=10 vs pos=10, ang=15 for easy) but paradoxically scores higher -- possibly because the model happens to be more decisive on harder-looking instances, or the random problem distribution is slightly more favorable. Both terminate after ~4 steps on average, making this a compact task.

### maze_2d/hard (46.9%)
11x11 maze with 4-directional movement. 53% of rollouts hit max_turns without solving, 47% terminate successfully. The long episode length (18.88 avg) means high token cost per sample. For reference, maze_2d/easy was previously evaluated at ~80%+ (too easy for useful RL signal).

### matchstick_equation (easy: 34.4%, hard: 0.0%)
Single-turn task: model sees a broken equation and must specify which matchstick to move. Easy requires 1 move (break_moves=1), hard requires 2 (break_moves=2). The model always terminates after exactly 2 steps (one move + stop). At 34.4% it's below the ideal range but could be useful if combined with other tasks.

### patch_reassembly (easy: 0.0%, hard: 0.0%)
Complete failure. The action regex extracts `('place', (3, 2, 0)` (missing closing paren) from model outputs like `('place', (3, 2, 0))`. The env then fails to parse the truncated action string. This is a **bug in the action extraction regex** in `sample-visgym.py` -- nested tuples break the greedy match. Would need a fix to the regex or the env's action parser before this task is usable.

### colorization, zoom_in_puzzle (not evaluated)
These environments require image datasets from `VisGym/inference-dataset` on HuggingFace. Download was blocked by IP-based rate limiting (429 Too Many Requests). Need an HF token to download. To retry:
```bash
HF_TOKEN=<your-token> uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='VisGym/inference-dataset', repo_type='dataset',
    local_dir='VisGym/inference/inference_dataset', allow_patterns=['partial_datasets/**'],
    token='<your-token>')
"
```

## Raw Data

All rollout JSON files and per-env logs are in `SkyRL/eval_results/`.
