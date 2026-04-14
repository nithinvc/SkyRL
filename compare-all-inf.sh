#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

exp_dirs=()
for d in inference-exps/*/; do
    [ -f "${d}summary.json" ] && exp_dirs+=("${d%/}")
done

if [ ${#exp_dirs[@]} -lt 2 ]; then
    echo "Need at least 2 experiment directories with summary.json, found ${#exp_dirs[@]}"
    exit 1
fi

echo "Comparing: ${exp_dirs[*]}"
uv run --isolated --with matplotlib python examples/train/visgym_relaxed/compare_experiments.py \
    "${exp_dirs[@]}" \
    --output inference-exps/summary-comparisons/all/
