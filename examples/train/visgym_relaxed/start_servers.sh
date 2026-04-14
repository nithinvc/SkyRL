#!/usr/bin/env bash
set -euo pipefail
set -x

# Persistent vLLM server launcher for fast generation iteration.
#
# Starts vLLM inference servers and blocks until Ctrl-C. The servers stay up
# so that run_visgym_generate.sh can reconnect on each run without re-loading
# model weights.
#
# Usage:
#   bash examples/train/visgym_relaxed/start_servers.sh
#
# Environment variable overrides:
#   MODEL_PATH             Model to serve (default: Qwen/Qwen3-VL-8B-Instruct)
#   NUM_INFERENCE_GPUS     Number of vLLM engine instances (default: 8)
#   MAX_MODEL_LEN          Maximum model context length (default: 60000)

: "${MODEL_PATH:=Qwen/Qwen3-VL-8B-Instruct}"
: "${NUM_INFERENCE_GPUS:=8}"
: "${MAX_MODEL_LEN:=60000}"

_SKYRL_USE_NEW_INFERENCE=1 \
uv run --isolated --extra fsdp \
  python examples/train/visgym_relaxed/start_servers.py \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=false \
  generator.inference_engine.num_engines="$NUM_INFERENCE_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  "$@"
