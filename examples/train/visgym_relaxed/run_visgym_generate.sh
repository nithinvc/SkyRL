#!/usr/bin/env bash
set -euo pipefail
set -x

# Eval-only VLM generation script for fast iteration.
#
# Connects to already-running vLLM servers (started by start_servers.sh)
# and runs SkyRLVLMGymGenerator through the evaluate() pipeline.
#
# Usage:
#   # 1. Start servers first (in another terminal):
#   bash examples/train/visgym_relaxed/start_servers.sh
#
#   # 2. Run generation (re-run after editing generator code):
#   PROXY_URL=http://... SERVER_URLS=http://...:8000,http://...:8001 \
#     bash examples/train/visgym_relaxed/run_visgym_generate.sh
#
# Environment variable overrides:
#   PROXY_URL       Router URL from start_servers.sh output
#   SERVER_URLS     Comma-separated server URLs from start_servers.sh output
#   MODEL_PATH      Model name (must match what servers are serving)
#   LOGGER          Logging backend (default: console)

: "${ENV_ID:=maze_2d/easy}"
: "${EVAL_DATA_DIR:=$HOME/data/visgym_maze_2d_easy_eval}"
: "${MODEL_PATH:=Qwen/Qwen3-VL-8B-Instruct}"
: "${LOGGER:=console}"
: "${N_SAMPLES_PER_PROMPT:=8}"

# Server connection - these must match start_servers.sh output
: "${PROXY_URL:?Set PROXY_URL to the router URL printed by start_servers.sh}"
: "${SERVER_URLS:?Set SERVER_URLS to the comma-separated server URLs printed by start_servers.sh}"

# Convert comma-separated SERVER_URLS to Python list format:
#   "http://a:8000,http://b:8001" -> "['http://a:8000','http://b:8001']"
IFS=',' read -ra _urls <<< "$SERVER_URLS"
_py_list="["
for i in "${!_urls[@]}"; do
  [ "$i" -gt 0 ] && _py_list+=","
  _py_list+="'${_urls[$i]}'"
done
_py_list+="]"

# Generate eval stub dataset if it does not exist
if [ ! -f "$EVAL_DATA_DIR/train.parquet" ]; then
  echo "=== Generating eval stub dataset for $ENV_ID ==="
  uv run examples/train/visgym_relaxed/visgym_dataset.py \
    --env_id "$ENV_ID" \
    --num_rows 64 \
    --seed \
    --output_dir "$EVAL_DATA_DIR"
fi

_SKYRL_USE_NEW_INFERENCE=1 \
uv run --isolated --extra fsdp \
  python examples/train/visgym_relaxed/visgym_generate.py \
  data.val_data="['$EVAL_DATA_DIR/train.parquet']" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=false \
  trainer.eval_interval=1 \
  trainer.logger="$LOGGER" \
  trainer.project_name="vlm_maze_2d_easy_generate" \
  trainer.run_name="visgym_generate_dev" \
  trainer.log_path="$HOME/exports/testing/logs" \
  trainer.export_path="$HOME/exports/testing" \
  trainer.ckpt_path="$HOME/exports/testing/ckpts" \
  trainer.dump_eval_results=false \
  trainer.max_prompt_length=2048 \
  generator.inference_engine.external_proxy_url="$PROXY_URL" \
  generator.inference_engine.external_server_urls="$_py_list" \
  generator.inference_engine.async_engine=true \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=1 \
  generator.max_turns=18 \
  generator.max_input_length=8192 \
  generator.n_samples_per_prompt="$N_SAMPLES_PER_PROMPT" \
  generator.is_vlm=true \
  generator.batched=false \
  environment.env_class=visgym \
  "$@"
