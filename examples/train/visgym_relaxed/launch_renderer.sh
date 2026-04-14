#!/usr/bin/env bash
set -euo pipefail
set -x

# Launch a pool of GPU-less render servers behind a vllm-router.
#
# Each server runs `vllm launch render` (CPU-only, no model weights loaded).
# The vllm-router load-balances across them, keeping render traffic
# completely separate from generate.
#
# Usage:
#   bash examples/train/visgym_relaxed/launch_renderer.sh
#
# Environment variable overrides:
#   MODEL_PATH          Model name (must match what GPU servers are serving)
#   NUM_RENDER_SERVERS  Number of render server processes (default: 8)
#   RENDER_BASE_PORT    First render server port; others use consecutive ports (default: 8090)
#   RENDER_ROUTER_PORT  Port for the render router (default: 8089)

: "${MODEL_PATH:=Qwen/Qwen3-VL-8B-Instruct}"
: "${NUM_RENDER_SERVERS:=64}"
: "${RENDER_BASE_PORT:=8090}"
: "${RENDER_ROUTER_PORT:=8089}"

NODE_IP="$(hostname -i)"
RENDER_SERVER_URLS=()
PIDS=()

cleanup() {
    echo "Shutting down render servers..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait
}
trap cleanup EXIT

# Kill any leftover processes on the render port range
for i in $(seq 0 $((NUM_RENDER_SERVERS - 1))); do
    port=$((RENDER_BASE_PORT + i))
    fuser -k "${port}/tcp" 2>/dev/null || true
done
fuser -k "${RENDER_ROUTER_PORT}/tcp" 2>/dev/null || true
sleep 1

for i in $(seq 0 $((NUM_RENDER_SERVERS - 1))); do
    port=$((RENDER_BASE_PORT + i))
    echo "=== Starting render server $i on port $port ==="
    CUDA_VISIBLE_DEVICES="" \
    HF_HUB_OFFLINE=1 \
    uv run --isolated --extra fsdp \
        vllm launch render "$MODEL_PATH" --port "$port" \
        --mm-processor-cache-gb 0 &
    PIDS+=($!)
    RENDER_SERVER_URLS+=("http://${NODE_IP}:${port}")
done

echo "=== Waiting for render servers to be ready ==="
for url in "${RENDER_SERVER_URLS[@]}"; do
    until curl -sf "${url}/health" > /dev/null 2>&1; do
        sleep 1
    done
    echo "  $url is healthy"
done

echo "=== Starting render router on port $RENDER_ROUTER_PORT ==="
uv run --isolated --extra fsdp \
    vllm-router \
    --host 0.0.0.0 \
    --port "$RENDER_ROUTER_PORT" \
    --policy round_robin \
    --prometheus-port 9091 \
    --worker-urls "${RENDER_SERVER_URLS[@]}" &
PIDS+=($!)

RENDER_ROUTER_URL="http://${NODE_IP}:${RENDER_ROUTER_PORT}"
until curl -sf "${RENDER_ROUTER_URL}/health" > /dev/null 2>&1; do
    sleep 1
done

echo ""
echo "=============================================="
echo "  Render router ready: $RENDER_ROUTER_URL"
echo "  Backends: ${NUM_RENDER_SERVERS} render servers"
echo "  Ports: ${RENDER_BASE_PORT}–$((RENDER_BASE_PORT + NUM_RENDER_SERVERS - 1))"
echo ""
echo "  Use with run_visgym_generate.sh:"
echo "    RENDER_URL=$RENDER_ROUTER_URL"
echo "=============================================="
echo ""

wait
