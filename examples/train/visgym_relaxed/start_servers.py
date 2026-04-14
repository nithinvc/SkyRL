"""
Persistent vLLM server launcher for fast generation iteration.

Starts vLLM inference servers via Ray + ServerGroup + VLLMRouter and blocks
until Ctrl-C.  The servers stay up so that the eval script
(visgym_generate.py) can reconnect on each run without re-loading model
weights.

Usage:
    uv run --isolated --extra fsdp \
        python examples/train/visgym_relaxed/start_servers.py \
        [config overrides...]

    # e.g. override number of engines:
    #   ... start_servers.py generator.inference_engine.num_engines=4
"""

import signal
import sys
import threading

import ray
from loguru import logger

from skyrl.backends.skyrl_train.inference_servers.server_group import ServerGroup
from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
from skyrl.backends.skyrl_train.inference_servers.vllm_router import VLLMRouter
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import initialize_ray

_shutdown_event = threading.Event()


def _on_signal(signum, _frame):
    logger.info(f"Received signal {signal.Signals(signum).name}, shutting down...")
    _shutdown_event.set()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])

    if not ray.is_initialized():
        initialize_ray(cfg)

    cli_args = build_vllm_cli_args(cfg)
    ie_cfg = cfg.generator.inference_engine

    server_group = ServerGroup(
        cli_args=cli_args,
        num_servers=ie_cfg.num_engines,
        enable_dp=ie_cfg.data_parallel_size > 1,
        distributed_executor_backend=ie_cfg.distributed_executor_backend,
    )
    server_infos = server_group.start()
    server_urls = [info.url for info in server_infos]

    router = VLLMRouter(server_urls=server_urls)
    proxy_url = router.start()

    print(flush=True)
    print("=" * 60, flush=True)
    print("vLLM servers ready. Use these values in run_visgym_generate.sh:", flush=True)
    print(f"  PROXY_URL={proxy_url}", flush=True)
    print(f"  SERVER_URLS={','.join(server_urls)}", flush=True)
    print("=" * 60, flush=True)
    print("Press Ctrl-C to shut down.", flush=True)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    _shutdown_event.wait()

    logger.info("Shutting down router...")
    router.shutdown()
    logger.info("Shutting down server group...")
    server_group.shutdown()
    logger.info("Done.")


if __name__ == "__main__":
    main()
