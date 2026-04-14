"""
Eval-only entrypoint for VLM generation with VisGym environments.

Connects to already-running vLLM servers (started by start_servers.sh) via
``external_proxy_url`` / ``external_server_urls`` and runs the
SkyRLVLMGymGenerator through the evaluate() pipeline.

Usage:
    _SKYRL_USE_NEW_INFERENCE=1 \
    uv run --isolated --extra fsdp \
        python examples/train/visgym_relaxed/visgym_generate.py \
        generator.inference_engine.external_proxy_url="http://..." \
        generator.inference_engine.external_server_urls="['http://...']" \
        [config overrides...]
"""

import asyncio
import multiprocessing as mp
import sys

import ray
from loguru import logger

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl.train.utils.utils import initialize_ray, validate_generator_cfg
from skyrl_gym.envs import register

mp.set_start_method("spawn", force=True)


@ray.remote(num_cpus=1)
def visgym_eval_entrypoint(cfg: SkyRLTrainConfig) -> dict:
    register(
        id="visgym",
        entry_point="examples.train.visgym_relaxed.env:VisGymEnv",
    )

    exp = EvalOnlyEntrypoint(cfg)
    return asyncio.run(exp.run())


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(visgym_eval_entrypoint.remote(cfg))
    logger.info(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
