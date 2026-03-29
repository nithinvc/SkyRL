"""
Custom entrypoint for VLM RL training with VisGym environments.

Subclasses BasePPOExp to wire in SkyRLGymTinkerGenerator (multi-modal)
instead of the default SkyRLGymGenerator (text-only).

Requires _SKYRL_USE_NEW_INFERENCE=1 (RemoteInferenceClient).

Usage:
    _SKYRL_USE_NEW_INFERENCE=1 uv run --isolated --extra fsdp \
        python examples/train/visgym/visgym_entrypoint.py [config overrides...]
"""

import asyncio
import multiprocessing as mp
import sys

import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.generators.base import GeneratorInterface
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

mp.set_start_method("spawn", force=True)


class VisGymVLMExp(BasePPOExp):
    """BasePPOExp subclass that uses the Tinker generator for VLM support."""

    def get_generator(self, cfg, tokenizer, inference_engine_client) -> GeneratorInterface:
        return self.get_tinker_generator(cfg, inference_engine_client, tokenizer)


@ray.remote(num_cpus=1)
def visgym_entrypoint(cfg: SkyRLTrainConfig):
    exp = VisGymVLMExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(visgym_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
