#!/bin/bash
set -x

# Colocated GRPO training+generation for Qwen3-VL on Geometry-3K
#
# First, prepare the dataset:
#   uv run examples/geometry-3k/geometry_3k_dataset.py --output_dir $HOME/data/geometry_3k
#
# Then run training:
#   bash examples/geometry-3k/run_geometry_3k.sh
#
# You can override defaults with environment variables:
#   NUM_GPUS=4 DATA_DIR=/path/to/data bash examples/geometry-3k/run_geometry_3k.sh

# tested on 8xH100 80GB

: "${DATA_DIR:="$HOME/data/geometry_3k"}"
: "${NUM_GPUS:=8}"
: "${LOGGER:=wandb}"  # change to "wandb" for W&B logging
: "${INFERENCE_BACKEND:=vllm}"

    # grpo_args = (
    #     "--advantage-estimator grpo "
    #     "--kl-loss-coef 0.00 "
    #     "--kl-loss-type low_var_kl "
    #     "--kl-coef 0.00 "
    #     "--entropy-coef 0.00 "
    #     "--eps-clip 0.2 "
    #     "--eps-clip-high 0.28 "
    # )

    # optimizer_args = (
    #     "--optimizer adam "
    #     "--lr 1e-6 "
    #     "--lr-decay-style constant "
    #     "--weight-decay 0.1 "
    #     "--adam-beta1 0.9 "
    #     "--adam-beta2 0.98 "
    # )


uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/test.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-VL-4B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=512 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.adam_betas=[0.9,0.98] \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.kl_loss_coef=0.0 \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=geometry-3k \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="geometry-3k" \
  trainer.run_name="geometry-3k-qwen3-4b-slime-hparams" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/data/skyrl/ckpts/geometry-3k-qwen3-4b-slime-hparams-ckpt" \
  $@