#!/bin/bash
set -x

# Development/testing run for Geometry-3K with smaller batch sizes
#
# First, prepare the dataset:
#   uv run examples/geometry-3k/geometry_3k_dataset.py --output_dir $HOME/data/geometry_3k
#
# Then run development training:
#   bash examples/geometry-3k/run_geometry_3k-dev.sh
#
# This script uses smaller batch sizes and the dev subset for faster iteration.

: "${DATA_DIR:="$HOME/data/geometry_3k"}"
: "${NUM_GPUS:=4}"
# : "${LOGGER:=console}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"

uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/test.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-VL-2B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=10 \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=true \
  trainer.eval_interval=2 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.use_sample_packing=false \
  trainer.ckpt_interval=5 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=geometry-3k \
  generator.n_samples_per_prompt=3 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="geometry-3k" \
  trainer.run_name="geometry-3k-dev" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/data/skyrl/ckpts/geometry-3k-dev_ckpt" \
  trainer.eval_before_train=false \
  trainer.dump_data_batch=true \
  +generator.engine_init_kwargs.max_model_len=16384 \
  $@