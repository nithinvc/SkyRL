set -x

# Dev training run: VLM RL on VisGym maze_2d/easy with Qwen3-VL-2B.
#
# Non-colocated setup: 2 GPUs for vLLM inference, 2 GPUs for FSDP training.
# Uses the Tinker generator (multi-modal) via custom entrypoint.
#
# Prerequisites:
#   uv run examples/train/visgym/visgym_dataset.py --output_dir ~/data/visgym
#
# Usage:
#   bash examples/train/visgym/run_visgym.sh

: "${DATA_DIR:="$HOME/data/visgym"}"
: "${NUM_INFERENCE_GPUS:=2}"
: "${NUM_TRAIN_GPUS:=2}"
: "${LOGGER:=console}"

_SKYRL_USE_NEW_INFERENCE=1 \
uv run --isolated --extra fsdp \
  python examples/train/visgym/visgym_entrypoint.py \
  data.train_data="['$DATA_DIR/train.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-VL-2B-Instruct" \
  trainer.placement.colocate_all=false \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_TRAIN_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_TRAIN_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.async_engine=true \
  trainer.epochs=3 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=256 \
  generator.sampling_params.temperature=1.0 \
  generator.max_turns=10 \
  generator.max_input_length=4096 \
  generator.n_samples_per_prompt=4 \
  generator.batched=false \
  trainer.algorithm.use_kl_loss=true \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  environment.env_class=visgym \
  trainer.logger="$LOGGER" \
  trainer.project_name="visgym_maze2d" \
  trainer.run_name="visgym_maze2d_dev" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.ckpt_path="$HOME/ckpts/visgym_maze2d_dev" \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=-1 \
  $@
