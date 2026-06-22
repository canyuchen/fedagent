#!/bin/bash
# Evaluate a (trained) checkpoint on the ALFWorld environment and collect
# per-episode trajectories.
#
# What it runs on: the held-out valid_seen split (eval_id_data_path). To collect
# trajectories over the TRAINING games for the `hardness` partition, use
# eval/batch_alfworld_eval.sh instead.
#
# Usage:
#   bash eval/eval_alfworld.sh [ENGINE] [PRETRAINED_MODEL_PATH]
#
# Arguments:
#   ENGINE                 rollout engine to use (default: vllm)
#   PRETRAINED_MODEL_PATH  path to the model / checkpoint to evaluate.
#                          Defaults to the base "Qwen/Qwen2.5-1.5B-Instruct";
#                          pass a local checkpoint directory to evaluate a
#                          trained FedAgent model.
set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Resolve this script's directory BEFORE any `cd`, so we can locate the
# trajectory-merging helper after switching into the verl-agent repo.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

paths_file="./config/paths.yaml"

read_yaml_path() {
    python3 -c "
from omegaconf import OmegaConf
conf = OmegaConf.load('$paths_file')
conf = OmegaConf.to_container(conf, resolve=True)
print(conf$1)
"
}

project_root=$(read_yaml_path "['project_root']")
verl_agent_repo=$(read_yaml_path "['repo']['verl_agent']")

# Evaluation configuration.
train_data_size=16 # Inference-only run (total_epochs=0): this only sizes the
                   # placeholder train.parquet that data_preprocess.prepare emits
                   # (it select()s this many rows). It is NOT a GRPO rollout group
                   # size. The number of episodes actually evaluated is val_data_size
                   # below.
val_data_size=32

# Pretrained model / checkpoint path. Override via the second positional
# argument to evaluate a trained checkpoint instead of the base model.
PRETRAINED_MODEL_PATH=${2:-"Qwen/Qwen2.5-1.5B-Instruct"}

cd ${verl_agent_repo}

# Data preprocessing (only used to indicate the modality and the data size).
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    --local_dir $project_root/data/verl-agent \

# Create the directory where rollout trajectories will be saved.
mkdir -p ${project_root}/output/inference/trajectories_alfworld

# Run inference using main_ppo_inference.py.
python3 -m verl.trainer.main_ppo_inference \
    algorithm.adv_estimator=gae \
    data.train_files=${project_root}/data/verl-agent/text/train.parquet \
    data.val_files=${project_root}/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$PRETRAINED_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$PRETRAINED_MODEL_PATH \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    +env.save_trajectories=True \
    +env.trajectory_save_dir=${project_root}/output/inference/trajectories_alfworld \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_alfworld_inference' \
    trainer.experiment_name='inference_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=0 \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.log_val_generations=10 \
    trainer.rollout_data_dir=${project_root}/output/inference \
    trainer.validation_data_dir=${project_root}/output/inference

echo "Inference complete. Results saved to: ${project_root}/output/inference"

# Merge all per-episode trajectories into a single JSON file.
echo "Merging trajectory files..."
python3 ${SCRIPT_DIR}/merge_trajectories.py \
    ${project_root}/output/inference/trajectories_alfworld \
    ${project_root}/output/inference/all_trajectories_alfworld.json
