set -x
ENGINE=${1:-vllm}
# VLLM V1 engine rejects XFORMERS; let vLLM pick its default backend.
# NOTE: Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True - vLLM V1's
# CuMemAllocator (needed for free_cache_engine / sleep mode) asserts against it.
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

train_data_size=32 # match GRPO and GiGPO configuration (16 x 8)
val_data_size=64


cd ${verl_agent_repo}

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    --local_dir $project_root/data/verl-agent \

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=${project_root}/data/verl-agent/text/train.parquet \
    data.val_files=${project_root}/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.prompt_length=4096 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.response_length=512 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=auto \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.webshop.use_small=True  \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name='ppo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.val_before_train=True $@

