#!/usr/bin/env bash
#
# batch_webshop_eval.sh: roll out a checkpoint over a WebShop data split and dump
# per-episode trajectories. Two modes (SPLIT):
#
#   SPLIT=train  (default)  the WHOLE training pool, in 128-goal batches, merged to
#     output/inference/all_trajectories.json. This is what the `hardness` partition
#     reads (docs/heterogeneity.md). WebShop trains on catalog goals[500:]; with
#     env.webshop.infer_special the env offsets start/end by +500, so batch b covers
#     goals[500 + b*128 : 500 + (b+1)*128]. Batching is needed (and possible) here
#     because the pool is large and start/end select an index window.
#
#   SPLIT=val               DEFAULT: reproduce the EXACT in-training validation set =
#     goals[0:VAL_SUBSET] (default 64), single pass, merged to
#     output/inference/all_trajectories_webshop_val.json. The FEDERATED trainer's
#     WebShop val set is goals[0:64] (fed_env_manager passes val_batch_size=64 ->
#     goal_idxs=range(64)), so VAL_SUBSET=64 rolls out exactly the goals the reported
#     val/success_rate is computed on. The default uses the windowed-val path
#     (env.webshop.start_idx=0/end_idx=VAL_SUBSET, no infer_special -> range(0,
#     VAL_SUBSET)); goals[0:N] indexes the seed-42-shuffled goal list, so it is
#     identical across clients/runs. (Note: the standalone make_envs path leaves the
#     env's val_batch_size at its 500 default, which is why we pass start/end here
#     instead of relying on a bare run; a bare run would draw a scattered 64-of-500,
#     NOT goals[0:64].)
#       Set VAL_TOTAL=N to instead SWEEP the held-out pool goals[0:N] in BATCH_SIZE
#       windows (symmetric with SPLIT=train); VAL_TOTAL=500 = the entire pool; max 500.
#
# Usage:
#   [VAR=value ...] bash eval/batch_webshop_eval.sh [ENGINE] [CHECKPOINT] [START_BATCH]
#
#   Examples:
#     bash eval/batch_webshop_eval.sh vllm /path/to/ckpt                  # full TRAIN pool (hardness)
#     SPLIT=val bash eval/batch_webshop_eval.sh vllm /path/to/ckpt        # exact in-training val (64, default)
#     SPLIT=val VAL_TOTAL=500 bash eval/batch_webshop_eval.sh vllm /ckpt  # sweep full held-out pool (batched)
#     bash eval/batch_webshop_eval.sh vllm /ckpt 12                       # resume TRAIN from batch 12
#
#   Positional args:
#     ENGINE       rollout engine                        (default: vllm)
#     CHECKPOINT   model/checkpoint to roll out          (default: Qwen/Qwen2.5-1.5B-Instruct)
#     START_BATCH  resume from this batch (train or val sweep)  (default: 0)
#
#   Env-var knobs (prefix the command, e.g. SPLIT=val VAL_TOTAL=500 bash ...):
#     SPLIT              train (default) | val
#     TOTAL_TRAIN_GOALS  train: len(goals) - 500 (default 6410; the shipped small
#                        catalog with use_small=True; env prints the count at startup).
#     VAL_SUBSET         val (default mode): exact in-training val = goals[0:VAL_SUBSET]
#                        (default 64 = data.val_batch_size; the federated val set).
#     VAL_TOTAL          val: if set, SWEEP held-out goals[0:VAL_TOTAL] in BATCH_SIZE
#                        windows instead (e.g. 500 = full pool; max 500).
#     BATCH_SIZE         window size for train and the VAL_TOTAL sweep (default 128).
set -x
SPLIT=${SPLIT:-train}
ENGINE=${1:-vllm}
PRETRAINED_MODEL_PATH=${2:-"Qwen/Qwen2.5-1.5B-Instruct"}
START_BATCH=${3:-0}
TOTAL_TRAIN_GOALS=${TOTAL_TRAIN_GOALS:-6410}
VAL_SUBSET=${VAL_SUBSET:-64}
VAL_TOTAL=${VAL_TOTAL:-}
BATCH_SIZE=${BATCH_SIZE:-128}
export VLLM_ATTENTION_BACKEND=XFORMERS

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

cd ${verl_agent_repo}

# One inference call. Args: <val_batch_size> <trajectory_dir> [extra +env args...]
# val_size is BOTH the parallel-env count AND the number of goals rolled out (the env
# draws this many from the selected goal window). The placeholder val parquet is sized
# to match so the trainer's val loop is exactly one pass: len(val_dataloader) =
# ceil(val_data_size / val_batch_size) = 1. Sizing it per call (rather than once) keeps
# a partial final batch correct: passing val_size = the window avoids drawing more
# envs than the window has goals (np.random.choice replace=False would otherwise throw).
run_infer() {
    local val_size="$1"; local out_dir="$2"; shift 2
    mkdir -p "${out_dir}"
    python3 -m examples.data_preprocess.prepare \
        --mode 'text' --train_data_size 4 --val_data_size ${val_size} \
        --local_dir $project_root/data/verl-agent
    python3 -m verl.trainer.main_ppo_inference \
        algorithm.adv_estimator=gae \
        data.train_files=${project_root}/data/verl-agent/text/train.parquet \
        data.val_files=${project_root}/data/verl-agent/text/test.parquet \
        data.train_batch_size=4 \
        data.val_batch_size=${val_size} \
        data.max_prompt_length=4096 \
        data.max_response_length=512 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$PRETRAINED_MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=4 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.use_invalid_action_penalty=True \
        actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path=$PRETRAINED_MODEL_PATH \
        critic.model.enable_gradient_checkpointing=False \
        critic.ppo_micro_batch_size_per_gpu=2 \
        critic.model.fsdp_config.param_offload=False \
        critic.model.fsdp_config.optimizer_offload=True \
        algorithm.use_kl_in_reward=False \
        env.env_name=Webshop \
        env.seed=0 \
        env.max_steps=15 \
        env.webshop.use_small=True \
        "$@" \
        +env.save_trajectories=True \
        +env.trajectory_save_dir=${out_dir} \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='verl_agent_webshop_eval' \
        trainer.experiment_name="webshop_${SPLIT}_qwen2.5_1.5b" \
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
}

if [ "$SPLIT" = "val" ]; then
    val_root="${project_root}/output/inference"
    val_out="${val_root}/all_trajectories_webshop_val.json"

    if [ -z "${VAL_TOTAL:-}" ]; then
        # DEFAULT: reproduce the EXACT in-training validation set = goals[0:VAL_SUBSET]
        # via the windowed-val path (start/end without infer_special -> goal_idxs =
        # range(0,VAL_SUBSET)). The FEDERATED trainer's WebShop val is goals[0:64]
        # (fed_env_manager passes val_batch_size=64), so VAL_SUBSET=64 matches the
        # goals the reported val/success_rate is computed on. Set VAL_TOTAL=N to sweep.
        echo "[batch-eval] WebShop VAL (in-training set): goals[0:${VAL_SUBSET}] (single run)"
        val_dir="${val_root}/trajectories_webshop_val"
        rm -rf "${val_dir}"
        run_infer "${VAL_SUBSET}" "${val_dir}" \
            +env.webshop.start_idx=0 \
            +env.webshop.end_idx="${VAL_SUBSET}"
        python3 ${SCRIPT_DIR}/merge_trajectories.py "${val_dir}" "${val_out}"
        echo "[batch-eval] done -> ${val_out}"
        exit 0
    fi

    # VAL_TOTAL set: SWEEP the held-out validation pool goals[0:VAL_TOTAL] in
    # BATCH_SIZE windows (symmetric with SPLIT=train). Each batch passes start/end
    # WITHOUT infer_special, so the env slices goals[start:end] of the val pool (no
    # +500 offset; windowed-val path in envs.py). VAL_TOTAL=500 = the whole pool.
    if [ "$VAL_TOTAL" -gt 500 ]; then
        echo "[batch-eval] WARNING: VAL_TOTAL=${VAL_TOTAL} > 500; clamping to 500 (goals>=500 are TRAINING goals, not held-out val)." >&2
        VAL_TOTAL=500
    fi
    val_batches=$(( (VAL_TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
    echo "[batch-eval] WebShop VAL sweep: held-out goals[0:${VAL_TOTAL}] -> ${val_batches} batches of ${BATCH_SIZE} (start at ${START_BATCH})"
    for batch_idx in $(seq $START_BATCH $((val_batches - 1))); do
        start_idx=$((batch_idx * BATCH_SIZE))
        end_idx=$(( (batch_idx + 1) * BATCH_SIZE ))
        [ $end_idx -gt $VAL_TOTAL ] && end_idx=$VAL_TOTAL
        echo "[batch-eval] val batch $((batch_idx + 1))/${val_batches}: held-out goals ${start_idx}-$((end_idx - 1))"
        run_infer "$((end_idx - start_idx))" "${val_root}/trajectories_webshop_val_${batch_idx}" \
            +env.webshop.start_idx=$start_idx \
            +env.webshop.end_idx=$end_idx
        if [ $? -ne 0 ]; then
            echo "[batch-eval] val batch ${batch_idx} failed; re-run with START_BATCH=${batch_idx} to resume" >&2
            exit 1
        fi
    done
    echo "[batch-eval] merging val batches -> all_trajectories_webshop_val.json"
    tmp="${val_root}/_tmp_all_trajectories_webshop_val"
    mkdir -p ${tmp}
    for batch_idx in $(seq 0 $((val_batches - 1))); do
        cp ${val_root}/trajectories_webshop_val_${batch_idx}/*.json ${tmp}/ 2>/dev/null || true
    done
    python3 ${SCRIPT_DIR}/merge_trajectories.py "${tmp}" "${val_out}"
    rm -rf ${tmp}
    echo "[batch-eval] done -> ${val_out}"
    exit 0
fi

# SPLIT=train: loop over the whole training pool in BATCH_SIZE batches.
total_batches=$(( (TOTAL_TRAIN_GOALS + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "[batch-eval] WebShop TRAIN: ${TOTAL_TRAIN_GOALS} goals -> ${total_batches} batches of ${BATCH_SIZE} (start at ${START_BATCH})"
mkdir -p ${project_root}/output/inference/trajectories

for batch_idx in $(seq $START_BATCH $((total_batches - 1))); do
    start_idx=$((batch_idx * BATCH_SIZE))
    end_idx=$(( (batch_idx + 1) * BATCH_SIZE ))
    [ $end_idx -gt $TOTAL_TRAIN_GOALS ] && end_idx=$TOTAL_TRAIN_GOALS
    echo "[batch-eval] batch $((batch_idx + 1))/${total_batches}: training goals ${start_idx}-$((end_idx - 1)) (catalog indices $((500 + start_idx))-$((500 + end_idx - 1)))"
    run_infer "$((end_idx - start_idx))" "${project_root}/output/inference/trajectories_${batch_idx}" \
        +env.webshop.infer_special=True \
        +env.webshop.start_idx=$start_idx \
        +env.webshop.end_idx=$end_idx
    if [ $? -ne 0 ]; then
        echo "[batch-eval] batch ${batch_idx} failed; re-run with START_BATCH=${batch_idx} to resume" >&2
        exit 1
    fi
done

echo "[batch-eval] merging all batches -> all_trajectories.json"
tmp="${project_root}/output/inference/_tmp_all_trajectories"
mkdir -p ${tmp}
for batch_idx in $(seq 0 $((total_batches - 1))); do
    cp ${project_root}/output/inference/trajectories_${batch_idx}/*.json ${tmp}/ 2>/dev/null || true
done
python3 ${SCRIPT_DIR}/merge_trajectories.py \
    ${tmp} \
    ${project_root}/output/inference/all_trajectories.json
rm -rf ${tmp}
echo "[batch-eval] done -> ${project_root}/output/inference/all_trajectories.json"
