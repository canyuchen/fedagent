#!/usr/bin/env bash
#
# batch_alfworld_eval.sh: roll out a checkpoint over an ALFWorld data split and dump
# per-episode trajectories. Two modes (SPLIT):
#
#   SPLIT=train  (default)  the WHOLE training split ($ALFWORLD_DATA/json_2.1.1/train),
#     in 128-game batches, merged to output/inference/all_trajectories_alfworld.json.
#     This is what the `hardness` partition reads (docs/heterogeneity.md). Passing
#     env.alfworld.start_idx/end_idx makes the inference env read the TRAIN directory
#     and slice game_files[start:end], so batch b covers training games [b*128 :
#     (b+1)*128].
#
#   SPLIT=val               DEFAULT: reproduce the EXACT in-training validation set:
#     the first VAL_SUBSET (default 64) games of the seed-42-shuffled valid_seen,
#     single pass, merged to output/inference/all_trajectories_alfworld_val.json.
#     ALFWorld val is contiguous (worker i -> game_files[i]), so env_num=VAL_SUBSET
#     rolls out game_files[0:VAL_SUBSET] = exactly the games the in-training
#     val/success_rate scores (VAL_SUBSET=64 = data.val_batch_size).
#       Set VAL_TOTAL=N to instead SWEEP the held-out valid_seen split games[0:N] in
#       BATCH_SIZE windows (symmetric with SPLIT=train): ALFWORLD_VAL_WINDOW=1 makes
#       env.alfworld.start_idx/end_idx slice valid_seen (NOT the TRAIN dir; windowed-val
#       branch in alfred_tw_env.py), env_num = the window, START_BATCH resumes. The env
#       prints the real valid_seen count and clamps, so a VAL_TOTAL overshoot just
#       re-rolls a few games in the last batch. valid_seen is small, so a sweep at
#       BATCH_SIZE=128 is ~2 batches; lower BATCH_SIZE (e.g. 64) for finer, memory-bounded ones.
#
# Usage:
#   [VAR=value ...] bash eval/batch_alfworld_eval.sh [ENGINE] [CHECKPOINT] [START_BATCH]
#
#   Examples:
#     bash eval/batch_alfworld_eval.sh vllm /path/to/ckpt                  # full TRAIN split (hardness)
#     SPLIT=val bash eval/batch_alfworld_eval.sh vllm /path/to/ckpt        # exact in-training val (64, default)
#     SPLIT=val VAL_TOTAL=140 bash eval/batch_alfworld_eval.sh vllm /ckpt  # sweep full valid_seen (batched)
#     bash eval/batch_alfworld_eval.sh vllm /ckpt 5                        # resume TRAIN from batch 5
#
#   Positional args:
#     ENGINE       rollout engine                        (default: vllm)
#     CHECKPOINT   model/checkpoint to roll out          (default: Qwen/Qwen2.5-1.5B-Instruct)
#     START_BATCH  resume from this batch (train or val sweep)  (default: 0)
#
#   Env-var knobs (prefix the command, e.g. SPLIT=val VAL_TOTAL=140 bash ...):
#     SPLIT              train (default) | val
#     ALFWORLD_DATA      ALFWorld data root              (default ~/.cache/alfworld)
#     TOTAL_TRAIN_GAMES  train: games under $ALFWORLD_DATA/json_2.1.1/train (default 3553;
#                        check: find "$ALFWORLD_DATA/json_2.1.1/train" -name game.tw-pddl | wc -l).
#     VAL_SUBSET         val (default mode): exact in-training val = the first VAL_SUBSET
#                        games of valid_seen (default 64 = data.val_batch_size).
#     VAL_TOTAL          val: if set, SWEEP valid_seen games[0:VAL_TOTAL] in BATCH_SIZE
#                        windows instead (e.g. 140 = full split).
#     BATCH_SIZE         window size for train and the VAL_TOTAL sweep (default 128).
set -x
SPLIT=${SPLIT:-train}
ENGINE=${1:-vllm}
PRETRAINED_MODEL_PATH=${2:-"Qwen/Qwen2.5-1.5B-Instruct"}
START_BATCH=${3:-0}
TOTAL_TRAIN_GAMES=${TOTAL_TRAIN_GAMES:-3553}
VAL_SUBSET=${VAL_SUBSET:-64}
VAL_TOTAL=${VAL_TOTAL:-}
BATCH_SIZE=${BATCH_SIZE:-128}
export VLLM_ATTENTION_BACKEND=XFORMERS
# ALFWorld reads its data from $ALFWORLD_DATA (default ~/.cache/alfworld; see
# download_data.sh / docs/installation.md).
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"

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
# val_size is BOTH the parallel-worker count AND the number of games rolled out (each
# worker runs game_files[worker_idx]). The placeholder val parquet is sized to match
# so the trainer's val loop is exactly one pass: len(val_dataloader) =
# ceil(val_data_size / val_batch_size) = 1. Sizing it per call keeps a partial final
# batch correct (val_size = the window, so workers don't over-run the game list).
run_infer() {
    local val_size="$1"; local out_dir="$2"; shift 2
    mkdir -p "${out_dir}"
    python3 -m examples.data_preprocess.prepare \
        --mode 'text' --train_data_size 16 --val_data_size ${val_size} \
        --local_dir $project_root/data/verl-agent
    python3 -m verl.trainer.main_ppo_inference \
        algorithm.adv_estimator=gae \
        data.train_files=${project_root}/data/verl-agent/text/train.parquet \
        data.val_files=${project_root}/data/verl-agent/text/test.parquet \
        data.train_batch_size=16 \
        data.val_batch_size=${val_size} \
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
        "$@" \
        +env.save_trajectories=True \
        +env.trajectory_save_dir=${out_dir} \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='verl_agent_alfworld_eval' \
        trainer.experiment_name="alfworld_${SPLIT}_qwen2.5_1.5b" \
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
    val_out="${val_root}/all_trajectories_alfworld_val.json"

    if [ -z "${VAL_TOTAL:-}" ]; then
        # DEFAULT: reproduce the EXACT in-training validation set = the first
        # VAL_SUBSET games of the seed-42-shuffled valid_seen (one shot, no windowing).
        # ALFWorld val is contiguous (worker i -> game_files[i]), so env_num=VAL_SUBSET
        # rolls out game_files[0:VAL_SUBSET] = the same games the in-training
        # val/success_rate scores. VAL_SUBSET=64 = data.val_batch_size. Set VAL_TOTAL=N
        # to sweep the whole split instead.
        echo "[batch-eval] ALFWorld VAL (in-training set): valid_seen games[0:${VAL_SUBSET}] (single run)"
        val_dir="${val_root}/trajectories_alfworld_val"
        rm -rf "${val_dir}"
        run_infer "${VAL_SUBSET}" "${val_dir}"
        python3 ${SCRIPT_DIR}/merge_trajectories.py "${val_dir}" "${val_out}"
        echo "[batch-eval] done -> ${val_out}"
        exit 0
    fi

    # VAL_TOTAL set: SWEEP the held-out valid_seen split in BATCH_SIZE windows
    # (symmetric with SPLIT=train). ALFWORLD_VAL_WINDOW=1 makes env.alfworld.start_idx/
    # end_idx slice the seed-42-shuffled valid_seen game_files (NOT the TRAIN dir; see
    # alfred_tw_env.py), so batch b covers valid_seen games[b*BATCH_SIZE :
    # (b+1)*BATCH_SIZE] and env_num = the window (one game per worker). The env clamps
    # the window to the real game count and prints it, so a VAL_TOTAL overshoot just
    # re-rolls a few games in the last batch (the worker fallback handles it).
    export ALFWORLD_VAL_WINDOW=1
    val_batches=$(( (VAL_TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
    echo "[batch-eval] ALFWorld VAL sweep: valid_seen[0:${VAL_TOTAL}] -> ${val_batches} batches of ${BATCH_SIZE} (start at ${START_BATCH})"
    for batch_idx in $(seq $START_BATCH $((val_batches - 1))); do
        start_idx=$((batch_idx * BATCH_SIZE))
        end_idx=$(( (batch_idx + 1) * BATCH_SIZE ))
        [ $end_idx -gt $VAL_TOTAL ] && end_idx=$VAL_TOTAL
        echo "[batch-eval] val batch $((batch_idx + 1))/${val_batches}: valid_seen games ${start_idx}-$((end_idx - 1))"
        run_infer "$((end_idx - start_idx))" "${val_root}/trajectories_alfworld_val_${batch_idx}" \
            +env.alfworld.start_idx=$start_idx \
            +env.alfworld.end_idx=$end_idx
        if [ $? -ne 0 ]; then
            echo "[batch-eval] val batch ${batch_idx} failed; re-run with START_BATCH=${batch_idx} to resume" >&2
            exit 1
        fi
    done
    echo "[batch-eval] merging val batches -> all_trajectories_alfworld_val.json"
    tmp="${val_root}/_tmp_all_trajectories_alfworld_val"
    mkdir -p ${tmp}
    for batch_idx in $(seq 0 $((val_batches - 1))); do
        cp ${val_root}/trajectories_alfworld_val_${batch_idx}/*.json ${tmp}/ 2>/dev/null || true
    done
    python3 ${SCRIPT_DIR}/merge_trajectories.py "${tmp}" "${val_out}"
    rm -rf ${tmp}
    echo "[batch-eval] done -> ${val_out}"
    exit 0
fi

# SPLIT=train: loop over the whole training split in BATCH_SIZE batches.
total_batches=$(( (TOTAL_TRAIN_GAMES + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "[batch-eval] ALFWorld TRAIN: ${TOTAL_TRAIN_GAMES} games -> ${total_batches} batches of ${BATCH_SIZE} (start at ${START_BATCH})"
mkdir -p ${project_root}/output/inference/trajectories_alfworld

for batch_idx in $(seq $START_BATCH $((total_batches - 1))); do
    start_idx=$((batch_idx * BATCH_SIZE))
    end_idx=$(( (batch_idx + 1) * BATCH_SIZE ))
    [ $end_idx -gt $TOTAL_TRAIN_GAMES ] && end_idx=$TOTAL_TRAIN_GAMES
    echo "[batch-eval] batch $((batch_idx + 1))/${total_batches}: training games ${start_idx}-$((end_idx - 1))"
    run_infer "$((end_idx - start_idx))" "${project_root}/output/inference/trajectories_alfworld_${batch_idx}" \
        +env.alfworld.start_idx=$start_idx \
        +env.alfworld.end_idx=$end_idx
    if [ $? -ne 0 ]; then
        echo "[batch-eval] batch ${batch_idx} failed; re-run with START_BATCH=${batch_idx} to resume" >&2
        exit 1
    fi
done

echo "[batch-eval] merging all batches -> all_trajectories_alfworld.json"
tmp="${project_root}/output/inference/_tmp_all_trajectories_alfworld"
mkdir -p ${tmp}
for batch_idx in $(seq 0 $((total_batches - 1))); do
    cp ${project_root}/output/inference/trajectories_alfworld_${batch_idx}/*.json ${tmp}/ 2>/dev/null || true
done
python3 ${SCRIPT_DIR}/merge_trajectories.py \
    ${tmp} \
    ${project_root}/output/inference/all_trajectories_alfworld.json
rm -rf ${tmp}
echo "[batch-eval] done -> ${project_root}/output/inference/all_trajectories_alfworld.json"
