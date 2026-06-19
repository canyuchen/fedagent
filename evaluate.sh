#!/usr/bin/env bash
#
# evaluate.sh: evaluate a trained FedAgent checkpoint and collect trajectories.
#
# Thin dispatcher over the eval/ harness scripts (eval/eval_webshop.sh /
# eval/eval_alfworld.sh) that runs a single trained checkpoint against the
# unperturbed environment and dumps per-episode trajectories + aggregate
# metrics (Success Rate / Task Score).
#
# Usage:
#   bash evaluate.sh <webshop|alfworld> <checkpoint> [engine]
#
# Arguments:
#   <webshop|alfworld>   which environment's eval harness to use
#   <checkpoint>         path to a trained model checkpoint directory
#   [engine]             optional rollout engine (default: vllm)
#
# Notes:
#   - Trained / aggregated checkpoints are saved as FSDP shards
#     (model_world_size_*_rank_*.pt), which the HF loader cannot read directly. If
#     <checkpoint> is (or contains) an FSDP-sharded actor dir, this script merges it
#     to HuggingFace format once via eval/convert_fsdp_to_hf.sh and evaluates the
#     merged copy; a base HF model dir or an already-merged dir is used as-is.
#   - Evaluation always runs on the STANDARD (unperturbed) environment so the
#     metric isolates post-aggregation generalization (see docs/reproducing.md).
#   - ALFWorld (eval/eval_alfworld.sh): the held-out valid_seen split.
#   - WebShop (eval/eval_webshop.sh): a slice of the TRAINING pool (goals[500:],
#     via infer_special) for a trajectory dump; this is NOT a held-out test number.
#     The reported WebShop val/success_rate is the in-training validation on the
#     held-out goals[0:500]. For the full training pool (used by the `hardness`
#     partition) run eval/batch_webshop_eval.sh.
#
set -euo pipefail

ENV="${1:-}"
CKPT="${2:-}"
ENGINE="${3:-vllm}"

if [[ -z "${ENV}" || -z "${CKPT}" ]]; then
  echo "Usage: bash evaluate.sh <webshop|alfworld> <checkpoint> [engine]" >&2
  exit 2
fi

case "${ENV}" in
  webshop|alfworld) ;;
  *) echo "Unknown environment: ${ENV} (expected webshop|alfworld)" >&2; exit 2 ;;
esac

if [[ ! -e "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------- #
# FSDP -> HF auto-conversion
# --------------------------------------------------------------------------- #
# Training / aggregation write the actor as FSDP shards (model_world_size_*_rank_*.pt)
# under an 'actor/' dir, not a HuggingFace model (configs set checkpoint.contents=
# [model], no 'hf_model'). The eval harness loads weights via HF from_pretrained
# (resume_mode=disable), so an FSDP checkpoint must be merged to HF first. If CKPT is
# (or contains) exactly one FSDP-sharded actor dir, merge it once and evaluate the
# merged dir; a base HF model dir or an already-merged dir passes through unchanged.
if [[ -d "${CKPT}" ]]; then
  # Only consider the actor (policy) shards. PPO checkpoints also write critic/ shards
  # with the same filename under the same global_step_N; matching both would look like
  # "multiple checkpoints". Restrict to dirs named 'actor'.
  mapfile -t _actor_dirs < <(find "${CKPT}" -maxdepth 5 -type f -name 'model_world_size_*_rank_0.pt' -exec dirname {} \; 2>/dev/null | grep -E '(^|/)actor$' | sort -u)
  if [[ "${#_actor_dirs[@]}" -gt 1 ]]; then
    echo "[evaluate] ${CKPT} contains multiple FSDP checkpoints:" >&2
    printf '  %s\n' "${_actor_dirs[@]}" >&2
    echo "[evaluate] pass a specific .../actor (or .../global_step_N) path." >&2
    exit 2
  elif [[ "${#_actor_dirs[@]}" -eq 1 ]]; then
    _actor="${_actor_dirs[0]}"
    _hf="${_actor}/hf_merged"
    if [[ -f "${_hf}/config.json" ]] && ls "${_hf}"/*.safetensors >/dev/null 2>&1; then
      echo "[evaluate] reusing merged HF checkpoint: ${_hf}"
    else
      echo "[evaluate] FSDP checkpoint detected; merging to HF (one-time)..."
      bash "${REPO_ROOT}/eval/convert_fsdp_to_hf.sh" "${_actor}" "${_hf}"
    fi
    CKPT="${_hf}"
  fi
fi

echo "[evaluate] env        = ${ENV}"
echo "[evaluate] checkpoint = ${CKPT}"
echo "[evaluate] engine     = ${ENGINE}"

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
# Dispatch to the matching eval/ harness. Those scripts take the rollout engine as
# the first positional argument and the checkpoint / model path as the second, and
# internally merge per-episode trajectory shards into a single JSON via
# eval/merge_trajectories.py. They read config/paths.yaml relative to the current
# directory, so run them from the repository root (this script's own directory).
EVAL_SCRIPT="${REPO_ROOT}/eval/eval_${ENV}.sh"

if [[ ! -x "${EVAL_SCRIPT}" && ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Eval harness not found: ${EVAL_SCRIPT}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
echo "[evaluate] exec: bash eval/eval_${ENV}.sh ${ENGINE} ${CKPT}"
exec bash "${EVAL_SCRIPT}" "${ENGINE}" "${CKPT}"
