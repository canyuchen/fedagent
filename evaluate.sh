#!/usr/bin/env bash
#
# evaluate.sh — evaluate a trained FedAgent checkpoint and collect trajectories.
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
#   - Evaluation always runs on the STANDARD (unperturbed) environment so the
#     metric isolates post-aggregation generalization (see docs/reproducing.md).
#   - WebShop test set: goals[0:500]; ALFWorld: valid_seen + valid_unseen.
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

echo "[evaluate] env        = ${ENV}"
echo "[evaluate] checkpoint = ${CKPT}"
echo "[evaluate] engine     = ${ENGINE}"

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
# Dispatch to the matching eval/ harness. Those scripts take the rollout engine
# as the first positional argument and the checkpoint / model path as the
# second, and internally merge the per-episode trajectory shards into a single
# JSON via eval/merge_trajectories.py. They read config/paths.yaml relative to
# the current directory, so run them from the repository root (this script's
# own directory).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${REPO_ROOT}/eval/eval_${ENV}.sh"

if [[ ! -x "${EVAL_SCRIPT}" && ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Eval harness not found: ${EVAL_SCRIPT}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
echo "[evaluate] exec: bash eval/eval_${ENV}.sh ${ENGINE} ${CKPT}"
exec bash "${EVAL_SCRIPT}" "${ENGINE}" "${CKPT}"
