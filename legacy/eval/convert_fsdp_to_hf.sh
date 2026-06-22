#!/usr/bin/env bash
#
# convert_fsdp_to_hf.sh: merge an FSDP-sharded verl-agent checkpoint into a standard
# HuggingFace model directory that evaluate.sh / from_pretrained can load.
#
# Training and federated aggregation save the actor as FSDP shards
# (model_world_size_*_rank_*.pt) under an 'actor/' dir, because the configs set
# actor.checkpoint.contents=[model] (no 'hf_model'). The eval harness, however,
# loads weights via HF from_pretrained (resume_mode=disable), which needs an
# HF-format directory. This script bridges that gap.
#
# It wraps the vendored upstream merger third_party/verl-agent/scripts/model_merger.py
# and reads the HF config + tokenizer that verl saves next to the shards (config.json
# in the actor dir), so no separate base-model path is needed. It runs on CPU.
#
# Usage:
#   bash eval/convert_fsdp_to_hf.sh <ACTOR_DIR> [TARGET_HF_DIR]
#
#   ACTOR_DIR      a checkpoint 'actor' dir holding model_world_size_*_rank_*.pt, e.g.
#                  <ckpt>/global_step_70/actor  or
#                  <ckpt>/aggregated/checkpoints/global_step_70/actor
#   TARGET_HF_DIR  where to write the merged HF model (default: <ACTOR_DIR>/hf_merged)
set -euo pipefail

ACTOR_DIR="${1:-}"
TARGET_DIR="${2:-}"
if [[ -z "${ACTOR_DIR}" ]]; then
  echo "Usage: bash eval/convert_fsdp_to_hf.sh <ACTOR_DIR> [TARGET_HF_DIR]" >&2
  exit 2
fi
if [[ ! -d "${ACTOR_DIR}" ]]; then
  echo "[convert] not a directory: ${ACTOR_DIR}" >&2
  exit 2
fi
ACTOR_DIR="$(cd "${ACTOR_DIR}" && pwd)"
TARGET_DIR="${TARGET_DIR:-${ACTOR_DIR}/hf_merged}"

if ! ls "${ACTOR_DIR}"/model_world_size_*_rank_0.pt >/dev/null 2>&1; then
  echo "[convert] no FSDP shards (model_world_size_*_rank_0.pt) found in ${ACTOR_DIR}" >&2
  echo "[convert] pass the 'actor' dir of a trained/aggregated checkpoint." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
verl_agent_repo=$(python3 -c "
from omegaconf import OmegaConf
conf = OmegaConf.to_container(OmegaConf.load('${REPO_ROOT}/config/paths.yaml'), resolve=True)
print(conf['repo']['verl_agent'])
")

echo "[convert] FSDP -> HF"
echo "[convert]   from: ${ACTOR_DIR}"
echo "[convert]   to:   ${TARGET_DIR}"
# model_merger.py imports `verl`, so run it from the verl-agent repo root. It reads
# the HF config/tokenizer from --local_dir (config.json saved beside the shards).
cd "${verl_agent_repo}"
python3 scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "${ACTOR_DIR}" \
    --target_dir "${TARGET_DIR}"

echo "[convert] done -> ${TARGET_DIR}"
