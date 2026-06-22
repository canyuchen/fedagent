#!/usr/bin/env bash
#
# reproduce.sh: one-command reproduction entry point for FedAgent.
#
# This is the entry point the README's 'Reproducing the paper' section uses. It resolves a
# named experiment to its canonical config, applies the requested hardware /
# parallelism overrides, and launches the federated runner.
#
# Usage:
#   bash reproduce.sh <experiment> [--gpus N] [--mode fed|serial]
#                                  [--fsdp on|off] [--single-gpu] [--slurm]
#
# Defaults (paper happy path):
#   - 4 x NVIDIA H100 (80 GB) on a single node
#   - non-SLURM: launched directly via scripts/start_federated.sh (cluster
#     users add --slurm to submit the same script through sbatch)
#   - the WebShop main GRPO config (Qwen2.5-1.5B-Instruct)
#
# Experiments:
#   webshop-main    WebShop  main table, GRPO, p-uniform, Qwen2.5-1.5B-Instruct
#   alfworld-main   ALFWorld main table, GRPO, p-uniform, Qwen2.5-1.5B-Instruct
#
# Flag -> config knob:
#   --gpus N        verl.trainer.n_gpus_per_node + rollout tensor_model_parallel_size
#                   (also limits CUDA_VISIBLE_DEVICES to the first N devices)   [default 4]
#   --mode          federated.training.parallel_workers (fed=N / serial=1)
#   --fsdp          verl.actor_rollout_ref.actor.fsdp_config.param_offload (on=True / off=False)
#   --single-gpu    n_gpus_per_node=1, tensor_model_parallel_size=1 (implies --gpus 1)
#   --slurm         submit the launcher via sbatch instead of running it locally
#
# How the overrides are applied
# ------------------------------
# The runner (tools/run_federated.py -> scripts/start_federated.sh ->
# core/custom_fed_server.py) takes ONLY a config name; it has no command-line
# dotlist override. Every knob above is read straight out of the YAML config at
# run time (core/fed/script_builder.py, round_orchestrator.py, aggregator.py).
# So when any override flag is given, this script copies the canonical config to
# a generated leaf under config/_reproduce_generated/, edits the few keys with
# OmegaConf, and hands the runner that generated config name. With no override
# flags it runs the canonical config directly.
#
set -euo pipefail

# Run from the repository root: the runner resolves both ./config/paths.yaml and
# ./config/<name>.yaml relative to the current directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
EXPERIMENT="${1:-}"
GPUS=4
MODE="fed"          # fed | serial
FSDP=""             # ""(leave config default) | on | off
SINGLE_GPU=0
USE_SLURM=0

if [[ -z "${EXPERIMENT}" || "${EXPERIMENT}" == "-h" || "${EXPERIMENT}" == "--help" ]]; then
  sed -n '2,40p' "$0"   # print the usage header above
  exit 0
fi
shift || true

# --------------------------------------------------------------------------- #
# Parse flags
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)       GPUS="$2"; shift 2 ;;
    --mode)       MODE="$2"; shift 2 ;;
    --fsdp)       FSDP="$2"; shift 2 ;;
    --single-gpu) SINGLE_GPU=1; shift ;;
    --slurm)      USE_SLURM=1; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

case "${MODE}" in
  fed|serial) ;;
  *) echo "--mode must be 'fed' or 'serial' (got '${MODE}')" >&2; exit 2 ;;
esac
if [[ -n "${FSDP}" && "${FSDP}" != "on" && "${FSDP}" != "off" ]]; then
  echo "--fsdp must be 'on' or 'off' (got '${FSDP}')" >&2; exit 2
fi

# --------------------------------------------------------------------------- #
# Resolve experiment -> canonical config name (relative to config/, no .yaml)
# --------------------------------------------------------------------------- #
# The runner looks the name up as ./config/<name>.yaml (resolve_paths.py /
# tools/run_federated.py). These are the real canonical leaves:
#   config/uniform/Qwen2.5-1.5B-Instruct/main/grpo/<file>.yaml
CONFIG_DIR="uniform/Qwen2.5-1.5B-Instruct/main/grpo"
case "${EXPERIMENT}" in
  webshop-main)
    CONFIG="${CONFIG_DIR}/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform"
    ;;
  alfworld-main)
    CONFIG="${CONFIG_DIR}/fed_alfworld_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform"
    ;;
  *)
    echo "Unknown experiment: ${EXPERIMENT}" >&2
    echo "Known: webshop-main, alfworld-main" >&2
    exit 2
    ;;
esac

CANONICAL_YAML="config/${CONFIG}.yaml"
if [[ ! -f "${CANONICAL_YAML}" ]]; then
  echo "Canonical config not found: ${CANONICAL_YAML}" >&2
  echo "(run from the repository root; check the config/ tree)" >&2
  exit 2
fi

# Single-GPU implies 1 GPU regardless of --gpus.
if [[ "${SINGLE_GPU}" -eq 1 ]]; then
  GPUS=1
fi

echo "[reproduce] experiment = ${EXPERIMENT}"
echo "[reproduce] config     = ${CONFIG}"
echo "[reproduce] gpus       = ${GPUS}  mode=${MODE}  fsdp=${FSDP:-<config default>}  slurm=${USE_SLURM}"

# --------------------------------------------------------------------------- #
# Apply overrides (only if any flag departs from the canonical config)
# --------------------------------------------------------------------------- #
# parallel_workers: fed -> N (default 4), serial -> 1. NOTE: in this code path
# the *effective* client concurrency is GPU-bound (available_gpus //
# n_gpus_per_node in core/fed/round_orchestrator.py); parallel_workers is the
# documented federated.training knob and is written here to match that contract.
if [[ "${MODE}" == "serial" ]]; then
  PARALLEL_WORKERS=1
else
  PARALLEL_WORKERS=4
fi

# Decide whether we must materialise an overridden config.
NEED_OVERRIDE=0
[[ "${GPUS}" -ne 4 ]] && NEED_OVERRIDE=1
[[ "${MODE}" != "fed" ]] && NEED_OVERRIDE=1
[[ -n "${FSDP}" ]] && NEED_OVERRIDE=1
[[ "${SINGLE_GPU}" -eq 1 ]] && NEED_OVERRIDE=1

RUN_CONFIG="${CONFIG}"   # the name handed to the runner

if [[ "${NEED_OVERRIDE}" -eq 1 ]]; then
  GEN_REL_DIR="_reproduce_generated"
  GEN_DIR="config/${GEN_REL_DIR}"
  mkdir -p "${GEN_DIR}"
  GEN_BASE="$(basename "${CONFIG}")_g${GPUS}_${MODE}${FSDP:+_fsdp-${FSDP}}"
  RUN_CONFIG="${GEN_REL_DIR}/${GEN_BASE}"
  GEN_YAML="config/${RUN_CONFIG}.yaml"

  # tensor_model_parallel_size tracks the GPU count for these single-node configs
  # (canonical = 4-way TP on 4 GPUs; --single-gpu / --gpus N scale it down).
  TP_SIZE="${GPUS}"

  # param_offload: only set when --fsdp was passed; otherwise keep config default.
  if [[ "${FSDP}" == "on" ]]; then PARAM_OFFLOAD="True"
  elif [[ "${FSDP}" == "off" ]]; then PARAM_OFFLOAD="False"
  else PARAM_OFFLOAD=""; fi

  echo "[reproduce] generating overridden config: ${GEN_YAML}"
  CANONICAL_YAML="${CANONICAL_YAML}" GEN_YAML="${GEN_YAML}" \
  N_GPUS="${GPUS}" TP_SIZE="${TP_SIZE}" \
  PARALLEL_WORKERS="${PARALLEL_WORKERS}" PARAM_OFFLOAD="${PARAM_OFFLOAD}" \
  python3 - <<'PY'
import os
from omegaconf import OmegaConf

src = os.environ["CANONICAL_YAML"]
dst = os.environ["GEN_YAML"]
conf = OmegaConf.load(src)

n_gpus = int(os.environ["N_GPUS"])
tp = int(os.environ["TP_SIZE"])
pw = int(os.environ["PARALLEL_WORKERS"])
param_offload = os.environ.get("PARAM_OFFLOAD", "")

# --gpus / --single-gpu
OmegaConf.update(conf, "verl.trainer.n_gpus_per_node", n_gpus, force_add=True)
OmegaConf.update(conf, "verl.actor_rollout_ref.rollout.tensor_model_parallel_size", tp, force_add=True)

# --mode (federated.training.parallel_workers)
OmegaConf.update(conf, "federated.training.parallel_workers", pw, force_add=True)

# --fsdp (only when explicitly requested)
if param_offload != "":
    OmegaConf.update(
        conf,
        "verl.actor_rollout_ref.actor.fsdp_config.param_offload",
        param_offload == "True",
        force_add=True,
    )

OmegaConf.save(conf, dst)
print(f"  wrote {dst}: n_gpus_per_node={n_gpus}, tensor_model_parallel_size={tp}, "
      f"parallel_workers={pw}" + (f", param_offload={param_offload}" if param_offload else ""))
PY
fi

# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #
# The runner's normal single-run path is scripts/start_federated.sh, which
# resolves the config name via tools/resolve_paths.py, sets up the output dir,
# and runs core/custom_fed_server.py. --gpus N additionally pins
# CUDA_VISIBLE_DEVICES to the first N devices. For SLURM the script ships with a
# commented #SBATCH block and is submitted with sbatch (see its header).
LAUNCHER="scripts/start_federated.sh"
LAUNCH_ARGS=(--verl-config "${RUN_CONFIG}" --gpus "${GPUS}")

if [[ "${USE_SLURM}" -eq 1 ]]; then
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "[reproduce] --slurm given but 'sbatch' not found on PATH" >&2
    exit 2
  fi
  echo "[reproduce] submitting via SLURM: sbatch ${LAUNCHER} ${LAUNCH_ARGS[*]}"
  # NOTE: scripts/start_federated.sh ships its #SBATCH directives commented out
  # so it also runs standalone; uncomment/adjust them (partition, --gres, mem,
  # time) for your cluster before relying on sbatch resource allocation.
  exec sbatch "${LAUNCHER}" "${LAUNCH_ARGS[@]}"
else
  echo "[reproduce] launching locally: bash ${LAUNCHER} ${LAUNCH_ARGS[*]}"
  exec bash "${LAUNCHER}" "${LAUNCH_ARGS[@]}"
fi
