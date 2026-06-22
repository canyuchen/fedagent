#!/bin/bash
# Launch one federated-learning training run (the FedAgent server in
# core/custom_fed_server.py). The server drives FedAvg-style training: it spawns
# one local-RL training subprocess per client per round and then aggregates the
# resulting model weights into the next round's global model. This script is the
# low-level per-run launcher; tools/run_federated.py (via
# scripts/smart_federated_runner.sh) is the higher-level orchestrator that calls it.
#
# Runs WITHOUT SLURM by default: simply execute it directly, e.g.
#   bash scripts/start_federated.sh --verl-config <NAME>
#
# GPU selection:
#   - Pass --gpus N to use the first N visible GPUs (sets CUDA_VISIBLE_DEVICES
#     to 0,1,...,N-1).
#   - Or export CUDA_VISIBLE_DEVICES yourself before calling this script.
#   - If neither is provided, all visible GPUs are used (the default).
#
# ---------------------------------------------------------------------------
# OPTIONAL: SLURM submission (only if you run on a SLURM cluster).
# These directives are commented out so the script runs standalone. To submit
# this file as a SLURM batch job, uncomment the block below (and adjust the
# partition / CPU / GPU / memory / time values for your cluster), then submit
# with:  sbatch scripts/start_federated.sh ...
# ---------------------------------------------------------------------------
# #SBATCH -c 32
# #SBATCH --gres=gpu:2
# #SBATCH --mem=512G
# #SBATCH -t 1-00:00:00
# #SBATCH --output=logs/job_%A_%a.out
# #SBATCH --error=logs/job_%A_%a.err
# ---------------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESOLVER="$PROJECT_ROOT_DIR/tools/resolve_paths.py"

SMART_RESUME=false
OUTPUT_DIR=""
CONFIG_PATH=""
VERL_CONFIG=""
NO_TIMESTAMP=false
TIMESTAMP=""
GPUS=""

usage() {
    cat >&2 <<EOF
Usage:
  $0 --verl-config NAME [--gpus N] [--no-timestamp] [--timestamp YYYYMMDD_HHMMSS]
  $0 --smart-resume --output-dir DIR --config PATH [--gpus N]
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --smart-resume)  SMART_RESUME=true; shift ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --config)        CONFIG_PATH="$2"; shift 2 ;;
        --verl-config)   VERL_CONFIG="$2"; shift 2 ;;
        --no-timestamp)  NO_TIMESTAMP=true; shift ;;
        --timestamp)     TIMESTAMP="$2"; shift 2 ;;
        --gpus)          GPUS="$2"; shift 2 ;;
        *)
            echo "unknown arg: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# GPU selection (default: all visible GPUs).
# --gpus N selects the first N devices; otherwise respect an externally provided
# CUDA_VISIBLE_DEVICES, and leave it unset (all GPUs) if neither is given.
if [ -n "$GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPUS - 1)))"
fi

if [ "$SMART_RESUME" = true ]; then
    if [ -z "$OUTPUT_DIR" ] || [ -z "$CONFIG_PATH" ]; then
        echo "ERROR: --smart-resume requires --output-dir and --config" >&2
        usage
        exit 1
    fi
    output_dir="$OUTPUT_DIR"
    training_config_path="$CONFIG_PATH"
    echo "Smart resume mode:"
    echo "  Output directory: $output_dir"
    echo "  Config path:      $training_config_path"
else
    if [ -z "$VERL_CONFIG" ]; then
        echo "ERROR: --verl-config required (or use --smart-resume)" >&2
        usage
        exit 1
    fi
    # --verl-config names a top-level training config; config/paths.yaml supplies
    # the config root and output root, and tools/resolve_paths.py joins them with
    # the name (<config.root>/<NAME>.yaml) and derives OUTPUT_DIR under <output.root>.
    # That resolver is the single source of truth so this launcher and
    # run_federated.py agree; it prints shell `NAME=value` assignment lines
    # (CONFIG_FILE, OUTPUT_DIR, META_INFO, ...) that we eval below.
    resolver_args=(--verl-config "$VERL_CONFIG")
    [ "$NO_TIMESTAMP" = true ] && resolver_args+=(--no-timestamp)
    [ -n "$TIMESTAMP" ] && resolver_args+=(--timestamp "$TIMESTAMP")
    if ! resolved=$(python3 "$RESOLVER" "${resolver_args[@]}"); then
        echo "ERROR: path resolver failed" >&2
        exit 1
    fi
    eval "$resolved"
    output_dir="$OUTPUT_DIR"
    training_config_path="$CONFIG_FILE"
    mkdir -p "$output_dir"
    echo "Normal mode:"
    echo "  Verl config:      $VERL_CONFIG"
    echo "  Output directory: $output_dir"
    echo "  Config path:      $training_config_path"
fi

echo "Starting Federated Learning Training..."
echo "======================================"
echo "Starting federated learning at $(date)" | tee "$output_dir/startup.log"

server_args=(--output-dir "$output_dir" --config "$training_config_path")
[ "$SMART_RESUME" = true ] && server_args=(--smart-resume "${server_args[@]}")

PYTHONPATH="${PROJECT_ROOT_DIR}:${PYTHONPATH}" python3 \
    "${PROJECT_ROOT_DIR}/core/custom_fed_server.py" "${server_args[@]}" \
    2>&1 | tee "$output_dir/server.log"
python_exit_code=${PIPESTATUS[0]}

if [ "$python_exit_code" -ne 0 ]; then
    echo "Error: Federated learning training failed with exit code $python_exit_code"
    echo "Check the logs in $output_dir for more details"
    exit "$python_exit_code"
fi

echo "Federated learning completed at $(date)" | tee -a "$output_dir/startup.log"
echo "Training completed! Check logs in $output_dir"
echo "Summary file: $output_dir/federated_training_summary.json"
