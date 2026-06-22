#!/bin/bash
# Thin launcher for the federated runner. All logic lives in
# tools/run_federated.py; this wrapper just forwards its arguments.
#
# Runs WITHOUT SLURM by default: simply execute it directly, e.g.
#   bash scripts/smart_federated_runner.sh --verl-config <NAME>
#
# ---------------------------------------------------------------------------
# OPTIONAL: SLURM submission (only if you run on a SLURM cluster).
# These directives are commented out so the script runs standalone. To submit
# this file as a SLURM batch job, uncomment the block below (and adjust the
# partition / CPU / GPU / memory / time values for your cluster), then submit
# with:  sbatch scripts/smart_federated_runner.sh ...
# ---------------------------------------------------------------------------
# #SBATCH -c 75
# #SBATCH --gres=gpu:2
# #SBATCH --mem=768G
# #SBATCH -t 1-00:00:00
# #SBATCH --output=logs/job_%A_%a.out
# #SBATCH --error=logs/job_%A_%a.err
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/../tools/run_federated.py" "$@"
