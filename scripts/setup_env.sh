#!/bin/bash
# Create / update / switch conda envs for FedAgent tasks.
#
# Usage:
#   bash scripts/setup_env.sh list
#   bash scripts/setup_env.sh create  <task> [env_name]
#   bash scripts/setup_env.sh update  <task> [env_name]
#   bash scripts/setup_env.sh switch  <task> [env_name]
#
# <task> is one of: webshop | alfworld
# Default env_name: fedagent-<task>
#
# Notes:
#   - 'create' makes a fresh env (python 3.10) then pip installs the reqs.
#   - 'update' only pip-installs the reqs into an existing env.
#   - 'switch' prints the `conda activate` command for the chosen task.
#     To actually switch in your shell, run:     source scripts/setup_env.sh switch <task>

set -e

ACTION="$1"
TASK="$2"
ENV_NAME="${3:-fedagent-${TASK}}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REQ_FILE="${PROJECT_ROOT}/${TASK}_requirements.txt"

case "$ACTION" in
    list)
        echo "Available task reqs in ${PROJECT_ROOT}:"
        ls "${PROJECT_ROOT}"/*_requirements.txt 2>/dev/null | xargs -n1 basename
        echo ""
        echo "Existing conda envs matching fedagent-*:"
        conda env list 2>/dev/null | awk '/^fedagent-/ {print "  " $1}'
        ;;

    create)
        [ -z "$TASK" ] && { echo "Error: task is required"; exit 1; }
        [ ! -f "$REQ_FILE" ] && { echo "Error: $REQ_FILE not found"; exit 1; }
        echo "Creating conda env: $ENV_NAME (python 3.10)"
        conda create -y -n "$ENV_NAME" python=3.10
        echo "Installing from $REQ_FILE ..."
        conda run -n "$ENV_NAME" pip install -r "$REQ_FILE"
        echo ""
        echo "Done. Activate with:  conda activate $ENV_NAME"
        ;;

    update)
        [ -z "$TASK" ] && { echo "Error: task is required"; exit 1; }
        [ ! -f "$REQ_FILE" ] && { echo "Error: $REQ_FILE not found"; exit 1; }
        echo "Updating $ENV_NAME from $REQ_FILE ..."
        conda run -n "$ENV_NAME" pip install -r "$REQ_FILE"
        echo ""
        echo "Done. Activate with:  conda activate $ENV_NAME"
        ;;

    switch)
        [ -z "$TASK" ] && { echo "Error: task is required"; exit 1; }
        # If sourced, actually activate. Otherwise just print the command.
        if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
            conda activate "$ENV_NAME"
        else
            echo "Run (sourced):  source scripts/setup_env.sh switch $TASK"
            echo "Or directly:    conda activate $ENV_NAME"
        fi
        ;;

    *)
        # Unknown/empty action: print this file's usage header (its lines 2-15,
        # the Usage/Notes block above) as help text, then exit non-zero.
        sed -n '2,15p' "$0"
        exit 1
        ;;
esac
