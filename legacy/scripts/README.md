# Scripts

Setup and launch scripts. Run them from the repository root.

| Path | What it does |
|---|---|
| `setup_env.sh` | Create / update / switch the per-task conda env, e.g. `bash scripts/setup_env.sh create webshop`. Installs torch first, then the task requirements. |
| `start_federated.sh` | Low-level per-run launcher for the federated server (spawns one local-RL subprocess per client per round). Usually invoked via `tools/run_federated.py`. |
| `smart_federated_runner.sh` | Higher-level orchestrator over `start_federated.sh` (smart resume / multi-run). |
| `verl-agent/{grpo,ppo}/run_{webshop,alfworld}.sh` | The vendored verl-agent base launch scripts each client runs (referenced by a config's `base_script_path`). |
| `plotting/plot_training_dynamics.py` | Plot per-round val / success-rate curves from `round_summary.json`. |

Usual path: `reproduce.sh` or `tools/run_federated.py` -> `start_federated.sh` -> the
per-client `verl-agent/` base script. See [`../docs/running.md`](../docs/running.md).
