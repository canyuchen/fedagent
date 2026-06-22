# `tools/`

Migration + operations tooling for the verl-0.8 FedAgent overlay.

- [`verl08_migration/`](verl08_migration/) — the maintained toolbox:
  - `aggregate_fedavg_fsdp.py` — the FSDP-sharded FedAvg aggregator invoked by
    [`fedagent/fed/run_fed.py`](../fedagent/fed/run_fed.py) each round.
  - `gen_paper_configs.py` — regenerates the 176-config paper matrix under
    [`fedagent/config/paper/`](../fedagent/config/README.md).
  - `gen_hardness_trajectories.py` — generates the `data/hardness/*.json` task-difficulty
    labels the hardness heterogeneity arm requires.
  - `summarize_fed_run.py` · `collect_fed_logs.sh` · `eval_alfworld_by_tasktype.py` —
    log / metrics / eval helpers.

The original verl-agent-0.3.1 tooling (`run_federated.py`, `resolve_paths.py`,
`generate_uniform_configs.py`, and the `aggregation/`, `env_heterogeneity/`,
`heterogeneity_test/`, `monitor/` toolboxes) has been archived to
[`../legacy/tools/`](../legacy/tools/).
