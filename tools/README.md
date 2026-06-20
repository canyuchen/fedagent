# Tools

Operational entry points and helpers for federated runs, config generation,
aggregation diagnostics, and heterogeneity data.

| Path | What it does |
|---|---|
| `run_federated.py` | CLI front-end (`--smart` / `--restart-resume` / direct) that resolves paths and launches the federated server. |
| `resolve_paths.py` | Single source of truth for output-dir / run-name derivation from a config. |
| `generate_uniform_configs.py` | Generate the curated config matrix. |
| `verify_train_val_disjoint.py` | Sanity check that the train and val task splits do not overlap. |
| `aggregation/` | Standalone aggregation verification / diagnostics (`check_aggregation.py`, `verify_aggregation.py`, `verl_fsdp_aggregation.py`, `create_fsdp_shards.py`, `fix_dtensor_loading.py`). |
| `env_heterogeneity/` | Generate the env-level holdout / distractor sets (`gen_holdout_{webshop,alfworld}.py`). |
| `heterogeneity_test/` | Task-level partition simulations + federated smoke tests (see its README). |
| `monitor/` | Live run / checkpoint health monitor (`checkpoint_monitor.py`). |

See [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for how these fit the
control plane.
