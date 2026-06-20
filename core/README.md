# Core: federated control plane

FedAgent's first-party control plane. It drives FedAvg / FedProx federated training
over the vendored verl-agent trainer: each round it selects clients, launches one
local-RL subprocess per client, collects their checkpoints, and aggregates them into
the next round's global model. Per-file detail is in
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

| File | Role |
|---|---|
| `custom_fed_server.py` | Server entry point; drives the whole round loop. |
| `fed/round_orchestrator.py` | Per-round scheduling: select clients, launch, collect, aggregate. |
| `fed/script_builder.py` | Renders each client's verl-agent launch script (env vars, partition kwargs, resume paths). |
| `fed/client_runner.py` | Launches and supervises one client's training subprocess. |
| `fed/aggregator.py` | Model aggregation (FedAvg / FedProx), incl. the FSDP-sharded path (with `utils/model_aggregation.py`). |
| `fed/checkpoint_manager.py`, `fed/session_manager.py`, `fed/config_helpers.py` | Checkpoint bookkeeping, resume / session state, config helpers. |
| `fed_ray_ppo_trainer.py`, `ppo_model_wrapper.py`, `extra_metrics.py` | Ray / PPO glue and metric hooks on the control side. |

In practice the entry chain is `tools/run_federated.py` ->
`scripts/start_federated.sh` -> `custom_fed_server.py`.
