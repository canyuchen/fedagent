# Legacy — original FedAgent (verl-agent-0.3.1 fork)

> [!WARNING]
> This is the **original** FedAgent implementation: a vendored, modified fork of
> **verl-agent 0.3.1**. It is **superseded** by the verl-0.8 overlay in
> [`../fedagent/`](../fedagent/README.md) and retained only as historical / paper
> reference. **Do not run this stack** — its entry points (`reproduce.sh`,
> `tools/run_federated.py` → `core/custom_fed_server.py`) target the old architecture and
> the old nested config schema. The maintained entry is `python -m fedagent.fed.run_fed`.

## What's here

| Path | Original role |
|---|---|
| [`core/`](core/) | federated control plane (round orchestrator, aggregator, checkpoint manager, `custom_fed_server`) |
| [`utils/`](utils/) | model aggregation (FedAvg, FSDP) + logging helpers |
| [`eval/`](eval/) | checkpoint evaluation + trajectory-collection harness |
| [`scripts/`](scripts/) | env setup + federated launchers (`start_federated.sh`, `setup_env.sh`, verl-agent launchers) |
| [`config/`](config/) | the paper config matrix in the **old** nested `federated:` / `verl:` / `data_preprocess:` schema |
| [`docs/`](docs/) | the original documentation suite (9 files), incl. [`extending.md`](docs/extending.md) and [`features.md`](docs/features.md) |
| [`tools/`](tools/) | the 0.3.1 runner (`run_federated.py`), config generator, path resolver, aggregation / heterogeneity / monitor toolboxes |
| `reproduce.sh` · `evaluate.sh` · `download_data.sh` | the original convenience entry points (target this legacy stack) |

What changed in the verl-0.8 migration, and why, is documented in
[`../fedagent/docs/migration.md`](../fedagent/docs/migration.md).
