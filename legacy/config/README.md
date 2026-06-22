# Configuration

> [!WARNING]
> **Legacy (verl-agent-0.3.1) configs — do not run these with the verl-0.8 runner.** These
> YAMLs use the original nested `federated:` / `verl:` / `data_preprocess:` schema consumed by
> `tools/run_federated.py`, retained as the paper's experiment ground-truth. The **maintained**
> configs are the verl-0.8 overlay's, under [`../fedagent/config/`](../../fedagent/config/README.md),
> driven by `python -m fedagent.fed.run_fed` — the **same 176-config matrix, mirrored 1:1** in
> structure + naming (only the file *contents* differ). See
> [`../fedagent/docs/migration.md`](../../fedagent/docs/migration.md).

Every experiment is a single YAML with three top-level blocks: `federated:`
(federation / aggregation / partition), `verl:` (the verl-agent trainer, passed
through to each client), and `data_preprocess:` (dataset sharding). The full field
reference is [`../docs/configuration.md`](../docs/configuration.md); a fully
annotated template is [`example.yaml`](example.yaml).

## First-time setup

```bash
cp paths.yaml.example paths.yaml   # then edit project_root / repo paths
```

`paths.yaml` is machine-specific and gitignored; `tools/resolve_paths.py` reads it.

## Layout

| Path | What's in it |
|---|---|
| `uniform/` | Main experiments, one dir per backbone (`Qwen2.5-1.5B/3B/7B-Instruct`, `Llama-3.2-3B-Instruct`); uniform (IID) task partition. |
| `task_heterogeneity/{grpo,ppo}/` | Task-level heterogeneity sweeps (Preference / Coverage / Hardness). |
| `env_heterogeneity/<variant>[_ppo]/` | Environment-level heterogeneity (`catalog_split`, `bm25_reweighting`, `lookalike_injection`, `rank_wrapper`, `field_subset_index`). |
| `decentralized/` | Federation-protocol sweeps (clients/round, local epochs, tasks/client). |
| `example.yaml` | Annotated schema reference. |
| `paths.yaml.example` | Path template (copy to `paths.yaml`). |

Configs are produced/curated by `tools/generate_uniform_configs.py`. Pick one and run
it via `reproduce.sh` or `tools/run_federated.py` (see
[`../docs/running.md`](../docs/running.md) and
[`../docs/reproducing.md`](../docs/reproducing.md)).
