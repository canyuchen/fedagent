# Running FedAgent

This is the reference for running FedAgent on your own hardware. A run is a single
**config YAML** handed to the federated runner; the config is the source of truth
for the hardware and parallelism knobs (GPU count, tensor-parallel size, FSDP
offload, client concurrency). This guide shows how to launch a run directly and
maps each run *mode* to the config knob behind it.

> **Two conda environments.** FedAgent uses `fedagent-webshop` and
> `fedagent-alfworld` (the two benchmarks have conflicting dependencies). Run every
> command inside the env that matches your benchmark. See
> [`installation.md`](installation.md).

> To reproduce the paper's specific table/figure runs with a one-command wrapper,
> see [`reproducing.md`](reproducing.md).

## How a run is launched

```
tools/run_federated.py             # Python runner: modes, resume, monitoring
  └─ scripts/start_federated.sh    # per-run launcher (GPU masking, path resolution)
       └─ core/custom_fed_server.py
            ├─ partitions the data into N clients
            ├─ for each round: samples M clients, runs E local epochs each via the
            │    verl-agent base script (federated.base_script_path, e.g.
            │    scripts/verl-agent/grpo/run_webshop.sh)
            └─ aggregates the client models (utils/model_aggregation.py)
```

`scripts/start_federated.sh` is the real launcher; it accepts `--verl-config NAME`
(a config path under `config/`, **without** the `.yaml` suffix) and `--gpus N`, and
resolves output paths through `tools/resolve_paths.py` + `config/paths.yaml`. Each
client's local RL update is run by the federated verl-agent trainer
(`verl.trainer.main_ppo_fed`, which serves both PPO and GRPO via `adv_estimator`).

The **config YAML is the source of truth** for every hardware knob below. The base
verl-agent scripts under `scripts/verl-agent/` are *templates* with their own
single-GPU defaults (e.g. `trainer.n_gpus_per_node=1`); the federated server
overrides them from the YAML at launch, so to change hardware you change the YAML,
not those templates.

## Launching a run

Hand the runner a config name (its path under `config/`, without `.yaml`) and a
round count:

```bash
conda activate fedagent-webshop

# re-runnable: creates/auto-detects the output dir and resumes where it left off
python tools/run_federated.py --restart-resume \
  uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform 70
```

For a single launch (no resume loop), the lower-level launcher takes the same
config name plus a device count:

```bash
bash scripts/start_federated.sh --verl-config <config-name> --gpus 4
```

(`scripts/smart_federated_runner.sh` is a thin wrapper around the Python runner.)

## Default configuration

The paper happy path is the canonical config as shipped:

- **Hardware:** 4 × NVIDIA H100 (80 GB) on a single node.
- **Rollout:** vLLM with `actor_rollout_ref.rollout.tensor_model_parallel_size: 4`
  and `gpu_memory_utilization: 0.5`.
- **Trainer:** `verl.trainer.n_gpus_per_node: 4`, `nnodes: 1`.
- **Precision:** bf16 throughout (vLLM rollout + FSDP-sharded actor/ref).
- **Federation:** `total_clients: 100`, `clients_per_round: 2`,
  `epochs_per_client: 3`, `total_rounds: 70` (= 210 total local epochs),
  `parallel_workers: 4`.

## Changing hardware: the config knobs

To run on different hardware, set the knobs below in your config YAML (copy the
closest config or [`config/example.yaml`](../config/example.yaml) and edit). A
copy-and-patch script is in the next section.

### 1. Client execution: parallel vs serial

- **What:** whether the *M* clients sampled in a round train concurrently (one
  worker process per client) or one after another.
- **Knob:** `federated.training.parallel_workers` (`4` = up to 4 concurrent
  clients; `1` = strictly serial).
- **Effective concurrency** is `min(parallel_workers, clients_per_round)`, and is
  further bounded by GPUs. With the main protocol's `clients_per_round: 2`, at most
  **2** clients run at once even though the shipped configs set `parallel_workers:
  4`; the headroom matters only for ablations that raise `clients_per_round`. The
  `centralized/` and `local_client*/` configs use `clients_per_round: 1`, so they
  are serial regardless.
- **When to go serial:** limited GPU count (each concurrent client needs its own
  device slice), or to isolate one client for debugging.

**Example** (serial, in the config):

```yaml
federated:
  training:
    parallel_workers: 1     # serial; >1 (default 4) runs clients concurrently
```

### 2. FSDP parameter / optimizer offload

- **What:** whether FSDP offloads parameters and/or optimizer state to CPU to fit
  larger backbones (or batches) into GPU memory, trading throughput for capacity.
- **Knobs:**
  - `verl.actor_rollout_ref.actor.fsdp_config.param_offload`
  - `verl.actor_rollout_ref.actor.fsdp_config.optimizer_offload`
  - For PPO, the critic mirrors these under `verl.critic.fsdp_config.*`.
  - The reference policy is offloaded by default
    (`verl.actor_rollout_ref.ref.fsdp_config.param_offload: true`); it is only used
    for log-probs and never updated.
- **Defaults:** the shipped configs (including 7B) set the actor, and the PPO
  critic, to `param_offload: false` and `optimizer_offload: false`, so offload is
  **off** for the trainable modules on H100; only the frozen reference policy is
  offloaded. Turn offload on when a backbone or batch no longer fits.
- **When to enable:** to run a larger backbone or longer rollouts on tighter
  memory; leave it off for maximum throughput on H100.

**Example** (turn offload on for the actor):

```yaml
verl:
  actor_rollout_ref:
    actor:
      fsdp_config:
        param_offload: true
        optimizer_offload: true
```

### 3. GPU count (including single-GPU)

- **What:** how many GPUs on the node a run uses, and the rollout tensor-parallel
  degree.
- **Knobs:** `verl.trainer.n_gpus_per_node` and, normally,
  `verl.actor_rollout_ref.rollout.tensor_model_parallel_size` (the vLLM
  tensor-parallel size must divide the model's attention heads and is typically set
  equal to the GPU count). For a **single-GPU** smoke/debug run, set both to `1`.
  At launch, `start_federated.sh --gpus N` masks devices
  (`CUDA_VISIBLE_DEVICES=0..N-1`); keep `--gpus N` consistent with
  `n_gpus_per_node`.
- **Note:** because tensor-parallel size and the number of concurrent clients both
  consume GPUs, with parallel clients you want roughly `N >= tensor_model_parallel_size
  * (clients running concurrently)`. If a round cannot fit, lower
  `clients_per_round` / `parallel_workers` or `tensor_model_parallel_size`.
- A single-GPU run is **not** paper-scale; expect to also drop `clients_per_round`,
  batch sizes, and/or `total_rounds`, and likely to enable offload (mode 2).

**Example** (single GPU): set the knobs, then match `--gpus`:

```yaml
verl:
  trainer:
    n_gpus_per_node: 1
  actor_rollout_ref:
    rollout:
      tensor_model_parallel_size: 1
```

```bash
bash scripts/start_federated.sh --verl-config <config-name> --gpus 1
```

### 4. Multi-node

- **What:** shard a single client's training across more than one node.
- **Knobs:** `verl.trainer.n_gpus_per_node` (GPUs per node) and
  `verl.trainer.nnodes` (node count; the shipped configs use `1`). Multi-node also
  needs the matching launch (a Ray cluster or multi-node SLURM allocation, which
  verl/verl-agent already understands).
- **When:** very large backbones or long rollouts that exceed a single node. The
  paper's results are all single-node.

**Example** (2 nodes × 8 GPUs):

```yaml
verl:
  trainer:
    nnodes: 2
    n_gpus_per_node: 8
```

### 5. SLURM vs local

- **Local (default):** `scripts/start_federated.sh` runs on the current box;
  `--gpus N` masks the first `N` visible GPUs. No scheduler is assumed, so the repo
  runs on a bare multi-GPU machine. (The per-client trainer is launched as
  `python3 -m verl.trainer.main_ppo_fed` and lets verl/Ray manage intra-node
  distribution.)
- **SLURM:** `scripts/start_federated.sh` and `scripts/smart_federated_runner.sh`
  ship with a commented `#SBATCH` header (partition / CPU / GPU / memory / time).
  Uncomment and adjust it for your cluster, then submit:
  ```bash
  sbatch scripts/start_federated.sh --verl-config <config-name> --gpus N
  ```
  The cluster-specific partition and `--gres` / `--mem` values are intentionally
  left for you to fill in.

## Hardware-knob summary

| Knob (YAML) | Effect | Default |
|---|---|---|
| `verl.trainer.n_gpus_per_node` | GPUs per node | `4` |
| `verl.actor_rollout_ref.rollout.tensor_model_parallel_size` | vLLM tensor-parallel degree (usually = GPU count) | `4` |
| `verl.trainer.nnodes` | node count (multi-node) | `1` |
| `federated.training.parallel_workers` | concurrent client workers (`1` = serial; effective = `min(workers, clients_per_round)`) | `4` |
| `...actor.fsdp_config.param_offload` / `optimizer_offload` | CPU offload of trainable modules (PPO also `critic.fsdp_config.*`) | `false` (ref policy offloaded) |
| `start_federated.sh --gpus N` | masks `CUDA_VISIBLE_DEVICES=0..N-1` at launch (keep equal to `n_gpus_per_node`) | `4` |

## A minimal config override

To run an existing config on different hardware without hand-editing it, copy it
and patch the few knobs with OmegaConf, then launch the generated config:

```bash
python - <<'PY'
from omegaconf import OmegaConf
base = "config/uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml"
c = OmegaConf.load(base)
OmegaConf.update(c, "verl.trainer.n_gpus_per_node", 1, force_add=True)
OmegaConf.update(c, "verl.actor_rollout_ref.rollout.tensor_model_parallel_size", 1, force_add=True)
OmegaConf.update(c, "federated.training.parallel_workers", 1, force_add=True)              # serial
OmegaConf.update(c, "verl.actor_rollout_ref.actor.fsdp_config.param_offload", True, force_add=True)  # offload on
OmegaConf.save(c, "config/_custom/webshop_1gpu.yaml")   # _custom/ is a gitignored scratch dir
PY

python tools/run_federated.py --restart-resume _custom/webshop_1gpu 70
```

## Worked examples

```bash
# Paper default (WebShop main, GRPO, 4 GPUs), from the fedagent-webshop env
python tools/run_federated.py --restart-resume \
  uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform 70

# ALFWorld main, from the fedagent-alfworld env
conda activate fedagent-alfworld
python tools/run_federated.py --restart-resume \
  uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_alfworld_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform 70

# Single launch on 2 GPUs (the config's n_gpus_per_node / TP should match)
bash scripts/start_federated.sh --verl-config <config-name> --gpus 2
```

To evaluate a trained checkpoint on the **unperturbed** environment and dump
trajectories plus aggregate metrics (run from the matching env):

```bash
bash evaluate.sh webshop  /path/to/checkpoint
bash evaluate.sh alfworld /path/to/checkpoint
```

A trained checkpoint is FSDP-sharded; `evaluate.sh` auto-converts it to HuggingFace
format on first use (or run `eval/convert_fsdp_to_hf.sh` manually). See
[eval/README.md](../eval/README.md).

## Compute budget

For planning a run (the full reproduction recipe and per-sweep breakdown live in
[`reproducing.md`](reproducing.md)):

| Sweep | Approx. GPU-hours | Wall clock (4 × H100) |
|---|---|---|
| WebShop, GRPO  | ~93  | ~24 h |
| WebShop, PPO   | ~117 | ~30 h |
| ALFWorld, GRPO | ~117 | ~30 h |
| ALFWorld, PPO  | ~140 | ~36 h |

Total reported compute across all experiments is **~1,800 H100 GPU-hours**. Serial
clients and FSDP offload both increase wall-clock time relative to these figures;
they trade speed for fitting on fewer / smaller GPUs.

## Validating an aggregation change

If you change how client models are combined, the FSDP-aware FedAvg / FedProx
implementation lives in `utils/model_aggregation.py`, and the diagnostic toolbox is
`tools/aggregation/` (in particular `check_aggregation.py` and
`verify_aggregation.py`). Run those after any aggregation change to confirm the
merged checkpoint is numerically correct before launching a full sweep. See
[`extending.md`](extending.md) for the aggregation extension point.

## See also

- [`installation.md`](installation.md): the two-conda-env setup (required).
- [`reproducing.md`](reproducing.md): the paper grid (every table/figure mapped to
  its config), the one-command reproduction wrapper, seeds, and the full compute
  breakdown.
- [`configuration.md`](configuration.md): the decoder for config filenames and a
  field reference for the `federated:` and `verl:` blocks.
- [`heterogeneity.md`](heterogeneity.md): the two-level heterogeneity suite
  (task-level vs environment-level) and how to select each variant.
