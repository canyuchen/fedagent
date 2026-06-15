# Running Experiments — Hardware & Scaling

This document is the power-user reference for the hardware and parallelism modes
FedAgent supports. The [README](../README.md) covers only the happy path (the
paper default: 4 × H100 + the `webshop-main` config); everything here maps a run
*mode* to a concrete `reproduce.sh` flag **and** to the underlying config knob it
overrides, so you can either drive it through `reproduce.sh` or set the same knob
by hand in the YAML.

> **Before you run anything:** FedAgent uses **two separate conda environments**
> — `fedagent-webshop` and `fedagent-alfworld` — because the two benchmarks have
> conflicting dependencies. Every `reproduce.sh` / `evaluate.sh` invocation must
> happen *inside the matching env*. See
> [`docs/installation.md`](installation.md) for the full setup.

## How a run is actually launched

Knowing the launch path makes the rest of this document concrete.
`reproduce.sh` resolves a named experiment (e.g. `webshop-main`) to a canonical
YAML config and then drives the federated stack:

```
reproduce.sh <experiment>
  └─ tools/run_federated.py            # Python runner: modes, resume, monitor
       └─ scripts/start_federated.sh   # per-run launcher (GPU masking, paths)
            └─ core/custom_fed_server.py
                 ├─ partitions the data into N clients
                 ├─ for each round: samples M clients, runs E local epochs each
                 │    via the verl-agent base script (federated.base_script_path,
                 │    e.g. scripts/verl-agent/grpo/run_webshop.sh)
                 └─ aggregates the client models (utils/model_aggregation.py)
```

`scripts/start_federated.sh` is the real launcher; it accepts `--verl-config NAME`
(a config path under `config/`, **without** the `.yaml` suffix) and `--gpus N`,
and resolves output paths through `tools/resolve_paths.py` + `config/paths.yaml`.
Each client's local RL update is run by the verl-agent trainer
(`verl.trainer.main_ppo`, which serves both PPO and GRPO via `adv_estimator`).

The **config YAML is the source of truth** for the hardware knobs below
(`verl.trainer.n_gpus_per_node`, rollout tensor-parallel size, FSDP offload,
`federated.training.parallel_workers`). The base verl-agent scripts under
`scripts/verl-agent/` are *templates* with their own hard-coded single-GPU
defaults (e.g. `trainer.n_gpus_per_node=1`); the federated server overrides them
from the YAML at launch, so to change hardware you change the YAML (or pass a
`reproduce.sh` flag that does it for you), not those templates.

## Default configuration

The paper happy path needs no flags:

```bash
conda activate fedagent-webshop
bash reproduce.sh webshop-main          # == --gpus 4, fed, FSDP per-config, torchrun
```

- **Hardware:** 4 × NVIDIA H100 (80 GB) on a single node.
- **Rollout:** vLLM with `actor_rollout_ref.rollout.tensor_model_parallel_size: 4`
  and `gpu_memory_utilization: 0.5`.
- **Trainer:** `verl.trainer.n_gpus_per_node: 4`, `nnodes: 1`.
- **Precision:** bf16 throughout (vLLM rollout + FSDP-sharded actor/ref).
- **Federation:** `total_clients: 100`, `clients_per_round: 2`,
  `epochs_per_client: 3`, `total_rounds: 70` (= 210 total local epochs),
  `parallel_workers: 4`.

The same defaults back `alfworld-main` (run it from the `fedagent-alfworld` env).

## Run-mode matrix

Each mode lists **what it changes**, the **config knob** behind it, and the
**`reproduce.sh` flag** that toggles it.

### 1. Client execution: parallel vs serial

- **What:** whether the *M* clients sampled in a round train concurrently (one
  worker process per client) or one after another.
- **Knob:** `federated.training.parallel_workers` (e.g. `4` = up to 4 concurrent
  clients; `1` = strictly serial).
- **Flag:** `--mode fed` (parallel, default) / `--mode serial` (sets
  `parallel_workers = 1`).
- **Note on the effective degree of parallelism:** concurrency is bounded by the
  number of clients in the round, i.e. `min(parallel_workers, clients_per_round)`.
  With the main protocol's `clients_per_round: 2`, at most **2** clients run at
  once even though the shipped configs set `parallel_workers: 4`. The headroom
  matters only for ablations that raise `clients_per_round` (see
  [`docs/reproducing.md`](reproducing.md) and the `decentralized` configs).
  The `centralized/` and `local_client*/` configs use `clients_per_round: 1`, so
  they are serial in practice regardless of the `parallel_workers` value.
- **When to use serial:** limited GPU count (each concurrent client needs its own
  slice of devices), or to isolate a single client's training for debugging.

### 2. FSDP parameter / optimizer offload on/off

- **What:** whether FSDP offloads parameters and/or optimizer state to CPU to fit
  larger backbones (or larger batches) into GPU memory, trading throughput for
  capacity.
- **Knobs:**
  - `verl.actor_rollout_ref.actor.fsdp_config.param_offload`
  - `verl.actor_rollout_ref.actor.fsdp_config.optimizer_offload`
  - For PPO, the critic mirrors these under `verl.critic.fsdp_config.*`.
  - The reference policy is offloaded by default
    (`verl.actor_rollout_ref.ref.fsdp_config.param_offload: true`) because it is
    only used for log-probs and never updated.
- **Paper defaults:** the shipped configs (including 7B) set the actor — and the
  PPO critic — to `param_offload: false` and `optimizer_offload: false`, i.e.
  **offload is off** for the trainable modules on H100; only the frozen reference
  policy is offloaded. Offload is the lever you turn **on** when a backbone or
  batch no longer fits.
- **Flag:** `--fsdp on|off` (`on` enables `param_offload`; with no flag the
  config defaults stand).
- **When to use:** enable offload to run a larger backbone or longer rollouts on
  tighter memory; leave it off for maximum throughput on H100.

### 3. Single-GPU

- **What:** collapse rollout tensor-parallelism and put everything on one device
  for smoke / debug runs.
- **Knobs:** `verl.actor_rollout_ref.rollout.tensor_model_parallel_size = 1` and
  `verl.trainer.n_gpus_per_node = 1` (and, on the launcher side,
  `CUDA_VISIBLE_DEVICES` masked to a single device).
- **Flag:** `--single-gpu` (forces 1 GPU regardless of `--gpus`).
- **When to use:** developing or verifying the pipeline end-to-end. A single-GPU
  run is **not** paper-scale — expect to also drop `clients_per_round`, batch
  sizes, and/or `total_rounds`, and very likely to enable offload (mode 2).

### 4. Variable GPU count (`--gpus N`)

- **What:** how many GPUs on the node a run uses. This sets both the device mask
  (`start_federated.sh --gpus N` exports `CUDA_VISIBLE_DEVICES=0,…,N-1`) and the
  trainer/rollout parallelism.
- **Knobs:** `verl.trainer.n_gpus_per_node` and, normally,
  `verl.actor_rollout_ref.rollout.tensor_model_parallel_size` (vLLM tensor-parallel
  size must divide the model's attention heads and is typically set equal to
  `N`).
- **Flag:** `--gpus N` (default `4`).
- **Note:** because TP and the number of concurrent clients (mode 1) both consume
  GPUs, with parallel clients you want roughly `N ≥ tensor_model_parallel_size ×
  (clients running concurrently)`. If a round can't fit, lower `clients_per_round`
  / `parallel_workers` or `tensor_model_parallel_size`.

### 5. Multi-node

- **What:** shard a single client's training across more than one node.
- **Knobs:** `verl.trainer.n_gpus_per_node` (GPUs per node) and
  `verl.trainer.nnodes` (node count). The shipped configs use `nnodes: 1`.
- **Flag:** not exposed through `reproduce.sh` — multi-node is a config-only path.
  Set `verl.trainer.nnodes` (and the matching multi-node launch — a Ray cluster /
  multi-node SLURM allocation that verl/verl-agent already understands) directly in
  the config and launcher.
- **When to use:** very large backbones or long rollouts that exceed a single
  node. The paper's results are all single-node (`nnodes: 1`).

### 6. Launcher: local (default) vs SLURM

- **What:** how the per-run launcher is dispatched.
- **Default — non-SLURM:** `scripts/start_federated.sh` runs directly on the
  current box; `--gpus N` masks the first `N` visible GPUs (otherwise all visible
  GPUs are used, or whatever `CUDA_VISIBLE_DEVICES` you export). No scheduler is
  assumed, so the repo runs on a bare multi-GPU machine out of the box. (The
  per-client trainer is launched as `python3 -m verl.trainer.main_ppo` and lets
  verl/Ray manage intra-node distribution — there is no literal `torchrun` in the
  training path; `torchrun` is used only by the offline FSDP aggregation
  utilities under `tools/aggregation/`, despite the loose "torchrun" wording in
  `reproduce.sh`'s comments.)
- **SLURM:** both `scripts/start_federated.sh` and
  `scripts/smart_federated_runner.sh` ship with a commented `#SBATCH` header
  (partition/CPU/GPU/memory/time). To submit as a batch job, **uncomment that
  block, adjust the resource values for your cluster, and `sbatch` the script.**
  The cluster-specific partition and `--gres`/`--mem` values from the original
  internal scripts are intentionally parameterized out — you must fill in your own.
- **Flag:** `--slurm` takes the SLURM path: `reproduce.sh` execs
  `sbatch scripts/start_federated.sh`. The non-SLURM default execs
  `bash scripts/start_federated.sh` directly on the current box.

## Flag → knob summary

| `reproduce.sh` flag | Underlying config knob (YAML) | Launcher effect | Default |
|---|---|---|---|
| `--gpus N` | `verl.trainer.n_gpus_per_node`; usually `verl.actor_rollout_ref.rollout.tensor_model_parallel_size` | `start_federated.sh --gpus N` → `CUDA_VISIBLE_DEVICES=0..N-1` | `4` |
| `--mode fed` / `--mode serial` | `federated.training.parallel_workers` (`fed`=config value, `serial`=`1`) | concurrent vs sequential client workers | `fed` |
| `--fsdp on` / `--fsdp off` | `...actor.fsdp_config.param_offload` (+ `optimizer_offload`; PPO also `critic.fsdp_config.*`) | CPU offload of trainable modules | per-config (off for actor/critic; ref offloaded) |
| `--single-gpu` | `n_gpus_per_node = 1`, `rollout.tensor_model_parallel_size = 1` | one masked device | off |
| (no flag) `nnodes` | `verl.trainer.nnodes` | multi-node allocation (Ray/SLURM) | `1` |
| `--slurm` | — (launcher backend) | `sbatch` the launcher (uncomment its `#SBATCH` block) vs run locally | off (local/torchrun) |

> **How overrides are applied:** for `--gpus` / `--mode` / `--fsdp` /
> `--single-gpu`, `reproduce.sh` writes an overridden copy of the chosen config
> under `config/_reproduce_generated/` (via OmegaConf), then launches the runner
> on that generated config. Without those flags it runs the canonical config
> unchanged.

## Worked examples

```bash
# Paper default (WebShop main, GRPO, 4xH100) — from the fedagent-webshop env
bash reproduce.sh webshop-main

# Clients run one-at-a-time (less peak GPU memory, slower wall clock)
bash reproduce.sh webshop-main --mode serial

# 2-GPU box: halve the device count (set TP/n_gpus accordingly)
bash reproduce.sh webshop-main --gpus 2

# 1-GPU smoke/debug run (collapses rollout TP; pair with smaller config knobs)
bash reproduce.sh webshop-main --single-gpu

# Tight memory: turn FSDP parameter offload on
bash reproduce.sh webshop-main --fsdp on

# ALFWorld on a SLURM cluster — from the fedagent-alfworld env
conda activate fedagent-alfworld
bash reproduce.sh alfworld-main --slurm
```

To evaluate a trained checkpoint on the **unperturbed** environment and dump
trajectories + aggregate metrics (run from the matching env):

```bash
bash evaluate.sh webshop  /path/to/checkpoint
bash evaluate.sh alfworld /path/to/checkpoint
```

## Compute budget

For planning a run, the headline numbers (full reproduction recipe and per-sweep
breakdown live in [`docs/reproducing.md`](reproducing.md)):

| Sweep | Approx. GPU-hours | Wall clock (4 × H100) |
|---|---|---|
| WebShop, GRPO  | ~93  | ~24 h |
| WebShop, PPO   | ~117 | — |
| ALFWorld, GRPO | ~117 | — |
| ALFWorld, PPO  | ~140 | — |

Total reported compute across all experiments is **~1,800 H100 GPU-hours**.
Serial-client mode (mode 1) and FSDP offload (mode 2) both *increase* wall-clock
time relative to these figures — they trade speed for fitting on fewer / smaller
GPUs.

## Validating an aggregation change

If you change how client models are combined, the FSDP-aware FedAvg / FedProx
implementation lives in `utils/model_aggregation.py`, and the diagnostic toolbox
is `tools/aggregation/` — in particular `check_aggregation.py` and
`verify_aggregation.py`. Run those after any aggregation change to confirm the
merged checkpoint is numerically correct before launching a full sweep. See
[`docs/extending.md`](extending.md) for the aggregation extension point.

## See also

- [`docs/installation.md`](installation.md) — the two-conda-env setup (required).
- [`docs/reproducing.md`](reproducing.md) — per-experiment recipes, seeds, and the
  full compute breakdown.
- [`docs/configuration.md`](configuration.md) — decoder for config filenames and a
  field reference for the `federated:` and `verl:` blocks.
- [`docs/heterogeneity.md`](heterogeneity.md) — the two-level heterogeneity suite
  (task-level vs environment-level) and how to select each variant.
