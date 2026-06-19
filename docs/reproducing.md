# Reproducing the Paper

This guide maps every figure and table in the paper to the exact config
directory, run command, and protocol that produces it. All numbers, seeds, and
cadences below are taken from the paper's Experimental Setup appendix and were
cross-checked against the released config tree under
[`config/`](../config/).

Read this together with:

- [`installation.md`](installation.md), the **two** conda environments
  (`fedagent-webshop`, `fedagent-alfworld`) you must create first.
- [`running.md`](running.md), how to run any config directly and the hardware
  knobs (GPU count, FSDP, client parallelism) behind the `reproduce.sh` flags below.
- [`heterogeneity.md`](heterogeneity.md), the two-level heterogeneity taxonomy
  that the task-level and env-level experiments instantiate.
- [`configuration.md`](configuration.md), the field-by-field config reference.

---

## Overview

### How experiments are organized

Every experiment is a single YAML config under `config/`. The directory
groups, and the paper artifacts they back, are:

| Config group | Backs | Section |
|---|---|---|
| [`config/uniform/`](../config/uniform/) | Table 1 (GRPO), the PPO appendix table, and the main training-dynamics figure | [Main table](#1-main-table--configuniform) |
| [`config/task_heterogeneity/`](../config/task_heterogeneity/) | The task-level heterogeneity figure (6 panels) | [Task-level heterogeneity](#2-task-level-heterogeneity--configtask_heterogeneity) |
| [`config/env_heterogeneity/`](../config/env_heterogeneity/) | The environment-level heterogeneity figure (GRPO + PPO) | [Environment-level heterogeneity](#3-environment-level-heterogeneity--configenv_heterogeneity) |
| [`config/decentralized/`](../config/decentralized/) | The decentralized / hyperparameter-sensitivity ablations | [Decentralized ablations](#4-decentralized--hyperparameter-sensitivity-ablations--configdecentralized) |

A config's filename is self-describing. For example:

```
fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform.yaml
        │       │      │           │          │         │              │           └ partition strategy
        │       │      │           │          │         │              └ |X_i| (min goals/client)
        │       │      │           │          │         └ epochs per client per round (E)
        │       │      │           │          └ communication rounds (T)
        │       │      │           └ clients sampled per round (M)
        │       │      └ total clients (N)
        │       └ RL algorithm (grpo | ppo)
        └ benchmark (webshop | alfworld)
```

The token at the end (`p-uniform`, `p-preference_omega-0.99`, `p-catalog_split_...`,
etc.) names the data-partition / environment-perturbation strategy and is the
only thing that changes between heterogeneity sweep points.

### How to launch a run

There are two equivalent entry points. The named wrapper drives this stack:

```
reproduce.sh <experiment>          # resolve a named experiment (+ flags) -> config
  └─ scripts/start_federated.sh    # per-run launcher (GPU masking, path resolution)
       └─ core/custom_fed_server.py
            ├─ partitions the data into N clients
            ├─ each round: samples M clients, runs E local epochs each via the
            │    verl-agent base script (federated.base_script_path)
            └─ aggregates the client models (utils/model_aggregation.py)
```

(`tools/run_federated.py`, entry point 2 below, is the re-runnable direct runner;
it launches the same `scripts/start_federated.sh`.)

**1. Named happy-path experiments**: `reproduce.sh` exposes two curated names
for the main-table GRPO runs on the default backbone:

```bash
conda activate fedagent-webshop
bash reproduce.sh webshop-main      # WebShop,  main table, GRPO, Qwen2.5-1.5B

conda activate fedagent-alfworld
bash reproduce.sh alfworld-main     # ALFWorld, main table, GRPO, Qwen2.5-1.5B
```

with optional hardware flags:

```bash
bash reproduce.sh <experiment> [--gpus N] [--mode fed|serial]
                               [--fsdp on|off] [--single-gpu] [--slurm]
```

The default is **4 × H100 (80 GB) on a single node, non-SLURM**; pass `--slurm`
on a cluster. Each flag is a convenience that overrides a config knob:

| Flag | Overrides (config knob) |
|---|---|
| `--gpus N` | `verl.trainer.n_gpus_per_node` + rollout `tensor_model_parallel_size` |
| `--mode serial` | `federated.training.parallel_workers = 1` |
| `--fsdp on` / `off` | actor (and PPO critic) `fsdp_config.param_offload` |
| `--single-gpu` | `n_gpus_per_node = 1`, `tensor_model_parallel_size = 1` |
| `--slurm` | submit the launcher via `sbatch` instead of running locally |

[`running.md`](running.md) documents these knobs in full and shows how to run any
config directly, without `reproduce.sh`.

> **Note.** `reproduce.sh` wires `webshop-main` and `alfworld-main` as named
> shortcuts (plus the `--gpus` / `--mode` / `--fsdp` / `--single-gpu` / `--slurm`
> overrides). To run any other config, point the federated runner at it directly
> (next paragraph).

**2. Any config, directly**: every other cell in every table/figure is run by
handing its **config name** to the federated runner. The config name is the path
**under `config/`, without the `.yaml`** suffix, passed as a positional argument:

```bash
# init + resume-loop an experiment (re-runnable; auto-detects the output dir)
python tools/run_federated.py --restart-resume <config-name> <N_iters>

# example: the WebShop main-table GRPO run
python tools/run_federated.py --restart-resume \
  uniform/Qwen2.5-1.5B-Instruct/main/grpo/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform 70
```

`tools/run_federated.py` is the runner: it resolves the config name via
`tools/resolve_paths.py`, sets up the output directory, and launches per-client
training through `scripts/start_federated.sh`. (`scripts/smart_federated_runner.sh`
is a thin wrapper around it.) The config is a **positional** argument;
`--config` / `--output-dir` apply only to the explicit `--direct-resume` mode.
Activate the conda environment that matches the benchmark in the filename
(`fed_webshop_*` → `fedagent-webshop`, `fed_alfworld_*` → `fedagent-alfworld`).

### Shared federation protocol

Unless an experiment overrides it (the decentralized ablations do, on purpose),
every run uses the default protocol, reflected directly in the config:

```yaml
federated:
  total_clients: 100        # N
  clients_per_round: 2      # M
  total_rounds: 70          # T   -> 210 total local epochs
  epochs_per_client: 3      # E
  aggregation_method: fedavg
  data_sharding:
    min_goals_per_client: 100   # |X_i|
    partition: { strategy: uniform }
```

- `N = 100` total clients; `M = 2` sampled per round; `E = 3` local epochs per
  selected client; `T = 70` rounds → **210 total local epochs**.
- `|X_i| = 100` task instructions per client (pools may overlap).
- Each local epoch draws minibatches of **64 tasks with replacement** (RL
  sampling, not a full traversal).
- Backbone for all sweeps: **Qwen2.5-1.5B-Instruct** (`actor_rollout_ref.model.path:
  Qwen/Qwen2.5-1.5B-Instruct`). The main table additionally reports
  Qwen2.5-3B, Qwen2.5-7B, and Llama-3.2-3B-Instruct.
- Benchmarks: **WebShop** and **ALFWorld**. RL algorithms: **GRPO** (main text)
  and **PPO** (appendix).
- Aggregation: **FedAvg** by default (FedProx is also available; see
  [Extension points](#extension-points)).
- Federated and centralized results report **mean ± std over 3 seeds**.

> Weights & Biases logging has been removed from this release; no W&B account or
> key is required.

---

## 1. Main table → `config/uniform/`

**Backs:** Table 1 (GRPO) and its PPO-appendix counterpart, plus the main
training-dynamics figure (`main_combined_val_success_rate.pdf`), which overlays
the Qwen2.5-1.5B FedAgent and Centralized validation-success curves.

### Layout

```
config/uniform/
  <model>/                       # Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct,
                                 # Qwen2.5-7B-Instruct, Llama-3.2-3B-Instruct
    main/        {grpo,ppo}/     # FedAgent     (seed 0)
    main_seed1/  {grpo,ppo}/     # FedAgent     (seed 1)
    main_seed2/  {grpo,ppo}/     # FedAgent     (seed 2)
    centralized/ {grpo,ppo}/     # Centralized baseline
    local_client1/ {grpo,ppo}/   # Local baseline, client index 21
    local_client2/ {grpo,ppo}/   # Local baseline, client index 42
    local_client3/ {grpo,ppo}/   # Local baseline, client index 84
```

Each leaf holds exactly two configs, one per benchmark (`fed_webshop_*.yaml`,
`fed_alfworld_*.yaml`).

### Row → config mapping

| Table row | Config subdir | Federation shape (filename) |
|---|---|---|
| **FedAgent** | `main/`, `main_seed1/`, `main_seed2/` | `total-100_cl-per-rd-2_rd-70_ep-per-cl-3` |
| **Centralized** | `centralized/` | `total-1_cl-per-rd-1_rd-1_ep-per-cl-210` (one client, one round, all 210 epochs) |
| **Local** | `local_client{1,2,3}/` | `total-100_cl-per-rd-1_rd-1_ep-per-cl-210` (a single client trained in isolation) |

The three baselines hold the **total optimization budget fixed at 210 local
epochs** so the comparison is compute-matched: Centralized trains one client on
the pooled dataset; Local trains one client on its own shard; FedAgent
distributes the same 210 epochs across `M=2` clients over `T=70` rounds with
FedAvg between rounds.

The **Local** rows in the paper are the three fixed client indices **21, 42, 84**; these are the goal shards that `local_client1/2/3` train on under the
`data_sharding.seed = 42` partition.

### Run

```bash
# FedAgent, WebShop, GRPO, default backbone (the webshop-main shortcut):
conda activate fedagent-webshop
bash reproduce.sh webshop-main

# FedAgent, ALFWorld, GRPO, default backbone:
conda activate fedagent-alfworld
bash reproduce.sh alfworld-main

# Any other cell, e.g. the Centralized WebShop-GRPO baseline:
python tools/run_federated.py --restart-resume uniform/Qwen2.5-1.5B-Instruct/centralized/grpo/fed_webshop_grpo_total-1_cl-per-rd-1_rd-1_ep-per-cl-210_min-goals-per-cl-100_p-uniform

# A larger backbone, e.g. FedAgent ALFWorld-GRPO with Qwen2.5-7B:
python tools/run_federated.py --restart-resume uniform/Qwen2.5-7B-Instruct/main/grpo/fed_alfworld_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform
```

### Notes

- **Models:** swap the `<model>` directory to reproduce other backbone blocks of
  Table 1. The model is pinned inside each config at
  `actor_rollout_ref.model.path`.
- **GRPO vs PPO:** the `grpo/` configs back the main table; the sibling `ppo/`
  configs back the PPO appendix table. For PPO, the critic uses
  `optimizer_offload=false` (offload off on H100), matching
  [`running.md`](running.md).
- **Seeds:** `main/` is seed 0; `main_seed1/` and `main_seed2/` are the other two
  of the 3 reported seeds. The Centralized and Local baselines are likewise run
  for 3 seeds (see [Seeds](#seeds-and-statistical-reporting)).
- **Training-dynamics figure:** `main_combined_val_success_rate.pdf` is built
  from the Qwen2.5-1.5B `main/grpo` and `centralized/grpo` validation curves
  (single-seed server curves, per-client population overlaid as scatter).

---

## 2. Task-level heterogeneity → `config/task_heterogeneity/`

**Backs:** the task-level heterogeneity figure
(`heterogeneous_combined_val_success_rate.pdf`), **6 panels**: two per sub-type,
one column per benchmark:

| Panels | Sub-type |
|---|---|
| (a), (b) | **Preference** |
| (c), (d) | **Coverage** |
| (e), (f) | **Hardness** |

Task-level heterogeneity enters the policy **through the prompt** (the task
descriptor is observable), so the federated objective is robust to it; this is
the paper's **Pattern A** axis.

### Layout

```
config/task_heterogeneity/
  grpo/ {webshop,alfworld}/
  ppo/  {webshop,alfworld}/
```

Each leaf holds the **6 sweep configs**: the two endpoints of each sub-type:

| Sub-type | Code strategy | Filename token | Endpoints (low → high heterogeneity) |
|---|---|---|---|
| **Preference** | `preference` | `p-preference_omega-*` | `omega = 0.01` (near-uniform) → `omega = 0.99` (extreme) |
| **Coverage** | `coverage` | `p-coverage_std-*` | `size_std = 256` (near-uniform) → `size_std = 1` (extreme) |
| **Hardness** | `hardness` | `p-hardness_success_std-*` | `success_std = 256` (near-uniform) → `success_std = 1` (extreme) |

> **Naming caveat (the one real collision).** The dispatch strategy, the paper
> name, and the filename token all agree at the word level (`preference` /
> **Preference**, `coverage` / **Coverage**, `hardness` / **Hardness** -- the code
> just lowercases the paper's title-case term; `hardness` is the lowercased
> *Hardness*, not a misspelling). The actual hazard is the **Preference knob's
> symbol**: filenames and configs use `omega` (env var `OMEGA`; e.g.
> `p-preference_omega-0.99`), but the codebase still accepts a **legacy alias
> `tau`/`TAU`** for the same knob (in `partition_strategy.py`, `omega` defaults to
> `tau` when `omega` is unset). Do **not** confuse that legacy `tau` with the
> paper's symbol $\tau$, which denotes the *task descriptor* (an unrelated
> concept: the observable, prompt-visible task identity), not the Dirichlet
> spread. Prefer `omega` everywhere; treat `tau` as deprecated. See
> [`heterogeneity.md`](heterogeneity.md).

### Run

```bash
# Preference, extreme heterogeneity, WebShop-GRPO:
conda activate fedagent-webshop
python tools/run_federated.py --restart-resume task_heterogeneity/grpo/webshop/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-preference_omega-0.99

# Coverage, high spread, ALFWorld-GRPO:
conda activate fedagent-alfworld
python tools/run_federated.py --restart-resume task_heterogeneity/grpo/alfworld/fed_alfworld_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-coverage_std-256
```

To reproduce a full panel, run **both** endpoints of the relevant sub-type for
the relevant benchmark, for 3 seeds each. The PPO appendix variant uses the
`ppo/` sibling configs.

### Notes

- The base federation shape is unchanged (`total-100_cl-per-rd-2_rd-70_ep-per-cl-3`);
  only the partition strategy differs from the uniform baseline.
- **Hardness** pre-labels each task success/fail with a reference checkpoint
  (zero-shot Qwen2.5-1.5B-Instruct on the training pool) before drawing
  per-client success-rate quotas; `|X_i|` is held constant.
- The partition is realized by `partition_dataset(strategy, ...)` in
  [`third_party/verl-agent/agent_system/environments/partition_strategy.py`](../third_party/verl-agent/agent_system/environments/partition_strategy.py),
  selected via `federated.data_sharding.partition.strategy`.

---

## 3. Environment-level heterogeneity → `config/env_heterogeneity/`

**Backs:** the environment-level heterogeneity figure
(`webshop_env_variants_combined_val_success_rate.pdf`), **GRPO on the left, PPO
on the right**.

Env-level heterogeneity enters through the **transition kernel** (the policy only
senses it through successor states; it is *not* observable from the prompt), so
the federated objective is **worst-case non-robust** to it. The task partition is
held **uniform** across all env-level runs, so any divergence is attributable to
the transition perturbation alone. WebShop's transition pipeline factors into
four stages, and the five variants perturb across them:

| Variant | Config subdir | Strategy / pool | Pipeline stage | Sweep token | Robustness pattern |
|---|---|---|---|---|---|
| **Catalog Split** | `catalog_split/` | `catalog_split` | content | `p-catalog_split_div-{0.0,0.3,0.7,1.0}_keep-0.7` | B/C |
| **Field-Subset Index** | `field_subset_index/` | `bm25_variant`, `fields_only` | encoding | `p-field_subset_index_N-{4,8}` | C |
| **BM25 Reweighting** | `bm25_reweighting/` | `bm25_variant`, extreme `(k1,b)` | matching | `p-bm25_reweighting_N-{4,8}` | C |
| **Lookalike Injection** | `lookalike_injection/` | `lookalike_injection` | content + matching | `p-lookalike_injection_N-{2,4}` | D (GRPO) → C (PPO) |
| **Rank Wrapper** | `rank_wrapper/` | `rank_wrapper` | rendering | `p-rank_wrapper_N-4` | D (GRPO) → C (PPO) |

The B/C/D labels are the paper's worst-case-degradation spectrum: Catalog Split
sits at Pattern B/C (mildest, the optimal policy still transfers); the two BM25
variants land at C; Lookalike Injection and Rank Wrapper hit Pattern D under
GRPO and are **rescued to C under PPO** (the source of the GRPO-vs-PPO contrast
in the figure).

### Layout

```
config/env_heterogeneity/                          # run configs (YAML)
  catalog_split/            catalog_split_ppo/
  field_subset_index/         field_subset_index_ppo/
  bm25_reweighting/         bm25_reweighting_ppo/
  lookalike_injection/      lookalike_injection_ppo/
  rank_wrapper/ rank_wrapper_ppo/

data/env_heterogeneity/                            # data (not configs)
  lookalike_data/                                  # offline lookalike pools (see below)
  holdout_{webshop,alfworld}_v1.json               # env-level OOD holdout sets
```

`data/env_heterogeneity/lookalike_data/` is **not** a run config; it holds the
pre-synthesized lookalike product pools (`lookalike_v_price.json`,
`lookalike_v_color.json`, `lookalike_v_size.json`, `lookalike_v_price_color.json`)
consumed by the `lookalike_injection*` runs.

### Run (WebShop env only)

```bash
conda activate fedagent-webshop

# Catalog Split, full divergence, GRPO:
python tools/run_federated.py --restart-resume env_heterogeneity/catalog_split/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-catalog_split_div-1.0_keep-0.7

# Lookalike Injection, GRPO vs PPO (the Pattern-D contrast):
python tools/run_federated.py --restart-resume env_heterogeneity/lookalike_injection/fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-lookalike_injection_N-4
python tools/run_federated.py --restart-resume env_heterogeneity/lookalike_injection_ppo/fed_webshop_ppo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-lookalike_injection_N-4
```

### Notes

- **GRPO vs PPO:** every variant has a `*_ppo` sibling directory; the figure
  plots both side by side. The GRPO directories sweep multiple points (Catalog
  Split over `env_div ∈ {0.0, 0.3, 0.7, 1.0}`, BM25/field-subset over `N ∈
  {4, 8}`, Lookalike over `N ∈ {2, 4}`); each `*_ppo` directory, however, holds
  only the **single** most-divergent sweep point used for the GRPO-vs-PPO
  comparison, do not expect a full PPO sweep.
- **Validation is always on the UNPERTURBED WebShop environment**: the eval
  harness forces all perturbation kwargs to `None`, so the metric isolates
  post-aggregation generalization rather than per-client overfitting.
- `SEARCH_RETURN_N = 200` is held fixed throughout to prevent target dropouts
  under aggressive filtering.
- Per-client variant assignment is deterministic given the partition seed; the
  exact per-variant seed offsets are documented under
  [Seeds](#seeds-and-statistical-reporting).

---

## 4. Decentralized / hyperparameter-sensitivity ablations → `config/decentralized/`

**Backs:** the decentralized / hyperparameter-sensitivity figure
(`decentralized_setting.pdf`). These sweeps justify the default
`(M = 2, E = 3, |X_i| = 100)` choice by varying **one** federation knob at a time
while holding the total optimization budget comparable.

### Layout and what each sweep varies

```
config/decentralized/
  selected_cl_change/  {grpo,ppo}/   # vary M (clients sampled per round)
  ep_per_round_change/ {grpo,ppo}/   # vary E (local epochs per round)
  samples_change/      {grpo,ppo}/   # vary |X_i| (tasks per client)
```

| Sweep | Knob | Configs present (filename shape) |
|---|---|---|
| `selected_cl_change/` | `clients_per_round` (M) | `cl-per-rd-1 ... rd-70 ... ep-per-cl-3`, `cl-per-rd-4 ... rd-70 ... ep-per-cl-3` (the `M=2` point is the uniform baseline) |
| `ep_per_round_change/` | `epochs_per_client` (E) | `rd-210 ... ep-per-cl-1`, `rd-42 ... ep-per-cl-5` (rounds scaled to keep ~210 total epochs; `E=3 / T=70` is the baseline) |
| `samples_change/` | `min_goals_per_client` (`|X_i|`) | `min-goals-per-cl-500`, `min-goals-per-cl-1000` (the `100` point is the uniform baseline) |

Each leaf holds the WebShop and ALFWorld variants of its sweep points.

### Run

```bash
# M = 4 clients/round, WebShop-GRPO:
conda activate fedagent-webshop
python tools/run_federated.py --restart-resume decentralized/selected_cl_change/grpo/fed_webshop_grpo_total-100_cl-per-rd-4_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform

# E = 5 local epochs (T = 42 rounds), ALFWorld-GRPO:
conda activate fedagent-alfworld
python tools/run_federated.py --restart-resume decentralized/ep_per_round_change/grpo/fed_alfworld_grpo_total-100_cl-per-rd-2_rd-42_ep-per-cl-5_min-goals-per-cl-100_p-uniform
```

### Notes

- The **baseline point** for each sweep (`M=2`, `E=3`, `|X_i|=100`) is *not*
  duplicated here; it is the corresponding uniform main-table run from
  [`config/uniform/`](#1-main-table--configuniform).
- `ep_per_round_change/` scales `total_rounds` (T) inversely with
  `epochs_per_client` (E) to hold the total local-epoch budget near 210, so the
  ablation isolates the round/epoch trade-off rather than total compute.
- PPO counterparts live in the `ppo/` siblings.

---

## Compute budget

Approximately **1,800 H100 GPU-hours** total across all reported experiments.
Per-config (single seed) estimates, on the default 4 × H100 node:

| Benchmark × algorithm | Wall-clock (4 × H100) | GPU-hours / config |
|---|---|---|
| WebShop GRPO  | ~24 h | ~93 |
| WebShop PPO   | ~29 h | ~117 |
| ALFWorld GRPO | ~29 h | ~117 |
| ALFWorld PPO  | ~35 h | ~140 |

GPU-hours = wall-clock × 4 GPUs. Multiply by **3 seeds** for each reported
mean ± std cell, and by the number of sweep points / backbones in a given
figure or table block. To shrink cost while developing, use `--single-gpu` for
a smoke run or `--mode serial` to reduce peak GPU demand (at the cost of
wall-clock); see [`running.md`](running.md).

---

## Evaluation protocol

### Test sets (held out from training)

| Benchmark | Validation set | Size |
|---|---|---|
| **WebShop** | `goals[0:500]` | 500 held-out goals |
| **ALFWorld** | `valid_seen` + `valid_unseen` | 140 + 134 = **274** episodes |

`tools/verify_train_val_disjoint.py` checks that the training shards and these
validation sets do not overlap. For env-level heterogeneity, validation always
runs on the **unperturbed** environment.

### Metrics

- **WebShop:** Task Score and Success Rate.
- **ALFWorld:** Success Rate, broken down by task type
  (Pick / Look / Clean / Heat / Cool / Pick2) plus an **All** aggregate.

### Cadence (during training)

Set in every config:

- `trainer.test_freq = 5`, validate every 5 communication rounds (the source of
  the training-dynamics curves).
- `federated.eval_only_final_round: true`, the headline table number is the
  validation at the final round; the intermediate `test_freq` evaluations
  populate the curves.
- In-training validation batch: `data.val_batch_size = 64` (uniform across all
  released GRPO and PPO configs).

### Standalone checkpoint evaluation

To evaluate a single trained checkpoint and dump per-episode trajectories +
aggregate metrics, use the post-hoc harness (separate from the in-training
validation path):

```bash
conda activate fedagent-webshop
bash evaluate.sh webshop  <checkpoint-dir>

conda activate fedagent-alfworld
bash evaluate.sh alfworld <checkpoint-dir>
```

`evaluate.sh` dispatches to `eval/eval_<env>.sh`, runs against the **standard
(unperturbed)** environment, and merges trajectory shards via
`eval/merge_trajectories.py`. The WebShop harness evaluates with
`val_data_size = 128`, `env.seed = 0`, sampling `temperature = 0.4`,
`do_sample = True`, and `test_freq = -1` (single eval pass, no training). Inspect
results with `eval/view_results.py`.

> The standalone harness's `val_data_size = 128` is the eval-only batch size and
> is **independent** of the in-training `val_batch_size = 64` used to draw the
> training-dynamics curves; do not conflate the two.

### Seeds and statistical reporting

Federated and centralized rows report **mean ± std over 3 random seeds** (std as
a subscript). The infrastructure seeds, set in the configs and the heterogeneity
constructors, are:

| Seed | Role |
|---|---|
| `env.seed = 0` | Training SimServer / ALFWorld step seed (also the eval seed) |
| `SimServer.seed = 1000` | Validation server seed, isolated from training RNG |
| `data_sharding.seed = 42` | Deterministic client→goal mapping (fixes Local indices 21/42/84) |
| `42 + client_id` | Per-client variant assignment (BM25 / Lookalike / Rank Wrapper) |
| `42 + 1000·client_id` | Catalog Split per-client distractor RNG (global seed 42 for the shared distance threshold) |
| `99999` | Held-out set generation (`tools/env_heterogeneity/gen_holdout_webshop.py` / `gen_holdout_alfworld.py`) |

Training-dynamics figures show single-seed server curves with the per-client
population overlaid as scatter points; table cells aggregate the 3 seeds.

---

## Extension points

To go beyond the released sweeps (full contract in
[`extending.md`](extending.md)):

1. **New environment / dataset**: add an env package under
   `third_party/verl-agent/agent_system/environments/env_package/` and register
   it.
2. **New heterogeneity construction**: add a strategy plus an `elif` branch in
   `partition_dataset()` in
   [`partition_strategy.py`](../third_party/verl-agent/agent_system/environments/partition_strategy.py),
   then select it via `federated.data_sharding.partition.strategy`.
3. **New RL algorithm**: implemented in the verl-agent trainer (PPO / GRPO /
   GiGPO / RLOO / DAPO are available upstream); set `verl.algorithm.adv_estimator`.
4. **New aggregation rule**: extend
   [`utils/model_aggregation.py`](../utils/model_aggregation.py) (FedAvg and
   FedProx today) and validate with
   `tools/aggregation/check_aggregation.py` and
   `tools/aggregation/verify_aggregation.py`.
