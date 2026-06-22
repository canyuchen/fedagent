# Key features, detailed

This expands each headline feature from the [README](../README.md#key-features)
with the concrete config keys, CLI flags, and source files that implement it.

Every experiment is a single YAML with three top-level blocks, `federated:`
(FedAgent's federation / aggregation / partition), `verl:` (the verl-agent trainer,
passed through to each client), and `data_preprocess:` (dataset sharding). See
[`config/example.yaml`](../config/example.yaml) for the fully annotated schema and
[configuration.md](configuration.md) for the complete field reference.

## Contents
1. [Algorithms, federated PPO & GRPO](#1-algorithms)
2. [Models, any HuggingFace backbone](#2-models)
3. [Environments, WebShop & ALFWorld](#3-environments)
4. [Two-level heterogeneity](#4-two-level-heterogeneity)
5. [Aggregation, FedAvg / FedProx](#5-aggregation)
6. [Decentralized setting](#6-decentralized-setting)
7. [FSDP & scaling](#7-fsdp--scaling)
8. [Client execution, serial / parallel](#8-client-execution)
9. [Extensibility](#9-extensibility)
10. [W&B-free logging](#10-wb-free-logging)

---

## 1. Algorithms

Federated **PPO** and **GRPO**, as drop-in federated counterparts of the verl-agent
trainers. Each selected client runs local PPO/GRPO updates on its own data; the
server aggregates the resulting weights each round. GRPO uses group rollouts and no
critic; PPO adds a critic that is aggregated alongside the actor.

**Configure**
- `verl.algorithm.adv_estimator: grpo` (GRPO) or `gae` (PPO).
- `federated.base_script_path`, the per-client base launch script for the env+algo,
  e.g. `scripts/verl-agent/grpo/run_webshop.sh` or `scripts/verl-agent/ppo/run_alfworld.sh`.
- `verl.env.rollout.n`, GRPO group size (rollouts per prompt).
- Each client runs the federated entry point `python -m verl.trainer.main_ppo_fed`.

Adding a new RL algorithm → [extending.md](extending.md).

## 2. Models

Any **HuggingFace** causal-LM backbone. The paper sweeps **Qwen2.5-1.5B / 3B /
7B-Instruct** and **Llama-3.2-3B-Instruct**. Backbones auto-download on first run.

**Configure**
- `verl.actor_rollout_ref.model.path` and `.tokenizer_path`, an HF id
  (e.g. `Qwen/Qwen2.5-1.5B-Instruct`) or a local path. For PPO, also set the matching
  `verl.critic.model.path`.
- Acquisition, cache location, the gated `Llama-3.2-3B`, and offline/air-gapped
  clusters → [installation.md#models](installation.md#models).

## 3. Environments

Real agent benchmarks **WebShop** (e-commerce search-and-buy) and **ALFWorld**
(embodied household tasks on TextWorld). The two have conflicting dependencies, so
each runs in its own conda env. The vendored verl-agent tree also bundles Sokoban /
GymCards / AppWorld, which are *not* part of the FedAgent experiments.

**Configure**
- `verl.env.env_name: Webshop` or `alfworld/AlfredTWEnv`.
- `verl.env.max_steps`, max agent steps per episode.
- `verl.env.webshop.use_small`, the small shipped catalog (default `true`) vs the
  full catalog (see [Data setup](installation.md)).

Adding a new environment / dataset → [extending.md](extending.md).

## 4. Two-level heterogeneity

The core research feature: a configurable suite of **client-partition strategies**
along two structurally distinct axes, all selected via
`federated.data_sharding.partition.{strategy, kwargs}`.

**Task-level**: clients differ in their *task distribution*, which the policy can
observe through the prompt:

| `strategy` | kwargs | What varies across clients |
|---|---|---|
| `preference` | `omega` | Dirichlet skew over product categories |
| `coverage` | `size_std` | dispersion of per-client pool sizes |
| `hardness` | `success_std` | dispersion of per-client task difficulty |

**Environment-level**: clients differ in the WebShop *transition kernel*, which is
hidden from the policy (sensed only through successor states):

| `strategy` | kwargs | Paper name (perturbed stage) |
|---|---|---|
| `catalog_split` | `env_div`, `keep_ratio` | Catalog Split (content) |
| `bm25_variant` + `variant_pool: fields_only` | `N` | Field-Subset Index (encoding) |
| `bm25_variant` (default pool) | `N` | BM25 Reweighting (matching) |
| `lookalike_injection` | `N` | Lookalike Injection (content + matching) |
| `rank_wrapper` | `N` | Rank Wrapper (rendering) |

`uniform` (i.i.d.) is the homogeneous baseline. The full construction, the paper
mapping, and the *stable → degrade → collapse* spectrum are in
[heterogeneity.md](heterogeneity.md); the strategy code lives in
`third_party/verl-agent/agent_system/environments/partition_strategy.py`, with the
bundled env-level data under [`data/env_heterogeneity/`](../../data/env_heterogeneity).

## 5. Aggregation

Server-side model combination each round is **FedAvg** (weighted parameter mean),
implemented in `utils/model_aggregation.py` / `core/fed/aggregator.py` (the
FSDP-sharded path and the aggregation verifiers live under `tools/aggregation/`).
**FedProx** is also available: it keeps each client near the round's global model by
adding a proximal term (μ/2)‖w − w^t‖² to that client's **local** training
objective, so it changes the client update — the server still aggregates by FedAvg.

**Configure**
- `federated.aggregation_method: fedavg` (default) or `fedprox`.
- `federated.fedprox_mu`, the FedProx proximal coefficient μ (FedProx only; μ=0 ≡ FedAvg).
- FedProx implementation: the proximal term lives in the verl actor
  (`verl/workers/actor/dp_actor.py`, `update_policy`); `core/fed/script_builder.py`
  bridges μ to each client through the `FEDPROX_MU` env var → `actor.fedprox_mu`.

Adding a new aggregation rule → [extending.md](extending.md).

## 6. Decentralized setting

The full federation protocol is configurable, and the paper's decentralized /
hyperparameter-sensitivity sweeps ship as ready-made config groups.

**Configure** (the `federated:` block):

| Symbol | Key | Meaning |
|---|---|---|
| `N` | `total_clients` | size of the client pool |
| `M` | `clients_per_round` | clients sampled & trained each round |
| `E` | `epochs_per_client` | local epochs per selected client |
| `T` | `total_rounds` | number of federated rounds |
| `\|Xᵢ\|` | `data_sharding.min_goals_per_client` | tasks per client |
| seed | `data_sharding.seed` | deterministic client → data assignment |

Ready-made sweeps under `config/decentralized/` (each varies one knob from the
table above while holding the others at the main-table defaults):

| Directory | Sweeps | What changes in the filename |
|---|---|---|
| `selected_cl_change/` | **M** (`clients_per_round`) | `cl-per-rd-1 / 2 / 4` |
| `samples_change/` | **\|Xᵢ\|** (`min_goals_per_client`) | `min-goals-per-cl-100 / 500 / 1000` |
| `ep_per_round_change/` | **E** (`epochs_per_client`), with **T** (`total_rounds`) adjusted to keep `E×T = 210` fixed | `ep-per-cl-1_rd-210`, `ep-per-cl-3_rd-70`, `ep-per-cl-5_rd-42` |

The config filename encodes the full protocol
(e.g. `…total-100_cl-per-rd-2_rd-70_ep-per-cl-3…`); the decoder is in
[configuration.md](configuration.md).

## 7. FSDP & scaling

Larger backbones (3B / 7B) train via **FSDP** with optional CPU offload, and runs
scale from a single GPU to multi-node / SLURM.

**Configure**
- `verl.actor_rollout_ref.actor.fsdp_config.param_offload` / `.optimizer_offload`
  (and `…ref.fsdp_config.param_offload`); toggle with `reproduce.sh --fsdp on|off`.
- `verl.trainer.n_gpus_per_node` / `nnodes`;
  `verl.actor_rollout_ref.rollout.tensor_model_parallel_size` (vLLM tensor-parallel).
- `reproduce.sh` flags: `--gpus N`, `--single-gpu`, `--slurm`.
- The full hardware / scaling matrix → [running.md](running.md).

## 8. Client execution

Within a round, clients can run **serially** (one at a time) or **concurrently**
across the available GPUs.

**Configure**
- `reproduce.sh --mode serial` runs clients one at a time; the default distributes
  them across GPUs.
- Related field: `federated.training.parallel_workers`. Effective concurrency and the
  GPU-mapping details are documented in
  [running.md](running.md).

## 9. Extensibility

FedAgent is built to be extended, not only reproduced. The extension points:

| Add… | Where | Guide |
|---|---|---|
| a new **environment / dataset** | `third_party/verl-agent` env package | [extending.md](extending.md) |
| a new **heterogeneity** (client partition) | `partition_strategy.py` | [heterogeneity.md](heterogeneity.md) |
| a new **RL algorithm** (beyond PPO/GRPO) | verl-agent trainer | [extending.md](extending.md) |
| a new **aggregation** (beyond FedAvg/FedProx) | `utils/model_aggregation.py` | [extending.md](extending.md) |

## 10. W&B-free logging

Weights & Biases is **removed** from this release, no tracking account or key is
needed. Metrics are written to console and JSON (`verl.trainer.logger: [console]`,
plus FedAgent's per-round / per-client JSON logs under the run's `output_dir`). You
can wire in your own tracker if desired.
