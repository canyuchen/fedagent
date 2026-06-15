# CHANGES — Modifications to the vendored verl-agent

This directory (`third_party/verl-agent/`) is a **modified vendored copy** of
**verl-agent** ([langfengQ/verl-agent](https://github.com/langfengQ/verl-agent)),
which is itself an extension of **veRL**
([volcengine/verl](https://github.com/volcengine/verl)). Both upstream projects
are licensed under the **Apache License, Version 2.0**; their original `LICENSE`
file is preserved alongside this source.

This file documents the changes FedAgent made to the upstream sources, as
required by **Section 4(b) of the Apache License, Version 2.0** ("You must cause
any modified files to carry prominent notices stating that You changed the
files"). It is a summary of the FedAgent additions/modifications and is not a
line-by-line diff.

> Upstream references:
> - verl-agent: Feng et al., "Group-in-Group Policy Optimization for LLM Agent
>   Reinforcement Learning", arXiv:2505.10978 (Apache-2.0).
> - veRL: ByteDance / the veRL authors (Apache-2.0).
>
> TODO(author): pin the exact upstream verl-agent commit/tag this copy was
> vendored from, and verify the modification list below against the final diff
> before public release.

## Summary of FedAgent modifications

FedAgent adapts verl-agent from single-trainer agent RL into a **federated**
agent-RL framework with **two levels of heterogeneity** (task-level and
environment-level). The changes are additive where possible: upstream
single-trainer entry points are left in place, and federated variants are added
alongside them.

### 1. Federated dataset/partition strategies (task-level heterogeneity)

- **`agent_system/environments/partition_strategy.py`** (new) — dataset
  partition strategies that split tasks across federated clients, including a
  non-IID category/task-type partition with a Laplace-smoothed global category
  marginal (e.g. ALFWorld task types, WebShop categories). The task-level
  strategies are `preference`, `coverage`, and `hardness` (see
  `docs/heterogeneity.md`).

### 2. Federated environment manager

- **`agent_system/environments/fed_env_manager.py`** (new) — federated
  environment managers (ALFWorld, Sokoban, GymCards, WebShop, AppWorld) that
  wire per-client task partitions and per-client environment variants into the
  rollout loop.

### 3. Federated PPO / GRPO training (federated trainer)

- **`verl/trainer/main_ppo_fed.py`** (new) — federated PPO/GRPO entry point
  (federated counterpart of upstream `verl/trainer/main_ppo.py`).
- **`verl/trainer/ppo/ray_trainer_fed.py`** (new) — Ray-based federated trainer
  implementing client-local updates and server-side aggregation across rounds.
- **`verl/utils/checkpoint/fsdp_checkpoint_manager_fed.py`** (new) — checkpoint
  manager for the federated trainer.
- **`verl/utils/tracking_fed.py`** (new) — experiment tracking for federated
  runs (per-client / per-round metrics).

### 4. Environment-level heterogeneity (transition-level env divergence)

Per-client environment variants that make the *same* task behave differently
across clients, implemented primarily for WebShop and selected via
`partition_strategy_env`:

- **BM25 variant heterogeneity** (`bm25_variant`) — each client's
  search backend is swapped to an in-memory BM25 searcher with a distinct
  `(fields, k1, b)` configuration, so retrieval results diverge per client while
  the catalog and goals are held fixed.
- **Lookalike (adversarial) variant heterogeneity**
  (`lookalike_injection`) — per-client injection of lookalike/distractor
  products as an adversarial environment perturbation.
- **Search-engine TYPE swap** — per-client swap of the search-engine backend
  type as an additional env-heterogeneity axis.
- Supporting env-heterogeneity tooling and holdout generation under
  `tools/env_heterogeneity/`.

### 5. Tests and supporting scripts (new)

- `test_alfworld_fed.py` and `test_eval_consistency.py` exercise the federated
  partition and environment-heterogeneity code paths. Runnable partition-strategy
  simulations also live under `tests/heterogenous/` in the release root.

## Files NOT modified

All other files retain their upstream content and their upstream copyright and
license headers. Where present, upstream `LICENSE`/`NOTICE` files (and the
nested environment packages' license files) are preserved.
