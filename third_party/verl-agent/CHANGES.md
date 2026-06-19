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
> Vendored from upstream **`langfengQ/verl-agent`** at commit
> **`a64f4e0905690823b21c4244354bfef63980b8f4`** (which embeds veRL `0.3.1.dev0`).
> FedAgent's modifications are layered on top of that snapshot; the summary below
> is not a line-by-line diff.

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

- FedAgent's tests and partition-strategy simulations live OUTSIDE this vendored
  tree, under `tests/heterogenous/` at the release root (`test_alfworld_fed.py`,
  `simulate_{preference,coverage,hardness}.py`, `test_data.py`); they import the
  vendored env package via `config/paths.yaml`. No test files are added inside
  `third_party/verl-agent/` itself.

### 6. Modified-in-place upstream files (environment layer)

The environment-level heterogeneity (section 4) and the federated-validation
machinery are implemented by editing existing upstream files in place, not only by
adding new ones. The known in-place edits:

- **`agent_system/environments/env_manager.py`** — per-env start_idx/end_idx and
  infer_special selection wired into env construction (plus defensive config access).
- **`agent_system/environments/__init__.py`** — exposes the federated env managers.
- **`agent_system/environments/env_package/webshop/envs.py`** — goal slicing,
  partition-strategy dispatch, infer_special, and the windowed validation path.
- **`.../webshop/webshop/web_agent_site/envs/web_agent_text_env.py`** and
  **`.../web_agent_site/engine/engine.py`** — env-level heterogeneity hooks (catalog
  double-track, lookalike injection, BM25 / search-engine variants) and fixed-seed
  goal shuffling.
- **`.../alfworld/alfworld/agents/environment/alfred_tw_env.py`** and
  **`.../alfworld/envs.py`** — federated game-file sharding, index-based batch
  selection, and held-out valid_seen windowing.

Each retains its upstream copyright/license header; the edits are FedAgent's.

## Files NOT modified

All files other than the additions in sections 1-5 and the in-place edits in
section 6 retain their upstream content and their upstream copyright and license
headers. This summary is not a line-by-line diff; the exhaustive list will be
regenerated against the pinned upstream commit (see the TODO above) before public
release. Where present, upstream `LICENSE`/`NOTICE` files (and the nested
environment packages' license files) are preserved.
