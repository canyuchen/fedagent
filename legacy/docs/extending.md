# Extending FedAgent

FedAgent is a **library first** and a reproduction script second. The federated
control loop, the heterogeneity constructors, the RL trainer, and the model
aggregation are deliberately decoupled so that you can replace any one of them
without touching the others. This document is the reference for doing exactly
that.

There are four extension points, each isolated to a small number of files:

| # | Extension point | Primary file(s) | Selected by |
|---|---|---|---|
| 1 | **New dataset / environment** | `third_party/verl-agent/agent_system/environments/env_package/<env>/` + `env_manager.py` / `fed_env_manager.py` | `verl.env.env_name` |
| 2 | **New heterogeneity strategy** | `third_party/verl-agent/agent_system/environments/partition_strategy.py` | `federated.data_sharding.partition.strategy` |
| 3 | **New RL algorithm** | verl-agent trainer (`verl/trainer/`) | `verl.algorithm.adv_estimator` |
| 4 | **New aggregation strategy** | `utils/model_aggregation.py` (+ `tools/aggregation/`) | `federated.aggregation_method` |

**How the layers fit together.** The federated server lives in `core/` and
`tools/run_federated.py`; it samples `M` clients per round, launches one
verl-agent training job per client (via `core/fed/script_builder.py` →
`scripts/verl-agent/<algo>/run_<env>.sh`), then aggregates their checkpoints
(via `core/fed/aggregator.py` → `utils/model_aggregation.py`). The vendored
`third_party/verl-agent` framework owns the environment, the rollout, and the RL
update. Because of this split, extension points 1–3 live inside the vendored
framework, while 4 lives in `utils/`. The server is environment-, heterogeneity-
and algorithm-agnostic: it moves whatever checkpoints the local trainer
produces.

> **Before you start.** `config/paths.yaml` is loaded at import time by several
> modules (including `partition_strategy.py`, which calls
> `OmegaConf.load(config/paths.yaml)` at module scope). Keep a valid copy
> present when developing, `cp config/paths.yaml.example config/paths.yaml`,
> or imports will fail before your code ever runs. There are **two** conda
> environments, `fedagent-webshop` and `fedagent-alfworld`; activate the one
> that matches the environment you are extending (see
> [`docs/installation.md`](installation.md)).

---

## 1. Add a dataset / environment

### Where

A self-contained environment lives in its own package under
`third_party/verl-agent/agent_system/environments/env_package/<env>/`, and is
wired into the framework by two dispatch functions in the same directory:

- `env_manager.py` → `make_envs(config)`, the **non-federated** entry point.
- `fed_env_manager.py` → `fed_make_envs(config, client_id, client_num)`, the
  **federated** entry point, which additionally performs the per-client data
  partition (see point 2).

`sokoban/` is the cleanest reference package; `webshop/` is the most fully
featured (it is the only env wired for the **five-variant** env-level
heterogeneity pipeline — Catalog Split / Field-Subset Index / BM25 Reweighting /
Lookalike Injection / Rank Wrapper; see §2. ALFWorld also has env-level
heterogeneity, but only via the single `env_disjoint` scene-disjoint strategy).

### The contract

A new environment package exposes three things, mirroring the existing ones
(see e.g. `env_package/sokoban/__init__.py`):

```python
# env_package/<env>/__init__.py
from .projection import <env>_projection      # text action -> env action
from .envs       import build_<env>_envs      # vectorized env constructor
```

1. **A vectorized env builder** `build_<env>_envs(...)`. Existing builders are
   Ray-backed (one `@ray.remote` worker per sub-environment) and are called with
   a `seed`, an `env_num` (= `config.data.train_batch_size`), a `group_n` (the
   GRPO/GiGPO group size, `1` for validation), and an `is_train` flag. Look at
   `build_sokoban_envs` in `env_package/sokoban/envs.py` for a minimal signature
   and at `build_webshop_envs` in `env_package/webshop/envs.py` for the full
   one. Train and validation envs are built separately, with the validation env
   given `seed = config.env.seed + 1000` (the SimServer validation seed used
   throughout FedAgent).

2. **A projection function** `<env>_projection(text_actions, ...) -> (actions, valids)`.
   It maps the model's raw text output to the discrete/structured action the env
   `step()` expects, and returns a parallel `valids` list flagging which actions
   parsed (these become `info['is_action_valid']`). See
   `env_package/sokoban/projection.py`.

3. **An `EnvironmentManager` subclass** of `EnvironmentManagerBase`
   (`agent_system/environments/base.py`). The base class already implements
   `reset()`, `step()`, `close()`, and a default `success_evaluator()` that
   reads `info['won']` on the last active step. At minimum you must implement
   `build_text_obs(...)`, which renders the env's observation into the prompt
   string the agent sees; override `step()` / `success_evaluator()` only if your
   env needs custom reward shaping or a non-`won` success signal.

4. **Two dispatch branches**, one in `make_envs` and one in `fed_make_envs`.
   Both dispatch on `config.env.env_name.lower()` via an `if/elif` chain and
   fall through to an error (`make_envs` prints "Environment not supported" and
   exits; add your branch *before* that). A branch builds the train and val
   envs, partials the projection function, wraps both in your manager, and
   returns `(envs, val_envs)`:

   ```python
   # in fed_make_envs(config, client_id, client_num), env_manager.py-style:
   elif "myenv" in config.env.env_name.lower():
       from agent_system.environments.env_package.myenv import (
           build_myenv_envs, myenv_projection,
       )
       group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1  # GRPO/GiGPO group size
       _envs = build_myenv_envs(
           seed=config.env.seed,
           env_num=config.data.train_batch_size,
           group_n=group_n, is_train=True,
       )
       _val_envs = build_myenv_envs(
           seed=config.env.seed + 1000,
           env_num=config.data.val_batch_size,
           group_n=1, is_train=False,
       )
       projection_f = partial(myenv_projection)
       envs     = MyEnvEnvironmentManager(_envs, projection_f, config,
                                          client_id=client_id, client_num=client_num)
       val_envs = MyEnvEnvironmentManager(_val_envs, projection_f, config,
                                          client_id=client_id, client_num=client_num)
   ```

5. **A data/preprocessing path.** The verl trainer reads
   `verl.data.train_files` / `verl.data.val_files` as parquet (each row carrying
   the prompt and the per-task metadata). For text-agent environments the
   parquet is a thin index of task ids/prompts; the heavy env state is loaded by
   the env workers. Follow the pattern in
   `third_party/verl-agent/examples/data_preprocess/` (e.g. `prepare.py`,
   `gsm8k.py`) to emit train/val parquet for the new dataset, then point the two
   `verl.data.*_files` fields at them. For WebShop/ALFWorld the data is fetched
   by `download_data.sh`; add your dataset there if it needs a download step.

6. **(Federated only) the partition surface.** `fed_make_envs` partitions the
   training pool per client. If your env's tasks are a flat list, the default
   `uniform` strategy works out of the box. If you want **Preference** (task-type)
   heterogeneity on the new env, expose a category label that
   `preference_partition` can read: the generic path reads `item[category_key]`
   (default `category_key='category'`), while ALFWorld derives the label from
   the trajectory file (`_preference_partition_alfworld`). For a brand-new env you
   either store a `preference` field on each item or add an env-specific label
   extractor analogous to `_preference_partition_alfworld`.

### Validate

Run a smoke pass under the matching conda env and confirm the env resolves and a
single round trains:

```bash
bash reproduce.sh <myenv-experiment> --single-gpu --mode serial
```

To guard against train/val leakage for the new dataset, mirror
`tools/verify_train_val_disjoint.py` (the WebShop/ALFWorld held-out check): in
FedAgent, WebShop evaluates on `goals[0:500]` and trains on `goals[500:]`, and
ALFWorld evaluates on `valid_seen(140)+valid_unseen(134)`. Your env needs an
analogous, explicit held-out split, and a check that no training task id leaks
into it.

---

## 2. Add a heterogeneity strategy

### Where

`third_party/verl-agent/agent_system/environments/partition_strategy.py`. Every
task-level partition funnels through one dispatch function,
`partition_dataset()`; env-level partitions are private helpers called directly
from `fed_env_manager.py` (they need `products`/`ins`/`goals`, which
`partition_dataset` does not carry).

### The contract, task-level (`partition_dataset` dispatch)

`partition_dataset` is the single entry point for the task-level axis:

```python
def partition_dataset(
    data: List[Any],
    strategy: str,
    client_id: int,
    client_num: int,
    min_samples_per_client: int,
    start_idx: int = 0,
    data_type: str = 'generic',   # 'generic' | 'webshop' | 'alfworld'
    **kwargs,
) -> Union[List[Any], Tuple[List[Any], int, int]]:
```

It dispatches on `strategy` and currently knows:

```
'uniform'                 -> uniform_partition           (returns (slice, start, end))
'preference'              -> preference_partition          # paper "Preference"
'coverage'                -> coverage_partition
'hardness'                -> hardness_partition[_alfworld]
'env_disjoint'            -> _env_disjoint_partition_alfworld   # ALFWorld env-level
'catalog_split'  -> raises (call the webshop helper directly; see below)
else                      -> raise ValueError("Unknown partition strategy: ...")
```

To add a strategy:

1. **Implement the partition function.** Match the shape of the existing
   strategies. The non-uniform task strategies all have the signature

   ```python
   def my_strategy_partition(
       data: List[Any],
       client_id: int,
       client_num: int,
       min_samples_per_client: int,
       start_idx: int = 0,
       **kwargs,
   ) -> List[Any]:
       ...
   ```

   and **return the list of items for `client_id`** (a *slice* of `data`, not
   indices). The single exception is `uniform_partition`, which returns the
   triple `(client_data_slice, start_slice, end_slice)`; `partition_dataset`
   special-cases it. Two invariants the existing strategies all honor, and you
   should too:

   - **Respect `start_idx`.** The first `start_idx` items are the held-out
     validation set; partition only `data[start_idx:]`
     (`total_train_data = data[start_idx:]`).
   - **Be deterministic in `client_id`.** Every client process runs this
     function independently and must agree on the global allocation. The
     existing code seeds a per-client RNG as `np.random.RandomState(42 + client_id)`
     (category) or seeds a *shared* `np.random.default_rng(42)` and indexes the
     result by `client_id` (coverage/hardness). Do **not** use Python's builtin
     `hash()` on strings for seeding (it is salted per interpreter); use
     `_spec_hash()` (sha256-based, defined at the top of the file) if you need a
     stable hash of a spec string.
   - **Guarantee the floor.** Top up to `min_samples_per_client` if your draw
     comes up short (see the tail of `_preference_partition_generic`).

2. **Add a dispatch branch** to `partition_dataset()` *and* update the
   `else`-branch error string (the supported-strategy list is asserted there).
   If your strategy also needs a `get_partition_info()` entry (used for
   logging/plots), add the parallel branch there.

3. **Select it from config.** Heterogeneity is chosen entirely through the YAML
   `federated.data_sharding.partition` block:

   ```yaml
   federated:
     data_sharding:
       seed: 42
       min_goals_per_client: 100
       partition:
         strategy: "my_strategy"   # the dispatch key
         kwargs:                   # forwarded verbatim as **kwargs
           my_param: 0.5
   ```

   `core/fed/script_builder.py` exports the strategy as `PARTITION_STRATEGY`
   into each client's launch script; `fed_env_manager.py` reads the `partition`
   block (`config.federated.data_sharding.partition.strategy` and `.kwargs`)
   and forwards `kwargs` to `partition_dataset`. A handful of legacy scalar env
   vars (`OMEGA`, `SIZE_STD`, `SUCCESS_STD`, `ENV_DIV`, …) are still honored for
   back-compat, but the `partition.kwargs` map is the canonical path, prefer
   it.

### The contract, env-level (transition-kernel) strategies

Env-level strategies perturb the **transition kernel** `T(s'|s,a)`, not the task
prompt, so they cannot run from a flat `data` list; they need the catalog and
goals. They are private helpers in `partition_strategy.py` invoked directly from
`fed_env_manager.py`'s WebShop branch, and each returns a per-client object that
becomes an `env_kwargs[...]` entry consumed by SimServer:

| Strategy | Helper | Returns / `env_kwargs` key | Transition stage |
|---|---|---|---|
| Catalog Split | `_distractor_disjoint_partition_webshop_v5` | `(catalog_asins, client_goal_idxs)` | content |
| Field-Subset Index | `_bm25_variant_partition_webshop` (`BM25_VARIANT_POOL=fields_only`) | `bm25_in_memory_config` | encoding |
| BM25 Reweighting | `_bm25_variant_partition_webshop` (extreme `k1,b`) | `bm25_in_memory_config` | matching |
| Lookalike Injection | `_lookalike_injection_partition_webshop` | `extra_products` | content + matching |
| Rank Wrapper | `_rank_wrapper_partition_webshop` | `search_engine_variant` | rendering |

> **Paper-variant ↔ runtime-key map (avoid two easy confusions).** The leftmost
> column above is the *paper* variant name; the runtime YAML `strategy:` keys are
> `catalog_split`, `bm25_variant`, `bm25_variant`, `lookalike_injection`,
> `rank_wrapper` respectively. Two snags:
> - **One key, two variants.** *Field-Subset Index* (V2, Stage 2) and *BM25
>   Reweighting* (V3, Stage 3) share the **same** dispatch key `bm25_variant` and
>   the same helper `_bm25_variant_partition_webshop`; they are told apart only by
>   the `BM25_VARIANT_POOL` env var (`fields_only` selects Field-Subset Index;
>   unset/`default` selects BM25 Reweighting with extreme `k1,b`).
> - **`v4`/`v5` are NOT paper Variants 4/5.** The `_v5` suffix on
>   `_distractor_disjoint_partition_webshop_v5` is an *implementation-iteration*
>   number: `v4` = the older `_distractor_disjoint_partition_webshop` (key
>   `distractor_disjoint`, legacy/superseded, all clients share `goals[500:]`);
>   `v5` = the current function (key `catalog_split`, per-client target floor,
>   uniform 100/client). **Both implement the one paper Variant 1 (Catalog
>   Split)** and are unrelated to paper Variant 4 (Lookalike Injection) or Variant
>   5 (Rank Wrapper). Only `catalog_split`/v5 backs the reported Catalog Split
>   numbers.

These share a fixed shape: assignment is **deterministic by `client_id`**
(`np.random.RandomState(base_seed + client_id)`) so FedAvg sees the same
per-client variant every round, and the **validation env is left unperturbed**
(all perturbation kwargs forced to `None`), so divergence is attributable to the
transition perturbation alone. To add a new env-level axis: write a
`_myvariant_disjoint_partition_webshop(client_id, client_num, N, base_seed, ...)`
helper returning a config dict, add a `MYVARIANT_VARIANTS_DEFAULT` pool next to
the existing ones, add an `elif` branch in the WebShop block of `fed_make_envs`
that calls it and threads the result into `env_kwargs`, and make sure SimServer
knows how to consume the new key.

> **Naming caveat (read this).** The task-level *dispatch keys* already match
> the paper at the key level (`preference` = paper *Preference*, `coverage` =
> *Coverage*, `hardness` = *Hardness* — `hardness` is the lowercased paper term,
> not a typo). The two places the names genuinely diverge, and can mislead, are
> the *knob parameters*:
>
> 1. **`omega` vs `tau` (a real symbol collision).** Preference's Dirichlet knob
>    is `omega` (paper symbol ω). Legacy configs and the `TAU`/`+data.tau` env
>    var are a back-compat *alias for the same knob* (`core/fed/script_builder.py`
>    falls back `kwargs.get('omega', kwargs.get('tau'))`). Beware: the paper's
>    symbol `tau` (τ) is the unrelated **task descriptor**, NOT the preference
>    knob — so the legacy code name `tau` collides with a different paper concept.
>    Prefer `omega` everywhere.
> 2. **`size_std`/`success_std` are misleadingly named — they are not standard
>    deviations.** Coverage's `size_std` and Hardness's `success_std` are the Beta
>    *concentration*, i.e. they **equal** the paper symbols ξ (Coverage) and ξ'
>    (Hardness) directly: a *larger* value means *lower* cross-client variance =
>    *more uniform* (high `size_std` ≈ high ξ ≈ near-uniform; low ≈ extreme).
>    Endpoints in code (matching the paper): `omega ∈ {0.01 near-uniform, 0.99
>    extreme}`, `size_std`/`success_std ∈ {256 near-uniform, 1 extreme}`.
>
> See [`docs/heterogeneity.md`](heterogeneity.md) for the full taxonomy, the
> two-level rationale (task-level enters via the prompt and is robust; env-level
> enters via the transition kernel and is worst-case non-robust), and the
> variant ↔ pipeline-stage mapping.

---

## 3. Add an RL algorithm (beyond PPO / GRPO)

### Where

The RL update lives entirely in the vendored verl-agent trainer
(`third_party/verl-agent/verl/trainer/`), selected by
`verl.algorithm.adv_estimator`. The federated layer never touches it.

### The contract

1. **Add the advantage estimator / loss** under the trainer's algorithm switch
   so `verl.algorithm.adv_estimator: my_algo` resolves. verl-agent ships PPO,
   GRPO, GiGPO, RLOO, and DAPO upstream; a new estimator is added alongside
   them in the trainer's advantage-computation dispatch. (FedAgent's paper uses
   GRPO for the main tables and PPO in the appendix; both are already wired.)

2. **Surface its hyperparameters** under the `verl:` config block (mirror the
   `verl.algorithm.*` and `verl.actor_rollout_ref.*` fields the existing
   estimators read).

3. **Provide a base launch script** `scripts/verl-agent/<algo>/run_<env>.sh`
   analogous to the existing `grpo/` and `ppo/` scripts. The federated server
   calls it once per client via `federated.base_script_path`
   (`core/fed/script_builder.py`), so the script is the contract between the
   server and the trainer, keep its argument/env-var interface identical to the
   GRPO/PPO ones.

4. **No federated changes are required.** `core/` aggregates whatever
   checkpoints the local trainer writes, and `utils/model_aggregation.py`
   operates on the FSDP shard layout, not on algorithm internals. The **one**
   thing to watch is the checkpoint *format*: if your algorithm adds trainable
   components beyond the actor, PPO already adds a **critic**, which
   `aggregate_verl_models` discovers via `_find_fsdp_critic_dir` and aggregates
   separately, make sure those extra components land under the same
   `checkpoints/global_step_<n>/<component>/` FSDP layout so aggregation finds
   and averages them. An actor-only algorithm needs nothing extra.

---

## 4. Add an aggregation strategy (beyond FedAvg / FedProx)

### Where

`utils/model_aggregation.py` is the live aggregation path. It is invoked once
per round from `core/fed/aggregator.py`, which reads
`federated.aggregation_method` and calls the module-level entry point:

```python
# utils/model_aggregation.py
def aggregate_round_models(
    round_num: int,
    client_results: List[Dict[str, Any]],   # per-client {client_id, model_path, success, ...}
    output_dir: Path,
    aggregation_method: str = 'fedavg',      # <- federated.aggregation_method
    n_gpus_per_node: int = 1,
    **kwargs,                                # extra rule-specific args
) -> Dict[str, str]:                         # {'actor': path, 'critic': path?}
```

Today both `'fedavg'` and `'fedprox'` dispatch to the same uniform FedAvg here
(and it raises on anything else). FedProx is not a distinct server rule: it adds a
proximal term to each **client's** local objective (`verl/workers/actor/dp_actor.py`),
leaving server aggregation as FedAvg. The heavy lifting is in the `ModelAggregator`
class in the same file.

### The contract

1. **Add a branch** for `aggregation_method == 'my_rule'` in
   `aggregate_round_models`, and update the `else: raise ValueError(...)`. Your
   branch returns a `{component_name: aggregated_path}` dict (at least
   `'actor'`, plus `'critic'` when present), each path pointing at a
   `global_step_0` FSDP directory, that is the format the next round's clients
   load from.

2. **Operate on FSDP-sharded checkpoints.** Clients save FSDP shards
   (`model_world_size_*_rank_*.pt` under
   `checkpoints/global_step_<n>/<component>/`). The two reusable primitives in
   `ModelAggregator` are:

   - `fedavg_aggregation(model_paths, output_path, weights=None, model_type="actor", n_gpus_per_node=1)`
     loads each client's shards, averages parameters (`average_models`,
     optionally weighted), then **reshards** to `n_gpus_per_node` and writes a
     fresh `global_step_0` directory. `aggregate_verl_models` is the
     higher-level driver that locates each client's actor (and critic) shard
     dir, checks that all clients are at a consistent `global_step`, and calls
     the shard aggregator per component.
   - For a server-side rule that needs the prior global model, accept it through
     `aggregate_round_models`'s `**kwargs` and load it alongside the client shards.

   Most new rules (trimmed mean, median, FedAvgM, per-client weighting by
   dataset size, …) can be implemented by writing a new `average_*` routine and
   reusing `fedavg_aggregation`'s load → average → reshard skeleton.

3. **Keep the resharding subprocess intact.** `reshard_model` /
   `direct_shard_aggregation` shell out via `torchrun` to
   `tools/aggregation/create_fsdp_shards.py` (a `subprocess.run` call in
   `_multi_gpu_fsdp_aggregation`). That script is a **runtime dependency**, not
   just a dev tool, do not change its CLI when extending aggregation.

4. **Weighting hook.** `fedavg_aggregation` and `average_models` already accept
   an optional `weights` list. If your rule is "FedAvg but weighted by
   `|X_i|`/round participation", you do not need a new dispatch branch, compute
   the weights from `client_results` and pass them through.

### Validate

The `tools/aggregation/` toolbox checks correctness against the
average-of-clients ground truth:

- `tools/aggregation/check_aggregation.py`, has a proper CLI; diff per-client
  vs aggregated weights:

  ```bash
  python tools/aggregation/check_aggregation.py \
      --aggregated-dir <round>/aggregated \
      --client-dirs <round>/client_14 <round>/client_81
  ```

- `tools/aggregation/verify_aggregation.py`, asserts the aggregated tensor is
  exactly the mean of the client tensors (to `tolerance=1e-6`). It currently
  hardcodes example paths in `main()` rather than taking flags, so point its
  `aggregated_path` / `client*_path` at your run before running it.
- `tools/aggregation/verl_fsdp_aggregation.py`, a standalone reference FSDP
  aggregation implementation to diff your output against.
- `tools/aggregation/fix_dtensor_loading.py`, DTensor-loading helper for
  reading sharded checkpoints saved under newer PyTorch.

Then add a regression check under `tools/heterogeneity_test/` covering the new rule
(it has no aggregation test yet; `tools/aggregation/check_aggregation.py` and
`tools/aggregation/verify_aggregation.py` are the existing correctness checks to
model it on).

---

## General notes

- The federated control loop (`core/`, `tools/run_federated.py`) and the
  aggregation (`utils/`) are decoupled from the RL trainer
  (`third_party/verl-agent`), so extension points 1–3 mostly live in the
  vendored framework while 4 lives in `utils/`.
- Reproduce / evaluate with the standard entry points, no extension needs a new
  runner:

  ```bash
  bash reproduce.sh <experiment> [--gpus N] [--mode fed|serial] [--fsdp on|off] [--single-gpu] [--slurm]
  bash evaluate.sh  <webshop|alfworld> <checkpoint>
  ```

  Default is 4 × H100, non-SLURM. See
  [`docs/running.md`](running.md) for the run-mode
  matrix and [`docs/configuration.md`](configuration.md) for the config-field
  reference. W&B has been removed from the public release; metrics are written
  under `output/`.
- Cross-references: [`docs/heterogeneity.md`](heterogeneity.md) (the two-level
  taxonomy and the knob-naming caveats: `omega` vs the legacy `tau` alias — which
  collides with the paper's task-descriptor τ — and `size_std`/`success_std`,
  which despite their names equal the paper's Beta-concentration ξ/ξ' directly),
  [`docs/installation.md`](installation.md) (the two conda environments),
  [`docs/reproducing.md`](reproducing.md) (config-group → paper-artifact
  mapping).
</content>
</invoke>
