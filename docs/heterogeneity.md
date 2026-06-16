# Heterogeneity Construction

This document is the conceptual core of FedAgent. It describes the **two-level
heterogeneity taxonomy** the paper builds on, *task-level* (axis 1) and
*environment-level* (axis 2), and, for each level, the concrete constructors,
the configuration knobs that select and parameterize them, and how to add your
own. It is meant to be read alongside the code it points at, so every path,
config key, strategy name, and function below has been written against the
release tree and can be opened directly.

Source: paper §Environment-Level Heterogeneity and the Heterogeneity
Construction Protocol appendix.

**Contents**

1. [Why two levels?](#why-two-levels)
2. [Task-level heterogeneity (axis 1)](#task-level-heterogeneity-axis-1)
3. [Environment-level heterogeneity (axis 2)](#environment-level-heterogeneity-axis-2)
4. [How a variant is selected (config -> env vars -> dispatch)](#how-a-variant-is-selected-config---env-vars---dispatch)
5. [What the experiments sweep](#what-the-experiments-sweep)
6. [Extension point: adding a new strategy](#extension-point-adding-a-new-strategy)

---

## Why two levels?

A FedAgent client trains on a *task-augmented MDP*: each episode is drawn from a
task distribution $\mathcal{D}_\tau$ and rolled out under a transition kernel
$P$. Heterogeneity across clients can therefore enter through **either** of two
structurally different channels, and the entire framework is organized around
keeping those two channels separable:

- A **task descriptor** $\tau$ enters the policy through its *input channel*: it is literally part of the prompt the agent reads. The policy can therefore
  condition on it, so $\tau$ is **observable**. Task-level heterogeneity is the
  *robust* case: it is **Pattern A** in the paper's taxonomy.
- A **transition kernel** $P$ is implicit in the dynamics. The policy never sees
  $P$ directly; it only senses it through the successor states the environment
  returns. $P$ is therefore **not observable**, and a perturbation of $P$ can be
  *worst-case non-robust*.

This single distinction, call it the **Input-Dynamics Asymmetry**: is what
drives the paper's asymmetric-robustness result. FedAgent ships independent
constructors for both levels precisely so the two axes can be swept on their own
and the asymmetry can be measured rather than assumed. Throughout the task-level
sweep the transition kernel is held fixed; throughout the environment-level
sweep the task partition is held **uniform**, so any divergence in the env-level
plots is attributable to the transition perturbation alone.

All heterogeneity constructors live in one file:

```
third_party/verl-agent/agent_system/environments/partition_strategy.py
```

and are wired into training through two seams: `partition_dataset(...)` (the
task-level dispatcher) and `fed_env_manager.py` (which applies the WebShop
environment-level perturbations). Section 4 walks the full path from a YAML key
to the function that runs.

---

## Task-level heterogeneity (axis 1)

Clients share **one** environment tuple but differ in their per-client task
distribution $\mathcal{D}_{\tau_i}$. The framework defines three operationally
separable sub-types, each governed by a *single* dispersion hyperparameter so
that one axis can be moved without disturbing the other two:

| Paper name | Code strategy | Filename token | Config kwarg | Exported env var | Endpoints (near-uniform -> extreme) | What it disperses |
|---|---|---|---|---|---|---|
| **Preference** (*what kind of task?*) | `preference` | `preference` | `omega` | `OMEGA` (+ `TAU` alias) | `0.01` -> `0.99` | per-client category marginal |
| **Coverage** (*how many tasks?*) | `coverage` | `coverage` | `size_std` | `SIZE_STD` | `1` -> `256` | per-client pool size |
| **Hardness** (*how hard are the tasks?*) | `hardness` | `hardness` | `success_std` | `SUCCESS_STD` | `1` -> `256` | per-client success-rate mix |

> **Naming caveat, read this once and the codebase stops being confusing.**
> *Preference* heterogeneity is spelled three different ways across the stack:
> the **code** dispatch strategy is `preference`, the **paper** calls it
> *Preference*, and the **config filename** uses the token `preference`. They are
> the same construction. Coverage is consistent everywhere. Hardness keeps its
> historical misspelling `hardness` in the code and filenames (the paper writes
> *Hardness*). When you grep, search for `preference` / `coverage` / `hardness`;
> when you read the paper, translate to *Preference* / *Coverage* / *Hardness*.

### Constructions

All three are implemented in `partition_strategy.py` and reached through
`partition_dataset(..., strategy=<name>, ...)`. Each has a WebShop and an
ALFWorld backend (the ALFWorld variant derives its category from the task-file
path rather than a `preference` field).

- **Preference, `preference_partition` (Dirichlet, $\omega$).**
  Each client's category distribution is drawn from a Dirichlet centered on the
  global category marginal $\pi$:

  ```python
  # partition_strategy.py, _preference_partition_generic / _preference_partition_alfworld
  omega = float(np.clip(omega, 1e-3, 1 - 1e-3))
  alpha_vec = pi * ((1.0 - omega) / omega)
  q = rng.dirichlet(alpha_vec)        # this client's per-category probabilities
  ```

  Per-category counts are then drawn by multinomial sampling from `q`. As
  $\omega \to 0$ the concentration $\alpha$ grows without bound and every client
  converges to the uniform global marginal (near-IID); as $\omega \to 1$ the
  concentration collapses and each client is pushed toward a one-hot vertex
  (one client = one category). `omega` is the canonical kwarg; older configs
  that pass only `tau` are aliased to it (`omega` wins if both are present), and
  the value is clipped into $(10^{-3}, 1-10^{-3})$ before use. Sweep endpoints
  used in the paper: `omega = 0.01` (near-uniform) and `omega = 0.99` (extreme).

- **Coverage, `coverage_partition` (Beta sizes, fixed overlap).**
  Each client's *pool size* is drawn from a Beta-shaped distribution while the
  cross-client overlap is held at `overlap_ratio = 1.3`, and the union of client
  pools is kept covering the dataset as far as possible. This changes the
  **spread** of how many tasks each client sees without changing the per-client
  mean or the global task mixture. `size_std` controls the spread (exported as
  `SIZE_STD`); endpoints `1` (nearly equal pool sizes) and `256` (extreme size
  imbalance).

- **Hardness, `hardness_partition` (success-rate quotas).**
  Tasks are first labelled success/fail by a reference checkpoint (the zero-shot
  backbone) recorded in a trajectories file; each client is then given a quota of
  "success" tasks drawn from a Normal-shaped distribution over
  `[0, min_goals_per_client]`, and the remainder of its fixed quota is filled
  with random tasks. The number of tasks per client stays constant, only the
  *difficulty mix* shifts. `success_std` controls the spread (exported as
  `SUCCESS_STD`); endpoints `1` (uniform difficulty) and `256` (extreme; some
  clients see almost only solvable tasks, others almost only hard ones).
  *Prerequisite:* the reference trajectories file is produced by the eval
  harness, run `bash eval/eval_webshop.sh` (resp. `eval/eval_alfworld.sh`) first
  to write `output/inference/all_trajectories.json` (resp.
  `all_trajectories_alfworld.json`), which `hardness_partition` reads by default.

By construction each axis offers (D1) target control of its own dispersion,
(D2) invariance of the *other* two measures, (D3) factor invariance (the global
mixture is preserved in expectation), and (D4) joint configurability, so the
three can be combined or varied one at a time. The formal statements and proofs
are in the paper appendix.

### Why task-level is the robust (Pattern A) case

Because $\tau$ is in the prompt, a single aggregated policy can read each
client's descriptor and act accordingly; FedAvg over task-heterogeneous clients
therefore converges to a policy that does well on the *union* of task
distributions. This is the empirical content of "task-level heterogeneity is
robust," and it is the contrast the environment-level axis is designed to break.

---

## Environment-level heterogeneity (axis 2)

Clients share the task distribution but differ in their **transition kernel**
$P_i$. The task partition is held **uniform** across every env-level experiment
(so the only thing that varies is $P$), and, critically, **validation is run
on the unperturbed environment** so that all clients are scored on the same
yardstick (see [§4](#how-a-variant-is-selected-config---env-vars---dispatch)).

WebShop's transition pipeline factors into **four stages**, and the five env
variants instantiate perturbations across them. The four stages, in pipeline
order, are:

1. **content**: *what is in the catalog* (the set of products the search can
   ever return);
2. **encoding**: *how a product is turned into indexed text* (which fields feed
   the retriever);
3. **matching**: *how a query is scored against that text* (the BM25 ranking
   function);
4. **rendering**: *how the ranked results are presented* to the agent (order /
   wrapping of the result page).

| Variant (paper) | Pipeline stage | Strategy key | Variant pool / data | Config dir | Pattern elicited |
|---|---|---|---|---|---|
| **Catalog Split** | content | `catalog_split` | per-client catalog floor + distractor pool | `config/env_heterogeneity/catalog_split/` | **B / C** |
| **Field-Subset Index** | encoding | `bm25_variant` (`variant_pool: fields_only`) | `BM25_VARIANTS_FIELDS_ONLY` | `config/env_heterogeneity/field_subset_index/` | **C** |
| **BM25 Reweighting** | matching | `bm25_variant` (default pool) | `BM25_VARIANTS_DEFAULT` (extreme `k1`,`b`) | `config/env_heterogeneity/bm25_reweighting/` | **C** |
| **Lookalike Injection** | content + matching | `lookalike_injection` | `data/env_heterogeneity/lookalike_data/*.json` | `config/env_heterogeneity/lookalike_injection/` | **D** (GRPO) -> **C** (PPO) |
| **Rank Wrapper** | rendering | `rank_wrapper` | `SEARCH_ENGINE_VARIANTS_DEFAULT` | `config/env_heterogeneity/rank_wrapper/` | **D** (GRPO) -> **C** (PPO) |

Each variant directory has a `*_ppo` sibling
(`catalog_split_ppo/`, `field_subset_index_ppo/`, ...), which is what produces the PPO half of
the env-heterogeneity figure
(`webshop_env_variants_combined_val_success_rate.pdf`, GRPO on the left, PPO on
the right).

The **B/C/D** column refers to the paper's robustness spectrum: **B/C** = the
optimal policy still largely transfers and FedAvg degrades gracefully; **C** =
divergence that is real but recoverable; **D** = the worst case where naive
aggregation breaks down under GRPO. The two strongest attacks (Lookalike
Injection and Rank Wrapper) land in **D** under GRPO but are *rescued back to C
under PPO*; this GRPO->PPO rescue is one of the paper's headline observations.

### What each variant does

- **Catalog Split, `catalog_split` (content).**
  Each client is assigned a different slice of the product catalog: a protected
  per-client floor of *target* ASINs (so every client can still complete its
  goals) plus a per-client *distractor* pool drawn so the catalogs diverge. The
  optimal "search -> click -> buy" behavior is unchanged by *which* products are
  present, so $\pi^\star$ stays essentially invariant; this is the mildest
  perturbation (Pattern B/C). The `_v5` algorithm differs from the older v4
  (`distractor_disjoint`): the task partition is `uniform` (100 goals/client,
  matching the main experiment) and the env partition protects only each
  client's own targets (~50-80 ASINs) against a ~920-item distractor pool, and
  it returns explicit `client_goal_idxs` so WebShop no longer hard-codes the
  held-out goal range.
  Config kwargs: `env_div` (divergence strength in $[0,1]$),
  `keep_ratio` (per-client distractor density), `search_return_n` (BM25 top-K).
  Implemented by `_distractor_disjoint_partition_webshop_v5`.

- **Field-Subset Index, `bm25_variant` with `variant_pool: fields_only` (encoding).**
  Each client indexes a *different subset of document fields*, so the same query
  ranks products differently and the agent must learn per-client query crafting.
  The pool is `BM25_VARIANTS_FIELDS_ONLY` (all share `k1=1.2, b=0.75`; only the
  fields differ): the first four entries are
  `full {name,Title,description,features,BulletPoints}`,
  `name {name,Title}`, `desc {description}`, and `bullets {BulletPoints}`
  (entries 5-8 extend it for `N>4`).
  Config kwargs: `N` (number of variants in play), `variant_pool: fields_only`,
  `search_return_n`. This is the **encoding**-stage perturbation and sits at
  Pattern **C**.

- **BM25 Reweighting, `bm25_variant` with the default pool (matching).**
  Same dispatch as Field-Subset, but with `variant_pool` omitted, so the pool is
  `BM25_VARIANTS_DEFAULT`: all variants index the **full** field set but use
  **extreme `(k1, b)` corners** that reshape TF saturation and length
  normalization. The first four entries (the `N=4` sweep) are the default
  `(k1=1.2, b=0.75)` plus the corners `(1.2, 0.00)`, `(0.3, 0.75)`, and
  `(5.0, 0.75)` (entries 5-8, e.g. `(0.1,0.75)`, `(1.2,1.00)`, `(2.0,0.50)`,
  `(0.3,0.00)`, extend it for `N>4`). Because only *ranking* changes, the
  catalog and fields are identical across clients; this is the **matching**-stage
  perturbation, Pattern **C**.
  Config kwargs: `N`, `search_return_n` (no `variant_pool`).
  Implemented by `_bm25_variant_partition_webshop`.

- **Lookalike Injection, `lookalike_injection` (content + matching).**
  The strongest *content* attack: each client gets a per-client set of synthetic
  **lookalike products** appended to the base 1000-product catalog. The
  lookalikes are tuned to fool BM25 ranking *and* to defeat one specific reward
  subterm, so the agent is forced to learn to check a particular attribute
  (price, color, ...) to filter out the fakes. Because different clients attack
  different attributes, their optimal policies diverge structurally; this spans
  **content + matching** and elicits Pattern **D** under GRPO (rescued to **C**
  under PPO). The default `N=2` covers the two reward-validated attacks
  (`v_price`, `v_color`); `N>=3` adds `v_size` and `v_price_color`. The lookalike
  product JSON lives in `data/env_heterogeneity/lookalike_data/`
  (`lookalike_v_price.json`, `lookalike_v_color.json`, `lookalike_v_size.json`,
  `lookalike_v_price_color.json`).
  Config kwargs: `N`, `search_return_n`.
  Implemented by `_lookalike_injection_partition_webshop` (file paths are
  resolved against `PROJECT_ROOT`, exported for you by the launcher).

- **Rank Wrapper, `rank_wrapper` (rendering).**
  Each client's search results are post-processed by a different *wrapper* on top
  of the same BM25 base, breaking any "trust the top position" heuristic while
  preserving the reward gradient (the target stays reachable in the candidate
  set, avoiding the degenerate case where some clients can never see a reward).
  The pool is `SEARCH_ENGINE_VARIANTS_DEFAULT` (`N=4`):
  `v_bm25_default` (control), `v_shuffled_topk` (shuffle the top 50),
  `v_inverted_topk` (reverse the top-K), and `v_partial_random` (50% of queries
  return random results). This is the **rendering**-stage perturbation, Pattern
  **D** under GRPO (rescued to **C** under PPO).
  Config kwargs: `N`, `search_return_n`.
  Implemented by `_rank_wrapper_partition_webshop`.

**Mapping back to the four stages:** Catalog Split = *content*, Field-Subset
Index = *encoding*, BM25 Reweighting = *matching*, Rank Wrapper = *rendering*,
and Lookalike Injection straddles *content + matching*. Together the five
variants cover all four transition stages.

> **ALFWorld env-level.** The env-level axis on ALFWorld is a *scene-disjoint*
> partition selected with `strategy: env_disjoint` (handled inside
> `partition_dataset`, ALFWorld-only). The five variants above are
> WebShop-specific because they perturb WebShop's retrieval pipeline.

### `search_return_n` and validation

Every WebShop env-level config sets `search_return_n: 200`. The env-level
machinery **requires** `WEBSHOP_SEARCH_RETURN_N >= 100` and aborts loudly
otherwise: raising the BM25 top-K keeps the rendered result page full even after
aggressive per-client filtering, so a target is never silently dropped before
the agent can act on it.

Validation is forced onto the **unperturbed** environment regardless of the
training perturbation. In `fed_env_manager.py` the validation env kwargs null
out every perturbation channel:

```python
# fed_env_manager.py (WebShop val env)
val_env_kwargs = {
    **env_kwargs,
    'catalog_filter_asins': None,   # Catalog Split off
    'bm25_in_memory_config': None,  # Field-Subset / BM25 Reweighting off -> default Lucene
    'extra_products': None,         # Lookalike Injection off
    'search_engine_variant': None,  # Rank Wrapper off
}
```

so all clients are scored on the **full Lucene index over the full
1000-product catalog**. The validation seed is `config.env.seed + 1000`
(i.e. `1000`, with `env.seed = 0`); this is what makes the cross-variant curves
comparable.

---

## How a variant is selected (config -> env vars -> dispatch)

Heterogeneity is selected entirely from the YAML config, through the
`federated.data_sharding.partition` block. The user-facing interface is the
same for **both** levels:

```yaml
federated:
  data_sharding:
    seed: 42
    min_goals_per_client: 100
    partition:
      strategy: "bm25_variant"   # the dispatch key
      kwargs:
        N: 4
        variant_pool: "fields_only"       # Field-Subset Index; omit for BM25 Reweighting
        search_return_n: 200
```

Under the hood there are **two different dispatch paths**, and it matters which
one a given strategy takes:

```
config.federated.data_sharding.partition.{strategy, kwargs}
        |
        v   core/fed/script_builder.py
  modify_script_for_federated()        -> export PARTITION_STRATEGY="<strategy>"
  _get_partition_strategy_env_vars()   -> export the kwargs as env vars:
        preference -> OMEGA, TAU
        coverage   -> SIZE_STD
        hardness   -> SUCCESS_STD
        catalog_split -> ENV_DIV, KEEP_RATIO, MIN_GOALS_PER_CLIENT,
                                  WEBSHOP_SEARCH_RETURN_N [, HOLDOUT_FILE]
        bm25_variant  -> N_VARIANTS [, BM25_VARIANT_POOL] [, WEBSHOP_SEARCH_RETURN_N]
        rank_wrapper-> N_VARIANTS [, WEBSHOP_SEARCH_RETURN_N]
        lookalike_injection -> N_VARIANTS, PROJECT_ROOT [, WEBSHOP_SEARCH_RETURN_N]
        env_disjoint (ALFWorld)-> ENV_DIV, FALLBACK [, HOLDOUT_FILE]
        |
        v   third_party/verl-agent/agent_system/environments/fed_env_manager.py
  reads PARTITION_STRATEGY (+ the env vars above) and EITHER
   (a) TASK-LEVEL: calls partition_dataset(strategy=..., ...)
       -> uniform | preference | coverage | hardness | env_disjoint
   (b) ENV-LEVEL (WebShop): applies the perturbation INLINE by calling the
       helper directly and stuffing the result into env_kwargs:
         catalog_split      -> env_kwargs['catalog_filter_asins'], ['client_goal_idxs']
         bm25_variant       -> env_kwargs['bm25_in_memory_config']
         lookalike_injection  -> env_kwargs['extra_products']
         rank_wrapper     -> env_kwargs['search_engine_variant']
```

> **Important correctness note (this is a common misreading).** The four WebShop
> **environment-level** strategies are **NOT** routed through
> `partition_dataset()`. `partition_dataset()` only dispatches the task-level
> strategies plus the ALFWorld `env_disjoint` split; it explicitly **raises** if
> you hand it `catalog_split`:
>
> ```python
> # partition_strategy.py, partition_dataset()
> elif strategy == 'catalog_split':
>     raise NotImplementedError(
>         "catalog_split: call _distractor_disjoint_partition_webshop_v5 "
>         "directly from fed_env_manager.py, not via partition_dataset()")
> ...
> else:
>     raise ValueError(f"Unknown partition strategy: {strategy}. Supported "
>         "strategies: uniform, preference, coverage, hardness, env_disjoint, "
>         "catalog_split")
> ```
>
> The WebShop env variants are applied **inline in `fed_env_manager.py`** because
> they need the loaded products / attributes / goals to build the per-client
> catalog, BM25 config, lookalikes, or search wrapper, and the result has to be
> threaded into `env_kwargs` rather than returned as a data slice. So: pick the
> strategy in YAML, and let `script_builder.py` + `fed_env_manager.py` route it.

The `variant_pool` kwarg is what distinguishes **Field-Subset Index** from
**BM25 Reweighting** even though both use `strategy: bm25_variant`:
`variant_pool: fields_only` selects `BM25_VARIANTS_FIELDS_ONLY` (encoding),
while omitting it (or `default`) selects `BM25_VARIANTS_DEFAULT` (matching).

Per-client variant assignment is **deterministic by `client_id`**
(`np.random.RandomState(base_seed + client_id)`, `base_seed = 42`) so the same
client keeps the same variant across rounds; this is required for FedAvg to
average comparable policies round to round.

---

## What the experiments sweep

The config groups under `config/` map to the paper's figures and tables; the
env-level and task-level sweeps relevant to this document are:

- **Task-level**: `config/task_heterogeneity/{grpo,ppo}/{webshop,alfworld}/`
  produce the heterogeneity-challenges figure
  (`heterogeneous_combined_val_success_rate.pdf`, six panels: a,b Preference;
  c,d Coverage; e,f Hardness). Sweep endpoints: Preference `omega ∈ {0.01, 0.99}`,
  Coverage `size_std ∈ {1, 256}`, Hardness `success_std ∈ {1, 256}`, encoded in
  the filenames as `preference_omega-*`, `coverage_std-*`, `hardness_success_std-*`.
- **Environment-level**: `config/env_heterogeneity/{catalog_split,field_subset_index,bm25_reweighting,lookalike_injection,rank_wrapper}/`
  (plus the `_ppo` siblings) produce the env-heterogeneity figure
  (`webshop_env_variants_combined_val_success_rate.pdf`, GRPO left / PPO right).
  Catalog Split additionally sweeps `env_div ∈ {0.0, 0.3, 0.7, 1.0}` at
  `keep_ratio = 0.7` (four YAMLs in `catalog_split/`); the BM25/field-subset variants
  sweep `N ∈ {4, 8}`; Lookalike Injection sweeps `N ∈ {2, 4}`; Rank Wrapper is
  `N = 4`. These multi-point sweeps exist only in the GRPO directories, each
  `*_ppo` sibling contains a single config (the most-divergent sweep point used
  for the GRPO-vs-PPO comparison), not the full sweep.

For the full config-to-artifact mapping (including the uniform/main tables and
the decentralized ablations) see [`docs/reproducing.md`](reproducing.md).

---

## Extension point: adding a new strategy

Where you add code depends on which level you are extending.

### A) A new *task-level* strategy (routed through `partition_dataset`)

1. Implement
   `my_strategy_partition(data, client_id, client_num, min_samples_per_client, start_idx=0, data_type='generic', **kwargs)`
   in `partition_strategy.py`, returning the same shape as the existing
   task-level strategies (the client's data slice; `uniform` additionally returns
   slice bounds).
2. Add an `elif strategy == 'my_strategy':` branch to **`partition_dataset()`**
   (and, if you want `get_partition_info()` to describe it, add a matching branch
   there) and update the `ValueError` "Supported strategies: ..." list so the
   error message stays accurate.
3. Teach the launcher to pass your kwarg through as an env var: add a
   `if strategy == 'my_strategy':` case to
   `_get_partition_strategy_env_vars()` in `core/fed/script_builder.py` that
   `export`s whatever `fed_env_manager.py` will read.
4. Select it from a config:
   `federated.data_sharding.partition.strategy: "my_strategy"` with the matching
   `kwargs`.

### B) A new *environment-level* (WebShop transition) variant

1. Implement a helper in `partition_strategy.py` that returns a per-client config
   object (mirroring `_bm25_variant_partition_webshop` /
   `_rank_wrapper_partition_webshop`): seed it deterministically with
   `np.random.RandomState(base_seed + client_id)` so assignment is stable across
   rounds.
2. Add a `if partition_strategy_env == 'my_env_variant':` block to
   `fed_env_manager.py` that calls your helper and writes the result into
   `env_kwargs[...]`, **and** add the matching key to the validation override
   (`val_env_kwargs[...] = None`) so validation stays on the unperturbed env.
3. Add the env-var export in `script_builder.py`
   (`_get_partition_strategy_env_vars`) for your kwargs.
4. If your variant needs the search backend to behave differently, wire the new
   `env_kwargs` key through the WebShop env package
   (`environments/env_package/webshop/`) where the existing
   `bm25_in_memory_config` / `extra_products` / `search_engine_variant` keys are
   consumed.

In both cases, keep the deterministic-by-`client_id` assignment and the
unperturbed-validation invariant; they are what make federated runs reproducible
and the cross-variant curves comparable.

See [`docs/extending.md`](extending.md) for the broader extension contract
(new environments/datasets, new RL algorithms, and new aggregation rules).
