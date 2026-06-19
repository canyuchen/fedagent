# Configuration Reference

Every FedAgent experiment is driven by a single YAML file. The file has four
top-level blocks:

| Block | Purpose | Documented in |
|---|---|---|
| `federated:` | FedAgent federation parameters, the contribution of this work (client population, rounds, aggregation, data sharding / heterogeneity) | §(b), in full below |
| `verl:` | Training knobs forwarded to the vendored `verl-agent` trainer (RL algorithm, backbone, batch sizes, rollout/FSDP) | §(c), FedAgent-relevant subset only; the rest is upstream |
| `data_preprocess:` | Parquet generation for the chosen task (mode + train/val sizes) | §(d) |
| ~~`wandb:`~~ | **Removed in this release.** See the note at the end of §(c). | |

The curated configs that reproduce every figure and table in the paper live
under `config/`. This document explains how to read those files and how
to write your own. A field-by-field annotated scaffold lives at
`config/example.yaml`; pair it with the tables here and a working file from
`config/` when authoring a new run.

The runnable, verified reference used throughout this document is:

```
config/env_heterogeneity/catalog_split/
  fed_webshop_grpo_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-catalog_split_div-0.3_keep-0.7.yaml
```

For the heterogeneity construction itself (what each partition strategy *does*,
the env-variant taxonomy, and the naming caveats) see
[`docs/heterogeneity.md`](heterogeneity.md). For how to launch a config see
[`docs/running.md`](running.md) and
[`docs/reproducing.md`](reproducing.md).

---

## (a) Filename naming convention

Config filenames are self-describing: the federation protocol can be read off
the name without opening the file. The scheme is

```
fed_<env>_<algo>_total-<N>_cl-per-rd-<k>_rd-<R>_ep-per-cl-<E>_min-goals-per-cl-<G>_p-<partition>[_<kw>...].yaml
```

| Token | Meaning | Maps to YAML field |
|---|---|---|
| `fed_` | fixed prefix (all configs are federated runs; the centralized and local baselines are special cases, see below) | (all runs are federated) |
| `<env>` | environment: `webshop` or `alfworld` | `verl.env.env_name` (`Webshop` / `alfworld/AlfredTWEnv`) |
| `<algo>` | RL algorithm: `grpo` or `ppo` | `verl.algorithm.adv_estimator` (`grpo` / `gae`) |
| `total-<N>` | total client population *N* | `federated.total_clients` |
| `cl-per-rd-<k>` | clients sampled per round *M* | `federated.clients_per_round` |
| `rd-<R>` | communication rounds *T* | `federated.total_rounds` |
| `ep-per-cl-<E>` | local epochs per selected client *E* | `federated.epochs_per_client` |
| `min-goals-per-cl-<G>` | minimum tasks (goals) per client \|X_i\| | `federated.data_sharding.min_goals_per_client` |
| `p-<partition>` | partition strategy *and* the filename spelling of its key hyperparameter(s) | `federated.data_sharding.partition.strategy` plus `partition.kwargs` (see the caveat below) |
| `[_<kw>...]` | optional trailing tokens spelling out the partition kwargs, e.g. `omega-0.99`, `std-256`, `N-4`, `div-0.3_keep-0.7` | `partition.kwargs.*` |

The default protocol, the one used for the main table, is therefore
`total-100_cl-per-rd-2_rd-70_ep-per-cl-3`: **100 clients, M = 2 sampled per
round, E = 3 local epochs per round, T = 70 rounds**, which is
`E × T = 210` total local epochs of training.

### Filename → partition decoder

The `p-<...>` segment is the only token that is *not* a verbatim copy of the
YAML value: the filename uses a human-friendly spelling, while `partition.strategy`
uses the code's internal dispatch key. The mapping (cross-check against the
heterogeneity doc):

| Filename `p-...` | `partition.strategy` | `partition.kwargs` keys | Paper name |
|---|---|---|---|
| `p-uniform` | `uniform` | (none) | homogeneous (IID baseline) |
| `p-preference_omega-<ω>` | `preference` | `omega` | **Preference** heterogeneity |
| `p-coverage_std-<s>` | `coverage` | `size_std` | **Coverage** heterogeneity |
| `p-hardness_success_std-<s>` | `hardness` | `success_std` | **Hardness** heterogeneity |
| `p-catalog_split_div-<d>_keep-<r>` | `catalog_split` | `env_div`, `keep_ratio`, `search_return_n` | Catalog Split (env, content) |
| `p-field_subset_index_N-<n>` | `bm25_variant` | `N`, `variant_pool: fields_only`, `search_return_n` | Field-Subset Index (env, encoding) |
| `p-bm25_reweighting_N-<n>` | `bm25_variant` | `N`, `search_return_n` | BM25 Reweighting (env, matching) |
| `p-lookalike_injection_N-<n>` | `lookalike_injection` | `N`, `search_return_n` | Lookalike Injection (env, content+matching) |
| `p-rank_wrapper_N-<n>` | `rank_wrapper` | `N`, `search_return_n` | Rank Wrapper (env, rendering) |

> **⚠️ Naming caveat (Preference: the `omega` vs. legacy `tau` knob).** The
> *strategy* key is consistent everywhere, `preference` in the code, the paper
> (*Preference*), and the filename, so a config reads
> `..._p-preference_omega-0.99.yaml` with `strategy: "preference"` and
> `kwargs.omega: 0.99`. The genuine source of confusion is the **hyperparameter
> name**, not the strategy name. The preference knob is `omega` (the Dirichlet
> spread/skew parameter $\omega$; the concentration is $\alpha = \pi\,(1-\omega)/\omega$,
> so larger $\omega$ means *more* heterogeneity), but older configs pass a kwarg
> called `tau`, which the preference partition aliases to `omega` when `omega` is
> absent (`omega` wins if both are present; see `partition_strategy.py`, where the
> per-backend helpers `_preference_partition_generic` / `_preference_partition_alfworld`
> apply `omega = tau`). Two things to keep straight: (1) prefer `omega` in new
> configs, `tau` only survives for backward compatibility; (2) this code `tau` is
> **unrelated** to the paper's symbol $\tau$, which denotes the *task descriptor*
> (the observable task input), not the preference-skew knob. See
> [`docs/heterogeneity.md`](heterogeneity.md#task-level-heterogeneity-axis-1).

> **Note (Field-Subset vs. BM25 Reweighting).** Both Field-Subset Index
> (`field_subset_index`) and BM25 Reweighting (`bm25_reweighting`) dispatch through
> the *same* strategy key, `bm25_variant`. They are distinguished by `variant_pool`:
> `field_subset_index` sets `variant_pool: "fields_only"` (vary which catalog fields
> enter the BM25 document text); `bm25_reweighting` omits `variant_pool` and instead
> perturbs the BM25 `k1`/`b` scoring corners. The directory (`field_subset_index` vs
> `bm25_reweighting`) and the filename (`p-field_subset_index_N-*` vs `p-bm25_reweighting_N-*`) keep
> them apart.

### Sweep endpoints in the filenames

For the task-heterogeneity figure each axis is swept between a near-uniform and
an extreme endpoint, visible directly in the filenames:

| Axis | Near-uniform | Extreme |
|---|---|---|
| Preference | `omega-0.01` | `omega-0.99` |
| Coverage | `std-256` (high Beta concentration → near-uniform) | `std-1` (low concentration → skewed) |
| Hardness | `success_std-256` | `success_std-1` |

For the env-heterogeneity figure, the `catalog_split` directory sweeps `env_div ∈
{0.0, 0.3, 0.7, 1.0}` at fixed `keep_ratio: 0.7`; the
`field_subset_index`/`bm25_reweighting`/`lookalike_injection`/`rank_wrapper`
directories sweep the variant count `N` (e.g. `N-2`, `N-4`,
`N-8`). These multi-point sweeps live in the **GRPO** directories only; each
`*_ppo` sibling ships a single config (the most-divergent sweep point used for
the GRPO-vs-PPO comparison), not the full sweep.

### Reading the baselines off the protocol tokens

The three rows of the main table are the *same* config family at different
federation settings (under `config/uniform/<model>/`):

| Baseline | Directory | Protocol tokens | Reading |
|---|---|---|---|
| **FedAgent** | `main/` | `total-100_cl-per-rd-2_rd-70_ep-per-cl-3` | 100 clients, federated |
| **Centralized** | `centralized/` | `total-1_cl-per-rd-1_rd-1_ep-per-cl-210` | 1 client, 1 round, all 210 epochs at once (no aggregation) |
| **Local** | `local_client{1,2,3}/` | `total-100_cl-per-rd-1_rd-1_ep-per-cl-210` | the paper's *Local Agent Training* baseline: one fixed client trains alone for all 210 epochs (no aggregation). These configs use `partition.strategy: uniform_single` with `kwargs.cl_id: 21 / 42 / 84` for `local_client1 / 2 / 3` respectively, i.e. always select that one client ID; the per-client data budget is the usual `min-goals-per-cl-100`. |

The `main_seed1/` and `main_seed2/` sibling directories are the additional two
seeds of the FedAgent main run (three seeds total).

---

## (b) The `federated:` block, FedAgent parameters

This block is the FedAgent contribution. It is consumed by the federation
orchestrator under `core/fed/` (`round_orchestrator.py`, `aggregator.py`,
`script_builder.py`) and `core/custom_fed_server.py`; it is *not* part of
upstream verl-agent. Below is the block from the reference config, annotated.

```yaml
federated:
  total_clients: 100            # N, total client population
  clients_per_round: 2          # M, clients sampled (uniformly, without replacement) each round
  total_rounds: 70              # T, communication rounds; E*T = 210 local epochs
  epochs_per_client: 3          # E, local epochs each selected client runs per round
  eval_only_final_round: true   # append one extra round that only validates (val_before_train), no FedAvg

  aggregation_method: "fedavg"  # "fedavg" | "fedprox"
  fedprox_mu: 0.01              # proximal coefficient μ; read only when aggregation_method == "fedprox"

  base_script_path: "scripts/verl-agent/grpo/run_webshop.sh"  # per-client verl-agent launch script
  output_dir: "./output"        # root for checkpoints, aggregated models, and metrics

  training:
    timeout_per_client: 3600    # seconds before a stuck client is killed
    max_retries: 3              # client-launch retries on transient failure
    parallel_workers: 4         # concurrent client processes (1 = strictly serial)

  logging:
    level: "INFO"
    save_client_logs: true
    save_metrics: true

  data_sharding:
    seed: 42                    # deterministic client→goal assignment (paper: data_sharding.seed = 42)
    min_goals_per_client: 100   # |X_i|, minimum tasks each client must receive
    partition:
      strategy: "catalog_split"   # dispatch key (see the decoder table in §a)
      kwargs:
        env_div: 0.3            # strategy-specific; here: env-heterogeneity strength in [0,1]
        keep_ratio: 0.7         # per-client distractor-pool density
        search_return_n: 200    # raise BM25 top-K so the filtered result page stays full

  environment:
    cuda_device: 0
    python_path: "/usr/bin/python3"

  rounds:
    wait_between_rounds: 5      # seconds to pause between rounds (lets GPUs drain)
    save_checkpoints: true      # persist the aggregated model each round

  max_rounds_to_keep_client_checkpoints: 2   # disk hygiene: prune per-client shards older than this many rounds
```

### Field table

| Field | Type | Meaning |
|---|---|---|
| `total_clients` | int | total client population *N* (filename `total-<N>`) |
| `clients_per_round` | int | clients sampled per round *M* (filename `cl-per-rd-<k>`) |
| `total_rounds` | int | communication rounds *T* (filename `rd-<R>`) |
| `epochs_per_client` | int | local epochs per selected client *E* (filename `ep-per-cl-<E>`) |
| `eval_only_final_round` | bool | if true, append a final round that runs validation only (no aggregation), gives a clean end-of-training number |
| `aggregation_method` | str | `fedavg` (default) or `fedprox`; defaults to `fedavg` if omitted |
| `fedprox_mu` | float | FedProx proximal coefficient μ; ignored unless `aggregation_method: fedprox`. FedProx engages only from round 2 onward (round 1 has no global anchor) |
| `base_script_path` | path | per-client verl-agent launch script: `scripts/verl-agent/{grpo,ppo}/run_{webshop,alfworld}.sh`, must match `<algo>` and `<env>` |
| `output_dir` | path | root for checkpoints, aggregated models, metrics |
| `training.timeout_per_client` | int (s) | kill a client that exceeds this wall-clock budget |
| `training.max_retries` | int | per-client relaunch attempts on failure |
| `training.parallel_workers` | int | concurrent client processes; `1` = serial (this is what `reproduce.sh --mode serial` sets) |
| `logging.level` | str | Python log level |
| `logging.save_client_logs` | bool | persist each client's stdout/stderr |
| `logging.save_metrics` | bool | persist per-round metrics |
| `data_sharding.seed` | int | RNG seed for the client→goal assignment (paper: 42) |
| `data_sharding.min_goals_per_client` | int | minimum tasks per client \|X_i\| (filename `min-goals-per-cl-<G>`) |
| `data_sharding.partition.strategy` | str | partition dispatch key, see the decoder table in §(a) and [`docs/heterogeneity.md`](heterogeneity.md) |
| `data_sharding.partition.kwargs` | map | strategy-specific args. Common keys: `omega` (Preference), `size_std` (Coverage), `success_std` (Hardness), `N` + `variant_pool` + `search_return_n` (BM25/lookalike/search env-variants), `env_div` + `keep_ratio` + `search_return_n` (Catalog Split). Omit the whole `kwargs` map for `uniform`. |
| `environment.cuda_device` | int | default CUDA device for the orchestrator process |
| `environment.python_path` | path | interpreter used to launch clients |
| `rounds.wait_between_rounds` | int (s) | pause between rounds |
| `rounds.save_checkpoints` | bool | persist the aggregated model each round |
| `max_rounds_to_keep_client_checkpoints` | int | retain per-client shards for only this many recent rounds (disk hygiene) |

### Where the partition strategy is actually applied

The strategy string is dispatched by `partition_dataset(strategy, ...)` in
`third_party/verl-agent/agent_system/environments/partition_strategy.py`. The
**task-level** strategies (`uniform`, `preference`, `coverage`, `hardness`) route
through that function directly.

The **WebShop env-level** strategies are special: `catalog_split` (and
the WebShop distractor path generally) is *not* dispatched through
`partition_dataset()`, calling it that way raises `NotImplementedError`. It is
invoked directly from the WebShop env manager (`fed_env_manager.py`) because it
needs the products / instructions / goals at construction time. The
`bm25_variant`, `lookalike_injection`, and
`rank_wrapper` variants are likewise wired into the WebShop search
backend rather than the generic slicing path. From a *config-authoring*
standpoint this distinction does not change anything, you still set
`partition.strategy` and `partition.kwargs`, but it explains why those keys do
not appear as `elif` branches that return data slices in `partition_dataset()`.
(The ALFWorld env-level analogue is `env_disjoint`, a scene-disjoint partition.)

To add your own heterogeneity, add a strategy plus a branch in
`partition_dataset()`; see §(e) and
[`docs/heterogeneity.md`](heterogeneity.md#extension-point-adding-a-new-strategy).

---

## (c) The `verl:` block, training knobs (FedAgent-relevant subset)

The `verl:` block is forwarded to the vendored verl-agent trainer. Only the
knobs that FedAgent users routinely touch are documented here; for the **full**
reference (every actor/critic/rollout/FSDP field) see upstream verl-agent and
veRL. Below is the block from the reference GRPO config, with the PPO-only
`critic:` sub-block shown afterward.

```yaml
verl:
  algorithm:
    adv_estimator: grpo         # grpo (main) | gae (PPO; the filename says "ppo")
    use_kl_in_reward: false

  data:
    train_files: data/verl-agent_webshop_grpo/text/train.parquet
    val_files:   data/verl-agent_webshop_grpo/text/test.parquet
    train_batch_size: 8         # GRPO uses 8; PPO uses 64 (see note)
    val_batch_size: 64
    max_prompt_length: 4096
    max_response_length: 512
    filter_overlong_prompts: true
    truncation: error
    return_raw_chat: true

  actor_rollout_ref:
    model:
      path: Qwen/Qwen2.5-1.5B-Instruct        # backbone; main table also 3B/7B + Llama-3.2-3B
      tokenizer_path: Qwen/Qwen2.5-1.5B-Instruct
      enable_gradient_checkpointing: true
      use_remove_padding: true
    actor:
      optim:
        lr: 1e-6                # local learning rate
      ppo_mini_batch_size: 64
      ppo_micro_batch_size_per_gpu: 8
      use_kl_loss: true
      kl_loss_coef: 0.01
      kl_loss_type: low_var_kl
      fsdp_config:
        param_offload: false    # set true (and optimizer_offload true) to fit smaller GPUs
        optimizer_offload: false
      use_invalid_action_penalty: true
      invalid_action_penalty_coef: 0.1
      checkpoint:
        contents: ['model']     # save weights only (keeps per-client checkpoints small)
    rollout:
      name: vllm
      tensor_model_parallel_size: 4   # vLLM TP; keep equal to n_gpus_per_node (paper: 4)
      gpu_memory_utilization: 0.5
      log_prob_micro_batch_size_per_gpu: 16
      enable_chunked_prefill: true
      enforce_eager: false
      free_cache_engine: false
      prompt_length: 4096
      max_model_len: 4096
      response_length: 512
      val_kwargs:
        temperature: 0.4
        do_sample: true
    ref:
      log_prob_micro_batch_size_per_gpu: 16
      fsdp_config:
        param_offload: true

  env:
    env_name: Webshop           # Webshop | alfworld/AlfredTWEnv
    seed: 0                     # env.seed = 0 (paper)
    max_steps: 15               # episode cap: WebShop 15, ALFWorld 50
    rollout:
      n: 8                      # rollouts per task = GRPO group size
    webshop:                    # WebShop-only sub-block (absent for ALFWorld)
      use_small: true           # use the shipped small WebShop catalog

  trainer:
    critic_warmup: 0
    logger: ['console']         # W&B removed, console only
    project_name: verl_agent_webshop_federated
    experiment_name: grpo_qwen2.5_1.5b_federated_catalog_split
    n_gpus_per_node: 4
    nnodes: 1
    save_freq: -1               # the federation layer manages checkpointing, not the trainer
    test_freq: 5                # validate every 5 rounds (paper: test_freq = 5)
    total_epochs: 100
    val_before_train: true      # one validation pass before any training (and for eval_only_final_round)
    save_dir: null
```

### PPO-only addition (`critic:` sub-block)

PPO configs set `algorithm.adv_estimator: gae` and add a `critic:` block (a
GRPO config has neither). From the PPO reference
(`config/uniform/Qwen2.5-1.5B-Instruct/main/ppo/...`):

```yaml
  critic:
    optim:
      lr: 1e-5                  # critic LR (10x the actor's 1e-6)
    model:
      path: Qwen/Qwen2.5-1.5B-Instruct
      use_remove_padding: true
      enable_gradient_checkpointing: true
      fsdp_config:
        param_offload: false
        optimizer_offload: false
    ppo_micro_batch_size_per_gpu: 4
```

### Field table (FedAgent-relevant subset)

| Field | Meaning / FedAgent note |
|---|---|
| `algorithm.adv_estimator` | `grpo` (main) or `gae` (PPO appendix). This is the real GRPO/PPO switch; the filename `<algo>` mirrors it (`grpo`/`ppo`) |
| `algorithm.use_kl_in_reward` | keep `false` (KL enters via the actor loss instead) |
| `data.train_files` / `data.val_files` | preprocessed parquet paths under `data/verl-agent_<env>_<algo>/text/`; must match `<env>` and `<algo>` |
| `data.train_batch_size` | **8 for GRPO, 64 for PPO**: the most important algorithm-dependent batch knob in this repo |
| `data.val_batch_size` | `64` in every shipped config (both GRPO and PPO) |
| `data.max_prompt_length` / `max_response_length` | sequence limits (4096 / 512) |
| `actor_rollout_ref.model.path` / `tokenizer_path` | backbone. Main table: `Qwen/Qwen2.5-1.5B-Instruct` (default) plus `Qwen2.5-3B`, `Qwen2.5-7B`, `Llama-3.2-3B`. Swapping the backbone = changing these two (and the matching `critic.model.path` for PPO) |
| `actor_rollout_ref.actor.optim.lr` | local learning rate (paper: `1e-6`) |
| `actor_rollout_ref.actor.ppo_mini_batch_size` / `ppo_micro_batch_size_per_gpu` | actor update batching |
| `actor_rollout_ref.actor.fsdp_config.param_offload` / `optimizer_offload` | FSDP CPU offload. Both `false` on 4×H100; set `true` to fit smaller GPUs (`reproduce.sh --fsdp on`). See [`docs/running.md`](running.md) |
| `actor_rollout_ref.actor.checkpoint.contents` | `['model']` keeps per-client checkpoints to weights only (important with up to T×M client checkpoints) |
| `actor_rollout_ref.rollout.name` | rollout engine (`vllm`) |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | vLLM tensor-parallel degree; keep equal to `trainer.n_gpus_per_node` (paper: 4). `reproduce.sh --gpus N` adjusts both |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | vLLM KV-cache memory fraction (`0.5`) |
| `actor_rollout_ref.rollout.val_kwargs` | validation sampling (`temperature: 0.4`, `do_sample: true`) |
| `env.env_name` | `Webshop` or `alfworld/AlfredTWEnv` |
| `env.seed` | env seed (paper: 0) |
| `env.max_steps` | per-episode step cap: WebShop 15, ALFWorld 50 |
| `env.rollout.n` | rollouts per task; this **is** the GRPO group size (8) |
| `env.webshop.use_small` | WebShop only: use the shipped small catalog. Absent in ALFWorld configs (there is no `env.webshop` block there) |
| `trainer.logger` | `['console']` only, W&B is removed (see note) |
| `trainer.n_gpus_per_node` / `nnodes` | hardware topology (paper: 4 / 1) |
| `trainer.test_freq` | evaluate every *N* rounds (paper: 5) |
| `trainer.val_before_train` | run one validation pass before training; also what `eval_only_final_round` reuses |
| `trainer.save_freq` | `-1`: the federation layer owns checkpointing, not the inner trainer |
| `trainer.save_dir` | `null`: paths are derived from `federated.output_dir` |
| `critic.*` (PPO only) | critic optimizer / model / batching; present only when `adv_estimator: gae` |

For any field not listed above, `actor.entropy_coeff`, `ulysses_sequence_parallel_size`,
KV-cache and chunked-prefill internals, the full critic schema, etc., consult
upstream verl-agent / veRL. FedAgent does not override those beyond the values
shown in the shipped configs.

> **Removed in this release: W&B.** There is no `wandb:` block and no
> `'wandb'` entry in `trainer.logger`; every shipped config logs to
> `console` only. Do not re-introduce a `wandb:` block, nothing reads it. If
> you want experiment tracking, add your own logger backend in the verl-agent
> trainer rather than relying on a config block.

### Evaluation cadence and seeds (config-visible)

- **Validation cadence:** `trainer.test_freq: 5` (every 5 rounds);
  `trainer.val_before_train: true` adds a pre-training pass and powers the
  `eval_only_final_round` end-of-training number.
- **Validation set size:** `data.val_batch_size: 64` in all shipped configs.
- **Held-out sets** (fixed by the harness, not these YAML keys): WebShop is
  evaluated on `goals[0:500]` (500 held-out tasks); ALFWorld on
  `valid_seen` (140) + `valid_unseen` (134) = 274. Env-level runs are always
  evaluated on the **unperturbed** environment, so the headline number measures
  whether training survived the heterogeneity, not performance on the perturbed
  variant.
- **Seeds:** `verl.env.seed: 0`; `federated.data_sharding.seed: 42`; the
  validation SimServer uses seed `1000` (set in the harness, not the YAML).

---

## (d) The `data_preprocess:` block

This small block controls one-time parquet generation for the chosen task and is
retained in the release:

```yaml
data_preprocess:
  mode: 'text'            # observation/encoding mode for the parquet build
  train_data_size: 8      # rows written to train.parquet (matches data.train_batch_size: GRPO 8 / PPO 64)
  val_data_size: 64       # rows written to test.parquet  (matches data.val_batch_size: 64)
  local_dir: null         # output dir; null = the default under data/
```

`train_data_size` / `val_data_size` track the corresponding `data.*` batch sizes
(so GRPO files carry `train_data_size: 8`, PPO files `train_data_size: 64`), and
`mode: 'text'` matches the `.../text/` segment of the parquet paths in
`verl.data.*`.

---

## (e) Library extension points

Where to plug in, with the file you touch:

| You want to add… | Where | Notes |
|---|---|---|
| a new env / dataset | `third_party/verl-agent/agent_system/environments/env_package/` (+ register it) | then point `verl.env.env_name` and `verl.data.*` at it |
| a new heterogeneity pattern | add a strategy + a branch in `partition_dataset()` in `partition_strategy.py` | select it via `federated.data_sharding.partition.strategy`; see [`docs/heterogeneity.md`](heterogeneity.md#extension-point-adding-a-new-strategy) |
| a new RL algorithm | the verl-agent trainer (PPO / GRPO / GiGPO / RLOO / DAPO are available upstream) | expose it through `verl.algorithm.adv_estimator` |
| a new aggregation rule | `utils/model_aggregation.py` (FedAvg and FedProx today) | select via `federated.aggregation_method`; validate with `tools/aggregation/check_aggregation.py` and `tools/aggregation/verify_aggregation.py` |

`check_aggregation.py` confirms that the aggregated weights equal the (weighted)
mean of the client shards (`--aggregated-dir`, `--client-dirs`); use it whenever
you change `model_aggregation.py`.

---

## Annotated example config

[`config/example.yaml`](../config/example.yaml) is a fully field-by-field
annotated config (the canonical WebShop / GRPO / uniform main-table run, with
every field commented inline and the partition-strategy options listed). Use it
as the field reference. To start a new run, copy the curated file under
`config/` closest to your `<env>` / `<algo>`, then edit the `federated.*`
protocol tokens and `partition.*` using the tables above.
