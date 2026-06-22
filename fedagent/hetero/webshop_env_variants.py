"""WebShop ENV-LEVEL heterogeneity variants 2-5 (paper Variants 2-5).

Ports the remaining four environment-level WebShop variants from verl-agent's
``agent_system/environments/partition_strategy.py`` into a clean numpy-only module
(mirrors fedagent/hetero/webshop_catalog_split.py, which ported paper Variant 1):

  * Variant 2 = Field-Subset Index  (Stage 2 encoding/index)
  * Variant 3 = BM25 Reweighting    (Stage 3 matching/score)
        -- BOTH served by ONE function `_bm25_variant_partition_webshop`, selected
           via the BM25_VARIANT_POOL env var (default -> V3; 'fields_only' -> V2);
           SimServer override key: env_kwargs['bm25_in_memory_config'].
  * Variant 4 = Lookalike Injection (Stages 1+3 catalog content + matching/score)
        -- `_lookalike_injection_partition_webshop`;
           SimServer override key: env_kwargs['extra_products'].
  * Variant 5 = Rank Wrapper        (Stage 4 rendering/ranking)
        -- `_rank_wrapper_partition_webshop`;
           SimServer override key: env_kwargs['search_engine_variant'].

The three partition functions plus their module-level variant pools / cache /
loader (`BM25_VARIANTS_DEFAULT`, `BM25_VARIANTS_FIELDS_ONLY`,
`LOOKALIKE_VARIANTS_DEFAULT`, `_LOOKALIKE_CACHE`, `_load_lookalikes`,
`SEARCH_ENGINE_VARIANTS_DEFAULT`) are copied VERBATIM from partition_strategy.py
(partition_strategy.py lines 2018-2233, revisions unchanged) so each client's
per-variant assignment is bit-identical to the 0.3.1 baseline. The science red
line: deterministic per-client-id assignment via RandomState(base_seed + client_id),
base_seed=42. The only additions are the thin public `*_for_client` wrappers at the
bottom (used by the verl-0.8 WebShop remote service); each returns the env_kwargs
dict the service should merge into the dict it passes to gym.make.

NAMING CAUTION (carried over from the source): the catalog-split module's '_v4' /
'_v5' tags are IMPLEMENTATION-REVISION numbers of paper Variant 1, NOT the paper's
Variant 4 (Lookalike Injection) or Variant 5 (Rank Wrapper) ported here.
"""
from typing import Any, Dict, List, Optional
import json
import os

import numpy as np


# ============================================================
# Environment-Level Heterogeneity (WebShop): BM25-variant search index/score
# strategy key: 'bm25_variant'  ->  SimServer override: bm25_in_memory_config.
# Dispatched by paper Algorithm 2 (EnvVariantPartition). This ONE function
# serves TWO paper variants, selected by the BM25_VARIANT_POOL env var:
#   * default pool (BM25_VARIANTS_DEFAULT)  -> paper Variant 3 "BM25 Reweighting"
#                                              (Stage 3 matching/score; sweeps k1/b).
#   * BM25_VARIANT_POOL=fields_only         -> paper Variant 2 "Field-Subset Index"
#                                              (Stage 2 encoding/index; varies the
#                                               field subset, fixed k1/b).
# (config keys: bm25_reweighting = V3 default pool; field_subset_index = V2 +
#  variant_pool=fields_only.)
# See docs/heterogeneity.md (BM25 Reweighting / Field-Subset Index).
#
# Each client is deterministically assigned (by client_id) to one of N
# (fields, k1, b) BM25 configs. SimServer in that client's worker swaps
# its search backend to InMemoryBM25Searcher with that config. Catalog,
# goals, reward, val env all UNCHANGED — only the search transition T(s'|s,a)
# differs across clients.
#
# The 4 default variants were selected as the most-divergent pairwise
# combination on real agent queries (mean Jaccard@10 ~ 0.65, top-1
# disagreement ~ 70%) during the heterogeneity study.
# ============================================================
BM25_VARIANTS_DEFAULT = [
    {'name': 'full',           'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 1.2, 'b': 0.75},
    {'name': 'full_b=0.0',     'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 1.2, 'b': 0.00},
    {'name': 'full_k1=0.3',    'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 0.3, 'b': 0.75},
    {'name': 'full_k1=5.0',    'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 5.0, 'b': 0.75},
    # N>=5 extension (deterministic ordering; existing N=4 yamls keep first 4 unchanged)
    {'name': 'full_k1=0.1',    'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 0.1, 'b': 0.75},
    {'name': 'full_b=1.0',     'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 1.2, 'b': 1.00},
    {'name': 'full_k1=2.0_b=0.5', 'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 2.0, 'b': 0.50},
    {'name': 'full_k1=0.3_b=0.0', 'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 0.3, 'b': 0.00},
]

# Field-Subset Index "field-subset" variant pool (mirrors doc §1's Lucene multi-index design
# but built on top of InMemoryBM25Searcher to skip the JDK/offline-indexing
# step). Same k1/b across variants; only the field subset that goes into the
# BM25 doc text differs. Selectable via env var BM25_VARIANT_POOL=fields_only.
BM25_VARIANTS_FIELDS_ONLY = [
    {'name': 'full',          'fields': ['name', 'Title', 'description', 'features', 'BulletPoints'], 'k1': 1.2, 'b': 0.75},
    {'name': 'name',          'fields': ['name', 'Title'],                                              'k1': 1.2, 'b': 0.75},
    {'name': 'desc',          'fields': ['description'],                                                'k1': 1.2, 'b': 0.75},
    {'name': 'bullets',       'fields': ['BulletPoints'],                                               'k1': 1.2, 'b': 0.75},
    # N>=5 extension
    {'name': 'features',      'fields': ['features'],                                                   'k1': 1.2, 'b': 0.75},
    {'name': 'name_bullets',  'fields': ['name', 'Title', 'BulletPoints'],                              'k1': 1.2, 'b': 0.75},
    {'name': 'desc_features', 'fields': ['description', 'features'],                                    'k1': 1.2, 'b': 0.75},
    {'name': 'no_name',       'fields': ['description', 'features', 'BulletPoints'],                    'k1': 1.2, 'b': 0.75},
]


def _bm25_variant_partition_webshop(
    client_id: int,
    client_num: int,
    N: int = 4,
    base_seed: int = 42,
    variants: Optional[List[Dict[str, Any]]] = None,
):
    """Return this client's (fields, k1, b) BM25 config dict.

    The dict is suitable as `env_kwargs['bm25_in_memory_config']` — SimServer
    will route through InMemoryBM25Searcher with these settings.

    Variant pool selection (lowest precedence first):
      1. BM25_VARIANTS_DEFAULT  -- paper Variant 3 "BM25 Reweighting"
         (Stage 3; extreme k1/b on full fields; config key bm25_reweighting)
      2. env BM25_VARIANT_POOL=fields_only -- paper Variant 2 "Field-Subset Index"
         (Stage 2; field-subset on default k1/b; config key field_subset_index)
      3. explicit `variants=` kwarg

    Assignment is deterministic by client_id so repeated launches converge
    on the same per-client variant (important for FedAvg cross-round consistency).
    """
    if variants is None:
        pool_name = os.environ.get('BM25_VARIANT_POOL', 'default').strip().lower()
        if pool_name == 'fields_only':
            variants = BM25_VARIANTS_FIELDS_ONLY
        else:
            variants = BM25_VARIANTS_DEFAULT
    pool = list(variants)
    if N > len(pool):
        raise ValueError(
            f"_bm25_variant_partition_webshop: requested N={N} but only "
            f"{len(pool)} variants defined (extend BM25_VARIANTS_DEFAULT or pass `variants=`)"
        )
    pool = pool[:N]
    rng = np.random.RandomState(base_seed + client_id)
    chosen = pool[rng.randint(N)]
    print(f"[BM25-VARIANT] client {client_id}/{client_num}: variant={chosen['name']} "
          f"fields={chosen['fields']} k1={chosen['k1']} b={chosen['b']}")
    return {
        'fields': list(chosen['fields']),
        'k1': float(chosen['k1']),
        'b': float(chosen['b']),
        '_variant_name': chosen['name'],  # bookkeeping; SimServer ignores keys it doesn't know
    }


# ============================================================
# Environment-Level Heterogeneity (WebShop): Lookalike Injection (adversarial)
#   This is paper Variant 4 = "Lookalike Injection", which spans
#   Transition-pipeline Stages 1+3 (catalog content injection + matching/score),
#   NOT Stage 4 — variant-number does NOT equal stage-number here.
#   Dispatched by paper Algorithm 2 (EnvVariantPartition).
#   strategy key: 'lookalike_injection'  ->  SimServer override: extra_products.
#   (Unrelated to the "v4 algo" Catalog-Split impl above; that v4 is an
#    impl-revision tag for paper Variant 1, not this paper Variant 4.)
# See docs/heterogeneity.md
#
# Each client deterministically assigned (by client_id) to one of N attribute-
# attack lookalike sets (price / color / ...). SimServer in that client's worker
# injects the lookalike products via env_kwargs['extra_products'] so the agent
# is forced to specifically check that attribute to filter out fakes — different
# variants force structurally different attribute-checking policies → π* divergence.
#
# Default N=2 covers the two reward-validated attacks (audit confirmed price and
# option/color attacks both flip reward components; material is not directly
# attackable since r_attribute fuzzy-matches text fields we keep identical).
# ============================================================
LOOKALIKE_VARIANTS_DEFAULT = [
    {'name': 'v_price',       'lookalike_file': 'data/env_heterogeneity/lookalike_data/lookalike_v_price.json'},
    {'name': 'v_color',       'lookalike_file': 'data/env_heterogeneity/lookalike_data/lookalike_v_color.json'},
    # N>=3 extension
    {'name': 'v_size',        'lookalike_file': 'data/env_heterogeneity/lookalike_data/lookalike_v_size.json'},
    {'name': 'v_price_color', 'lookalike_file': 'data/env_heterogeneity/lookalike_data/lookalike_v_price_color.json'},
]

_LOOKALIKE_CACHE = {}


def _load_lookalikes(file_path):
    if file_path not in _LOOKALIKE_CACHE:
        with open(file_path) as f:
            _LOOKALIKE_CACHE[file_path] = json.load(f)
    return _LOOKALIKE_CACHE[file_path]


def _lookalike_injection_partition_webshop(
    client_id: int,
    client_num: int,
    N: int = 2,
    base_seed: int = 42,
    project_root: Optional[str] = None,
    variants: Optional[List[Dict[str, Any]]] = None,
):
    """Return this client's adversarial lookalike list (raw products).

    Implements paper Variant 4 "Lookalike Injection" (Transition-pipeline
    Stages 1+3, NOT Stage 4); strategy key 'lookalike_injection'.

    The list is suitable as `env_kwargs['extra_products']` — SimServer will
    append it to the base 1000-product catalog before BM25 indexing.

    Assignment is deterministic by client_id so repeated launches converge on
    the same per-client variant.
    """
    pool = list(variants) if variants is not None else list(LOOKALIKE_VARIANTS_DEFAULT)
    if N > len(pool):
        raise ValueError(
            f"_lookalike_injection_partition_webshop: requested N={N} "
            f"but only {len(pool)} variants defined (extend LOOKALIKE_VARIANTS_DEFAULT)"
        )
    pool = pool[:N]
    rng = np.random.RandomState(base_seed + client_id)
    chosen = pool[rng.randint(N)]
    file_path = chosen['lookalike_file']
    if not os.path.isabs(file_path):
        if project_root is None:
            project_root = os.environ.get('PROJECT_ROOT', os.getcwd())
        file_path = os.path.join(project_root, file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"[LOOKALIKE-VARIANT] {file_path} does not exist.\n"
            f"  The lookalike data ships under data/env_heterogeneity/lookalike_data/; ensure it is present."
        )
    lookalikes = _load_lookalikes(file_path)
    print(f"[LOOKALIKE-VARIANT] client {client_id}/{client_num}: variant={chosen['name']} "
          f"|lookalikes|={len(lookalikes)} file={os.path.basename(file_path)}")
    return lookalikes


# ============================================================
# Environment-Level Heterogeneity (WebShop): search-engine TYPE swap
#   This is paper Variant 5 = "Rank Wrapper" (Transition-pipeline Stage 4,
#   rendering/ranking); dispatched by paper Algorithm 2 (EnvVariantPartition).
#   strategy key: 'rank_wrapper'  ->  SimServer override: search_engine_variant.
#   (Unrelated to the "v5 algo" Catalog-Split impl above; that v5 is an
#    impl-revision tag for paper Variant 1, not this paper Variant 5.)
# See docs/heterogeneity.md (search backend axis).
#
# Each variant breaks a different baseline-policy assumption:
#   v_bm25_default   -- control (BM25 ranking trustable)
#   v_shuffled_topk  -- BM25 ranks top-50 then shuffles → "click position 1" fails
#   v_inverted_topk  -- BM25 returns top-K reversed → forces "skip front, click later"
#   v_partial_random -- 50% queries return random → forces "verify each result"
#
# All 4 preserve reward gradient (target reachable in candidate set), avoiding
# the v_random pitfall where 25% of clients can never get reward signal.
# ============================================================
SEARCH_ENGINE_VARIANTS_DEFAULT = [
    {'name': 'v_bm25_default',  'type': 'bm25_default'},
    {'name': 'v_shuffled_topk', 'type': 'bm25_shuffle', 'shuffle_k': 50},
    {'name': 'v_inverted_topk', 'type': 'bm25_invert'},
    {'name': 'v_partial_random','type': 'bm25_partial', 'random_prob': 0.5},
]


def _rank_wrapper_partition_webshop(
    client_id: int,
    client_num: int,
    N: int = 4,
    base_seed: int = 42,
    variants: Optional[List[Dict[str, Any]]] = None,
):
    """Return this client's search-engine variant config.

    Implements paper Variant 5 "Rank Wrapper" (Transition-pipeline Stage 4);
    strategy key 'rank_wrapper'.

    Result is a dict suitable as `env_kwargs['search_engine_variant']` —
    SimServer routes through `init_search_engine(search_engine_variant=...)`
    which builds an InMemoryBM25 base and wraps it per the type field.
    """
    pool = list(variants) if variants is not None else list(SEARCH_ENGINE_VARIANTS_DEFAULT)
    if N > len(pool):
        raise ValueError(
            f"_rank_wrapper_partition_webshop: requested N={N} "
            f"but only {len(pool)} variants defined"
        )
    pool = pool[:N]
    rng = np.random.RandomState(base_seed + client_id)
    chosen = pool[rng.randint(N)]
    print(f"[RANK-WRAPPER] client {client_id}/{client_num}: variant={chosen['name']} "
          f"type={chosen['type']}")
    out = {k: v for k, v in chosen.items() if k != 'name'}
    # per-client unique seed so shuffle/random differ across clients of the same variant
    out['seed'] = base_seed + client_id
    return out


# --------------------------------------------------------------------------- #
# Thin public API for the verl-0.8 WebShop remote service (the only additions).
#
# Each wrapper realizes one paper variant for one client and returns the
# env_kwargs fragment the service must merge into the dict it passes to
# gym.make (WebAgentTextEnv / SimServer). Unlike Variant 1 (Catalog Split), none
# of Variants 2-5 partitions the task/goal set, so NO per-client goal_idxs are
# produced here — the task split stays uniform (the service keeps its default
# goals[start_idx:] slicing). Each wrapper's return contract is documented on
# the function. Defaults for N mirror fed_env_manager.py:
#   bm25_variant -> N=4, lookalike_injection -> N=2, rank_wrapper -> N=4.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
# Repo root (contains data/env_heterogeneity/lookalike_data/). Used as the default
# project_root for resolving the relative lookalike_file paths above when neither
# the PROJECT_ROOT env var nor an explicit project_root kwarg is supplied. Mirrors
# webshop_catalog_split.py's __file__-relative DEFAULT_DATA_DIR convention.
DEFAULT_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


def bm25_variant_for_client(
    client_id: int,
    client_num: int,
    *,
    N: int = 4,
    base_seed: int = 42,
    variant_pool: Optional[str] = None,
    variants: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """env_kwargs fragment for paper Variant 2 (Field-Subset Index) / Variant 3 (BM25 Reweighting).

    Return contract:
        {'bm25_in_memory_config': {'fields': [...], 'k1': float, 'b': float,
                                   '_variant_name': str}}
    The service should merge this into the kwargs it passes to gym.make; SimServer
    routes 'bm25_in_memory_config' through InMemoryBM25Searcher. No goal_idxs (the
    task split stays uniform). Variant 2 vs 3 is selected by the variant pool:
      * variant_pool='fields_only' (or env BM25_VARIANT_POOL=fields_only) -> V2
      * default pool -> V3
    Passing `variant_pool` here temporarily sets the BM25_VARIANT_POOL env var that
    the verbatim partition reads, so the wrapper stays a pure pass-through of the
    bit-identical assignment logic (`variants=` still takes highest precedence).
    """
    if variants is None and variant_pool is not None:
        # The verbatim fn reads BM25_VARIANT_POOL from the env; set it for this call
        # then restore, so the assignment stays bit-identical to fed_env_manager's path.
        prev = os.environ.get('BM25_VARIANT_POOL')
        os.environ['BM25_VARIANT_POOL'] = variant_pool
        try:
            cfg = _bm25_variant_partition_webshop(
                client_id=client_id, client_num=client_num,
                N=N, base_seed=base_seed, variants=None,
            )
        finally:
            if prev is None:
                os.environ.pop('BM25_VARIANT_POOL', None)
            else:
                os.environ['BM25_VARIANT_POOL'] = prev
    else:
        cfg = _bm25_variant_partition_webshop(
            client_id=client_id, client_num=client_num,
            N=N, base_seed=base_seed, variants=variants,
        )
    return {'bm25_in_memory_config': cfg}


def lookalike_injection_for_client(
    client_id: int,
    client_num: int,
    *,
    N: int = 2,
    base_seed: int = 42,
    project_root: Optional[str] = None,
    variants: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """env_kwargs fragment for paper Variant 4 (Lookalike Injection).

    Return contract:
        {'extra_products': [<raw product dict>, ...]}
    The service should merge this into the kwargs it passes to gym.make; SimServer
    appends 'extra_products' to the base 1000-product catalog before BM25 indexing.
    No goal_idxs (the task split stays uniform).

    project_root resolves the relative lookalike_file paths
    (data/env_heterogeneity/lookalike_data/lookalike_v_*.json). Precedence:
    explicit `project_root` kwarg > PROJECT_ROOT env var (read inside the verbatim
    fn) > this module's repo-root DEFAULT_PROJECT_ROOT fallback.
    """
    if project_root is None:
        project_root = os.environ.get('PROJECT_ROOT') or DEFAULT_PROJECT_ROOT
    extra_products = _lookalike_injection_partition_webshop(
        client_id=client_id, client_num=client_num,
        N=N, base_seed=base_seed, project_root=project_root, variants=variants,
    )
    return {'extra_products': extra_products}


def rank_wrapper_for_client(
    client_id: int,
    client_num: int,
    *,
    N: int = 4,
    base_seed: int = 42,
    variants: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """env_kwargs fragment for paper Variant 5 (Rank Wrapper).

    Return contract:
        {'search_engine_variant': {'type': str, ...variant-specific keys...,
                                   'seed': int}}
    The service should merge this into the kwargs it passes to gym.make; SimServer
    routes 'search_engine_variant' through init_search_engine(search_engine_variant=...)
    (builds an InMemoryBM25 base, then wraps it per 'type'). No goal_idxs (the task
    split stays uniform). The per-client 'seed' (= base_seed + client_id) makes the
    shuffle/partial-random behaviours differ across clients sharing a variant.
    """
    cfg = _rank_wrapper_partition_webshop(
        client_id=client_id, client_num=client_num,
        N=N, base_seed=base_seed, variants=variants,
    )
    return {'search_engine_variant': cfg}
