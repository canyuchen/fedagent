#!/usr/bin/env python3
"""Single source of truth for federated runner paths.

Resolves META_INFO / OUTPUT_DIR / CONFIG_FILE from a verl config name and
paths.yaml, so that scripts/start_federated.sh and
scripts/smart_federated_runner.sh agree on the naming convention.
"""

import argparse
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


def load_yaml(path):
    conf = OmegaConf.load(path)
    return OmegaConf.to_container(conf, resolve=True)


def get(conf, dotted, default=None):
    cur = conf
    for key in dotted.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_meta_info(conf):
    fed = conf.get('federated', {}) or {}
    total_clients = fed.get('total_clients')
    clients_per_round = fed.get('clients_per_round')
    total_rounds = fed.get('total_rounds')
    epochs_per_client = fed.get('epochs_per_client')
    min_goals = get(conf, 'federated.data_sharding.min_goals_per_client')
    model_path = get(conf, 'verl.actor_rollout_ref.model.path', '') or ''
    model_name = os.path.basename(model_path)
    strategy = get(conf, 'federated.data_sharding.partition.strategy', '') or ''
    kwargs = get(conf, 'federated.data_sharding.partition.kwargs', {}) or {}
    shuffle_seed = (
        get(conf, 'federated.data_sharding.shuffle_seed')
        or get(conf, 'data.shuffle_seed')
        or '42'
    )

    base = (
        f"{model_name}_total-{total_clients}_cl-per-rd-{clients_per_round}"
        f"_rd-{total_rounds}_ep-per-cl-{epochs_per_client}"
        f"_min-goals-per-cl-{min_goals}_p-{strategy}"
    )

    def require(key):
        val = kwargs.get(key)
        if val is None:
            raise SystemExit(
                f"ERROR: partition strategy '{strategy}' requires "
                f"federated.data_sharding.partition.kwargs.{key}"
            )
        return val

    if strategy == 'preference':
        # Preference Heterogeneity (paper PreferencePartition, Dirichlet, symbol omega).
        # The knob is `omega`; `tau` is the OLD name of the SAME knob (legacy yamls).
        # It is NOT a different algorithm -- both name the Dirichlet preference
        # partition; downstream partition_strategy.py aliases tau -> omega.
        # CAUTION: this legacy code knob `tau` is unrelated to the paper symbol tau
        # (the task descriptor); they only share a letter.
        # We branch on which key is present purely so the two naming conventions
        # produce DISTINCT output-dir suffixes (`_p-preference_omega-<v>` vs the
        # legacy `_p-preference_tau-<v>`) and never collide for different omega values.
        if 'omega' in kwargs:
            base_no_strategy = base[:-len(f"_p-{strategy}")]
            meta = f"{base_no_strategy}_p-preference_omega-{kwargs['omega']}"
        else:
            meta = f"{base}_tau-{require('tau')}"
    elif strategy == 'coverage':
        # Coverage Heterogeneity (paper CoveragePartition, Beta-distributed pool
        # sizes, symbol xi). NOTE: despite the name, `size_std` is NOT a standard
        # deviation -- it is the Beta CONCENTRATION (alpha=mu*s, beta=(1-mu)*s in
        # partition_strategy.generate_client_sizes), so it equals the paper's xi
        # directly: a LARGER size_std == LARGER xi == LOWER variance == MORE UNIFORM.
        # Configs sweep size_std in {1, 256}: size_std=256 == near-uniform pool sizes
        # (xi=256); size_std=1 == extreme size imbalance (xi=1).
        meta = f"{base}_std-{require('size_std')}"
    elif strategy == 'hardness':
        # Hardness Heterogeneity (paper HardnessPartition, symbol xi'; 'hardness'
        # is the lowercased paper term, not a typo). Like coverage, `success_std`
        # is the Beta CONCENTRATION, not a standard deviation, and equals xi'
        # directly: a LARGER success_std == LOWER variance == MORE UNIFORM. Configs
        # sweep success_std in {1, 256}: 256 == near-uniform difficulty (xi'=256),
        # 1 == extreme difficulty imbalance across clients (xi'=1).
        meta = f"{base}_success_std-{require('success_std')}"
    elif strategy == 'uniform_single':
        meta = f"{base}_cl-{require('cl_id')}"
    elif strategy == 'distractor_disjoint':
        # Environment-Level Heterogeneity on WebShop = paper Variant 1 "Catalog Split".
        # 'distractor_disjoint' is the LEGACY/superseded implementation of that variant
        # (function _distractor_disjoint_partition_webshop: all clients share goals[500:],
        # full-target floor). Internally nicknamed the "v4 algo" -- that is an
        # implementation-iteration number of the catalog-split code, NOT the paper's
        # Variant 4 (Lookalike Injection). Kept only for old runs; the paper's reported
        # Catalog Split numbers use 'catalog_split' below. Knobs: env_div, keep_ratio.
        meta = f"{base}_div-{require('env_div')}_keep-{require('keep_ratio')}"
    elif strategy == 'catalog_split':
        # Environment-Level Heterogeneity on WebShop = paper Variant 1 "Catalog Split"
        # (Stage 1 / content; the paper's Algorithm 1 'CatalogSplitPartition').
        # This is the CURRENT implementation (function
        # _distractor_disjoint_partition_webshop_v5: per-client target floor,
        # uniform 100 goals/client). Internally nicknamed the "v5 algo" -- an
        # implementation-iteration number of the catalog-split code, NOT the paper's
        # Variant 5 (Rank Wrapper). Knobs: env_div, keep_ratio. See docs/heterogeneity.md.
        meta = f"{base}_div-{require('env_div')}_keep-{require('keep_ratio')}"
    elif strategy == 'env_disjoint':
        # Env-level heterogeneity (AlfWorld scene-disjoint)
        fallback = kwargs.get('fallback', 'skip')
        meta = f"{base}_div-{require('env_div')}_fb-{fallback}"
    elif strategy == 'bm25_variant':
        # Env-level heterogeneity (WebShop BM25 variants). N and variant_pool
        # distinguish Field-Subset Index (variant_pool=fields_only) from
        # BM25 Reweighting (default pool), so both must enter the output name.
        meta = f"{base}_N-{require('N')}"
        vp = kwargs.get('variant_pool')
        if vp:
            meta = f"{meta}_{vp}"
    elif strategy in ('lookalike_injection', 'rank_wrapper'):
        # Environment-Level Heterogeneity on WebShop, assigned per client from a pool
        # of N variants (paper Algorithm 2 'EnvVariantPartition'; deterministic
        # per-client index RandomState(base_seed + client_id).randint(N)):
        #   lookalike_injection = paper Variant 4 "Lookalike Injection"
        #       (Stages 1+3, content+matching), fn _lookalike_injection_partition_webshop;
        #   rank_wrapper        = paper Variant 5 "Rank Wrapper"
        #       (Stage 4, rendering), fn _rank_wrapper_partition_webshop.
        # N (pool size) enters the output name.
        meta = f"{base}_N-{require('N')}"
    else:
        meta = base

    if str(shuffle_seed) != '42':
        meta = f"{meta}_shuffle-{shuffle_seed}"
    return meta


def resolve(verl_config, paths_file, no_timestamp, timestamp):
    paths = load_yaml(paths_file)
    project_root = paths['project_root']
    config_root = paths['config']['root']
    output_root = paths['output']['root']

    fields = {
        'PROJECT_ROOT': project_root,
        'CONFIG_ROOT': config_root,
        'OUTPUT_ROOT': output_root,
    }

    if verl_config:
        config_file = f"{config_root}/{verl_config}.yaml"
        if not Path(config_file).is_file():
            sys.stderr.write(f"ERROR: config file not found: {config_file}\n")
            sys.exit(1)

        conf = load_yaml(config_file)
        meta_info = build_meta_info(conf)
        config_base = re.sub(r'_total.*', '', verl_config)

        if no_timestamp:
            output_dir = f"{output_root}/{config_base}_{meta_info}"
        else:
            ts = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"{output_root}/{config_base}_{meta_info}_{ts}"

        fields.update({
            'VERL_CONFIG': verl_config,
            'CONFIG_FILE': config_file,
            'META_INFO': meta_info,
            'CONFIG_BASE': config_base,
            'OUTPUT_DIR': output_dir,
        })
    return fields


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verl-config', default=None,
                    help='Config name relative to config root, without .yaml')
    ap.add_argument('--no-timestamp', action='store_true')
    ap.add_argument('--timestamp', default=None,
                    help='Fixed timestamp (overrides default now)')
    ap.add_argument('--paths-file', default='./config/paths.yaml')
    ap.add_argument('--format', choices=['shell', 'field'], default='shell')
    ap.add_argument('--field', default=None,
                    help='With --format=field, emit this single field')
    args = ap.parse_args()

    fields = resolve(args.verl_config, args.paths_file,
                     args.no_timestamp, args.timestamp)

    if args.format == 'field':
        if not args.field:
            sys.stderr.write("ERROR: --format=field requires --field NAME\n")
            sys.exit(1)
        if args.field not in fields:
            sys.stderr.write(
                f"ERROR: unknown field {args.field}. "
                f"Available: {sorted(fields)}\n"
            )
            sys.exit(1)
        print(fields[args.field])
    else:
        for k, v in fields.items():
            print(f"{k}={shlex.quote(str(v))}")


if __name__ == '__main__':
    main()
