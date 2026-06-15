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
        # New: Dirichlet PreferencePartition with ω.
        # Distinguished from legacy tau-based Logit-Normal by suffix `_p-preference_omega-`
        # so output dirs are not aliased between the two algorithms.
        if 'omega' in kwargs:
            base_no_strategy = base[:-len(f"_p-{strategy}")]
            meta = f"{base_no_strategy}_p-preference_omega-{kwargs['omega']}"
        else:
            meta = f"{base}_tau-{require('tau')}"
    elif strategy == 'coverage':
        meta = f"{base}_std-{require('size_std')}"
    elif strategy == 'hardness':
        meta = f"{base}_success_std-{require('success_std')}"
    elif strategy == 'uniform_single':
        meta = f"{base}_cl-{require('cl_id')}"
    elif strategy == 'distractor_disjoint':
        # Env-level heterogeneity (WebShop, v4)
        meta = f"{base}_div-{require('env_div')}_keep-{require('keep_ratio')}"
    elif strategy == 'catalog_split':
        # Env-level heterogeneity (WebShop, v5: per-client target floor)
        # See docs/heterogeneity.md
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
        # Env-level heterogeneity (WebShop lookalike-injection / rank-wrapper)
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
