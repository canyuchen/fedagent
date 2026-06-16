#!/usr/bin/env python3
"""Generate the uniform config matrix.

Layout: config/uniform/[model]/[variant]/[algo]/fed_{task}_{algo}_*.yaml
  4 models × 7 variants × 2 algos × 2 tasks = 112 yamls.

The existing Qwen2.5-1.5B-Instruct/main/{grpo,ppo}/*.yaml (4 files) are kept as
the templates; they are also re-generated here so 'main' is consistent with the
other variants.

Run from repo root:
    python3 tools/generate_uniform_configs.py            # dry-run, preview diff
    python3 tools/generate_uniform_configs.py --write    # write files
    python3 tools/generate_uniform_configs.py --models Qwen2.5-3B-Instruct --variants centralized --write
"""
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJ_ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = PROJ_ROOT / "config" / "uniform"

# ---------------------------------------------------------------------------
# Per-(algo, task) parallel-client overrides (added 2026-05-29 from
# docs/running.md implementation matrix).
#
# Default for all models is Sequential (TP=4, n_gpus_per_node=4).
# Override entry shape: {(algo, task): {"tp_size": <int>, "n_gpus_per_node": <int>}}
# - Missing entry → falls back to the model's tp_size / hardcoded n_gpus=4.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-model hyperparams (default TP=4; parallel_overrides differ per-(algo, task))
# ---------------------------------------------------------------------------
MODELS: dict[str, dict[str, Any]] = {
    "Qwen2.5-1.5B-Instruct": {
        "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "tp_size": 4,
        "gpu_mem_util": 0.5,
        "log_prob_micro": 16,
        "ref_log_prob_micro": 16,
        "actor_micro_webshop": 8,
        "actor_micro_alfworld": 16,
        "critic_micro_webshop": 4,
        "critic_micro_alfworld": 4,
        "enforce_eager": False,
        "free_cache_engine": False,
        "experiment_short": "qwen2.5_1.5b",
        # 2026-06-13: standardized everything on Sequential (TP=4, n_gpus=4, mem=0.5).
        # Test C (TP=2) kept hitting Ray ActorUnavailable; Test B (TP=1, n=2) also hit the 80GB cap (val KV spike).
        # Test A's clean history came from a different cluster (sdiao NVIDIA) and is not guaranteed to transfer.
        # Sequential is the only reliable choice validated on this cluster (env_heterogeneity) at 0.01-0.13 err/R.
        # NO parallel_overrides -> default TP=4, n_gpus=4.
    },
    "Qwen2.5-3B-Instruct": {
        "model_path": "Qwen/Qwen2.5-3B-Instruct",
        "tp_size": 4,
        "gpu_mem_util": 0.5,
        "log_prob_micro": 8,
        "ref_log_prob_micro": 8,
        "actor_micro_webshop": 4,
        "actor_micro_alfworld": 8,
        "critic_micro_webshop": 2,
        "critic_micro_alfworld": 4,
        "enforce_eager": False,
        "free_cache_engine": False,
        "experiment_short": "qwen2.5_3b",
        # 2026-06-13: standardized on Sequential (TP=4, n_gpus=4, mem=0.5). NO parallel_overrides.
    },
    "Llama-3.2-3B-Instruct": {
        "model_path": "meta-llama/Llama-3.2-3B-Instruct",
        "tp_size": 4,
        "gpu_mem_util": 0.5,
        "log_prob_micro": 8,
        "ref_log_prob_micro": 8,
        "actor_micro_webshop": 4,
        "actor_micro_alfworld": 8,
        "critic_micro_webshop": 2,
        "critic_micro_alfworld": 4,
        "enforce_eager": False,
        "free_cache_engine": False,
        "experiment_short": "llama3.2_3b",
        # 2026-06-13: standardized on Sequential (TP=4, n_gpus=4, mem=0.5). NO parallel_overrides.
    },
    "Qwen2.5-7B-Instruct": {
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "tp_size": 4,
        "gpu_mem_util": 0.5,
        "log_prob_micro": 8,
        "ref_log_prob_micro": 8,
        "actor_micro_webshop": 8,
        "actor_micro_alfworld": 8,
        "critic_micro_webshop": 4,
        "critic_micro_alfworld": 4,
        "enforce_eager": False,
        "free_cache_engine": False,
        "experiment_short": "qwen2.5_7b",
        # 7B Test C real torch.OutOfMemoryError (78.36 GiB / 80 GiB cap), no mitigation.
        # All (algo, task) → Sequential TP=4 forced. NO parallel_overrides.
    },
}

# ---------------------------------------------------------------------------
# Per-variant federation structure
# total optim work = 70 rounds × 3 ep = 210 epochs per client in 'main';
# centralized & local_client* match this by 1 round × 210 ep.
# ---------------------------------------------------------------------------
VARIANTS: dict[str, dict[str, Any]] = {
    "main": {
        "total_clients": 100, "clients_per_round": 2, "total_rounds": 70, "epochs_per_client": 3,
        "partition_strategy": "uniform", "partition_kwargs": None,
        "shuffle_seed": None,  # not set → defaults to data_sharding.seed (42)
    },
    "main_seed1": {
        "total_clients": 100, "clients_per_round": 2, "total_rounds": 70, "epochs_per_client": 3,
        "partition_strategy": "uniform", "partition_kwargs": None,
        "shuffle_seed": 21,
    },
    "main_seed2": {
        "total_clients": 100, "clients_per_round": 2, "total_rounds": 70, "epochs_per_client": 3,
        "partition_strategy": "uniform", "partition_kwargs": None,
        "shuffle_seed": 84,
    },
    "centralized": {
        "total_clients": 1, "clients_per_round": 1, "total_rounds": 1, "epochs_per_client": 210,
        "partition_strategy": "uniform", "partition_kwargs": None,
        "shuffle_seed": None,
    },
    "local_client1": {
        "total_clients": 100, "clients_per_round": 1, "total_rounds": 1, "epochs_per_client": 210,
        "partition_strategy": "uniform_single", "partition_kwargs": {"cl_id": 21},
        "shuffle_seed": None,
    },
    "local_client2": {
        "total_clients": 100, "clients_per_round": 1, "total_rounds": 1, "epochs_per_client": 210,
        "partition_strategy": "uniform_single", "partition_kwargs": {"cl_id": 42},
        "shuffle_seed": None,
    },
    "local_client3": {
        "total_clients": 100, "clients_per_round": 1, "total_rounds": 1, "epochs_per_client": 210,
        "partition_strategy": "uniform_single", "partition_kwargs": {"cl_id": 84},
        "shuffle_seed": None,
    },
}

# ---------------------------------------------------------------------------
# Per-task constants (env, prompt size, val sizes — match current audit dim "task")
# ---------------------------------------------------------------------------
TASKS = {
    "webshop": {
        "env_name": "Webshop",
        "max_steps": 15,
        # Reverted 2026-05-12: 3072 broke GRPO multi-turn rollout (sequence_length=3160 > max_length=3072
        # raised by actor FSDP forward). Original 4096 is the safe ceiling for accumulated prompts
        # across max_steps=15 turns. Static prompt audit (max ≈ 2628) missed turn-level growth.
        "max_prompt_length": 4096,
        "max_model_len": 4096,
        "prompt_length": 4096,
        "val_data_size": 64,    # 2026-06-09: 128 → 64 (match alfworld; halve val time)
        "val_batch_size": 64,
        "use_small": True,  # webshop.use_small
    },
    "alfworld": {
        "env_name": "alfworld/AlfredTWEnv",
        "max_steps": 50,
        # Reverted 2026-05-12: alfworld 50 max_steps multi-turn carries same overflow risk
        # observed in webshop GRPO. Keep original 2048.
        "max_prompt_length": 2048,
        "max_model_len": 2048,
        "prompt_length": 2048,
        "val_data_size": 64,
        "val_batch_size": 64,
        "use_small": None,  # alfworld has no webshop.use_small
    },
}

# ---------------------------------------------------------------------------
# Per-algo constants
# ---------------------------------------------------------------------------
ALGOS = {
    "grpo": {
        "adv_estimator": "grpo",
        "train_batch_size": 8,
        "train_data_size": 8,
        "has_critic": False,
    },
    "ppo": {
        "adv_estimator": "gae",
        "train_batch_size": 64,
        "train_data_size": 64,
        "has_critic": True,
    },
}


def build_yaml(model: str, variant: str, algo: str, task: str) -> dict:
    M = MODELS[model]
    V = VARIANTS[variant]
    T = TASKS[task]
    A = ALGOS[algo]

    # Resolve parallel-client overrides for this (algo, task) pair.
    # Fallback: Sequential (model's tp_size, n_gpus=4).
    overrides = M.get("parallel_overrides", {}).get((algo, task), {})
    tp_size = overrides.get("tp_size", M["tp_size"])
    n_gpus_per_node = overrides.get("n_gpus_per_node", 4)
    # 2026-06-09: webshop TP=2 needs lower gpu_mem_util — 1.5B/3B/Llama at 0.6
    # crashed in val (allocated 68→72 GB / 80 GB cap → OOM via Raylet kill).
    # Override per-(algo, task) when present.
    gpu_mem_util = overrides.get("gpu_mem_util", M["gpu_mem_util"])

    base_script = f"scripts/verl-agent/{algo}/run_{task}.sh"
    parquet_dir = f"data/verl-agent_{task}_{algo}/text"
    verl_proj = f"verl_agent_{task}_federated"
    exp_name = f"{algo}_{M['experiment_short']}_uniform_{variant}"

    data_sharding = {
        "seed": 42,
        "min_goals_per_client": 100,
        "partition": {"strategy": V["partition_strategy"]},
    }
    if V["partition_kwargs"]:
        data_sharding["partition"]["kwargs"] = V["partition_kwargs"]
    if V["shuffle_seed"] is not None:
        data_sharding["shuffle_seed"] = V["shuffle_seed"]

    actor_micro = M[f"actor_micro_{task}"]
    cfg: dict[str, Any] = {
        "federated": {
            "total_clients": V["total_clients"],
            "clients_per_round": V["clients_per_round"],
            "total_rounds": V["total_rounds"],
            "epochs_per_client": V["epochs_per_client"],
            "eval_only_final_round": True,
            "aggregation_method": "fedavg",
            "fedprox_mu": 0.01,
            "base_script_path": base_script,
            "output_dir": "./output",
            "training": {"timeout_per_client": 3600, "max_retries": 3, "parallel_workers": 4},
            "logging": {"level": "INFO", "save_client_logs": True, "save_metrics": True},
            "data_sharding": data_sharding,
            "environment": {"cuda_device": 0, "python_path": "/usr/bin/python3"},
            "rounds": {"wait_between_rounds": 5, "save_checkpoints": True},
            "max_rounds_to_keep_client_checkpoints": 2,
        },
        "verl": {
            "algorithm": {"adv_estimator": A["adv_estimator"], "use_kl_in_reward": False},
            "data": {
                "train_files": f"{parquet_dir}/train.parquet",
                "val_files": f"{parquet_dir}/test.parquet",
                "train_batch_size": A["train_batch_size"],
                "val_batch_size": T["val_batch_size"],
                "max_prompt_length": T["max_prompt_length"],
                "max_response_length": 512,
                "filter_overlong_prompts": True,
                "truncation": "error",
                "return_raw_chat": True,
            },
            "actor_rollout_ref": {
                "model": {
                    "path": M["model_path"],
                    "tokenizer_path": M["model_path"],
                    "enable_gradient_checkpointing": True,
                    "use_remove_padding": True,
                },
                "actor": {
                    "optim": {"lr": "1e-6"},
                    "ppo_mini_batch_size": 64,
                    "ppo_micro_batch_size_per_gpu": actor_micro,
                    "use_kl_loss": True,
                    "kl_loss_coef": 0.01,
                    "kl_loss_type": "low_var_kl",
                    "fsdp_config": {"param_offload": False, "optimizer_offload": False},
                    "use_invalid_action_penalty": True,
                    "invalid_action_penalty_coef": 0.1,
                    "checkpoint": {"contents": ["model"]},
                },
                "rollout": {
                    "name": "vllm",
                    "tensor_model_parallel_size": tp_size,
                    "gpu_memory_utilization": gpu_mem_util,
                    "log_prob_micro_batch_size_per_gpu": M["log_prob_micro"],
                    "enable_chunked_prefill": True,
                    "enforce_eager": M["enforce_eager"],
                    "free_cache_engine": M["free_cache_engine"],
                    # FP8 KV cache removed: 0 speedup (BF16 KV already had 12-17x
                    # headroom, never the bottleneck) + 7B PPO NaN crash. KV stays BF16/auto.
                    "prompt_length": T["prompt_length"],
                    "max_model_len": T["max_model_len"],
                    "response_length": 512,
                    "val_kwargs": {"temperature": 0.4, "do_sample": True},
                },
                "ref": {
                    "log_prob_micro_batch_size_per_gpu": M["ref_log_prob_micro"],
                    "fsdp_config": {"param_offload": True},
                },
            },
            "env": {
                "env_name": T["env_name"],
                "seed": 0,
                "max_steps": T["max_steps"],
                "rollout": {"n": 8},
            },
            "trainer": {
                "critic_warmup": 0,
                "logger": ["console"],
                "project_name": verl_proj,
                "experiment_name": exp_name,
                "n_gpus_per_node": n_gpus_per_node,
                "nnodes": 1,
                "save_freq": -1,
                "test_freq": 5,
                "total_epochs": 100,
                "val_before_train": True,
                "save_dir": None,
            },
        },
        # Weights & Biases is removed in this release: runs log to the console
        # only (see trainer.logger above). No wandb block is emitted.
        "data_preprocess": {
            "mode": "text",
            "train_data_size": A["train_data_size"],
            "val_data_size": T["val_data_size"],
            "local_dir": None,
        },
    }

    if T["use_small"]:
        cfg["verl"]["env"]["webshop"] = {"use_small": True}

    if A["has_critic"]:
        critic_micro = M[f"critic_micro_{task}"]
        cfg["verl"]["critic"] = {
            "optim": {"lr": "1e-5"},
            "model": {
                "use_remove_padding": True,
                "path": M["model_path"],
                "enable_gradient_checkpointing": True,
                "fsdp_config": {"param_offload": False, "optimizer_offload": False},
            },
            "ppo_micro_batch_size_per_gpu": critic_micro,
        }

    return cfg


def output_path(model: str, variant: str, algo: str, task: str) -> Path:
    fname = (
        f"fed_{task}_{algo}_total-{VARIANTS[variant]['total_clients']}"
        f"_cl-per-rd-{VARIANTS[variant]['clients_per_round']}"
        f"_rd-{VARIANTS[variant]['total_rounds']}"
        f"_ep-per-cl-{VARIANTS[variant]['epochs_per_client']}"
        f"_min-goals-per-cl-100_p-uniform.yaml"
    )
    return OUT_BASE / model / variant / algo / fname


# YAML dump styling: keep quoted strings + flow style for short lists
class _Dumper(yaml.SafeDumper):
    pass


def _str_rep(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    # quote things that look like floats/exponents (1e-6, 1e-5) so they stay strings
    if data.replace("e-", "").replace(".", "").isdigit():
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_Dumper.add_representer(str, _str_rep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="actually write yaml files (default: dry-run)")
    parser.add_argument("--models", nargs="*", default=list(MODELS.keys()))
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()))
    parser.add_argument("--algos", nargs="*", default=["grpo", "ppo"])
    parser.add_argument("--tasks", nargs="*", default=["webshop", "alfworld"])
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing yamls")
    args = parser.parse_args()

    plan: list[tuple] = []
    for model in args.models:
        for variant in args.variants:
            for algo in args.algos:
                for task in args.tasks:
                    plan.append((model, variant, algo, task))

    print(f"Planning {len(plan)} yamls")
    print(f"  models:   {args.models}")
    print(f"  variants: {args.variants}")
    print(f"  algos:    {args.algos}")
    print(f"  tasks:    {args.tasks}")
    print()

    n_write = n_skip = n_exist = 0
    for m, v, algo, t in plan:
        path = output_path(m, v, algo, t)
        cfg = build_yaml(m, v, algo, t)

        rel = path.relative_to(PROJ_ROOT)
        if path.exists() and not args.overwrite:
            n_exist += 1
            if not args.write:
                print(f"  [exists] {rel}")
            continue

        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            text = yaml.dump(cfg, Dumper=_Dumper, sort_keys=False, default_flow_style=False, width=120)
            path.write_text(text)
            n_write += 1
            print(f"  [write]  {rel}")
        else:
            n_skip += 1
            print(f"  [dry]    {rel}")

    print()
    if args.write:
        print(f"✓ Wrote {n_write} yamls. Skipped {n_exist} already-existing (use --overwrite to force).")
    else:
        print(f"Dry-run: would write {n_skip} new yamls, would skip {n_exist} existing.")
        print(f"Re-run with --write to apply.")


if __name__ == "__main__":
    main()
