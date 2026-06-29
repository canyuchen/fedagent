#!/usr/bin/env python3
"""Emit 0.5B / 1-GPU heterogeneity smoke configs for EVERY WebShop partition strategy, with the
paper's faithful kwargs (strong-skew arm so the per-client difference is maximally visible).
Each run = 2 clients / 1 round / 1 step: proves the service shards clients DIFFERENTLY and trains.
Ports are banded so concurrent-safe; output_dir unique per strategy."""
import os

OUT = "/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent/_scratch/gpu_verify"
M05 = "/projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"

# (tag, partition_strategy, extra-yaml-lines)  -- catalog_split + preference already run in phase 1.
STRATS = [
    ("task_disjoint", "task_disjoint", ["env_div: 0.7", "keep_ratio: 0.7"]),          # ENV-het (full catalog, disjoint goals)
    ("coverage",      "coverage",      ["size_std: 256"]),                            # TASK-het (category coverage skew)
    ("hardness",      "hardness",      ["success_std: 256",
                                        "trajectories_file: data/hardness/qwen2.5-1.5b_webshop_trajectories.json"]),
    ("bm25_field_subset", "bm25_field_subset", ["variant_n: 4"]),                     # TRANSITION-het (field-subset index)
    ("bm25_reweight",     "bm25_reweight",     ["variant_n: 4"]),                     # TRANSITION-het (BM25 reweight)
    ("lookalike",         "lookalike",         ["variant_n: 2"]),                     # TRANSITION-het (lookalike injection)
    ("rank_wrapper",      "rank_wrapper",      ["variant_n: 4"]),                     # TRANSITION-het (rank wrapper)
]

TEMPLATE = """# AUTO-GEN het smoke: {strat} ({tag}). 0.5B / 1-GPU / 2cl / 1rd. Confirms per-client shards differ + trains.
env_kind: webshop
env_spec: config/envs/webshop_15.yaml
val_env_spec: ""
output_dir: /tmp/xbb9020_het_{tag}
model_path: {m05}

total_clients: 2
clients_per_round: 2
total_rounds: 1
epochs_per_round: 1
base_seed: 42

n_gpus_per_node: 1
total_training_steps: 1
save_freq: 100000
test_freq: -1
wait_between_clients: 8
min_goals_per_client: 100
partition_strategy: {strat}
{extra}
search_return_n: 200
webshop_pool_size: 8
webshop_base_port: {port}

client_overrides:
  - data.train_batch_size=8
  - data.max_prompt_length=2048
  - data.max_response_length=4096
  - actor_rollout_ref.actor.ppo_mini_batch_size=8
  - actor_rollout_ref.rollout.n=8
  - actor_rollout_ref.rollout.prompt_length=2048
  - actor_rollout_ref.rollout.response_length=4096
  - actor_rollout_ref.rollout.max_model_len=6144
  - actor_rollout_ref.rollout.gpu_memory_utilization=0.6
"""

port = 9200
for tag, strat, extra in STRATS:
    cfg = TEMPLATE.format(tag=tag, strat=strat, m05=M05, port=port,
                          extra="\n".join(extra))
    path = os.path.join(OUT, f"het_webshop_{tag}.yaml")
    with open(path, "w") as f:
        f.write(cfg)
    print(f"wrote {path} (strategy={strat}, port={port})")
    port += 16
