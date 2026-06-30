#!/usr/bin/env python
"""Faithful standalone eval: drive run_fed's OWN eval functions (load_cfg -> start_val_service ->
_build_eval -> stream -> summarize_val_dump) for a single unperturbed val-only pass, pinned to a
GPU subset. This is byte-for-byte the code path run_fed's per-round eval uses -- no reimplementation.

Usage: python standalone_eval.py <eval_config.yaml> [gpu_ids=2,3]
Prints EVAL wall-clock + parsed val metrics. Used by run_complete.sh to measure whether eval
hides under the 2 parallel 1-GPU training clients."""
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, "/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent")
from fedagent.fed import run_fed as R  # noqa: E402

cfg_path = sys.argv[1]
gpu_ids = sys.argv[2] if len(sys.argv) > 2 else "2,3"

args = SimpleNamespace(config=cfg_path, model_path=None, output_dir=None, rounds=None, clients=None)
cfg = R.load_cfg(args)
print(f"[standalone-eval] cfg loaded: model={cfg.model_path}", flush=True)
print(f"[standalone-eval] val_env_spec={cfg.val_env_spec} gpu_ids={gpu_ids}", flush=True)

# env_base exactly as run() builds it (so the eval subprocess gets PYTHONPATH/VERL_CFG/history-len)
env_base = os.environ.copy()
env_base["PYTHONPATH"] = f"{R.REPO_ROOT}:{env_base.get('PYTHONPATH', '')}".rstrip(":")
env_base["VERL_CFG"] = R.verl_cfg_dir()
env_base.update(R.history_length_env(cfg))

val_url = R.val_service_url(cfg)
print(f"[standalone-eval] starting UNPERTURBED val service -> {val_url}", flush=True)
vs = R.start_val_service(cfg, env_base)
print("[standalone-eval] val service healthy; launching eval pass", flush=True)

t0 = time.time()
cmd, env, log_path, dump_dir = R._build_eval(cfg, cfg.model_path, 1, env_base, val_url, gpu_ids=gpu_ids)
print(f"[standalone-eval] eval log -> {log_path}", flush=True)
rc = R.stream(cmd, env, log_path, tag="standalone-eval")
dt = time.time() - t0
print(f"[standalone-eval] EVAL DONE rc={rc} wall={dt:.0f}s", flush=True)

m = R.summarize_val_dump(dump_dir)
print(f"[standalone-eval] metrics={m}", flush=True)

try:
    R.stop_services([vs])
except Exception as e:  # teardown noise must not mask the result
    print(f"[standalone-eval] stop_services note: {e}", flush=True)
sys.exit(0 if rc == 0 else 1)
