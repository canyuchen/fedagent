# Acceleration benchmark — configs, drivers, helpers

The reproducibility artifacts behind the acceleration workstream
([`fedagent/docs/acceleration.md`](../../../fedagent/docs/acceleration.md) +
[`acceleration_results.md`](../../../fedagent/docs/acceleration_results.md)). Moved here out of the
(gitignored, 1.3 TB) `_scratch/` so the GPU-validated results stay reproducible from the repo. These
are **one-off experiment scaffolding**, not maintained APIs.

> **Paths.** Configs/drivers were authored to run from `_scratch/accel/` and still reference that
> path for `output_dir` / per-config logs (run *output*, intentionally gitignored — never committed).
> To re-run after the move: invoke `run_fed`/the driver with the config from **this** dir and point
> `output_dir` at any gitignored scratch location. Source lives in the repo; outputs do not.

## Configs → documented result

| config(s) / driver | result | doc |
|---|---|---|
| `p3_1gpu_A/B.yaml`, `p3_eval_2gpu.yaml`, `webshop_val_64.yaml`, `run_complete.sh`, `run_2x1gpu.sh`, `standalone_eval.py` | "2 train on 1 GPU + 2 GPU eval" layout: t1(1)=995s, eval 407s **hidden**, the 3-job ZMQ deadlock + fix; **not** the fast path | §7.7 / §Lever #3 |
| `p3_2gpu_worker_A/B.yaml`, `run_worker3.sh` | #3 + `eval_mode=worker` (hot-engine eval) = **845s/round**, the recommended fast path | §7.7 |
| `p3_2gpu_A/B.yaml`, `p3_4gpu.yaml`, `run_p3.sh`, `run_p3_baseline.sh` | GPU-scaling + client-parallel: t1(4)=558, t1(2)=725, #3 2×2=727s (−35%) | §Lever #3 |
| `ws_eval_inline/parallel.yaml`, `ws_xround_*.yaml`, `paper_ws_mode*.yaml`, `run_evalmode.sh`, `run_paper_modes*.sh`, `run_paper_4card.sh` | eval-mode sweep (inline / parallel / shared / worker) + cross-round persistence | §7.4 |
| `ws_clientend*.yaml` | client-end eval "circles" (per-client val marks) | §7.4 |
| `webshop_prewarm_on/off.yaml` | lever #2 — env-service pre-warm | §Lever #2 |
| `tinyguess_*.yaml`, `persist_full.yaml`, `subproc_full.yaml`, `run_persistent_smoke.sh` | persistent-trainer vs subprocess equivalence smokes | §7.1 / §7.2 |
| `ppo_*.yaml`, `xround_*.yaml`, `run_xround_*.sh`, `run_ppo_persist_smoke.sh` | PPO + cross-round persistence (critic reload) | §7.2 |

## Helpers
- `standalone_eval.py` — faithful standalone val pass (drives `run_fed._build_eval` → the same verl
  val-only path the loop uses); used by `run_complete.sh` to measure hidden-eval cost.
- `cmp_hf.py` — tensor-by-tensor HF safetensors compare (cross-mode weight equivalence).
- `critic_diag.py` — PPO critic-reload diagnostics.
