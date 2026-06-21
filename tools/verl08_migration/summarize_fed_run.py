#!/usr/bin/env python
"""Summarize fedagent.fed.run_fed output dirs: per-round reward, and compare conditions.

Reads each run's round_*/client_*/training.log, parses verl's per-step metrics
(critic/rewards/mean by default), and reports per round the mean-over-clients of the
round's mean and max step reward. With multiple LABEL=DIR args it prints a comparison
table -- e.g. the A/B/C decomposition:
    catalog_split (env+task het) vs task_disjoint (task het) vs homogeneous (IID).
A-B isolates the env-heterogeneity effect, B-C the task-heterogeneity effect.

Run on the node where the logs live (compute node /tmp):
    python summarize_fed_run.py A=/tmp/.../scaled_env B=/tmp/.../scaled_task C=/tmp/.../scaled_homog
"""
import glob
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root -> import fedagent
from fedagent.fed.metrics_logger import parse_training_log  # noqa: E402

KEY = "critic/rewards/mean"


def run_rounds(run_dir, key):
    """round -> {client -> (mean_reward_over_steps, max_reward)}"""
    rounds = {}
    for log in sorted(glob.glob(os.path.join(run_dir, "round_*", "client_*", "training.log"))):
        m = re.search(r"round_(\d+)[/\\]client_(\d+)", log)
        if not m:
            continue
        rnd, cl = int(m.group(1)), int(m.group(2))
        vals = [e["metrics"][key] for e in parse_training_log(log) if key in e["metrics"]]
        if vals:
            rounds.setdefault(rnd, {})[cl] = (sum(vals) / len(vals), max(vals))
    return rounds


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    key = KEY
    for a in sys.argv[1:]:
        if a.startswith("--key="):
            key = a.split("=", 1)[1]
    runs = {}
    for a in args:
        label, _, d = a.partition("=")
        runs[label] = run_rounds(d, key)

    all_rounds = sorted({r for rr in runs.values() for r in rr})
    print(f"\nmetric = {key}   (per round: mean[max] of clients' step-averaged reward)\n")
    header = "round | " + " | ".join(f"{lbl:>18}" for lbl in runs)
    print(header)
    print("-" * len(header))
    for r in all_rounds:
        cells = []
        for lbl, rr in runs.items():
            if r in rr:
                means = [v[0] for v in rr[r].values()]
                maxes = [v[1] for v in rr[r].values()]
                cells.append(f"{sum(means)/len(means):.3f}[{max(maxes):.2f}]")
            else:
                cells.append("-")
        print(f"{r:>5} | " + " | ".join(f"{c:>18}" for c in cells))

    # final-round deltas (the decomposition), if A/B/C present
    if {"A", "B", "C"} <= set(runs) and all_rounds:
        rlast = all_rounds[-1]
        def m(lbl):
            v = runs[lbl].get(rlast, {})
            return sum(x[0] for x in v.values()) / len(v) if v else float("nan")
        a, b, c = m("A"), m("B"), m("C")
        print(f"\nfinal round {rlast}:  A(env+task)={a:.3f}  B(task)={b:.3f}  C(iid)={c:.3f}")
        print(f"  env-het effect  (A-B) = {a-b:+.3f}   <- negative => env heterogeneity HURTS under FedAvg")
        print(f"  task-het effect (B-C) = {b-c:+.3f}   <- ~0 => task heterogeneity is FedAvg-robust")


if __name__ == "__main__":
    main()
