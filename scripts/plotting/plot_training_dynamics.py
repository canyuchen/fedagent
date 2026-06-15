#!/usr/bin/env python3
"""Plot aggregated (federated) performance training dynamics from a run's logs.

Reads the per-round / per-client metric logs written under an experiment
directory and plots the aggregated (global FedAvg model) trajectory over
training rounds. Two modes:

    (default)        aggregated curve only
    --with-clients   additionally overlay each client's per-round local
                     trajectory as a faint segment diverging from the shared
                     global point (visualizes client heterogeneity)

The plotted values are the **raw logged numbers** — no padding to a fixed
horizon, no interpolation, and no smoothing/nudging is applied. Each round's
aggregated value is the mean across the participating clients of the metric at
local step 0, i.e. the pre-local-training evaluation of the global model
*entering* that round (the FedAvg-aggregated model from the previous round).
Each client's end-of-round value is its metric at its last logged local step.

Expected layout (written by the federated runner):

    <experiment_dir>/round_<N>/client_<C>/json_logs/metrics.json

where metrics.json is a list of {"step": int, "metrics": {<name>: float, ...}}.

Usage:

    python scripts/plotting/plot_training_dynamics.py <experiment_dir> \\
        [--metric val/success_rate] [--with-clients] [--out FIG.pdf] \\
        [--round-stride N] [--percent] [--title STR]

Run it once without and once with --with-clients to get both figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

StepData = Dict[str, object]
Experiment = Dict[str, Dict[str, List[StepData]]]


def load_experiment(folder: Path) -> Experiment:
    """Load metrics.json for every round_*/client_* under `folder`.

    Returns {round_name: {client_name: [step_data, ...]}}.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"experiment dir not found: {folder}")
    rounds = sorted(
        d for d in folder.iterdir() if d.is_dir() and d.name.startswith("round_")
    )
    if not rounds:
        raise ValueError(f"no round_* directories in {folder}")
    data: Experiment = {}
    for round_dir in rounds:
        clients = sorted(
            d for d in round_dir.iterdir()
            if d.is_dir() and d.name.startswith("client_")
        )
        data[round_dir.name] = {}
        for client_dir in clients:
            mfile = client_dir / "json_logs" / "metrics.json"
            if mfile.exists():
                with open(mfile, "r", encoding="utf-8") as f:
                    data[round_dir.name][client_dir.name] = json.load(f)
    return data


def _round_index(round_name: str) -> int:
    return int(round_name.split("_")[1])


def infer_round_stride(data: Experiment) -> int:
    """X-axis stride (steps per round) so rounds tile without overlap.

    Equal to (max local step observed in any client/round) + 1, floored at 1.
    Only used by --with-clients to place per-client local steps inside a round.
    """
    max_step = 0
    for round_data in data.values():
        for client_data in round_data.values():
            for step_data in client_data:
                max_step = max(max_step, int(step_data.get("step", 0)))
    return max(max_step + 1, 1)


def aggregated_curve(data: Experiment, metric: str) -> List[Tuple[int, float]]:
    """Aggregated (global model) value per round.

    For each round, the mean across participating clients of `metric` at local
    step 0 — the global model entering the round. Raw values only.

    Returns [(round_num, value), ...] sorted by round.
    """
    pairs: List[Tuple[int, float]] = []
    for round_name in sorted(data, key=_round_index):
        vals: List[float] = []
        for client_data in data[round_name].values():
            for step_data in client_data:
                if int(step_data.get("step", 0)) == 0 and metric in step_data.get("metrics", {}):
                    vals.append(float(step_data["metrics"][metric]))
                    break  # step 0 only
        if vals:
            pairs.append((_round_index(round_name), float(np.mean(vals))))
    return pairs


def _last_metric_step(client_data: List[StepData], metric: str) -> Optional[Tuple[int, float]]:
    """The (step, value) at the client's largest local step that logs `metric`."""
    best: Optional[Tuple[int, float]] = None
    for step_data in client_data:
        metrics = step_data.get("metrics", {})
        if metric in metrics:
            s = int(step_data.get("step", 0))
            if best is None or s > best[0]:
                best = (s, float(metrics[metric]))
    return best


def plot_training_dynamics(
    folder: str,
    metric: str = "val/success_rate",
    *,
    with_clients: bool = False,
    out_path: Optional[str] = None,
    round_stride: Optional[int] = None,
    as_percent: bool = False,
    title: Optional[str] = None,
) -> str:
    """Render the aggregated training-dynamics figure and save .pdf + .png."""
    data = load_experiment(Path(folder))
    agg = aggregated_curve(data, metric)
    if not agg:
        raise ValueError(f"no step-0 values for metric '{metric}' in {folder}")

    scale = 100.0 if as_percent else 1.0
    fig, ax = plt.subplots(figsize=(10, 6))

    if with_clients:
        stride = round_stride if round_stride is not None else infer_round_stride(data)
        agg_xy = {r: ((r - 1) * stride, v) for r, v in agg}
        clients = sorted({int(c.split("_")[1]) for rd in data.values() for c in rd})
        palette = plt.cm.tab20(np.linspace(0, 1, max(len(clients), 1)))
        cmap = {c: palette[i % len(palette)] for i, c in enumerate(clients)}

        for round_name in sorted(data, key=_round_index):
            r = _round_index(round_name)
            if r not in agg_xy:
                continue
            x_agg, y_agg = agg_xy[r]
            for cname, cdata in sorted(data[round_name].items()):
                last = _last_metric_step(cdata, metric)
                if last is None or last[0] == 0:
                    continue  # nothing logged after the shared step-0 point
                step, y_client = last
                c = int(cname.split("_")[1])
                ax.plot(
                    [x_agg, x_agg + step], [y_agg * scale, y_client * scale],
                    color=cmap[c], linewidth=2.0, alpha=0.45, zorder=1,
                )
                ax.plot(
                    [x_agg + step], [y_client * scale], marker="o", markersize=6,
                    linestyle="None", color=cmap[c], alpha=0.7, zorder=2,
                )
        for r in range(2, max(agg_xy) + 1):  # round boundaries
            ax.axvline((r - 1) * stride, color="gray", linestyle="--",
                       alpha=0.35, linewidth=1.0, zorder=0)
        xs = [(r - 1) * stride for r, _ in agg]
        ax.set_xlabel("Training step (rounds tiled by local steps)", fontsize=14)
    else:
        xs = [r for r, _ in agg]
        ax.set_xlabel("Round", fontsize=14)

    ys = [v * scale for _, v in agg]
    ax.plot(xs, ys, marker="s", linewidth=3.0, markersize=8, color="tab:red",
            label="Aggregated (global model)", zorder=3)

    ax.set_ylabel(metric + (" (%)" if as_percent else ""), fontsize=14)
    ax.set_title(title or Path(folder).name, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc="best")
    fig.tight_layout()

    out = Path(out_path) if out_path else Path(folder) / (
        f"training_dynamics{'_with_clients' if with_clients else ''}.pdf"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {Path(folder).name}: {len(agg)} rounds, metric={metric}, "
          f"with_clients={with_clients} -> {out}")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "experiment_dir",
        help="run directory containing round_*/client_*/json_logs/metrics.json",
    )
    ap.add_argument("--metric", default="val/success_rate",
                    help="metric key to plot (default: val/success_rate)")
    ap.add_argument("--with-clients", action="store_true",
                    help="overlay per-client per-round local trajectories")
    ap.add_argument("--out", default=None,
                    help="output figure path (.pdf; a matching .png is written too)")
    ap.add_argument("--round-stride", type=int, default=None,
                    help="x-axis steps per round for --with-clients "
                         "(default: inferred from the logs)")
    ap.add_argument("--percent", action="store_true",
                    help="scale the metric to a percentage (x100)")
    ap.add_argument("--title", default=None, help="figure title (default: run dir name)")
    args = ap.parse_args()
    plot_training_dynamics(
        args.experiment_dir, args.metric,
        with_clients=args.with_clients, out_path=args.out,
        round_stride=args.round_stride, as_percent=args.percent, title=args.title,
    )


if __name__ == "__main__":
    main()
