"""Generate the AlfWorld holdout-scenes list for env-level OOD eval.

Runs once. Output is committed alongside the env-level yamls at
data/env_heterogeneity/holdout_alfworld_v1.json.
yaml configs reference this file via federated.data_sharding.partition.kwargs.holdout_file.

Strategy:
  Pick scenes spread across all 4 room types (kitchen, living_room, bedroom,
  bathroom) so the OOD eval set covers diverse env types. For each room type,
  pick 2 scenes (1 small, 1 medium) from the train pool.
"""
import os
import json
import random
from pathlib import Path
from collections import defaultdict, Counter


REPO_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_DIR = REPO_ROOT / "data" / "env_heterogeneity"
ALFWORLD_DATA = Path(os.path.expandvars("$ALFWORLD_DATA")) / "json_2.1.1" / "train"
TASK_TYPES_USED = {
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
}


def room_type(scene_id: str) -> str:
    """ALFRED FloorPlan id → room type."""
    try:
        n = int(scene_id)
    except ValueError:
        return "unknown"
    if 1 <= n <= 30:
        return "kitchen"
    if 200 <= n <= 230:
        return "living_room"
    if 300 <= n <= 330:
        return "bedroom"
    if 400 <= n <= 430:
        return "bathroom"
    return "other"


def main(holdout_seed: int = 99999, per_room_type: int = 2):
    if not ALFWORLD_DATA.exists():
        raise FileNotFoundError(
            f"$ALFWORLD_DATA/json_2.1.1/train not found at {ALFWORLD_DATA}. "
            "Set ALFWORLD_DATA env var or update this script's path."
        )

    # Scan: count trials per scene across the FedAgent-effective filter
    # (movable/Sliced excluded, only allowed task_types, solvable=True)
    scene_to_count = Counter()
    scene_to_specs = defaultdict(set)
    for d in os.listdir(ALFWORLD_DATA):
        full = ALFWORLD_DATA / d
        if not full.is_dir():
            continue
        if "movable" in d or "Sliced" in d:
            continue
        parts = d.rsplit("-", 1)
        if len(parts) != 2:
            continue
        spec, scene = parts
        for tr in os.listdir(full):
            tp = full / tr
            if not (tr.startswith("trial_") and tp.is_dir()):
                continue
            gp = tp / "game.tw-pddl"
            tj = tp / "traj_data.json"
            if not (gp.exists() and tj.exists()):
                continue
            try:
                gd = json.load(open(gp))
                if not gd.get("solvable", False):
                    continue
            except Exception:
                continue
            try:
                td = json.load(open(tj))
                if td.get("task_type") not in TASK_TYPES_USED:
                    continue
            except Exception:
                continue
            scene_to_count[scene] += 1
            scene_to_specs[scene].add(spec)

    # Group by room type and pick (small, medium) per type
    by_rt = defaultdict(list)
    for scene, count in scene_to_count.items():
        by_rt[room_type(scene)].append((scene, count))

    rng = random.Random(holdout_seed)
    holdout = []
    per_rt_chosen = {}
    for rt in sorted(by_rt):
        if rt in ("unknown", "other"):
            continue
        # Sort by trial count: prefer scenes with smaller / medium count to limit OOD set size
        candidates = sorted(by_rt[rt], key=lambda x: x[1])
        rng.shuffle(candidates)  # break ties / add seed-based variance
        chosen = sorted(candidates[: per_room_type], key=lambda x: x[0])
        per_rt_chosen[rt] = [c[0] for c in chosen]
        holdout.extend(c[0] for c in chosen)
    holdout = sorted(set(holdout), key=lambda s: int(s) if s.isdigit() else 0)

    total_holdout_trials = sum(scene_to_count[s] for s in holdout)
    total_holdout_specs = len(set().union(*(scene_to_specs[s] for s in holdout)))

    output = {
        "version": "v1",
        "seed": holdout_seed,
        "n_holdout_scenes": len(holdout),
        "n_holdout_trials": total_holdout_trials,
        "n_holdout_specs_touched": total_holdout_specs,
        "scenes": holdout,
        "per_room_type": per_rt_chosen,
        "per_scene_trial_count": {s: scene_to_count[s] for s in holdout},
        "comment": (
            "Reserved FloorPlans for OOD env eval. No client training trial in "
            "these scenes (the env_disjoint partition function applies this filter "
            "before per-spec top-k selection). Pick covers all 4 room types so OOD "
            "eval is balanced across env semantics."
        ),
    }

    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HOLDOUT_DIR / "holdout_alfworld_v1.json"
    json.dump(output, open(out_path, "w"), indent=2, sort_keys=False)
    print(f"Wrote {out_path}")
    print(f"  holdout scenes ({len(holdout)}): {holdout}")
    print(f"  total holdout trials: {total_holdout_trials}")
    print(f"  per room type: {per_rt_chosen}")


if __name__ == "__main__":
    main()
