#!/usr/bin/env python3
"""verify_train_val_disjoint.py — confirm Webshop & Alfworld train and val splits don't overlap.

WEBSHOP (analytical, by construction in
third_party/verl-agent/agent_system/environments/env_package/webshop/envs.py):
    val   goal_idxs = [0, val_batch_size)   (standard val pool, list(range(val_batch_size)))   # ~line 235
    train goal_idxs ⊂ [500, len(goals))     (uniform-partition slice per client, start_idx=500) # ~lines 257, 264
Note: the val length is bounded by the runtime `val_batch_size` field. In the
yamls this is set directly as verl.data.val_batch_size (currently 64) and is
ALSO mirrored by data_preprocess.val_data_size: 64, which core/fed/script_builder.py
feeds into the run script as val_data_size=... -> data.val_batch_size. This
script's --webshop-val-size stands in for that same val pool size.

This script re-implements uniform_partition (defined in
third_party/verl-agent/agent_system/environments/partition_strategy.py,
currently around line 250) and verifies, for the given
(val_size, client_num, min_goals_per_client), that
val ∩ train_per_client = ∅ for all clients, and that no client's train slice
crosses below index 500. (Line numbers drift as that file changes — search for
'def uniform_partition' if the reference is stale.)

ALFWORLD (empirical, file system walk):
    train    : $ALFWORLD_DATA/json_2.1.1/train/
    val (id) : $ALFWORLD_DATA/json_2.1.1/valid_seen/
    val (ood): $ALFWORLD_DATA/json_2.1.1/valid_unseen/

Walks all three directories, collects every game.tw-pddl, and verifies trial-id
sets are disjoint between train and val_{seen,unseen}. With --check-content,
also SHA1-hashes file contents to catch the "same file copied into two dirs"
failure mode.

Exit: 0 = all disjoint, 1 = overlap found, 2 = setup error.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable

# Default ALFWORLD data root, equivalent to the shell expansion
# ${ALFWORLD_DATA:-$HOME/.cache/alfworld}. The ALFWORLD_DATA environment variable
# (read in main() below) overrides this; otherwise we fall back to the standard
# per-user cache location rather than any machine-specific absolute path.
DEFAULT_ALFWORLD_DATA = os.path.expanduser("~/.cache/alfworld")

# -----------------------------------------------------------------------------
# Webshop
# -----------------------------------------------------------------------------

def uniform_partition_slice(total: int, client_id: int, client_num: int,
                            min_samples: int) -> tuple[int, int]:
    """Mirror of uniform_partition in
    third_party/verl-agent/agent_system/environments/partition_strategy.py
    (currently def at ~line 250, body through ~line 344).

    Reproduces only the start/end slice arithmetic (including the
    min_samples_per_client growth that lets a client's slice extend on both
    sides). Returns (start_slice, end_slice) — both relative to start_idx=500,
    i.e. real goal indices are [500+start, 500+end).
    """
    base = total // client_num
    start = client_id * base
    end = start + base
    if end - start < min_samples:
        needed = min_samples - (end - start)
        left = start
        right = total - end
        if left + right >= needed:
            left_extra = min(needed // 2, left)
            right_extra = min(needed - left_extra, right)
            if right_extra < (needed - left_extra):
                left_extra += min(needed - left_extra - right_extra,
                                  left - left_extra)
            start -= left_extra
            end += right_extra
        elif left > 0:
            start = 0
            end += min(needed - left, right)
        elif right > 0:
            end += min(needed, right)
        else:
            start, end = 0, total
    return max(0, start), min(total, end)


def check_webshop(val_size: int, clients: int, min_per_client: int,
                  total_goals: int) -> bool:
    print("== Webshop ==")
    print(f"  val_size             = {val_size}")
    print(f"  client_num           = {clients}")
    print(f"  min_goals_per_client = {min_per_client}")
    print(f"  total_goals (assumed)= {total_goals}")
    print(f"  start_idx (train)    = 500  (hardcoded envs.py:257,264,273)")

    val_idxs = set(range(val_size))
    train_per_client: dict[int, set[int]] = {}
    train_pool = total_goals - 500
    if train_pool <= 0:
        print(f"  ❌ total_goals={total_goals} ≤ 500; train pool empty.")
        return False

    for cid in range(clients):
        s, e = uniform_partition_slice(train_pool, cid, clients, min_per_client)
        train_per_client[cid] = set(range(500 + s, 500 + e))
        idxs = train_per_client[cid]
        rng = f"[{min(idxs)}, {max(idxs) + 1})" if idxs else "EMPTY"
        print(f"  client {cid}: train {rng}  size={len(idxs)}")

    train_all = set().union(*train_per_client.values()) if train_per_client else set()
    overlap = val_idxs & train_all
    below_500 = {i for i in train_all if i < 500}

    print(f"  |val|                = {len(val_idxs)}")
    print(f"  |train (∪ clients)|  = {len(train_all)}")
    print(f"  |val ∩ train|        = {len(overlap)}")
    print(f"  |train indices < 500|= {len(below_500)}")

    # Inter-client overlap (informational — uniform with min_samples allows it).
    cids = sorted(train_per_client)
    inter_msgs = []
    for i, a in enumerate(cids):
        for b in cids[i + 1:]:
            n = len(train_per_client[a] & train_per_client[b])
            if n:
                inter_msgs.append(f"client {a}∩{b}={n}")
    if inter_msgs:
        print(f"  inter-client overlap (expected if min>base): {', '.join(inter_msgs)}")

    ok = not overlap and not below_500
    if not ok:
        if overlap:
            print(f"  ❌ FAIL val ∩ train = {sorted(overlap)[:10]}...")
        if below_500:
            print(f"  ❌ FAIL train indices below 500: {sorted(below_500)[:10]}...")
    else:
        print(f"  ✅ PASS val ∩ train = ∅; all train indices ≥ 500")
    print()
    return ok


# -----------------------------------------------------------------------------
# Alfworld
# -----------------------------------------------------------------------------

def collect_games(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("game.tw-pddl"))


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_alfworld(data_root: Path, check_content: bool) -> bool:
    print("== Alfworld ==")
    splits = {
        "train":        data_root / "json_2.1.1" / "train",
        "valid_seen":   data_root / "json_2.1.1" / "valid_seen",
        "valid_unseen": data_root / "json_2.1.1" / "valid_unseen",
    }
    games = {name: collect_games(p) for name, p in splits.items()}
    for name, path in splits.items():
        print(f"  {name:<13s}: {path} → {len(games[name])} game.tw-pddl")

    if not games["train"]:
        print(f"  ❌ train dir missing or empty: {splits['train']}")
        return False
    if not games["valid_seen"]:
        print(f"  ❌ valid_seen missing or empty: {splits['valid_seen']}")
        return False

    # game.tw-pddl layout: .../<task_id>/<trial_id>/game.tw-pddl
    #   task_id  = pick_and_place_simple-Knife-None-SideTable-3
    #   trial_id = trial_T20190918_184236_557252
    def trial_id(p: Path) -> str: return p.parent.name
    def task_id(p: Path) -> str:  return p.parent.parent.name

    def setify(paths: list[Path], fn) -> set[str]:
        return {fn(p) for p in paths}

    trials = {name: setify(g, trial_id) for name, g in games.items()}
    tasks  = {name: setify(g, task_id)  for name, g in games.items()}

    fail = False
    print()
    print("  -- trial-id disjointness (DEMAND: 0) --")
    for v in ("valid_seen", "valid_unseen"):
        n = len(trials["train"] & trials[v])
        flag = "❌" if n else "✅"
        print(f"    {flag} |train ∩ {v}| (trial_id) = {n}")
        if n:
            fail = True
            print(f"       sample: {list(trials['train'] & trials[v])[:5]}")

    print()
    print("  -- task-id intersection (BY DESIGN: nonzero for valid_seen) --")
    for v in ("valid_seen", "valid_unseen"):
        n = len(tasks["train"] & tasks[v])
        print(f"    |train ∩ {v}| (task_id)  = {n}   "
              f"(valid_seen: SAME task types, different trials; "
              f"valid_unseen: DIFFERENT layouts)")

    if check_content:
        print()
        print("  -- content disjointness (SHA1 of game.tw-pddl) --")
        train_hashes = {sha1_of(p): str(p) for p in games["train"]}
        for v in ("valid_seen", "valid_unseen"):
            collisions = []
            for p in games[v]:
                h = sha1_of(p)
                if h in train_hashes:
                    collisions.append((p, train_hashes[h]))
            if collisions:
                fail = True
                print(f"    ❌ {v}: {len(collisions)} game files share SHA1 with train")
                for v_path, t_path in collisions[:3]:
                    print(f"       {v_path}\n         == {t_path}")
            else:
                print(f"    ✅ {v}: 0 SHA1 collisions with train ({len(games[v])} files compared)")

    print()
    if fail:
        print("  ❌ Alfworld FAIL")
    else:
        print("  ✅ Alfworld PASS: train ⊥ valid_seen/valid_unseen at trial-id "
              "(and content, if --check-content)")
    print()
    return not fail


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--skip-webshop", action="store_true")
    ap.add_argument("--skip-alfworld", action="store_true")
    ap.add_argument("--webshop-val-size", type=int, default=128,
                    help="Webshop val goal count (default: 128 — matches uniform yaml)")
    ap.add_argument("--webshop-clients", type=int, default=4,
                    help="N total clients in uniform partition (default: 4)")
    ap.add_argument("--webshop-min-per-client", type=int, default=100,
                    help="min_goals_per_client (default: 100)")
    ap.add_argument("--webshop-total-goals", type=int, default=6910,
                    help="Total len(goals) when use_small=True with synthetic goals. "
                         "Default 6910 is empirical for items_shuffle_1000.json + "
                         "items_ins_v2_1000.json; override if your env differs.")
    ap.add_argument("--alfworld-data",
                    default=os.environ.get("ALFWORLD_DATA", DEFAULT_ALFWORLD_DATA),
                    help=f"ALFWORLD_DATA root (default: $ALFWORLD_DATA or {DEFAULT_ALFWORLD_DATA})")
    ap.add_argument("--check-content", action="store_true",
                    help="Also SHA1-hash Alfworld game.tw-pddl files to catch "
                         "content-identical files placed across splits")
    args = ap.parse_args()

    passes = []
    if not args.skip_webshop:
        passes.append(check_webshop(args.webshop_val_size, args.webshop_clients,
                                    args.webshop_min_per_client,
                                    args.webshop_total_goals))
    if not args.skip_alfworld:
        passes.append(check_alfworld(Path(args.alfworld_data), args.check_content))

    if not passes:
        print("Nothing to check (both envs skipped).", file=sys.stderr)
        return 2
    if all(passes):
        print("=== ✅ All disjointness checks passed ===")
        return 0
    print("=== ❌ At least one disjointness check FAILED ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
