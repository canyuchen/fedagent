# Acceleration across environments — WebShop vs ALFWorld

**What this doc answers in one line:** *which acceleration choices transfer from WebShop to ALFWorld,
which flip, and the single principle that predicts it.*

This is the self-contained cross-environment synthesis. The per-environment detail lives in
[`acceleration.md`](./acceleration.md) (WebShop levers + analysis),
[`acceleration_results.md`](./acceleration_results.md) (WebShop numbers), and
[`alfworld_testing.md`](./alfworld_testing.md) (ALFWorld strategy + §6 results). Both are
Qwen2.5-1.5B-Instruct, 4×H100, GRPO (G=8), paper settings.

> ### ⚠️ Read this first — what is and isn't comparable
> **Absolute wall-clock seconds are NOT comparable across the two environments.** They differ in
> val size (WebShop eval-mode sweep n=500 vs ALFWorld n=48), episode length (15 vs 50 turns), and
> per-step env weight. **Compare the *rankings*, the *relative %* penalties, and the *mechanisms* —
> never "ALFWorld 3509s vs WebShop 2493s".** Where a number's *metric* matters (per-step vs full-run
> wall), it is labelled inline.

---

## 1. At a glance

| Axis | WebShop (15-turn) | ALFWorld (50-turn) | Cross-env verdict |
|---|---|---|---|
| **Eval-mode — fastest** | `parallel` (2493s, n=500) | **`worker`** (3509s, n=48) | **FLIPS** — worker overtakes parallel |
| **Eval-mode — slowest** | `shared` (3316s, n=500) | **`inline`** (4738s, n=48) | **FLIPS** — inline worst, not shared |
| **Eval-mode — structure** | parallel < worker < inline < shared | **worker < parallel < shared < inline** | decoupled-beats-coupled holds; order within shifts |
| **1-GPU penalty** | **+37%** (t1 wall, 995/725) | **+38% per-step** (534/387); +21% 1-step-wall | **TRANSFERS** (~identical per-step) |
| **GPU↔rollout coupling** | rollout GPU-sensitive¹ | **gen FLAT across 1/2/4 GPU (env-bound)** | ALFWorld-specific finding |
| **2-job concurrency (ZMQ fix)** | PASS (3-job) | PASS (2-job, both rc=0) | **TRANSFERS** |
| **Persistent trainer (#4)** | −43%/round, −62% cross-round | not isolated this round² | predicted smaller share² |

¹ Inferred (no `_TW_LOCK`, lighter env), not separately decomposed for WebShop this round.
² ALFWorld's bigger rollout term shrinks cold-start's *share* of wall-clock, so #4's *relative* win is
predicted smaller; not measured in isolation yet.

**Three sentences:** Decoupling eval from the training critical path wins on **both** envs, but on
ALFWorld `worker` (cross-round cold-start amortization) overtakes `parallel` and `inline` becomes the
*worst* mode — the ranking flips. The per-GPU training penalty is essentially the **same** (~+38%), yet
the *reason it doesn't get worse* differs: ALFWorld's rollout is **env-latency-bound** (generation time
is flat across GPU count). The concurrency fix is environment-agnostic and holds on both.

---

## 2. Eval-mode ranking — the big flip

Same 4-mode sweep (inline / parallel / shared / worker), each = eval running at a different place
relative to training. Full wall-clock of a 2-client × 2-round run, eval every round:

| Rank | WebShop (n=500) | ALFWorld (n=48) |
|---|---|---|
| 1 (fastest) | parallel 2493s | **worker 3509s** |
| 2 | worker 2637s | parallel 3620s |
| 3 | inline 3090s | shared 4560s |
| 4 (slowest) | shared 3316s | **inline 4738s** |

**What stays the same:** the two **eval-decoupled** modes (`worker`, `parallel`) beat the two
**eval-coupled** modes (`shared`, `inline`). Whether eval sits on the 4-GPU training critical path is the
dominant factor in both envs.

**What flips, and why:**
- **`worker` overtakes `parallel`.** ALFWorld's eval engine cold-start (vLLM init + CUDA-graph capture +
  loading the 8810-game service) is *expensive*. `worker` pays it **once** (persistent cross-round) and
  keeps all 4 GPUs for training; `parallel` hides eval but trains on only 2 GPUs (+30%/step). When eval
  is heavy, amortizing the cold-start beats hiding it.
- **`inline` becomes worst (not `shared`).** `inline` re-spins that expensive eval engine **every round**
  on the critical path. On WebShop the eval was light enough that inline's re-spin was cheap and
  `shared`'s 0.3-util KV throttle was the worst sin; on ALFWorld the heavy per-round re-spin dominates,
  so `inline` sinks below even throttled-`shared`.

> **Comparability caveat.** WebShop's "shared slowest" was specifically a **large-val (n=500)** effect;
> ALFWorld ran n=48. So the shared↔inline ordering is partly val-size, not pure env. The robust,
> val-size-independent claim is the **mechanism**: ALFWorld's heavy *per-eval cold-start* is what makes
> `inline` the loser and rewards `worker`'s amortization.

---

## 3. GPU scaling — the part that transfers (with a sharper mechanism)

**1-GPU penalty is ~the same on both envs.**

| | WebShop | ALFWorld |
|---|---|---|
| 4-GPU | 558s (t1 wall) | 298.4s/step · 778s wall |
| 2-GPU | 725s (t1 wall) | 386.9s/step · 865s wall |
| 1-GPU | 995s (t1 wall) | 534.5s/step · 1050s wall |
| **1-GPU vs 2-GPU** | **+37%** (wall) | **+38% per-step**; +21% 1-step-wall |

The clean, like-for-like figure is **per-step ≈ +38% on ALFWorld ≈ +37% on WebShop** — the penalty does
**not** narrow on ALFWorld. (ALFWorld's lower *wall* figure, +21%, is a single-step-probe artifact: ~490s
of fixed overhead — service load + Ray/vLLM init + teardown — doesn't scale and dilutes one step. Over a
real multi-step run the wall penalty climbs back to the per-step +38%.)

**The new mechanism (ALFWorld only, measured):** split each step into rollout vs training —

| GPUs | gen (rollout) | update_actor (training) |
|---|---|---|
| 1 | 228.3s | 140.0s |
| 2 | 225.3s | 92.2s |
| 4 | 219.3s | 43.3s |
| scaling | **FLAT (−4%)** | **~linear (3.2×)** |

`gen` is **flat across GPU count** → ALFWorld rollout is **env-latency-bound**: the `_TW_LOCK`-serialized,
`pool_size=8`-throttled TextWorld service gates generation, not GPU compute. Only `update_actor` scales.
**Practical lever:** to speed ALFWorld rollout, add **env workers (`pool_size`)**, not GPUs — 40–73% of
every step is GPUs idling on the env service. (WebShop, with no `_TW_LOCK` and a lighter env, is expected
to be less env-bound here, but its gen/compute split wasn't isolated this round.)

---

## 4. Concurrency / the ZMQ fix — environment-agnostic

The FSDP→vLLM weight-transfer deadlock (every isolated Ray cluster picks the same first job id
`01000000` → identical `/tmp` ZMQ socket → 44-min hang) and its fix (`VERL_RAY_JOB_ID` per verl
subprocess + a 2-line verl honor-override patch) live entirely in the **env-agnostic verl/Ray plane**.

| | WebShop | ALFWorld |
|---|---|---|
| Test | 3 concurrent jobs (client-parallel + eval∥train) | 2 concurrent training jobs, GPUs {0,1}+{2,3} |
| Result | PASS (rc=0) after fix | **PASS** (both rc=0; A 392s, B 473s) |

ALFWorld is the *stronger* stress test — its slow service cold-start widens the socket race window — and
the fix holds. This is the expected outcome: nothing about the bug or the fix touches the env service.

---

## 5. The principle (why all of the above follows)

ALFWorld differs from WebShop along three axes — **longer episodes (50 vs 15 turns)**, **heavier
per-step env (TextWorld + a process-global `_TW_LOCK`)**, **larger/heavier eval**. Each one shifts where
the wall-clock goes:

```
            WebShop  ────────────────►  ALFWorld
 cost moves FROM:  GPU compute    TO:  eval-engine cold-start  +  env-latency (rollout)
```

That single shift predicts every result above:
- **eval-cold-start grows** → the mode that *amortizes* it (`worker`) wins and the mode that *repeats* it
  (`inline`) loses → **eval-mode ranking flips**.
- **rollout becomes env-latency-bound** → adding GPUs stops helping generation (`gen` flat) → the lever
  for rollout becomes `pool_size`, and the per-GPU *training* penalty is unchanged (it was never about
  rollout).
- **the trainer plane is untouched** → the concurrency fix transfers verbatim.

**Decision rule for a new environment:** estimate (a) eval-engine cold-start cost and (b) how
env-latency-bound the rollout is. High (a) → prefer `worker`/`parallel`, avoid `inline`. High (b) →
scale `pool_size` before GPUs, and expect 1-GPU training to stay viable in *relative* terms even though
the per-step penalty doesn't shrink.

---

## 6. Settled vs open

**Settled (measured both envs):** eval-mode ranking + the decouple-eval principle; ~+38% per-step 1-GPU
penalty; ALFWorld rollout env-bound (`gen` flat); ZMQ concurrency fix env-agnostic.

**Open / not yet isolated:**
- **Persistent-trainer (#4) relative win on ALFWorld** — predicted smaller (cold-start is a smaller
  *share* of ALFWorld's bigger wall), but not A/B'd this round.
- **WebShop gen/compute decomposition** — to confirm WebShop rollout is genuinely *less* env-bound than
  ALFWorld's flat gen (currently inferred from "no `_TW_LOCK`").
- **Full-val ALFWorld numbers** — these used n=48; the in-loop `valid_seen` is 140 and the offline set is
  274 (`tools/verl08_migration/eval_alfworld_by_tasktype.py`).
- **Multi-step steady-state walls** — the scaling probe was 1 step; a multi-round run confirms the wall
  penalty converges to the per-step +38%.

---

## Provenance & see also
- **WebShop numbers:** [`acceleration_results.md`](./acceleration_results.md),
  [`acceleration.md`](./acceleration.md) §7.4 (eval modes) / §7.7 (layouts) / §Lever #3.
- **ALFWorld numbers:** [`alfworld_testing.md`](./alfworld_testing.md) §6 (predictions resolved +
  scorecard); [`EXPERIMENTS.md`](../EXPERIMENTS.md) "ALFWorld acceleration economics (2026-06-30)".
- **Configs:** `tools/verl08_migration/accel/webshop/`, `…/accel/alfworld/`,
  `…/accel/client_parallel/` (each has a README).
- **The fix:** `tools/verl08_migration/patches/` (`VERL_RAY_JOB_ID` honor-override).
- Chinese version: [`acceleration_cross_env_cn.md`](./acceleration_cross_env_cn.md).
