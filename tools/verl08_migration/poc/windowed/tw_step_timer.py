"""Standalone textworld step-time micro-benchmark (no vLLM, no lock). Isolates the raw cost of one
AlfredTWEnv .step() — the crux of whether the service's process-global _TW_LOCK (which serializes
ALL concurrent episodes' steps) is the slowdown vs legacy's parallel per-env Ray actors."""
import os, sys, time, random, statistics

ENGINE = "/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent/fedagent/envs/alfworld/engine"
sys.path.insert(0, ENGINE)
os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
import yaml
from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

CONFIG = ENGINE + "/agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
config = yaml.safe_load(open(CONFIG))
env_type = config["env"]["type"]
print(f"env_type={env_type}; building base env (walks $ALFWORLD_DATA)...", flush=True)
t0 = time.time()
base = get_environment(env_type)(config, train_eval="train")
tw = base.init_env(batch_size=1)
print(f"build={time.time()-t0:.1f}s", flush=True)


def unwrap(out):
    obs, info = (out if isinstance(out, tuple) and len(out) == 2 else (out, {}))
    return obs, info


def adm_of(info):
    cmds = info.get("admissible_commands", [[]])
    c0 = cmds[0] if cmds else []
    return [c for c in c0 if c != "help"]


t0 = time.time(); obs, info = unwrap(tw.reset()); t_reset = time.time() - t0
adm = adm_of(info)
times, resets, steps = [], 0, 0
for i in range(150):
    if not adm:
        obs, info = unwrap(tw.reset()); adm = adm_of(info); resets += 1
        continue
    a = random.choice(adm)
    t0 = time.time()
    o, score, done, info = tw.step([a])
    times.append(time.time() - t0); steps += 1
    adm = adm_of(info)
    if (done[0] if isinstance(done, (list, tuple)) else done):
        obs, info = unwrap(tw.reset()); adm = adm_of(info); resets += 1

ts = sorted(times)
mean = statistics.mean(times)
print(f"\nreset={t_reset*1000:.0f}ms  ({resets} resets during loop)")
print(f"STEP over {steps}:  mean={mean*1000:.1f}ms  median={statistics.median(times)*1000:.1f}ms  "
      f"p90={ts[int(0.9*len(ts))]*1000:.1f}ms  max={max(times)*1000:.1f}ms  min={min(times)*1000:.1f}ms")
print(f"\n=> 160 serialized steps (windowed/client/step) ≈ {mean*160:.1f}s of _TW_LOCK-serialized env time")
print(f"=> if parallelized over 16 workers (legacy-style) ≈ {mean*160/16:.1f}s")
