"""B3 airtight: overlay *_for_client(env_goals=real) == original partition_dataset(data=real).

Both operate on the SAME real env.server.goals (seed-42 shuffled). The overlay's partition fns
are byte-for-byte copies of the original, so feeding them the env's real goals must reproduce the
original's selection. We compare the SELECTED GOAL multiset (scientific quantity) and idx set.
"""
import os, sys, types, importlib.util, hashlib
from collections import Counter
REPO = "/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party/verl-agent/agent_system/environments/env_package/webshop/webshop"))

# --- real env goals (the order the original partitions) ---
import gym
from web_agent_site.envs import WebAgentTextEnv  # noqa
env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=None)
G = env.unwrapped.server.goals
print(f"server.goals: {len(G)}")

def tid(g):
    a = g.get("asin")
    if g.get("goal_options"):
        return f"{a}_{abs(int(hashlib.md5(str(sorted(g['goal_options'].items())).encode()).hexdigest(),16))}"
    return a

# --- load ORIGINAL partition_strategy.py in isolation (stub matplotlib/seaborn, paths.yaml) ---
for m in ("matplotlib", "matplotlib.pyplot", "seaborn"):
    sys.modules.setdefault(m, types.ModuleType(m))
import omegaconf
_orig_load = omegaconf.OmegaConf.load
omegaconf.OmegaConf.load = lambda p, *a, **k: (  # paths.yaml is absent here; only .project_root is read
    omegaconf.OmegaConf.create({"project_root": REPO}) if str(p).endswith("paths.yaml") else _orig_load(p, *a, **k))
PS = os.path.join(REPO, "third_party/verl-agent/agent_system/environments/partition_strategy.py")
spec = importlib.util.spec_from_file_location("orig_ps", PS)
ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)

def orig_idxs(strategy, **kw):
    sl = ps.partition_dataset(data=G, strategy=strategy, client_id=0, client_num=2,
                              min_samples_per_client=100, start_idx=500, data_type="webshop", **kw)
    out = []
    for g in sl:
        try: out.append(G.index(g))
        except ValueError: pass
    return sorted(out)

# --- PREFERENCE ---
from fedagent.hetero.webshop_task import preference_for_client
ov = preference_for_client(0, 2, omega=0.5, min_goals_per_client=100, env_goals=G, start_idx=500)
og = orig_idxs("preference", tau=0.5)
ov_tids = Counter(tid(G[i]) for i in ov); og_tids = Counter(tid(G[i]) for i in og)
print(f"\nPREFERENCE omega=0.5:  overlay|idx|={len(ov)} orig|idx|={len(og)}  "
      f"idx_set_equal={set(ov)==set(og)}  selected_goal_multiset_equal={ov_tids==og_tids}")

# --- COVERAGE ---
from fedagent.hetero.webshop_coverage import coverage_for_client
ovc = coverage_for_client(0, 2, size_std=1.0, min_goals_per_client=100, env_goals=G, start_idx=500)
ogc = orig_idxs("coverage", size_std=1.0)
print(f"COVERAGE size_std=1.0: overlay|idx|={len(ovc)} orig|idx|={len(ogc)}  "
      f"idx_set_equal={set(ovc)==set(ogc)}  "
      f"selected_goal_multiset_equal={Counter(tid(G[i]) for i in ovc)==Counter(tid(G[i]) for i in ogc)}")

pref_ok = (ov_tids == og_tids)
cov_ok = (Counter(tid(G[i]) for i in ovc) == Counter(tid(G[i]) for i in ogc))
print("\nRESULT:", "EQUIVALENT (overlay selection == original)" if (pref_ok and cov_ok)
      else "DIVERGENCE -> inspect")
