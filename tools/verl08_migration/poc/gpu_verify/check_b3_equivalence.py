"""B3 lynchpin: prove the overlay's webshop goal-serving == original verl-agent.

Claims under test (all must hold for catalog_split to be already-equivalent):
  C1. server.goals order is REPRODUCIBLE run-to-run (same seed 42 -> same order).
  C2. catalog_filter_asins does NOT change server.goals order/length
      (goals come from full_products; filter only touches search/click).
  C3. v5 client_goal_idxs is a contiguous range whose VALUES are order-independent,
      and the catalog is a SET derived from full products -> overlay == original inputs.
"""
import os, sys
REPO = "/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party/verl-agent/agent_system/environments/env_package/webshop/webshop"))
import gym
from web_agent_site.envs import WebAgentTextEnv  # noqa

def goals_asins(**kw):
    env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=None, **kw)
    g = [x.get("asin") for x in env.unwrapped.server.goals]
    try: env.close()
    except Exception: pass
    return g

print("=== C1: reproducible run-to-run (no filter) ===")
E1 = goals_asins(seed=0)
E2 = goals_asins(seed=7)            # different env seed; goal shuffle uses its own seed(42)
same12 = E1 == E2
print(f"len E1={len(E1)} E2={len(E2)}  identical_order={same12}")

print("=== C3 inputs: v5 catalog + idxs from raw products (overlay == original call) ===")
from fedagent.hetero.webshop_catalog_split import _distractor_disjoint_partition_webshop_v5, load_webshop_data
products, ins = load_webshop_data(None)
cat0, idx0 = _distractor_disjoint_partition_webshop_v5(products=products, ins=ins,
    client_id=0, client_num=2, min_goals_per_client=100, env_div=0.7, keep_ratio=0.7, base_seed=42)
contiguous = idx0 == list(range(idx0[0], idx0[-1] + 1))
print(f"|catalog0|={len(cat0)} |idx0|={len(idx0)} range={idx0[0]}..{idx0[-1]} contiguous={contiguous}")

print("=== C2: catalog filter does NOT change goal order/length ===")
E3 = goals_asins(seed=0, catalog_filter_asins=cat0)
same13 = E1 == E3
print(f"len E3={len(E3)}  order_identical_to_unfiltered={same13}")

ok = same12 and same13 and contiguous
print("RESULT:", "EQUIVALENT (no B3 code change needed for catalog_split)" if ok
      else "DIVERGENCE -> investigate")
