"""B3 diagnostic: how does the env's served goal order relate to the overlay's goal list?"""
import os, sys, random
REPO = "/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party/verl-agent/agent_system/environments/env_package/webshop/webshop"))
import gym
from web_agent_site.envs import WebAgentTextEnv  # noqa
env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=None)
u = env.unwrapped
E = [g.get("asin") for g in u.server.goals]            # env's served (shuffled) order
from fedagent.hetero.webshop_catalog_split import _generate_goal_asins_for_partition, load_webshop_data
products, ins = load_webshop_data(None)
U = _generate_goal_asins_for_partition(products, ins)  # overlay (unshuffled, reverted)
n = min(len(U), len(E))
def m(a, b): return sum(1 for i in range(n) if a[i] == b[i]) / n
print(f"len U={len(U)} E={len(E)}  same_multiset={sorted(U)==sorted(E)}")
print(f"[t1] U (unshuffled) vs E            : {m(U,E):.4f}")
U2 = list(U); random.Random(42).shuffle(U2)
print(f"[t2] Random(42).shuffle(U) vs E     : {m(U2,E):.4f}")
U3 = list(U); random.seed(42); random.shuffle(U3)
print(f"[t3] seed(42)+shuffle(U) vs E       : {m(U3,E):.4f}")
# Is E recoverable from U at all? (same multiset => yes, but order is env-internal)
# Robust fix implication: if t1/t2/t3 all low but same_multiset, must read env.server.goals at runtime.
print("CONCLUSION:", "reproducible-shuffle" if max(m(U,E),m(U2,E),m(U3,E))>0.9 else "NOT reproducible -> need runtime env-goal mapping")
