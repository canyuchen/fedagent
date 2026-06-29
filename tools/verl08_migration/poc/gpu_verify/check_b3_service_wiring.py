"""B3 service wiring: exercise the ACTUAL webshop_service.server deferral + runtime partition."""
import os, sys
REPO = "/gpfs/projects/b1222/userdata/canyu/kangyu/fedagent"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party/verl-agent/agent_system/environments/env_package/webshop/webshop"))
# configure the service exactly as run_fed would for a preference client
os.environ.update(PARTITION_STRATEGY="preference", CLIENT_ID="0", CLIENT_NUM="2",
                  OMEGA="0.5", MIN_GOALS_PER_CLIENT="100", FEDAGENT_LOG_GOAL_ID="1")
import fedagent.webshop_service.server as S
from collections import Counter

assert S._DEFERRED_TASK_PARTITION == "preference", S._DEFERRED_TASK_PARTITION
assert S.CLIENT_GOAL_IDXS is None, "should be deferred (None at import)"
assert S.CATALOG_ASINS is None, "task-level -> full catalog"
print("import-time: deferred OK (CLIENT_GOAL_IDXS=None, CATALOG_ASINS=None)")

# emulate _lifespan: make one env, compute partition + taskids from its real goals
env = S._make_env(0)
G = S._server_goals(env)
S._compute_task_partition(G)
idxs = S.CLIENT_GOAL_IDXS
assert idxs and all(500 <= i < len(G) for i in idxs), "idxs out of range"
served = Counter(G[i].get("category", "?") for i in idxs)
top, n = served.most_common(1)[0]
print(f"runtime partition: |idxs|={len(idxs)} top_cat={top} {n}/{len(idxs)}={n/len(idxs)*100:.0f}%")

# goal-id logging: task_ids built from real goals, options-hash form
tids = [S._goal_taskid(g) for g in G]
assert tids[idxs[0]] and "_" in str(tids[idxs[0]]), "task_id should be asin_optionshash"
print(f"goal_taskid sample: {tids[idxs[0]]!r}  (built {len(tids)} ids)")
print("SERVICE WIRING: PASS")
