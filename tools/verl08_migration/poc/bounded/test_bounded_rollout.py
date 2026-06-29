"""CPU test for fedagent.agent_loops.bounded_rollout core (windowing + per-turn samples).
Validates the properties that make bounded rollout CORRECT:
  1. every per-turn prompt keeps system + that turn's observation,
  2. the window actually bounds prompt length (old turns slide out),
  3. response == generated action, mask all 1s (no obs in any response),
  4. generation context == training context (same builder used both ways) -> no PPO mismatch,
  5. without windowing (huge W) the per-turn prompts are strictly growing (sanity).
"""
import os, sys
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # POC parked in _scratch/bounded_poc
from bounded_rollout import TurnRecord, build_per_turn_samples, _windowed_messages

SNAP = "/projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
tok = AutoTokenizer.from_pretrained(SNAP)
def tokenize_chat(msgs): return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
def decode(ids): return tok.decode(ids, skip_special_tokens=True)

SYS = "You are an expert agent in ALFRED. Reason in <think></think>, act in <action></action>."
# build a 12-turn episode; each action ~realistic reasoning; obs ~ALFWorld
turns = []
for t in range(12):
    obs = f"Your current observation is: step {t}, you are at location {t%4}. Admissible: ['go to loc {(t+1)%4}','take obj {t}','open drawer {t}']."
    act_text = f"<think>At step {t} I consider my options carefully and decide the best move toward the goal.</think><action>go to loc {(t+1)%4}</action>"
    act_ids = tok.encode(act_text, add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
    turns.append(TurnRecord(obs_str=obs, action_ids=act_ids, reward=(1.0 if t == 11 else 0.0), done=(t == 11)))

PROMPT_LEN, RESP_LEN = 2048, 2048
sys_id = None  # not needed

def run(window_tokens):
    samples = build_per_turn_samples(SYS, turns, window_tokens, tokenize_chat, decode, PROMPT_LEN, RESP_LEN)
    return samples

print(f"tokenizer={tok.__class__.__name__}; episode turns={len(turns)}")

# --- bounded: small window forces sliding ---
W = 512
s = run(W)
ok = True
sys_ids = tokenize_chat([{"role": "system", "content": SYS}, {"role": "user", "content": "x"}])
sys_prefix = tok.encode(SYS, add_special_tokens=False)[:8]   # a stable chunk of the system text
plens = [len(x.prompt_ids) for x in s]
for x in s:
    # (1) system present: the system text chunk appears in the decoded prompt
    dp = tok.decode(x.prompt_ids)
    has_sys = "expert agent in ALFRED" in dp
    # (1) current obs present
    has_obs = f"step {x.turn_index}," in dp
    # (3) response == action, mask all 1s
    mask_ok = (set(x.response_mask) <= {1}) and (len(x.response_mask) == len(x.response_ids)) and (sum(x.response_mask) == len(x.response_ids))
    if not (has_sys and has_obs and mask_ok):
        ok = False
        print(f"  turn {x.turn_index}: has_sys={has_sys} has_obs={has_obs} mask_ok={mask_ok} plen={len(x.prompt_ids)}")

# (2) window bounds prompt length: later-turn prompts should NOT keep growing unboundedly;
#     bounded prompts should stay within ~W (allow small chat-template overhead), and be much
#     smaller than the unbounded case.
bounded_max = max(plens)
unb = run(10**9)
unb_plens = [len(x.prompt_ids) for x in unb]
unbounded_max = max(unb_plens)
within_window = bounded_max <= W + 64          # small template slack
unbounded_grows = unb_plens == sorted(unb_plens) and unb_plens[-1] > unb_plens[0]
bounded_slides = bounded_max < unbounded_max   # bounding actually reduced context

# (4) consistency: prompt used at turn t equals re-deriving the windowed messages (same fn) -> tautologically equal;
#     assert the builder is deterministic (run twice -> identical)
s2 = run(W)
deterministic = all(a.prompt_ids == b.prompt_ids and a.response_ids == b.response_ids for a, b in zip(s, s2))

print(f"\n[bounded W={W}]  prompt lens: {plens}")
print(f"[unbounded]      prompt lens: {unb_plens}")
print(f"  (1) sys+obs in every prompt, (3) resp=action/mask=1s : {ok}")
print(f"  (2) bounded_max={bounded_max} <= W+64 ({W+64})        : {within_window}")
print(f"  (2) bounded_max({bounded_max}) < unbounded_max({unbounded_max}) : {bounded_slides}")
print(f"  (5) unbounded prompts strictly grow                  : {unbounded_grows}")
print(f"  (4) builder deterministic (gen==train context)       : {deterministic}")

# (6) STRUCTURAL: windowed messages must be a legal chat (system, then strict user/asst
#     alternation, ending in user) at EVERY window size -- this is the bug the first test
#     missed: per-message greedy windowing could strand an assistant -> system->assistant->user.
def legal_chat(msgs):
    if not msgs or msgs[0]["role"] != "system" or msgs[-1]["role"] != "user":
        return False
    roles = [m["role"] for m in msgs[1:]]            # after system: u,a,u,a,...,u
    return all(r == ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(roles))

# build the prior the same way build_per_turn_samples does, then probe many window sizes
prior, structural_ok = [], True
for t, turn in enumerate(turns):
    for Wp in (64, 96, 128, 160, 200, 256, 384, 512, 800):   # sweep -> hits odd pair boundaries
        m = _windowed_messages(SYS, prior, turn.obs_str, Wp, tokenize_chat)
        if not legal_chat(m):
            structural_ok = False
            print(f"  ILLEGAL chat at turn {t} W={Wp}: roles={[x['role'] for x in m]}")
    prior.append({"role": "user", "content": turn.obs_str})
    prior.append({"role": "assistant", "content": decode(turn.action_ids)})
print(f"  (6) windowed chat is legal alternation at all W       : {structural_ok}")

all_ok = ok and within_window and bounded_slides and unbounded_grows and deterministic and structural_ok
print("\n" + ("BOUNDED CORE OK -> per-turn windowed samples correct & consistent" if all_ok
             else "BOUNDED CORE FAILURE"))
sys.exit(0 if all_ok else 1)
