"""Verify fedagent GymTextAgentLoop's incremental concat/mask invariant against a
one-shot re-tokenization (the thing VAGEN does instead). The agent loop builds
response_ids/response_mask incrementally with obs_tokens = new_ids[len(cur_ids):].
This is only correct if the chat template is a clean PREFIX-EXTENSION each turn, i.e.

    prompt_ids + response_ids == cur_ids   (final)
    and new_ids[:len(cur_ids)] == cur_ids  (each turn, before taking the delta)

If that ever fails, the obs delta is garbage and the response_mask misaligns SILENTLY
(training would compute log_probs over a different sequence than was generated).
Replicates gym_text_agent_loop.run()'s tokenization logic exactly. CPU-only.
"""
import sys
from transformers import AutoTokenizer

SNAP = "/projects/b1222/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
tok = AutoTokenizer.from_pretrained(SNAP)

def tokenize_chat(messages):
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

SYS = ("You are an expert agent operating in the ALFRED Embodied Environment. Reason in "
       "<think></think> then act in <action></action>.")
# realistic multi-turn actions (with the think/action format) + ALFWorld-ish observations
ACTIONS = [
    "<think>I should look around to find the apple.</think><action>go to countertop 1</action>",
    "<think>The apple is not here, check the fridge.</think><action>open fridge 1</action>",
    "<think>Found it. Take the apple.</think><action>take apple 1 from fridge 1</action>",
    "<think>Now heat it using the microwave.</think><action>go to microwave 1</action>",
    "<think>Place the apple to heat.</think><action>heat apple 1 with microwave 1</action>",
    "<think>Deliver to the countertop.</think><action>go to countertop 1</action>",
]
OBS = [
    "Your current observation is: On the countertop 1 you see a knife 1, a mug 2. Admissible: ['go to fridge 1','open drawer 1'].",
    "Your current observation is: The fridge 1 is open. In it you see a apple 1, a tomato 2. Admissible: ['take apple 1 from fridge 1'].",
    "Your current observation is: You pick up the apple 1 from the fridge 1. Admissible: ['go to microwave 1','close fridge 1'].",
    "Your current observation is: You arrive at microwave 1. It is closed. Admissible: ['heat apple 1 with microwave 1'].",
    "Your current observation is: You heat the apple 1 using the microwave 1. Admissible: ['go to countertop 1'].",
]

def simulate(actions, obss, force_imend=True):
    messages = [{"role": "system", "content": SYS},
                {"role": "user", "content": "Your task is to: heat some apple and put it on countertop."}]
    prompt_ids = tokenize_chat(messages)
    cur_ids = list(prompt_ids)
    response_ids, response_mask = [], []
    prefix_ok = True
    for t, act in enumerate(actions):
        # emulate model gen: tokens of the action text, optionally + <|im_end|> like a real gen
        gen = tok.encode(act, add_special_tokens=False)
        if force_imend:
            gen = gen + [tok.convert_tokens_to_ids("<|im_end|>")]
        response_ids += gen
        response_mask += [1] * len(gen)
        cur_ids = cur_ids + gen
        text = tok.decode(gen, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": text})
        if t >= len(obss):
            break
        messages.append({"role": "user", "content": obss[t]})
        new_ids = tokenize_chat(messages)
        # the invariant the delta relies on:
        if new_ids[:len(cur_ids)] != cur_ids:
            prefix_ok = False
            # find first mismatch
            for i, (a, b) in enumerate(zip(new_ids, cur_ids)):
                if a != b:
                    print(f"  [turn {t}] PREFIX MISMATCH at idx {i}: new={a}({tok.decode([a])!r}) cur={b}({tok.decode([b])!r})")
                    break
        obs_tokens = new_ids[len(cur_ids):] if len(new_ids) > len(cur_ids) else []
        response_ids += obs_tokens
        response_mask += [0] * len(obs_tokens)
        cur_ids = new_ids
    return prompt_ids, response_ids, response_mask, cur_ids, prefix_ok

print(f"tokenizer loaded: {tok.__class__.__name__}, vocab={len(tok)}")
all_ok = True
for force_imend in (True, False):
    prompt_ids, response_ids, response_mask, cur_ids, prefix_ok = simulate(ACTIONS, OBS, force_imend)
    inv1 = (prompt_ids + response_ids == cur_ids)             # train seq == generated seq
    inv2 = (len(response_ids) == len(response_mask))          # mask aligned to response
    # reconstruct: action tokens (mask=1) should decode to the actions; obs (mask=0) to obs
    act_tokens = [t for t, m in zip(response_ids, response_mask) if m == 1]
    act_decoded = tok.decode(act_tokens, skip_special_tokens=True)
    n_actions_found = sum(1 for a in ACTIONS if a.split("</think>")[1] [:20] in act_decoded.replace(" ", "")[:99999] or a[:15] in act_decoded)
    print(f"\n=== force_imend={force_imend} ===")
    print(f"  prefix-extension holds each turn : {prefix_ok}")
    print(f"  INV1 prompt+response == cur_ids  : {inv1}")
    print(f"  INV2 len(mask)==len(response)    : {inv2}")
    print(f"  #action-tokens={len(act_tokens)} #obs-tokens={sum(1 for m in response_mask if m==0)} total_resp={len(response_ids)}")
    print(f"  masked-in text starts: {act_decoded[:80]!r}")
    all_ok = all_ok and inv1 and inv2 and prefix_ok

print("\n" + ("ALL INVARIANTS HOLD -> incremental delta == one-shot retokenize (no silent misalign)" if all_ok
             else "INVARIANT VIOLATION -> incremental delta can misalign; needs assertion/one-shot retokenize"))
sys.exit(0 if all_ok else 1)
