"""Bounded (sliding-window, per-turn) rollout — an OPT-IN alternative to the default
concat-chat ``GymTextAgentLoop``.

WHY THIS EXISTS
---------------
The default ``gym_text`` agent loop concatenates a whole episode into ONE training
sample, so the model both *generates* and *trains* with the FULL accumulated context
(grows to ~8.6k tokens on ALFWorld). Generation cost is ~73% of step time because each
decoded token attends to the entire history (long-context decode is memory-bandwidth
bound). The original verl-agent 0.3.1 baseline did NOT do this: it re-rendered each turn
as a fresh prompt truncated to ``max_model_len`` (2048) and trained PER TURN. That is a
*bounded* rollout: each generation sees only a recent window, so every token is cheap.

This module reproduces the bounded layout as an opt-in mode:

  * GENERATION: at turn t the prompt is ``system + (most-recent turns that fit W tokens) +
    obs_t``. Old turns slide out of the window -> short context -> fast decode.
  * TRAINING: each turn becomes its OWN sample ``(windowed_prompt, action_t)`` with the
    response = the generated action tokens only (mask all 1s; the env observation lives in
    the *next* sample's prompt, never in a response, so nothing to mask).

CORRECTNESS (the thing a naive sliding window gets wrong)
---------------------------------------------------------
PPO/GRPO require the training-time forward (old_log_prob / log_prob) to see the SAME
context the behavior policy used at generation. A naive "window the generation but keep
the full concat for training" breaks this: training attends to tokens the policy never
saw -> the importance ratio is wrong -> biased gradient. Per-turn samples avoid it by
construction: sample_t's prompt IS exactly the windowed context used to generate action_t,
and verl's standard full-causal forward over that single (short) sequence reproduces the
generation context exactly.

GRPO caveat (documented, handled at the worker/advantage layer, NOT here): the original
broadcasts ONE group-normalized *episode* advantage to all of an episode's turns. verl's
GRPO group-normalizes per *sample* by ``uid``; with per-turn samples that reweights the
group statistics by episode length. The faithful fix is to group/normalize at the episode
level then broadcast -- a worker/advantage concern. PPO (GAE) is unaffected: per-turn
samples each carry their own step reward, exactly the original per-turn PPO layout.

This file contains the PURE, CPU-testable core (windowing + per-turn sample construction).
The verl worker/manager wiring that emits these as multiple samples per episode lives in
``bounded_worker.py`` and is selected via
``actor_rollout_ref.rollout.agent.agent_loop_manager_class`` (verl's native hook -- no fork).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TurnRecord:
    """One environment interaction, as recorded during rollout."""
    obs_str: str                      # observation shown to the policy BEFORE it acted this turn
    action_ids: List[int] = field(default_factory=list)   # tokens the policy generated (exact gen ids)
    reward: float = 0.0               # env reward for this step (per-turn PPO credit)
    done: bool = False


@dataclass
class PerTurnSample:
    """One training sample = (windowed prompt, generated action). response_mask is all-1s
    because the response is purely the model's action; observations are only ever in prompts."""
    prompt_ids: List[int]
    response_ids: List[int]
    response_mask: List[int]
    reward: float
    turn_index: int


def _windowed_messages(
    system_content: str,
    prior: List[Dict[str, str]],   # alternating [user(obs0), assistant(act0), user(obs1), ...] BEFORE current obs
    current_obs: str,
    window_tokens: int,
    tokenize_chat: Callable[[List[Dict[str, str]]], List[int]],
) -> List[Dict[str, str]]:
    """Build the windowed chat for generating the current turn.

    Always keeps the system message and the current observation (the policy must see *what
    it is acting on*). Prepends the most-recent prior turns as COMPLETE (user-obs,
    assistant-action) PAIRS while the tokenized prompt stays within ``window_tokens``. We
    drop whole *pairs* from the front -- never a lone message -- so the chat never becomes an
    illegal `system -> assistant -> user` sequence (greedy per-message windowing could strand
    an assistant turn without its preceding user obs). Mirrors the original's 2048 truncation,
    which dropped oldest history first. If even system+current_obs exceeds the budget we still
    return them (a single turn cannot be split); response_length / max_model_len stay the hard
    ceilings. ``prior`` is alternating [user0, asst0, user1, asst1, ...] in complete pairs
    (build_per_turn_samples appends them together), so pairing is exact.
    """
    system_msg = {"role": "system", "content": system_content}
    current_msg = {"role": "user", "content": current_obs}
    pairs = [prior[i:i + 2] for i in range(0, len(prior), 2)]   # [[user_t, asst_t], ...]
    chosen: List[Dict[str, str]] = []
    # accept whole turn-pairs newest-first while within budget
    for pair in reversed(pairs):
        candidate = [system_msg] + pair + chosen + [current_msg]
        if len(tokenize_chat(candidate)) <= window_tokens:
            chosen = pair + chosen
        else:
            break
    return [system_msg] + chosen + [current_msg]


def build_per_turn_samples(
    system_content: str,
    turns: List[TurnRecord],
    window_tokens: int,
    tokenize_chat: Callable[[List[Dict[str, str]]], List[int]],
    decode: Callable[[List[int]], str],
    prompt_length: int,
    response_length: int,
) -> List[PerTurnSample]:
    """Reconstruct the per-turn training samples for a completed bounded episode.

    For each turn t, the prompt is the windowed context the policy saw (system + recent
    turns within ``window_tokens`` + obs_t), and the response is exactly the action tokens
    the policy generated. This is the offline reconstruction used by tests and by the
    worker to assemble samples; during live rollout the same ``_windowed_messages`` builds
    the prompt that is actually sent to the inference server, so generation == training.
    """
    samples: List[PerTurnSample] = []
    prior: List[Dict[str, str]] = []   # grows with (user obs, assistant action) as turns complete
    for t, turn in enumerate(turns):
        msgs = _windowed_messages(system_content, prior, turn.obs_str, window_tokens, tokenize_chat)
        prompt_ids = tokenize_chat(msgs)
        resp = list(turn.action_ids)
        samples.append(PerTurnSample(
            prompt_ids=prompt_ids[-prompt_length:],
            response_ids=resp[:response_length],
            response_mask=[1] * len(resp[:response_length]),
            reward=float(turn.reward),
            turn_index=t,
        ))
        # advance history: this turn's obs (user) then the action the policy took (assistant)
        prior.append({"role": "user", "content": turn.obs_str})
        prior.append({"role": "assistant", "content": decode(turn.action_ids)})
    return samples
