"""verl wiring for BOUNDED (sliding-window, per-turn) rollout — OPT-IN, default OFF.

Selected via verl's native hook (NO fork):
    actor_rollout_ref.rollout.agent.agent_loop_manager_class=fedagent.agent_loops.bounded_worker.BoundedAgentLoopManager

Architecture (why a custom manager+worker is required):
  verl's stock AgentLoopWorker maps 1 episode -> 1 training sample (one concat sequence,
  full-causal attention). Bounded rollout needs 1 episode -> K per-turn samples, each a
  short windowed (prompt, action). So we:
    * BoundedGymTextAgentLoop.run_episode_bounded(): windowed per-turn generation ->
      List[AgentLoopOutput] (one per turn). The prompt sent to the server each turn is the
      sliding window, so DECODE IS CHEAP -- this is where the gen speedup comes from.
    * BoundedAgentLoopWorker._run_agent_loop(): run the episode, then reuse the STOCK
      per-output _agent_loop_postprocess() on EACH turn -> List[_InternalAgentLoopOutput].
    * _postprocess(): flatten the per-episode lists AND expand the per-row non_tensor
      (uid / raw_prompt / index) to per-turn, so GRPO grouping + reward keys stay aligned.

CORRECTNESS: each sample's prompt IS exactly the windowed context used to generate its
action, so verl's standard full-causal forward over that short sequence reproduces the
behavior-policy context -> old_log_prob matches -> PPO/GRPO ratio is unbiased. (Contrast
with windowing only generation while training on the full concat, which is biased.)

STATUS / CAVEATS (must be GPU-validated before trusting numbers):
  [ ] batch-contract: per-turn EXPANDS rollout batch size (N -> sum_e K_e, variable per
      step). PPO mini-batching is per-sequence so it should tolerate this; verify no
      train_batch_size divisibility assert trips.
  [ ] GRPO faithfulness: the original broadcasts ONE group-normalized *episode* advantage
      to all of an episode's turns. verl group-norms per-sample by uid; per-turn samples
      reweight group stats by episode length. PPO (GAE, per-turn step rewards) is faithful
      as-is; GRPO needs episode-level grouping (TODO in _expand_non_tensor / advantage).
  [ ] reward placement: per-turn step reward at each action's last token (PPO/GAE). The
      invalid-action penalty is applied per turn here (coef * is_invalid).
The pure windowing/sample core is unit-tested in _scratch/gpu_verify/test_bounded_rollout.py.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List
from uuid import uuid4

import numpy as np
import ray

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    register,
)

from fedagent.agent_loops.gym_text_agent_loop import GymTextAgentLoop
from fedagent.envs.registry import make_env
from uuid import uuid4 as _uuid4

# POC parked in _scratch/bounded_poc/ and run via PYTHONPATH=_scratch/bounded_poc, so
# bounded_rollout is a top-level module here (not fedagent.agent_loops.bounded_rollout).
from bounded_rollout import _windowed_messages


@register("gym_text_bounded")
class BoundedGymTextAgentLoop(GymTextAgentLoop):
    """Windowed, per-turn variant of GymTextAgentLoop. ``run()`` (single-sample concat) is
    inherited unchanged so this class is harmless if invoked the stock way; the bounded
    worker calls ``run_episode_bounded`` instead to get per-turn samples."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # window in TOKENS for the generation prompt (system+recent turns+current obs).
        # Default mirrors the original verl-agent ALFWorld budget (max_model_len=2048).
        self._window_tokens = int(os.environ.get("FEDAGENT_ROLLOUT_WINDOW", "2048"))

    def _tokenize_chat_sync(self, messages: List[Dict[str, Any]]) -> List[int]:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
        )

    async def run_episode_bounded(self, sampling_params: Dict[str, Any], **kwargs) -> List[AgentLoopOutput]:
        env_name = kwargs.get("env_name", "TinyGuess")
        env = make_env(env_name, kwargs.get("config", {}) or {})
        seed = int(kwargs.get("seed", 0))
        max_turns = int(kwargs.get("max_turns", 6))
        outputs: List[AgentLoopOutput] = []
        try:
            sys_obs = await env.system_prompt()
            init_obs, _ = await env.reset(seed=seed)
            system_content = sys_obs["obs_str"]
            prior: List[Dict[str, str]] = []   # completed [user(obs), assistant(action), ...]
            cur_obs = init_obs["obs_str"]
            success = False
            for t in range(max_turns):
                msgs = _windowed_messages(system_content, prior, cur_obs,
                                          self._window_tokens, self._tokenize_chat_sync)
                prompt_ids = self._tokenize_chat_sync(msgs)
                # safety: never exceed the server context window
                if len(prompt_ids) >= self._max_ctx - 1:
                    prompt_ids = prompt_ids[-(self._max_ctx - 1):]
                out = await self.server_manager.generate(
                    request_id=_uuid4().hex, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
                action_ids = out.token_ids
                text = self.tokenizer.decode(action_ids, skip_special_tokens=True)
                obs, reward, done, info = await env.step(text)
                success = bool(info.get("success", success))
                invalid = 0.0 if info.get("is_action_valid", True) else self._invalid_penalty
                resp = action_ids[: self.response_length]
                outputs.append(AgentLoopOutput(
                    prompt_ids=prompt_ids[-self.prompt_length:],
                    response_ids=resp,
                    response_mask=[1] * len(resp),
                    num_turns=1,
                    reward_score=float(reward) - invalid,   # per-turn (GAE-natural) credit
                    metrics={},
                    extra_fields={"turn_scores": [], "tool_rewards": [],
                                  "reward_extra_info": {"traj_success": float(success)}},
                ))
                prior.append({"role": "user", "content": cur_obs})
                prior.append({"role": "assistant", "content": text})
                cur_obs = obs["obs_str"]
                if done or len(prior) // 2 >= max_turns:
                    break
        finally:
            await env.close()
        return outputs


class BoundedAgentLoopWorker(AgentLoopWorker):
    """Worker that expands each episode into its per-turn samples."""

    async def _run_agent_loop(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        import hydra
        from verl.experimental.agent_loop.agent_loop import (
            _agent_loop_registry, DictConfigWrap, ToolListWrap,
        )
        from verl.utils.rollout_trace import rollout_trace_attr
        with rollout_trace_attr(step=trajectory["step"], sample_index=trajectory["sample_index"],
                                rollout_n=trajectory["rollout_n"], validate=trajectory["validate"],
                                name="agent_loop", trace=trace):
            agent_loop = hydra.utils.instantiate(
                config=_agent_loop_registry[agent_name],
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client, tokenizer=self.tokenizer, processor=self.processor,
                dataset_cls=self.dataset_cls, data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )
            turn_outputs = await agent_loop.run_episode_bounded(sampling_params, **kwargs)
            # reuse the STOCK per-output postprocess (correct padding/masks/position_ids) per turn
            return [await self._agent_loop_postprocess(o, trajectory["validate"], **kwargs)
                    for o in turn_outputs]

    def _postprocess(self, inputs, input_non_tensor_batch=None, validate=False):
        # inputs is a list of per-episode lists -> flatten, and expand the per-episode
        # non_tensor rows to per-turn so uid/raw_prompt/index stay aligned with samples.
        flat, repeats = [], []
        for ep in inputs:
            ep_list = ep if isinstance(ep, list) else [ep]
            flat.extend(ep_list)
            repeats.append(len(ep_list))
        expanded_nt = None
        if input_non_tensor_batch is not None:
            expanded_nt = {k: np.repeat(v, repeats, axis=0) for k, v in input_non_tensor_batch.items()}
        return super()._postprocess(flat, input_non_tensor_batch=expanded_nt, validate=validate)


class BoundedAgentLoopManager(AgentLoopManager):
    """Manager that uses the per-turn worker. Select via
    ``actor_rollout_ref.rollout.agent.agent_loop_manager_class=fedagent.agent_loops.bounded_worker.BoundedAgentLoopManager``."""

    def __init__(self, *args, **kwargs):
        # mirror verl's AgentLoopManagerTQ pattern: set the worker class BEFORE super().__init__
        self.agent_loop_workers_class = ray.remote(BoundedAgentLoopWorker)
        super().__init__(*args, **kwargs)
