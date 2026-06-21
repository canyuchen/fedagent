"""FedProx proximal term for verl 0.8 (non-fork, one-method monkeypatch).

FedProx anchors each client's drifting local weights w to the round-start global model
w_t by adding mu*(w - w_t) to the actor gradient before every optimizer step. In
FedAgent's subprocess-per-round design each client-round is a FRESH process that loads
the aggregated model, so w_t is simply the params at the first optimizer step -- no
external per-round reset is needed.

Seam (verl 0.8): verl/workers/engine/fsdp/transformer_impl.py
  - FSDPEngine.optimizer_step() clips grads then calls optimizer.step().
We wrap it: snapshot w_t on the first call (params still == the loaded global model),
then on every call add the proximal grad per LOCAL shard (FSDP1 sharded view / FSDP2
DTensor -> elementwise on each shard is correct). GRPO has no critic and the ref
policy never steps, so patching the base engine's optimizer_step affects only the
actor. Mirrors verl-agent 0.3.1 dp_actor.update_policy (snapshot + grad.add_).

Enabled by run_fed via env var FEDPROX_MU>0 (so aggregation_method=fedprox uses it;
plain FedAvg leaves mu=0 = no-op).
"""
import os

_PATCHED = False


def enable_fedprox(mu: float) -> bool:
    """Monkeypatch FSDPEngine.optimizer_step to add the FedProx proximal gradient. Idempotent."""
    global _PATCHED
    if mu is None or mu <= 0 or _PATCHED:
        return False
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    _orig_optimizer_step = FSDPEngine.optimizer_step

    def optimizer_step(self):
        snap = getattr(self, "_fedprox_w_t", None)
        if snap is None:
            # first step of this round-process: params still == loaded global model w_t
            snap = {n: p.detach().clone() for n, p in self.module.named_parameters()}
            self._fedprox_w_t = snap
        for n, p in self.module.named_parameters():
            if p.grad is not None and n in snap:
                # per-local-shard: grad += mu * (w - w_t)   (FSDP/DTensor elementwise-safe)
                p.grad.add_(p.data - snap[n].to(p.grad.device), alpha=mu)
        return _orig_optimizer_step(self)

    FSDPEngine.optimizer_step = optimizer_step
    _PATCHED = True
    print(f"[fedprox] enabled: proximal mu={mu} (FSDPEngine.optimizer_step patched)", flush=True)
    return True


def maybe_enable_from_env() -> bool:
    """Enable FedProx if FEDPROX_MU>0 in the environment. Call before run_ppo()."""
    try:
        mu = float(os.environ.get("FEDPROX_MU", "0") or "0")
    except ValueError:
        mu = 0.0
    return enable_fedprox(mu)


def worker_setup():
    """Ray worker_process_setup_hook entry: runs in EVERY Ray worker at startup so the
    actor-engine worker (separate process from the agent-loop workers) gets the patch.
    Gated on FEDPROX_MU, so it is a no-op for plain FedAvg runs. Wire via the Hydra
    config: ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=fedagent.fedprox.worker_setup
    """
    maybe_enable_from_env()
