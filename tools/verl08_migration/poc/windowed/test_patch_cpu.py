"""CPU unit-test for the windowed_manager monkeypatch + adjust_batch logic (no GPU/ray)."""
import numpy as np
import torch
from tensordict import TensorDict

# importing the module applies the scoped monkeypatch
import fedagent.agent_loops.windowed_manager as wm
from verl.protocol import DataProto


def mk(n, tag=None, val=0):
    """tiny DataProto with n rows; one tensor key + one non_tensor key."""
    batch = TensorDict({"x": torch.arange(n).reshape(n, 1) + val}, batch_size=[n])
    nt = {"uid": np.array([f"u{i}" for i in range(n)], dtype=object)}
    dp = DataProto(batch=batch, non_tensor_batch=nt, meta_info={})
    if tag is not None:
        dp.meta_info[wm._TAG] = tag
    return dp


def test_divisor():
    class C(dict):
        __getattr__ = dict.get
        def get(self, k, d=None): return dict.get(self, k, d)
    cfg = C(actor_rollout_ref=C(rollout=C(n=4, log_prob_micro_batch_size_per_gpu=None,
                                          log_prob_use_dynamic_bsz=False),
                                actor=C(ppo_mini_batch_size=4, use_dynamic_bsz=False,
                                        ppo_micro_batch_size_per_gpu=2)),
            trainer=C(n_gpus_per_node=4, nnodes=1))
    d = wm._compute_size_divisor(cfg)
    # lcm(ppo_mini*n=16, ws=4, micro*ws=8) = 16
    assert d == 16, d
    print("[ok] divisor lcm(16,4,8) =", d)


def test_adjust():
    d = mk(13)
    out = wm._adjust_to_divisor(d, 16)
    assert len(out) == 16, len(out)
    # first 13 preserved, +3 dups (idx 0,1,2)
    xs = out.batch["x"].squeeze(1).tolist()
    assert xs[:13] == list(range(13)) and xs[13:] == [0, 1, 2], xs
    # already-divisible -> unchanged identity
    d2 = mk(16)
    assert wm._adjust_to_divisor(d2, 16) is d2
    print("[ok] adjust 13->16 dup idx", xs[13:])


def test_slice_patch():
    # tagged + slice(0, k<len) -> returns self (no truncation)
    t = mk(7, tag=16)
    assert wm._windowed_slice is DataProto.slice  # patched
    r = t.slice(0, 4)
    assert len(r) == 7, len(r)
    # untagged -> normal slice
    u = mk(7)
    assert len(u.slice(0, 4)) == 4
    # tagged but start!=0 (REMAX baseline) -> normal slice
    assert len(t.slice(3, None)) == 4
    print("[ok] slice: tagged(0,4)->7 keep; untagged(0,4)->4; tagged(3,None)->4")


def test_union_patch():
    # self(small, untagged) .union(other tagged, different len) -> adopt+pad other
    self_b = mk(3)                      # batch.repeat(n) stand-in
    other = mk(13, tag=16, val=100)     # the per-turn expansion
    out = self_b.union(other)
    assert len(out) == 16, len(out)     # adopted (13) + padded to 16
    assert wm._TAG not in out.meta_info  # tag popped
    assert out.batch["x"].squeeze(1).tolist()[0] == 100  # adopted other's data, not self's
    # matched-size union with tagged other -> NOT triggered (guard len==len) -> normal union
    a = mk(5); b = mk(5, tag=16)
    m = a.union(b)
    assert len(m) == 5 and "x" in m.batch.keys()
    print("[ok] union: (3).union(tagged13)->16 adopt+pad+pop; matched(5,5)->normal")


if __name__ == "__main__":
    test_divisor()
    test_adjust()
    test_slice_patch()
    test_union_patch()
    print("\nALL CPU PATCH TESTS PASSED")
