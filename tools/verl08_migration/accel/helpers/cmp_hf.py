"""Compare two HF safetensors models tensor-by-tensor -> max|Δ| / mean|Δ|. Used to settle whether the
worker vs inline 1.5B runs produced the SAME aggregated weights (val diff = pure eval noise) or
genuinely diverged (different training trajectories)."""
import sys
from safetensors import safe_open
import torch

A, B = sys.argv[1], sys.argv[2]
def load(p):
    sd = {}
    with safe_open(p, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    return sd
a, b = load(A), load(B)
ka, kb = set(a), set(b)
if ka != kb:
    print("KEY DIFF only-A:", list(ka - kb)[:4], "only-B:", list(kb - ka)[:4])
rows = []
for k in sorted(ka & kb):
    d = (a[k].float() - b[k].float()).abs()
    if d.numel():
        rows.append((float(d.max()), float(d.mean()), k))
rows.sort(reverse=True)
gmax = max(r[0] for r in rows)
print(f"params={len(rows)}  GLOBAL max|Δ|={gmax:.3e}")
print("top 5 divergent:")
for mx, mn, k in rows[:5]:
    print(f"  max|Δ|={mx:.3e} mean={mn:.3e}  {k}")
print("VERDICT:", "EQUIVALENT (≤1e-4, bf16 noise)" if gmax <= 1e-4 else
      f"DIVERGED ({gmax:.2e}) -> different training trajectories")
