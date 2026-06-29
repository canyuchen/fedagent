"""Diagnose the PPO critic diff: load rank0 critic shards from both A/B arms, compare per-param,
handle empty tensors, print top-divergent params + shapes -> value head or backbone?"""
import sys, torch
from pathlib import Path
S = Path(sys.argv[1]); X = Path(sys.argv[2])
def load(d):
    f = sorted(d.glob("model_world_size_*_rank_0.pt"))[0]
    sd = torch.load(f, map_location="cpu", weights_only=False)
    return sd.get("model", sd) if isinstance(sd, dict) else sd
a, b = load(S), load(X)
print(f"A params={len(a)} B params={len(b)}")
ka, kb = set(a), set(b)
if ka != kb:
    print("KEY DIFF only-A:", list(ka-kb)[:5], " only-B:", list(kb-ka)[:5])
rows, empties = [], []
for k in sorted(ka & kb):
    va, vb = a[k], b[k]
    if hasattr(va, "shape") and tuple(va.shape) != tuple(getattr(vb, "shape", ())):
        rows.append((float("inf"), k, f"SHAPE {tuple(va.shape)} vs {tuple(vb.shape)}")); continue
    try:
        d = (va.float() - vb.float()).abs()
    except Exception as e:
        rows.append((-1, k, f"ERR {e}")); continue
    if d.numel() == 0:
        empties.append((k, tuple(va.shape))); continue
    rows.append((float(d.max()), k, f"shape={tuple(va.shape)} mean={float(d.mean()):.3e}"))
rows.sort(reverse=True, key=lambda r: r[0])
print("\nTOP 8 divergent critic params:")
for m, k, info in rows[:8]:
    print(f"  max|d|={m:.3e}  {k}  ({info})")
print("\nEMPTY tensors (numel==0):")
for k, sh in empties[:8]:
    print(f"  {k}  shape={sh}")
worst = rows[0][1] if rows else ""
print(f"\nworst param = {worst}")
print("  -> VALUE HEAD" if any(t in worst.lower() for t in ["value","v_head","score","lm_head"]) else "  -> BACKBONE (concerning)")
