# Evaluation and trajectory collection

Scripts that roll out a (trained) checkpoint and dump per-episode trajectories.
Run them from the repository root, inside the matching environment (WebShop or
ALFWorld; see [docs/installation.md](../docs/installation.md)). There are two layers:

- **Single checkpoint, one pass** ([`../evaluate.sh`](../evaluate.sh)): a quick eval
  on the unperturbed environment that prints Success Rate / Task Score. Covered in
  [docs/running.md](../docs/running.md) and [docs/reproducing.md](../docs/reproducing.md).
- **A whole split, batched** (`batch_webshop_eval.sh`, `batch_alfworld_eval.sh`):
  loop a train or validation split in windows and merge to one JSON. This is how the
  `hardness` partition gets its per-task difficulty labels.

## Files

| File | Purpose |
|---|---|
| [`../evaluate.sh`](../evaluate.sh) | dispatcher: `bash evaluate.sh <webshop\|alfworld> <ckpt>` |
| `eval_webshop.sh`, `eval_alfworld.sh` | the one-pass harness `evaluate.sh` calls |
| `batch_webshop_eval.sh`, `batch_alfworld_eval.sh` | batched train / val sweeps (below) |
| `merge_trajectories.py` | merge per-episode shards into a single JSON |
| `view_results.py` | summarize rollout / validation results (`-f` parquet, `-d` dir) |
| `convert_fsdp_to_hf.sh` | merge FSDP-sharded checkpoint into a HuggingFace model dir |

## Checkpoint format (FSDP -> HF)

Training and federated aggregation save the actor as **FSDP shards**
(`model_world_size_*_rank_*.pt`) under an `actor/` dir, not a HuggingFace model (the
configs set `actor.checkpoint.contents=[model]`). The rollout harness loads weights via
HF `from_pretrained`, so a trained checkpoint must be merged to HF format first.

- `evaluate.sh` handles this automatically: if the checkpoint is (or contains) a single
  FSDP-sharded `actor/` dir, it merges to `<actor>/hf_merged/` once and evaluates that.
- For the batch sweeps, or to convert manually:

  ```bash
  bash eval/convert_fsdp_to_hf.sh <ckpt>/global_step_N/actor   # -> .../actor/hf_merged
  ```

A base HuggingFace model id (e.g. `Qwen/Qwen2.5-1.5B-Instruct`) or an already-merged
directory is used as-is, with no conversion needed.

## Batched train / val sweeps

`batch_{webshop,alfworld}_eval.sh` choose what to roll out via the `SPLIT` env var.
Knobs are env vars set before the command; `ENGINE` / `CHECKPOINT` / `START_BATCH`
are positional. Each script's header lists every option.

### Training pool (for the `hardness` partition)

```bash
bash eval/batch_webshop_eval.sh  vllm /path/to/checkpoint   # -> output/inference/all_trajectories.json
bash eval/batch_alfworld_eval.sh vllm /path/to/checkpoint   # -> output/inference/all_trajectories_alfworld.json
```

Loops the whole training split in `BATCH_SIZE` (128) windows and merges. The
`hardness` partition reads these files; see [docs/heterogeneity.md](../docs/heterogeneity.md).

### Validation set

```bash
# default: the EXACT in-training validation set (the goals val/success_rate is computed on)
SPLIT=val bash eval/batch_webshop_eval.sh  vllm /path/to/checkpoint
SPLIT=val bash eval/batch_alfworld_eval.sh vllm /path/to/checkpoint

# sweep the full held-out pool, in batches
SPLIT=val VAL_TOTAL=500 bash eval/batch_webshop_eval.sh  vllm /path/to/checkpoint   # WebShop goals[0:500]
SPLIT=val VAL_TOTAL=140 bash eval/batch_alfworld_eval.sh vllm /path/to/checkpoint   # ALFWorld valid_seen
```

Both write `output/inference/all_trajectories_{webshop,alfworld}_val.json`.

- **Default** (`SPLIT=val`, no `VAL_TOTAL`): the in-training validation set, i.e.
  `goals[0:VAL_SUBSET]` (default 64 = `data.val_batch_size`). WebShop rolls out
  `goals[0:64]` (the federated val set, via the windowed-val path); ALFWorld takes the
  first 64 of the seed-shuffled `valid_seen`. Both match the goals the reported
  `val/success_rate` is computed on.
- **Sweep** (`VAL_TOTAL=N`): cover held-out `goals[0:N]` in `BATCH_SIZE` windows, with
  bounded memory. Lower `BATCH_SIZE` for smaller batches.

All sweeps accept `START_BATCH` to resume after an interruption.

## Output

Per-episode shards are written under `output/inference/` and merged into a single
`all_trajectories*.json` per run. The train-pool files feed the `hardness` partition;
the `_val` files are for inspection and standalone validation scoring.
