"""Generate the WebShop reserved-distractor list for Environment-Level OOD evaluation.

This builds the set of distractor product ASINs that are held out of every
client's training catalog and injected only into the unperturbed validation
catalog, so the WebShop Environment-Level Heterogeneity experiments (paper
Variant 1, "Catalog Split") are scored on out-of-distribution (OOD) products
that no client trained on. The same list is consumed by both catalog-split
implementations exposed via the HOLDOUT_FILE env var: the current `catalog_split`
strategy (function `_distractor_disjoint_partition_webshop_v5`) and the legacy
`distractor_disjoint` strategy (`_distractor_disjoint_partition_webshop`).

Runs once. Output is committed alongside the Environment-Level yaml configs at
data/env_heterogeneity/holdout_webshop_v1.json (the `_v1` suffix is this artifact's
dataset revision number, NOT the paper's Variant 1). yaml configs reference the
file via federated.data_sharding.partition.kwargs.holdout_file, which script_builder
exports as HOLDOUT_FILE for the partition strategy to read.
"""
import json
import random
from pathlib import Path
from collections import defaultdict


REPO_ROOT = Path(__file__).resolve().parents[2]
WEBSHOP_DATA = (
    REPO_ROOT / "third_party" / "verl-agent" / "agent_system" / "environments"
    / "env_package" / "webshop" / "webshop" / "data"
)
HOLDOUT_DIR = REPO_ROOT / "data" / "env_heterogeneity"


def main(holdout_seed: int = 99999, per_category: int = 6):
    products = json.load(open(WEBSHOP_DATA / "items_shuffle_1000.json"))
    ins = json.load(open(WEBSHOP_DATA / "items_ins_v2_1000.json"))

    # Targets are ASINs that have a non-empty 'instruction' (will be referenced by some goal)
    target_asins = {asin for asin, entry in ins.items() if entry.get("instruction")}
    distractor_asins = sorted({p["asin"] for p in products} - target_asins)

    # Stratified sample: per_category items per product category
    asin_to_cat = {p["asin"]: p["category"] for p in products}
    by_cat = defaultdict(list)
    for d in distractor_asins:
        by_cat[asin_to_cat[d]].append(d)

    rng = random.Random(holdout_seed)
    holdout = []
    cat_counts = {}
    for cat in sorted(by_cat):
        pool = sorted(by_cat[cat])
        rng.shuffle(pool)
        picked = pool[:per_category]
        holdout.extend(picked)
        cat_counts[cat] = len(picked)

    output = {
        # Dataset revision tag for this holdout artifact (revision 1). This is a
        # bookkeeping field only -- it is written to the JSON but never read by the
        # runtime -- and it is unrelated to the paper's Environment-Level "Variant"
        # numbering (this distractor set serves paper Variant 1, Catalog Split).
        "version": "v1",
        "seed": holdout_seed,
        "n_holdout": len(holdout),
        "asins": sorted(holdout),
        "per_category_count": cat_counts,
        "comment": (
            "Reserved distractor ASINs for Environment-Level out-of-distribution (OOD) "
            "evaluation of paper Variant 1 (Catalog Split). "
            "Inject into the unperturbed validation (eval_unseen) catalog only; "
            "never include in any client training catalog."
        ),
        "stats": {
            "total_products": len(products),
            "target_asins": len(target_asins),
            "all_distractors": len(distractor_asins),
            "partition_distractors": len(distractor_asins) - len(holdout),
        },
    }

    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HOLDOUT_DIR / "holdout_webshop_v1.json"
    json.dump(output, open(out_path, "w"), indent=2, sort_keys=False)
    print(f"Wrote {out_path}")
    print(f"  total holdout = {len(holdout)} distractor ASINs")
    print(f"  per category  = {cat_counts}")
    print(f"  stats         = {output['stats']}")


if __name__ == "__main__":
    main()
