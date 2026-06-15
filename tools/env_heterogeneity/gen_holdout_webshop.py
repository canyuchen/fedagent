"""Generate the WebShop holdout-distractor list for env-level OOD eval.

Runs once. Output is committed alongside the env-level yamls at
config/env_heterogeneity/holdout_webshop_v1.json.
yaml configs reference this file via federated.data_sharding.partition.kwargs.holdout_file.
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
HOLDOUT_DIR = REPO_ROOT / "config" / "env_heterogeneity"


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
        "version": "v1",
        "seed": holdout_seed,
        "n_holdout": len(holdout),
        "asins": sorted(holdout),
        "per_category_count": cat_counts,
        "comment": (
            "Reserved distractor ASINs for OOD env eval. "
            "Inject into eval_unseen catalog only; never include in any client training catalog."
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
