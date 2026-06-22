# Heterogeneity simulations and smoke tests

Task-level partition-strategy **simulations** (the figures showing how each strategy
splits tasks across clients) plus two federated **smoke tests**. The env-level analog
is [`../env_heterogeneity/`](../env_heterogeneity/); the strategies themselves are
documented in [`../../docs/heterogeneity.md`](../../docs/heterogeneity.md).

Run everything **from the repository root** (the scripts load `./config/paths.yaml`
to locate the vendored env package).

## Partition simulations

Visualize how each task-level strategy distributes tasks across 100 clients:

| Script | Strategy | Knob |
|---|---|---|
| `simulate_preference.py` | Preference (Dirichlet) | `omega` |
| `simulate_coverage.py` | Coverage (Beta pool sizes) | `size_std` |
| `simulate_hardness.py` | Hardness (success-rate quotas) | `success_std` |

```bash
python tools/heterogeneity_test/simulate_preference.py --dataset webshop
python tools/heterogeneity_test/simulate_coverage.py   --dataset webshop
python tools/heterogeneity_test/simulate_hardness.py   --dataset webshop
```

`simulate_heterogenous.sh` runs all three. Each sweeps a few values of its knob and
writes high-resolution PNGs into a per-strategy subdirectory. The synthetic data
mirrors the real WebShop category mix (~2,400 samples over
beauty / electronics / fashion / garden / grocery).

## Smoke tests

| Script | Checks |
|---|---|
| `test_alfworld_fed.py` | the federated ALFWorld env builds and rolls out |
| `test_data.py` | data loading / partitioning |
