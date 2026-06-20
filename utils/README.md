# Utils

Shared first-party helpers.

| File | What it does |
|---|---|
| `model_aggregation.py` | The model-aggregation implementation (FedAvg / FedProx), including FSDP-sharded weight averaging. Used by `core/fed/aggregator.py` and the federated server. |
| `colored_logging.py` | Console logging helpers. |

The aggregation verification / diagnostics toolbox lives separately under
[`../tools/aggregation/`](../tools/aggregation/).
