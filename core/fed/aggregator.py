"""Model aggregation + round summary persistence.

Extracted from FederatedServer. Owns the per-round aggregation of client
models (actor + optional critic) and the round summary JSON that tracks
which aggregated model came out of a round.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from utils.model_aggregation import aggregate_round_models


class Aggregator:
    def __init__(self, config: Dict, output_dir, aggregated_models: Dict,
                 training_metrics: Dict, round_clients: Dict, logger):
        self.config = config
        self.output_dir = Path(output_dir)
        self.aggregated_models = aggregated_models  # shared ref, mutated
        self.training_metrics = training_metrics    # shared ref, mutated
        self.round_clients = round_clients          # shared ref, read
        self.logger = logger

    def aggregate_models(self, round_num: int,
                         client_results: List[Dict[str, Any]]) -> str:
        """Aggregate the client models and return the path of the aggregated main (actor) model."""
        self.logger.info(f"Aggregating models for round {round_num}")

        successful = [r for r in client_results if r['success'] and r.get('model_path')]
        if not successful:
            msg = f"No successful clients in round {round_num}. Cannot proceed with aggregation."
            self.logger.error(msg)
            raise RuntimeError(msg)

        expected = len(client_results)
        actual = len(successful)
        if actual < expected:
            msg = (f"Insufficient successful clients for aggregation in round {round_num}. "
                   f"Expected {expected} clients but only {actual} clients succeeded.")
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Aggregation rule. 'fedavg' (uniform model averaging) is the default and
        # is the only rule used for every reported result in the paper. 'fedprox'
        # is implemented but unused experimentally (it exists solely for the
        # mu=0.01 ablation). FedProx needs a reference model to anchor its
        # proximal term, so we pass the previous round's aggregated global model
        # as `global_model_path`; round 1 has no previous global model, hence the
        # `round_num > 1` guard (it stays None on the first round).
        aggregation_method = self.config['federated'].get('aggregation_method', 'fedavg')
        global_model_path = None
        if aggregation_method == 'fedprox' and round_num > 1:
            global_model_path = self.aggregated_models.get(round_num - 1)

        try:
            n_gpus_per_node = self.config.get('verl', {}).get('trainer', {}).get('n_gpus_per_node', 1)
            aggregated = aggregate_round_models(
                round_num=round_num,
                client_results=successful,
                output_dir=self.output_dir,
                aggregation_method=aggregation_method,
                n_gpus_per_node=n_gpus_per_node,
                global_model_path=global_model_path,
                mu=self.config['federated'].get('fedprox_mu', 0.01),
            )

            if not aggregated:
                msg = f"Failed to aggregate models for round {round_num}. No aggregated models returned."
                self.logger.error(msg)
                raise RuntimeError(msg)

            main_model_path = aggregated.get('actor', list(aggregated.values())[0])
            self.aggregated_models[round_num] = self._normalize_to_global_step(main_model_path)

            if 'critic' in aggregated:
                critic_path = aggregated['critic']
                self.aggregated_models[f"{round_num}_critic"] = critic_path
                self.logger.info(f"Critic model aggregation completed: {critic_path}")

            self.logger.info(f"Model aggregation completed: {main_model_path}")
            self.logger.info(f"Aggregated models: {list(aggregated.keys())}")
            return main_model_path

        except RuntimeError:
            raise
        except Exception as e:
            msg = f"Unexpected error during model aggregation for round {round_num}: {str(e)}"
            self.logger.error(msg)
            raise RuntimeError(msg)

    def save_round_summary(self, round_num: int,
                           client_results: List[Dict[str, Any]]):
        """Write this round's training summary to a JSON file."""
        summary = {
            'round_num': round_num,
            'timestamp': datetime.now().isoformat(),
            'selected_clients': self.round_clients[round_num],
            'client_results': client_results,
            'aggregated_model': self.aggregated_models.get(round_num),
        }
        summary_file = self.output_dir / f"round_{round_num}" / "round_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        self.training_metrics[round_num] = summary

    def _normalize_to_global_step(self, model_path: str) -> str:
        """Normalize a checkpoint path so it contains a `global_step_` segment.

        VERL's `resume_from_path` requires the path to include a `global_step_`
        field. If the supplied path is already a `global_step_X` directory it is
        returned unchanged; otherwise the concrete step is parsed from
        `latest_checkpointed_iteration.txt` and appended to build the full path.
        """
        if not model_path or 'global_step_' in str(model_path):
            return model_path

        checkpoints_dir = Path(model_path)
        latest_iter_file = checkpoints_dir / "latest_checkpointed_iteration.txt"
        if not latest_iter_file.exists():
            self.logger.error(f"Latest checkpointed iteration file not found: {latest_iter_file}")
            return model_path

        latest_step = latest_iter_file.read_text().strip()
        global_step_dir = checkpoints_dir / f"global_step_{latest_step}"
        if not global_step_dir.exists():
            self.logger.error(f"Global step directory not found: {global_step_dir}")
            return model_path

        return str(global_step_dir)
