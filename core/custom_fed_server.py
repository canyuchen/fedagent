#!/usr/bin/env python3
"""
Federated learning server.

Drives federated training by invoking per-client shell scripts (one training
subprocess per client per round), then aggregating the resulting model weights.
"""

import os
import json
import random
import subprocess
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil
import yaml
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from utils.model_aggregation import ModelAggregator, aggregate_round_models
from omegaconf import OmegaConf
import re

try:
    # Keeps `self.logger.info(...)` from clobbering the tqdm bar.
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:  # tqdm < 4.40
    logging_redirect_tqdm = nullcontext

from core.fed.aggregator import Aggregator
from core.fed.checkpoint_manager import CheckpointManager
from core.fed.client_runner import ClientRunner
from core.fed.config_helpers import extract_dataset_name, get_shuffle_seed, load_config
from core.fed.round_orchestrator import RoundOrchestrator
from core.fed.script_builder import ScriptBuilder
from core.fed.session_manager import SessionManager

_PATHS_YAML = "./config/paths.yaml"


def _load_path_cfg():
    """Lazily load config/paths.yaml relative to the current working directory.

    Deferred out of module import time so that `import core.custom_fed_server`
    from outside the repository root does not crash; the path config is only
    needed when resolving the default config path. Raises a clear
    FileNotFoundError if the file is missing.
    """
    if not os.path.exists(_PATHS_YAML):
        raise FileNotFoundError(
            f"[custom_fed_server] config/paths.yaml not found from cwd={os.getcwd()}; "
            f"copy config/paths.yaml.example to config/paths.yaml and run from the "
            f"repository root."
        )
    return OmegaConf.load(_PATHS_YAML)


def _default_config_path():
    """Resolve the default federated config path from config/paths.yaml.

    Kept as a function (rather than a module-level constant) so the paths.yaml
    load only happens when a caller actually relies on the default, not at
    import time.
    """
    path_cfg = _load_path_cfg()
    return os.path.join(path_cfg['config']['root'], "federated_verl_config.yaml")


class FederatedServer:
    def __init__(self, config_path: str = None,
                 session_id: str = None, resume_session: str = None,
                 output_dir: str = None):
        """Initialize the federated learning server.

        Args:
            config_path: Path to the config file. When None, it is resolved from
                config/paths.yaml (config.root / federated_verl_config.yaml).
            session_id: Session ID (auto-generated when omitted).
            resume_session: Session ID to resume.
            output_dir: Output directory (defaults to fed.output_dir from the
                config file when omitted).
        """
        if config_path is None:
            config_path = _default_config_path()
        self.config_path = config_path
        self.config = load_config(config_path)
        self.dataset_name = extract_dataset_name(config_path)

        fed_config = self.config['federated']
        self.total_clients = fed_config['total_clients']
        cpr = fed_config.get('clients_per_round', 1) or 1
        self.clients_per_round = cpr if cpr > 0 else 1
        self.total_rounds = fed_config['total_rounds']
        self.epochs_per_client = fed_config['epochs_per_client']
        # Optional: after total_rounds training rounds, run one extra round_{N+1}
        # whose clients only do val_before_train (total_epochs=0) and skip aggregation.
        # Lets plots include the post-training eval as a real x = total_rounds * stride point.
        self.eval_only_final_round = bool(fed_config.get('eval_only_final_round', False))
        self.effective_total_rounds = self.total_rounds + (1 if self.eval_only_final_round else 0)
        self.base_script_path = Path(fed_config['base_script_path'])

        (self.base_output_dir, self.output_dir,
         self.session_id, self.is_resume) = self._resolve_session_paths(
            output_dir, session_id, resume_session
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()

        # Shared mutable state — refs get passed to helpers, so these dicts are
        # mutated in place (never reassigned) to keep the refs valid.
        self.round_clients: Dict[int, List[int]] = {}
        self.aggregated_models: Dict[Any, Any] = {}
        self.training_metrics: Dict[int, Any] = {}
        self.current_round = 0
        random.seed(fed_config['data_sharding']['seed'])

        self._instantiate_helpers()

        if self.is_resume:
            self.load_session_state()

        self._log_init_info()

    def _resolve_session_paths(self, output_dir_arg, session_id_arg, resume_session):
        """Decide (base_output_dir, output_dir, session_id, is_resume).

        Precedence:
          1. output_dir arg given → use it directly; is_resume iff it exists.
          2. resume_session given → base/{resume_session}, always is_resume=True.
          3. Else generate a timestamped session_id (+ optional shuffle suffix).
        """
        fed_config = self.config['federated']

        if output_dir_arg:
            base = Path(output_dir_arg)
            return base, base, base.name, base.exists()

        base = Path(fed_config['output_dir'])

        if resume_session:
            return base, base / resume_session, resume_session, True

        if session_id_arg:
            sid = session_id_arg
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sid = f"federated_session_{ts}"
            seed = get_shuffle_seed(self.config)
            if seed is not None and seed != 42:
                sid += f"_shuffle-{seed}"
        return base, base / sid, sid, False

    def _instantiate_helpers(self):
        """Wire the 6 extracted components (created after logger + shared state exist)."""
        self.checkpoint_manager = CheckpointManager(
            config=self.config,
            output_dir=self.output_dir,
            logger=self.logger,
        )
        self.script_builder = ScriptBuilder(
            config=self.config,
            output_dir=self.output_dir,
            dataset_name=self.dataset_name,
            total_clients=self.total_clients,
            base_script_path=self.base_script_path,
            aggregated_models=self.aggregated_models,
            checkpoint_manager=self.checkpoint_manager,
            logger=self.logger,
        )
        self.client_runner = ClientRunner(
            checkpoint_manager=self.checkpoint_manager,
            logger=self.logger,
        )
        self.session_manager = SessionManager(
            output_dir=self.output_dir,
            logger=self.logger,
        )
        self.aggregator = Aggregator(
            config=self.config,
            output_dir=self.output_dir,
            aggregated_models=self.aggregated_models,
            training_metrics=self.training_metrics,
            round_clients=self.round_clients,
            logger=self.logger,
        )
        self.round_orchestrator = RoundOrchestrator(
            config=self.config,
            output_dir=self.output_dir,
            total_clients=self.total_clients,
            clients_per_round=self.clients_per_round,
            epochs_per_client=self.epochs_per_client,
            round_clients=self.round_clients,
            script_builder=self.script_builder,
            client_runner=self.client_runner,
            logger=self.logger,
            train_rounds=self.total_rounds,
            eval_only_final_round=self.eval_only_final_round,
        )

    def _log_init_info(self):
        mode = "Resuming" if self.is_resume else "Starting new"
        self.logger.info(f"{mode} session: {self.session_id}")
        self.logger.info(f"FederatedServer initialized with config: {self.config_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Extracted dataset name: {self.dataset_name}")
        self.logger.info(
            f"Total clients: {self.total_clients}, "
            f"Clients per round: {self.clients_per_round}"
        )
        self.logger.info(
            f"Total rounds: {self.total_rounds}, "
            f"Epochs per client: {self.epochs_per_client}"
        )
        if self.eval_only_final_round:
            self.logger.info(
                f"eval_only_final_round=True → extra eval-only round_{self.effective_total_rounds} "
                "(total_epochs=0, val_before_train, no aggregation)"
            )

    def save_session_state(self):
        """Save the session state (delegates to SessionManager; this method binds the server-side fields)."""
        self.session_manager.save({
            'session_id': self.session_id,
            'round_clients': self.round_clients,
            'aggregated_models': self.aggregated_models,
            'training_metrics': self.training_metrics,
            'current_round': getattr(self, 'current_round', 0),
            'timestamp': datetime.now().isoformat(),
        })

    def load_session_state(self):
        """Load the session state, restore the fields onto self, and repair aggregated-model paths."""
        state = self.session_manager.load()
        if state is None:
            self.current_round = 0
            return

        try:
            # Mutate shared dicts in place so Aggregator / RoundOrchestrator see updates
            self.round_clients.clear()
            self.round_clients.update(state.get('round_clients', {}))
            self.aggregated_models.clear()
            # JSON loads dict keys as strings; re-coerce numeric round keys to int
            # so downstream `aggregated_models.get(round_num - 1)` (int lookups) hit.
            # Keys like "2_critic" stay as strings.
            for k, v in state.get('aggregated_models', {}).items():
                if isinstance(k, str) and k.isdigit():
                    self.aggregated_models[int(k)] = v
                else:
                    self.aggregated_models[k] = v
            self.training_metrics.clear()
            self.training_metrics.update(state.get('training_metrics', {}))
            self.current_round = state.get('current_round', 0)

            # Repair aggregated-model paths: VERL's resume_from_path requires a global_step_ segment.
            for round_num, model_path in list(self.aggregated_models.items()):
                if not model_path or 'global_step_' in str(model_path):
                    continue
                ck_dir = Path(model_path)
                if ck_dir.name == 'aggregated':
                    ck_dir = ck_dir / 'checkpoints'
                latest_iter_file = ck_dir / "latest_checkpointed_iteration.txt"
                if not latest_iter_file.exists():
                    self.logger.error(f"Latest checkpointed iteration file not found: {latest_iter_file}")
                    continue
                latest_step = latest_iter_file.read_text().strip()
                global_step_dir = ck_dir / f"global_step_{latest_step}"
                if global_step_dir.exists():
                    self.aggregated_models[round_num] = str(global_step_dir)
                    self.logger.info(
                        f"Fixed aggregated model path for round {round_num}: {global_step_dir}"
                    )
                else:
                    self.logger.error(f"Global step directory not found: {global_step_dir}")

            self.logger.info(f"Resuming from round {self.current_round}")
            self.logger.info(f"Completed rounds: {list(self.round_clients.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to apply loaded session state: {str(e)}")
            self.current_round = 0

    def list_available_sessions(self) -> List[str]:
        """List the available sessions."""
        return self.session_manager.list_sessions(self.base_output_dir)

    def setup_logging(self):
        """Configure logging."""
        log_file = self.output_dir / "federated_training.log"

        # Use the colored-logging helper.
        from utils.colored_logging import setup_colored_logging
        self.logger = setup_colored_logging(
            level=logging.INFO,
            log_file=log_file
        )

    def find_latest_completed_round(self) -> int:
        """
        Find the highest round number that has an aggregated checkpoint.

        Returns:
            int: The highest completed round, or 0 if none is found.
        """
        completed_rounds = []

        # Iterate over every round directory.
        for round_dir in self.output_dir.iterdir():
            if not round_dir.is_dir() or not round_dir.name.startswith('round_'):
                continue

            try:
                round_num = int(round_dir.name.split('_')[1])

                # Check whether this round has an aggregated checkpoint.
                aggregated_dir = round_dir / "aggregated"
                if aggregated_dir.exists():
                    aggregated_checkpoints_dir = aggregated_dir / "checkpoints"
                    if aggregated_checkpoints_dir.exists():
                        latest_iteration_file = aggregated_checkpoints_dir / "latest_checkpointed_iteration.txt"
                        if latest_iteration_file.exists():
                            completed_rounds.append(round_num)
                            continue

                # Eval-only final round has no aggregated checkpoint; treat
                # round_summary.json (written after all clients log step=0 metrics)
                # as the completion marker.
                if (self.eval_only_final_round
                        and round_num > self.total_rounds
                        and (round_dir / "round_summary.json").exists()):
                    completed_rounds.append(round_num)

            except (ValueError, IndexError):
                continue

        if not completed_rounds:
            return 0
        return max(completed_rounds)

    # ------------------------------------------------------------------ smart resume
    def smart_resume_training(self):
        """Smart resume: detect the parts already completed and skip them."""
        self.logger.info("Starting smart resume training...")
        latest_completed = self.find_latest_completed_round()
        self.checkpoint_manager.cleanup_old_round_client_checkpoints_on_resume(latest_completed)

        start_round = latest_completed + 1
        if latest_completed == 0:
            self.logger.info(f"No completed rounds found; starting from round {start_round}")
        else:
            self.logger.info(
                f"Latest completed round: {latest_completed}; starting from round {start_round}"
            )

        self._load_prior_aggregated_models(start_round)
        self.logger.info(f"Loaded {len(self.aggregated_models)} aggregated model(s)")
        self.logger.debug(f"aggregated_models = {self.aggregated_models}")

        # Round-level progress bar. `initial=start_round-1` makes the bar show
        # completed rounds from resumed runs, and ETA reflects only remaining
        # rounds. `pbar.update(1)` is deferred to after _process_round succeeds
        # so an early break does not inflate the completed count.
        with logging_redirect_tqdm(), tqdm(
            desc="Federated rounds",
            total=self.effective_total_rounds,
            initial=max(start_round - 1, 0),
            unit="round",
            dynamic_ncols=True,
            smoothing=0.3,
        ) as pbar:
            round_start_ts = time.time()
            for round_num in range(start_round, self.effective_total_rounds + 1):
                self.current_round = round_num
                pbar.set_postfix(round=round_num, refresh=False)
                if not self._process_round(round_num):
                    break
                now = time.time()
                pbar.set_postfix(
                    round=round_num,
                    last_s=f"{now - round_start_ts:.0f}",
                    refresh=False,
                )
                round_start_ts = now
                pbar.update(1)

        self.save_final_summary()
        self.logger.info("Smart federated learning training completed!")
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def _load_prior_aggregated_models(self, start_round: int):
        """Populate self.aggregated_models: scan the aggregated checkpoints of rounds 1..start_round-1;
        if none are found, fall back to any round on disk that has a checkpoint.
        """
        self.logger.info("Loading aggregated models from completed rounds...")
        for round_num in range(1, start_round):
            self._load_one_round_aggregated_model(round_num)

        if self.aggregated_models:
            return

        self.logger.warning(
            "No aggregated models found in completed rounds, "
            "searching for latest available checkpoint..."
        )
        available_rounds = []
        for round_dir in self.output_dir.iterdir():
            if not round_dir.is_dir() or not round_dir.name.startswith('round_'):
                continue
            try:
                rn = int(round_dir.name.split('_')[1])
            except (ValueError, IndexError):
                continue
            ck_dir = round_dir / "aggregated" / "checkpoints"
            if (ck_dir / "latest_checkpointed_iteration.txt").exists():
                available_rounds.append(rn)

        if not available_rounds:
            return

        latest_round = max(available_rounds)
        self.logger.info(f"Found latest available round with checkpoint: {latest_round}")
        ck_dir = self.output_dir / f"round_{latest_round}" / "aggregated" / "checkpoints"
        latest_iter = (ck_dir / "latest_checkpointed_iteration.txt").read_text().strip()
        # NOTE: original code did not prepend `global_step_` in this fallback path — preserved.
        checkpoint_dir = ck_dir / latest_iter
        if checkpoint_dir.exists():
            actor_model = checkpoint_dir / "actor" / "model_world_size_1_rank_0.pt"
            if actor_model.exists():
                self.aggregated_models[latest_round] = str(actor_model)
                self.logger.info(f"Loaded latest available aggregated model: {actor_model}")

    def _load_one_round_aggregated_model(self, round_num: int):
        """Load aggregated actor (and optional critic) model for a single round."""
        ck_dir = self.output_dir / f"round_{round_num}" / "aggregated" / "checkpoints"
        if not ck_dir.exists():
            return

        latest_iter_file = ck_dir / "latest_checkpointed_iteration.txt"
        if not latest_iter_file.exists():
            return
        latest_iter = latest_iter_file.read_text().strip()

        checkpoint_dir = ck_dir / f"global_step_{latest_iter}"
        if not checkpoint_dir.exists():
            subdirs = [d for d in ck_dir.iterdir() if d.is_dir()]
            if not subdirs:
                return
            checkpoint_dir = subdirs[0]
            self.logger.info(
                f"Using checkpoint directory: {checkpoint_dir} (file content was incomplete)"
            )

        actor_model = checkpoint_dir / "actor" / "model_world_size_1_rank_0.pt"
        if actor_model.exists():
            self.aggregated_models[round_num] = str(actor_model)
            self.logger.info(f"Loaded aggregated actor model for round {round_num}: {actor_model}")
            critic_model = checkpoint_dir / "critic" / "model_world_size_1_rank_0.pt"
            if critic_model.exists():
                self.aggregated_models[f"{round_num}_critic"] = str(critic_model)
                self.logger.info(f"Loaded aggregated critic model for round {round_num}: {critic_model}")
            return

        model_files = list(checkpoint_dir.rglob("*.pt"))
        if not model_files:
            return
        self.aggregated_models[round_num] = str(model_files[0])
        self.logger.info(f"Loaded aggregated model for round {round_num}: {model_files[0]}")
        if len(model_files) > 1:
            self.logger.info(
                f"Found {len(model_files)} model files in round {round_num}: "
                f"{[str(f) for f in model_files]}"
            )

    def _process_round(self, round_num: int) -> bool:
        """Run one round of smart-resume. Returns False on KeyboardInterrupt (caller breaks)."""
        round_status = self.round_orchestrator.check_round_completion_status(round_num)

        self.logger.info(
            f"=== Round {round_num}/{self.effective_total_rounds}"
            f"{' [eval-only]' if self._is_eval_only_round(round_num) else ''} === "
            f"completed={round_status['round_completed']}, "
            f"weights_exist={round_status['server_weights_exist']}, "
            f"done={round_status['completed_clients']}, "
            f"pending={round_status['pending_clients']}"
        )

        if round_status['round_completed']:
            self._load_completed_round_model(round_num)
            return True

        selected_clients = self.round_clients.get(round_num) \
            or self.round_orchestrator.select_clients(round_num)
        clients_to_train = [c for c in selected_clients if c in round_status['pending_clients']]
        self.logger.info(f"Round {round_num}: will train clients {clients_to_train}")

        try:
            if not clients_to_train and round_status['completed_clients']:
                self._handle_aggregation_only_round(round_num, round_status)
                return True
            if not clients_to_train:
                self.logger.info(f"No clients need training in round {round_num}")
                return True
            self._handle_full_round(round_num, selected_clients, round_status, clients_to_train)
            return True

        except KeyboardInterrupt:
            self.logger.info(f"Training interrupted at round {round_num}")
            self.save_session_state()
            self.logger.info(
                f"Session state saved. You can resume with session ID: {self.session_id}"
            )
            return False

        except Exception as e:
            self.logger.error(f"Error in round {round_num}: {str(e)}")
            self.save_session_state()
            self.logger.error(
                f"Session state saved. You can resume with session ID: {self.session_id}"
            )
            raise

    def _load_completed_round_model(self, round_num: int):
        """Round already fully done — load the aggregated model path from disk."""
        self.logger.info(f"Round {round_num} already completed, skipping to next round")
        aggregated_dir = self.output_dir / f"round_{round_num}" / "aggregated"
        if not aggregated_dir.exists():
            return
        model_files = []
        for ext in ("*.pth", "*.pt", "*.safetensors", "*.bin"):
            model_files.extend(list(aggregated_dir.glob(ext)))
        if not model_files:
            return

        actor_model = aggregated_dir / "aggregated_actor_model.pth"
        self.aggregated_models[round_num] = (
            str(actor_model) if actor_model.exists() else str(model_files[0])
        )
        self.logger.info(f"Loaded existing aggregated model: {self.aggregated_models[round_num]}")

    def _handle_aggregation_only_round(self, round_num: int, round_status: dict):
        """All clients already trained in this round — only aggregation may remain."""
        self.logger.info(
            f"All clients completed in round {round_num}, checking if aggregation is needed"
        )

        if self._is_eval_only_round(round_num):
            self.logger.info(
                f"Eval-only round {round_num}: all clients have metrics, no aggregation needed"
            )
            all_client_results = self._collect_completed_client_results(
                round_num, round_status['completed_clients']
            )
            self.aggregator.save_round_summary(round_num, all_client_results)
            self.save_session_state()
            return

        if os.environ.get('CENTRALIZED_RESUME_EPOCH', 'false').lower() == 'true':
            self.logger.info(
                f"Centralized resume epoch mode: skipping aggregation for round {round_num}"
            )
            self.logger.info("Using existing checkpoints as resume path for next training session")
            self.save_session_state()
            return

        if round_status['server_weights_exist']:
            self.logger.info(f"Round {round_num} already has aggregated model, skipping")
            return

        self.logger.info("No aggregated model found, performing aggregation")
        all_client_results = self._collect_completed_client_results(
            round_num, round_status['completed_clients']
        )
        if not all_client_results:
            msg = f"No valid client results found for aggregation in round {round_num}"
            self.logger.error(msg)
            raise RuntimeError(msg)
        self._finalize_round(round_num, all_client_results)
        self.logger.info(f"Round {round_num} aggregation completed")

    def _handle_full_round(self, round_num: int, selected_clients: list,
                           round_status: dict, clients_to_train: list):
        """Some clients still need training — run them, then aggregate."""
        previous_model_path = self._resolve_previous_model_path(round_num)

        available_gpus = self.round_orchestrator.get_available_gpus()
        self.logger.info(
            f"Using smart parallel mode for {len(clients_to_train)} clients "
            f"with {available_gpus} GPUs"
        )
        client_results = self.round_orchestrator.run_smart_parallel_client_training(
            clients_to_train, round_num, previous_model_path,
            available_gpus=available_gpus,
        )

        all_client_results = self._collect_completed_client_results(
            round_num, round_status['completed_clients']
        )
        all_client_results.extend(client_results)

        if self._is_eval_only_round(round_num):
            # Eval-only: clients ran val_before_train with total_epochs=0, no new
            # weights to aggregate. Save a round summary so resume sees completion.
            self.aggregator.save_round_summary(round_num, all_client_results)
            self.save_session_state()
        else:
            self._finalize_round(round_num, all_client_results)

        successful = sum(1 for r in all_client_results if r['success'])
        self.logger.info(
            f"Round {round_num} completed: {successful}/{len(selected_clients)} clients successful"
        )

        if round_num < self.effective_total_rounds:
            wait_time = self.config['federated']['rounds']['wait_between_rounds']
            self.logger.info(f"Waiting {wait_time} seconds before next round...")
            time.sleep(wait_time)

    def _is_eval_only_round(self, round_num: int) -> bool:
        return self.eval_only_final_round and round_num > self.total_rounds

    def _resolve_previous_model_path(self, round_num: int):
        """Normalize previous aggregated model path to a global_step_X directory."""
        previous_model_path = self.aggregated_models.get(round_num - 1)
        self.logger.debug(f"Round {round_num}: previous_model_path = {previous_model_path}")
        self.logger.debug(f"Round {round_num}: aggregated_models = {self.aggregated_models}")

        if not previous_model_path:
            self.logger.warning(f"No previous model path found for round {round_num}")
            return None

        if 'global_step_' in str(previous_model_path):
            self.logger.info(f"Using checkpoint directory: {previous_model_path}")
            return previous_model_path

        # Convert .../global_step_Y/actor/model_world_size_1_rank_0.pt → .../global_step_Y
        checkpoint_dir = Path(previous_model_path).parent.parent
        self.logger.info(f"Using checkpoint directory: {checkpoint_dir}")
        return str(checkpoint_dir)

    def _collect_completed_client_results(self, round_num: int,
                                          completed_clients: list) -> list:
        """Build client_results entries for clients that already finished on disk."""
        results = []
        for client_id in completed_clients:
            client_dir = self.output_dir / f"round_{round_num}" / f"client_{client_id}"
            model_path = self.checkpoint_manager.find_client_model_path(client_dir)
            if model_path:
                results.append({
                    'success': True,
                    'client_id': client_id,
                    'round_num': round_num,
                    'model_path': model_path,
                    'log_file': str(client_dir / "training.log"),
                })
        return results

    def _finalize_round(self, round_num: int, all_client_results: list):
        """Common tail: aggregate, save summary/state, cleanup old round checkpoints.

        Raises RuntimeError if aggregation fails, after saving session state.
        """
        # Run cleanup BEFORE aggregation to free disk for the ~13G of shards the
        # aggregator is about to write. If cleanup stayed after aggregation, a
        # disk-full failure would skip the cleanup and the job could never
        # recover — which is exactly what happened at round 42. Wrapped in its
        # own try so a cleanup hiccup never blocks aggregation.
        # NOTE: with max_rounds_to_keep_client_checkpoints=2 at round N this
        # keeps {N-1, N} — i.e. N-1's aggregated stays available as the resume
        # fallback if round N's aggregation itself fails.
        try:
            self.checkpoint_manager.cleanup_old_round_client_checkpoints(round_num)
        except Exception as e:
            self.logger.warning(
                f"Pre-aggregation cleanup for round {round_num} failed: {e}"
            )

        try:
            self.aggregator.aggregate_models(round_num, all_client_results)
            self.aggregator.save_round_summary(round_num, all_client_results)
            self.save_session_state()
        except RuntimeError as e:
            self.logger.error(f"Round {round_num} aggregation failed: {str(e)}")
            self.save_session_state()
            raise

    # ------------------------------------------------------------------
    def run_federated_training(self):
        """
        Run federated learning training (using smart resume).
        """
        self.logger.info("Starting federated learning training with smart resume")
        self.smart_resume_training()

    def save_final_summary(self):
        """Save the final training summary."""
        final_summary = {
            'total_rounds': self.total_rounds,
            'total_clients': self.total_clients,
            'clients_per_round': self.clients_per_round,
            'epochs_per_client': self.epochs_per_client,
            'round_clients': self.round_clients,
            'aggregated_models': self.aggregated_models,
            'training_metrics': self.training_metrics,
            'completion_time': datetime.now().isoformat()
        }

        summary_file = self.output_dir / "federated_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)

        self.logger.info(f"Final summary saved to {summary_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Federated learning server')
    parser.add_argument('--config', type=str,
                       default=None,
                       help='Path to the config file')
    parser.add_argument('--session-id', type=str, default=None,
                       help='Session ID (optional; auto-generated if not provided)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Session ID to resume')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory path (optional; falls back to the path in the config file if not provided)')
    parser.add_argument('--list-sessions', action='store_true',
                       help='List the available sessions')
    parser.add_argument('--smart-resume', action='store_true',
                       help='Enable smart resume mode: auto-detect completed parts and skip them')

    args = parser.parse_args()

    config_path = args.config if args.config is not None else _default_config_path()

    if args.list_sessions:
        # List the available sessions.
        temp_server = FederatedServer(config_path=config_path)
        sessions = temp_server.list_available_sessions()
        if sessions:
            print("Available sessions:")
            for session in sessions:
                print(f"  {session}")
        else:
            print("No available sessions found")
        return

    # Create the federated learning server.
    server = FederatedServer(
        config_path=config_path,
        session_id=args.session_id,
        resume_session=args.resume,
        output_dir=args.output_dir
    )

    # Run federated learning training.
    if args.smart_resume:
        server.logger.info("Using smart resume mode")
        server.smart_resume_training()
    else:
        server.run_federated_training()


if __name__ == "__main__":
    main()
