"""Checkpoint / model path management.

Extracted from FederatedServer. Owns all file-system operations relating to
client and aggregated checkpoints, plus the logic for finding model paths.
"""

import shutil
from pathlib import Path
from typing import Optional


class CheckpointManager:
    def __init__(self, config, output_dir, logger):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger

    def find_client_model_path(self, client_dir: Path) -> Optional[str]:
        """Locate a client's trained model and return the client directory path for aggregation.

        Searches the client's `checkpoints/` directory (preferring the latest
        `global_step_*/actor` checkpoint), then a few fallback subdirectories.
        Returns the client directory itself (not the model file) because the
        aggregation routine consumes the directory.
        """
        checkpoints_dir = client_dir / "checkpoints"
        if checkpoints_dir.exists():
            global_step_dirs = [d for d in checkpoints_dir.iterdir()
                                if d.is_dir() and d.name.startswith('global_step_')]
            if global_step_dirs:
                latest_step_dir = max(global_step_dirs,
                                      key=lambda x: int(x.name.split('_')[2]))
                actor_dir = latest_step_dir / "actor"
                if actor_dir.exists():
                    model_files = []
                    for ext in ["*.pt", "*.pth", "*.safetensors", "*.bin"]:
                        model_files.extend(list(actor_dir.glob(ext)))
                    if model_files:
                        self.logger.info(f"Found model files in client directory: {client_dir}")
                        return str(client_dir)

            model_files = []
            for ext in ["*.pth", "*.pt", "*.safetensors", "*.bin"]:
                model_files.extend(list(checkpoints_dir.glob(ext)))
            if model_files:
                self.logger.info(f"Found model files in client directory: {client_dir}")
                return str(client_dir)

        for subdir in ("models", "output", "logs", "wandb"):
            path = client_dir / subdir
            if path.exists():
                model_files = []
                for ext in ["*.pth", "*.pt", "*.safetensors", "*.bin"]:
                    model_files.extend(list(path.glob(ext)))
                if model_files:
                    self.logger.info(f"Found model files in client directory: {client_dir}")
                    return str(client_dir)

        self.logger.warning(f"No model files found in {client_dir}")
        return None

    def _find_latest_checkpoint(self, checkpoints_dir: Path) -> Optional[Path]:
        """Find the most recent `global_step_*` checkpoint under checkpoints_dir."""
        try:
            if not checkpoints_dir.exists():
                return None

            global_step_dirs = [d for d in checkpoints_dir.iterdir()
                                if d.is_dir() and d.name.startswith('global_step_')]
            if not global_step_dirs:
                return None

            global_step_dirs.sort(key=lambda x: int(x.name.split('_')[2]))
            latest_checkpoint = global_step_dirs[-1]

            actor_dir = latest_checkpoint / "actor"
            if actor_dir.exists():
                model_files = list(actor_dir.glob("*.pt")) + list(actor_dir.glob("*.pth"))
                if not model_files:
                    model_files = list(actor_dir.glob("model_world_size_*_rank_*.pt"))

                if model_files:
                    latest_iteration_file = checkpoints_dir / "latest_checkpointed_iteration.txt"
                    if latest_iteration_file.exists():
                        try:
                            latest_iteration = latest_iteration_file.read_text().strip()
                            if latest_iteration in str(latest_checkpoint):
                                return latest_checkpoint
                        except Exception as e:
                            self.logger.warning(f"Failed to read latest_checkpointed_iteration.txt: {e}")
                    return latest_checkpoint
            return None
        except Exception as e:
            self.logger.warning(f"Error finding latest checkpoint in {checkpoints_dir}: {e}")
            return None

    def _cleanup_old_global_step_checkpoints(self, checkpoints_dir: Path,
                                             latest_checkpoint: Path) -> None:
        """Delete stale `global_step` checkpoints, keeping only the latest one."""
        try:
            if not checkpoints_dir.exists():
                return
            global_step_dirs = [d for d in checkpoints_dir.iterdir()
                                if d.is_dir() and d.name.startswith('global_step_')]
            if len(global_step_dirs) <= 1:
                return

            for checkpoint_dir in global_step_dirs:
                if checkpoint_dir != latest_checkpoint:
                    try:
                        shutil.rmtree(checkpoint_dir)
                        self.logger.info(f"Cleaned up old checkpoint: {checkpoint_dir}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete old checkpoint {checkpoint_dir}: {e}")
            self.logger.info(f"Cleanup completed. Kept latest checkpoint: {latest_checkpoint}")
        except Exception as e:
            self.logger.warning(f"Error cleaning up old checkpoints in {checkpoints_dir}: {e}")

    def _find_critic_model_path(self, actor_model_path: str) -> Optional[str]:
        """Given an actor model path, locate the matching critic model path."""
        try:
            actor_path = Path(actor_model_path)

            if 'global_step_' in str(actor_path) and actor_path.name.startswith('global_step_'):
                critic_dir = actor_path / "critic"
                if critic_dir.exists():
                    model_files = list(critic_dir.glob("*.pt")) + list(critic_dir.glob("*.pth"))
                    if not model_files:
                        model_files = list(critic_dir.glob("model_world_size_*_rank_*.pt"))
                    if model_files:
                        self.logger.info(f"Found critic model in checkpoint directory: {critic_dir}")
                        return str(critic_dir)
                    self.logger.warning(f"Critic directory exists but no model files found: {critic_dir}")
                else:
                    self.logger.warning(f"Critic directory not found in checkpoint: {critic_dir}")

            elif 'global_step_' in str(actor_path) and 'actor' in str(actor_path):
                critic_path = str(actor_path).replace('/actor/', '/critic/')
                if Path(critic_path).exists():
                    self.logger.info(f"Found critic model at: {critic_path}")
                    return critic_path
                self.logger.warning(f"Critic model not found at expected path: {critic_path}")

            elif 'aggregated' in str(actor_path):
                critic_path = str(actor_path).replace('aggregated_actor_model.pth',
                                                      'aggregated_critic_model.pth')
                if Path(critic_path).exists():
                    self.logger.info(f"Found aggregated critic model at: {critic_path}")
                    return critic_path
                self.logger.warning(f"Aggregated critic model not found at expected path: {critic_path}")

            return None
        except Exception as e:
            self.logger.error(f"Error finding critic model path: {str(e)}")
            return None

    def ensure_final_model_save(self, client_dir: Path, client_id: int, round_num: int):
        """Make sure the final weights are persisted once training finishes.

        If no model is found in the usual checkpoint locations, fall back to
        copying the latest model file out of the client's W&B run directory.
        """
        self.logger.info(f"Ensuring final model save for client {client_id} in round {round_num}")

        model_path = self.find_client_model_path(client_dir)
        if model_path:
            self.logger.info(f"Model already saved at: {model_path}")
            return

        wandb_dir = client_dir / "wandb"
        if wandb_dir.exists():
            for run_dir in wandb_dir.iterdir():
                if run_dir.is_dir():
                    files_dir = run_dir / "files"
                    if files_dir.exists():
                        model_files = []
                        for ext in ["*.pth", "*.pt", "*.safetensors", "*.bin"]:
                            model_files.extend(list(files_dir.glob(ext)))
                        if model_files:
                            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                            checkpoints_dir = client_dir / "checkpoints"
                            checkpoints_dir.mkdir(exist_ok=True)
                            target_path = checkpoints_dir / f"client_{client_id}_round_{round_num}_final.pth"
                            shutil.copy2(latest_model, target_path)
                            self.logger.info(f"Copied model from wandb to: {target_path}")
                            return

        self.logger.warning(f"Could not ensure final model save for client {client_id}")

    def cleanup_old_checkpoints_for_client(self, checkpoints_dir: Path,
                                           client_id: int, round_num: int):
        """Trim a single client's checkpoint directory, keeping only the latest `global_step`.

        Generic per-client trim helper: it is NOT specific to any run mode,
        despite the "Centralized resume epoch mode:" prefix baked into its log
        lines below. That prefix is misleading here: "Centralized resume epoch
        mode" is a distinct runtime mode gated by the CENTRALIZED_RESUME_EPOCH
        env var (used in round_orchestrator.py / script_builder.py /
        run_federated.py / custom_fed_server.py), and this method is not tied
        to it. Verified via repo-wide grep that this method currently has NO
        caller; kept for potential reuse (noted 2026-04-18). If revived, drop
        or adjust the "Centralized resume epoch mode:" log prefix so the logs
        are not misleading.
        """
        try:
            global_step_dirs = [d for d in checkpoints_dir.iterdir()
                                if d.is_dir() and d.name.startswith('global_step_')]
            if len(global_step_dirs) <= 1:
                return
            global_step_dirs.sort(key=lambda x: int(x.name.split('_')[2]))
            latest_step_dir = global_step_dirs[-1]
            for old_dir in global_step_dirs[:-1]:
                shutil.rmtree(old_dir)
                self.logger.info(
                    f"Centralized resume epoch mode: deleted old checkpoint "
                    f"{old_dir.name} for client {client_id} in round {round_num}"
                )
            self.logger.info(
                f"Centralized resume epoch mode: kept latest checkpoint "
                f"{latest_step_dir.name} for client {client_id} in round {round_num}"
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to cleanup old checkpoints for client {client_id} "
                f"in round {round_num}: {e}"
            )

    def cleanup_old_round_client_checkpoints(self, current_round: int):
        """Delete client and aggregated checkpoints from earlier rounds, keeping only the last few rounds."""
        max_keep = self.config['federated'].get('max_rounds_to_keep_client_checkpoints', 2)
        self.logger.info(f"Cleaning up old round checkpoints, keeping last {max_keep} rounds")

        rounds_to_keep = list(range(max(1, current_round - max_keep + 1), current_round + 1))
        self.logger.info(f"Rounds to keep: {rounds_to_keep}")

        self._cleanup_rounds_outside(rounds_to_keep, log_prefix="")
        self.logger.info(f"Checkpoint cleanup completed for rounds before {rounds_to_keep[0]}")

    def cleanup_old_round_client_checkpoints_on_resume(self, latest_completed_round: int):
        """Clean up old-round checkpoints when resuming.

        Takes latest_completed_round explicitly so this module doesn't need to
        know about session-state tracking.
        """
        max_keep = self.config['federated'].get('max_rounds_to_keep_client_checkpoints', 2)

        if latest_completed_round == 0:
            return

        rounds_to_keep = list(range(max(1, latest_completed_round - max_keep + 1),
                                    latest_completed_round + 1))
        self.logger.info(
            f"Resume cleanup: keeping last {max_keep} round(s) {rounds_to_keep}"
        )

        self._cleanup_rounds_outside(rounds_to_keep, log_prefix="Resume cleanup: ")

    def _cleanup_rounds_outside(self, rounds_to_keep, log_prefix: str = ""):
        for round_dir in self.output_dir.iterdir():
            if not round_dir.is_dir() or not round_dir.name.startswith('round_'):
                continue
            try:
                round_num = int(round_dir.name.split('_')[1])
            except (ValueError, IndexError):
                continue

            if round_num in rounds_to_keep:
                continue

            self.logger.info(f"{log_prefix}cleaning up checkpoints for round {round_num}")

            for client_dir in round_dir.iterdir():
                if not client_dir.is_dir() or not client_dir.name.startswith('client_'):
                    continue
                ck_dir = client_dir / "checkpoints"
                if ck_dir.exists():
                    shutil.rmtree(ck_dir)
                    self.logger.info(f"{log_prefix}deleted client checkpoints: {ck_dir}")

            aggregated_dir = round_dir / "aggregated"
            if aggregated_dir.exists():
                agg_ck = aggregated_dir / "checkpoints"
                if agg_ck.exists():
                    shutil.rmtree(agg_ck)
                    self.logger.info(f"{log_prefix}deleted aggregated checkpoints: {agg_ck}")
                for item in aggregated_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                        self.logger.info(f"{log_prefix}deleted aggregated file: {item}")
