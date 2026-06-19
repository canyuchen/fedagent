"""Per-round orchestration: client selection, GPU allocation, training dispatch,
and completion status detection.

Extracted from FederatedServer. Owns the logic that decides, for a given
round, which clients to run, which GPU each should run on, and whether a
round has already finished (used by smart-resume).
"""

import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List


class RoundOrchestrator:
    def __init__(self, config: Dict, output_dir, total_clients: int,
                 clients_per_round: int, epochs_per_client: int,
                 round_clients: Dict, script_builder, client_runner, logger,
                 train_rounds: int = None, eval_only_final_round: bool = False):
        self.config = config
        self.output_dir = Path(output_dir)
        self.total_clients = total_clients
        self.clients_per_round = clients_per_round
        self.epochs_per_client = epochs_per_client
        self.round_clients = round_clients  # shared ref, mutated in select_clients
        self.script_builder = script_builder
        self.client_runner = client_runner
        self.logger = logger
        # train_rounds: number of normal training rounds (config.federated.total_rounds).
        # eval_only_final_round: when True, round_{train_rounds+1} runs val_before_train
        # only (epochs=0) — no FedAvg, no client checkpoint.
        self.train_rounds = train_rounds
        self.eval_only_final_round = eval_only_final_round

    def _is_eval_only_round(self, round_num: int) -> bool:
        return (self.eval_only_final_round
                and self.train_rounds is not None
                and round_num > self.train_rounds)

    def _epochs_for_round(self, round_num: int) -> int:
        return 0 if self._is_eval_only_round(round_num) else self.epochs_per_client

    # ------------------------------------------------------------------ selection
    def select_clients(self, round_num: int) -> List[int]:
        """Select the clients that participate in this round.

        Two strategies:
          - 'uniform_single': the paper's "Local Agent Training" baseline. The same
            single client (config kwargs.cl_id, e.g. paper indices 21/42/84) is
            selected every round, modelling one client training alone with no
            federation/aggregation.
          - any other strategy (uniform, preference, coverage, hardness, the
            env-level keys, etc.): sample M = clients_per_round clients without
            replacement, seeded deterministically per round (base_seed + round_num
            - 1) so the same clients are re-selected on resume.
        """
        strategy = self.config['federated']['data_sharding']['partition']['strategy']

        # 'uniform_single' == paper "Local Agent Training" baseline: pin the same
        # single client (cl_id) for every round (no random sampling, no federation).
        if strategy == 'uniform_single':
            cl_id = self.config['federated']['data_sharding']['partition']['kwargs']['cl_id']
            selected = [cl_id]
            self.round_clients[round_num] = selected
            self.logger.info(f"Round {round_num}: Using uniform_single strategy, selected client {cl_id}")
            return selected

        base_seed = self.config['federated']['data_sharding']['seed']
        round_seed = base_seed + round_num - 1
        random.seed(round_seed)
        selected = random.sample(range(self.total_clients), self.clients_per_round)
        self.round_clients[round_num] = selected
        self.logger.info(f"Round {round_num}: Selected clients {selected} (seed: {round_seed})")
        return selected

    # ------------------------------------------------------------------ GPU
    def detect_available_gpus(self) -> int:
        """Auto-detect how many GPUs are usable in the current environment."""
        try:
            import torch
            if not torch.cuda.is_available():
                self.logger.warning("CUDA is not available, using CPU only")
                return 0
            gpu_count = torch.cuda.device_count()

            available = 0
            failed = []
            for i in range(gpu_count):
                try:
                    with torch.cuda.device(i):
                        test_tensor = torch.tensor([1.0], device=f'cuda:{i}')
                        del test_tensor
                    available += 1
                except Exception as e:
                    failed.append((i, str(e)))
            if failed:
                for i, err in failed:
                    self.logger.warning(f"GPU {i} unavailable: {err}")
            return available
        except ImportError:
            self.logger.warning("PyTorch not available, cannot detect GPUs")
            return 0
        except Exception as e:
            self.logger.error(f"Error detecting GPUs: {str(e)}")
            return 0

    def get_available_gpus(self) -> int:
        """Prefer auto-detection; fall back to the configured count on failure."""
        detected = self.detect_available_gpus()
        if detected > 0:
            self.logger.info(f"Detected {detected} available GPUs")
            return detected
        config_gpus = self.config['federated']['environment'].get('available_gpus', 1)
        self.logger.warning(f"GPU auto-detection failed, using config fallback: {config_gpus}")
        return config_gpus

    def smart_gpu_allocation(self, selected_clients: List[int],
                             available_gpus: int) -> Dict[int, List[int]]:
        """Allocate GPUs intelligently: assign by per-client GPU demand, and fall back to sequential execution when GPUs are scarce."""
        gpus_per_client = self.config.get('verl', {}).get('trainer', {}).get('n_gpus_per_node', 1)

        self.logger.info(
            f"GPU allocation: {len(selected_clients)} clients, "
            f"{available_gpus} available GPUs, {gpus_per_client} GPUs per client"
        )

        total_needed = len(selected_clients) * gpus_per_client
        max_concurrent = available_gpus // gpus_per_client

        if total_needed > available_gpus:
            self.logger.warning(
                f"Insufficient GPUs for parallel training: need {total_needed} GPUs for "
                f"{len(selected_clients)} clients (each needs {gpus_per_client} GPUs), "
                f"but only {available_gpus} GPUs available"
            )
            self.logger.info(f"Will use sequential training: max {max_concurrent} clients can run concurrently")
            if max_concurrent == 0:
                msg = (f"Cannot run any client: each client needs {gpus_per_client} GPUs, "
                       f"but only {available_gpus} GPUs available")
                self.logger.error(msg)
                raise RuntimeError(msg)

        if gpus_per_client == 1:
            allocation: Dict[int, List[int]] = {}
            for i, client_id in enumerate(selected_clients):
                allocation.setdefault(i % available_gpus, []).append(client_id)
            return allocation

        allocation = {}
        current_gpu = 0
        for client_id in selected_clients:
            if current_gpu + gpus_per_client > available_gpus:
                allocation.setdefault(0, []).append(client_id)
                self.logger.info(
                    f"Client {client_id} queued for sequential execution on GPU 0 "
                    "(insufficient GPUs for parallel)"
                )
                continue

            client_gpus = list(range(current_gpu, current_gpu + gpus_per_client))
            primary_gpu = client_gpus[0]
            allocation.setdefault(primary_gpu, []).append(client_id)
            self.logger.info(f"Client {client_id} allocated GPUs: {client_gpus}")
            current_gpu += gpus_per_client
        return allocation

    # ------------------------------------------------------------------ training dispatch
    def run_sequential_client_training(self, selected_clients: List[int],
                                       round_num: int,
                                       previous_model_path: str = None
                                       ) -> List[Dict[str, Any]]:
        """Run client training sequentially (one client per GPU at a time).

        NOTE: This is a self-contained, fully-serial fallback path. The live
        training path used by FederatedServer is run_smart_parallel_client_training
        (see core/custom_fed_server.py), which shares GPUs across clients and only
        falls back to sequential execution per-GPU. This method currently has no
        caller; it is kept for reuse (e.g. debugging or single-GPU runs that want
        strictly one client at a time).
        """
        client_results = []
        available_gpus = self.get_available_gpus()
        self.logger.info(f"Available GPUs: {available_gpus}")

        client_scripts = {
            cid: self.script_builder.create_client_script(
                cid, round_num, self._epochs_for_round(round_num), previous_model_path
            )
            for cid in selected_clients
        }

        base_cuda = self.config['federated']['environment'].get('cuda_device', 0)

        for i, client_id in enumerate(selected_clients):
            self.logger.info(f"\n{'='*30}")
            self.logger.info(f"Starting client {client_id} ({i+1}/{len(selected_clients)})")
            self.logger.info(f"{'='*30}")

            if available_gpus > 1:
                gpu_id = i % available_gpus
                cuda_device = base_cuda + gpu_id
                self.logger.info(
                    f"Assigning GPU {cuda_device} (base={base_cuda}, offset={gpu_id}) "
                    f"to client {client_id}"
                )
            else:
                cuda_device = base_cuda
                self.logger.info(f"Using single GPU {cuda_device} for client {client_id}")

            original = os.environ.get('CUDA_VISIBLE_DEVICES')
            os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
            try:
                result = self.client_runner.run_client_training(
                    client_id, round_num, client_scripts[client_id]
                )
                client_results.append(result)
                if result['success']:
                    self.logger.info(f"Client {client_id} completed successfully")
                else:
                    self.logger.error(f"Client {client_id} failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                self.logger.error(f"Client {client_id} failed with exception: {str(e)}")
                client_results.append({
                    'success': False, 'client_id': client_id,
                    'round_num': round_num, 'error': str(e),
                })
            finally:
                if original is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original
                else:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)

            if i < len(selected_clients) - 1:
                wait = self.config['federated'].get('wait_between_clients', 5)
                if wait > 0:
                    self.logger.info(f"Waiting {wait} seconds before next client...")
                    time.sleep(wait)

        return client_results

    def run_smart_parallel_client_training(self, selected_clients: List[int],
                                           round_num: int,
                                           previous_model_path: str = None,
                                           available_gpus: int = None
                                           ) -> List[Dict[str, Any]]:
        """Run client training in a smart parallel mode (GPUs are shared; within each GPU, clients run sequentially).

        available_gpus: GPU count already probed by the caller; if None, probe once here.
        """
        if available_gpus is None:
            available_gpus = self.get_available_gpus()

        gpu_allocation = self.smart_gpu_allocation(selected_clients, available_gpus)
        self.logger.info("GPU Allocation Plan:")
        for gpu_id, clients in gpu_allocation.items():
            self.logger.info(f"  GPU {gpu_id}: Clients {clients}")

        client_scripts = {}
        for gpu_id, clients in gpu_allocation.items():
            for client_id in clients:
                client_scripts[client_id] = self.script_builder.create_client_script(
                    client_id, round_num, self._epochs_for_round(round_num),
                    previous_model_path, gpu_id,
                )

        gpus_per_client = self.config.get('verl', {}).get('trainer', {}).get('n_gpus_per_node', 1)
        max_concurrent = available_gpus // gpus_per_client
        max_workers = min(max_concurrent, len(selected_clients))
        self.logger.info(
            f"Launching {max_workers} worker(s) for {len(selected_clients)} clients "
            f"({gpus_per_client} GPU/client, max concurrent: {max_concurrent})"
        )

        client_results: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_gpu = {
                executor.submit(
                    self.run_gpu_sequential_training,
                    gpu_id, clients, client_scripts, round_num, available_gpus,
                ): gpu_id
                for gpu_id, clients in gpu_allocation.items()
            }
            for future in as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                try:
                    gpu_results = future.result()
                    client_results.extend(gpu_results)
                    self.logger.info(f"GPU {gpu_id} completed with {len(gpu_results)} clients")
                except Exception as e:
                    self.logger.error(f"GPU {gpu_id} failed with exception: {str(e)}")
                    for client_id in gpu_allocation[gpu_id]:
                        client_results.append({
                            'success': False, 'client_id': client_id,
                            'round_num': round_num,
                            'error': f"GPU {gpu_id} failed: {str(e)}",
                        })
        return client_results

    def run_gpu_sequential_training(self, gpu_id: int, clients: List[int],
                                    client_scripts: Dict[int, str],
                                    round_num: int,
                                    available_gpus: int) -> List[Dict[str, Any]]:
        """Run several clients sequentially on a specified GPU."""
        gpu_results = []
        base_cuda = self.config['federated']['environment'].get('cuda_device', 0)
        gpus_per_client = self.config.get('verl', {}).get('trainer', {}).get('n_gpus_per_node', 1)

        self.logger.info(
            f"GPU {gpu_id} (base={base_cuda}): sequential training for clients {clients} "
            f"({gpus_per_client} GPU/client, {available_gpus} total)"
        )

        for i, client_id in enumerate(clients):
            self.logger.info(f"GPU {gpu_id}: Starting client {client_id} ({i+1}/{len(clients)})")
            cuda_device = self._resolve_cuda_device(gpu_id, client_id, base_cuda,
                                                    gpus_per_client, available_gpus)

            original = os.environ.get('CUDA_VISIBLE_DEVICES')
            os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
            try:
                result = self.client_runner.run_client_training(
                    client_id, round_num, client_scripts[client_id]
                )
                gpu_results.append(result)
                if result['success']:
                    self.logger.info(f"GPU {gpu_id}: Client {client_id} completed successfully")
                else:
                    self.logger.error(
                        f"GPU {gpu_id}: Client {client_id} failed: {result.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                self.logger.error(f"GPU {gpu_id}: Client {client_id} failed with exception: {str(e)}")
                gpu_results.append({
                    'success': False, 'client_id': client_id,
                    'round_num': round_num, 'error': str(e),
                })
            finally:
                if original is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original
                else:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)

            if i < len(clients) - 1:
                wait = self.config['federated'].get('wait_between_clients', 5)
                if wait > 0:
                    self.logger.info(f"GPU {gpu_id}: Waiting {wait} seconds before next client...")
                    time.sleep(wait)

        self.logger.info(f"GPU {gpu_id}: Completed all {len(clients)} clients")
        return gpu_results

    def _resolve_cuda_device(self, gpu_id: int, client_id: int, base_cuda: int,
                             gpus_per_client: int, available_gpus: int):
        if gpus_per_client > 1:
            if gpus_per_client <= available_gpus:
                devices = [str(base_cuda + j) for j in range(gpus_per_client)]
                return ",".join(devices)
            self.logger.warning(
                f"Client {client_id}: GPU insufficient for parallel mode, "
                f"falling back to single GPU {base_cuda} (may affect performance)"
            )
            return base_cuda
        return base_cuda + gpu_id

    # ------------------------------------------------------------------ completion status
    def check_round_completion_status(self, round_num: int) -> Dict[str, Any]:
        """Check the completion status of a given round.

        Returns:
            {
                'round_completed': bool,
                'server_weights_exist': bool,
                'client_status': Dict[int, Dict],
                'completed_clients': List[int],
                'pending_clients': List[int],
            }
        """
        round_dir = self.output_dir / f"round_{round_num}"
        selected_clients = self.round_clients.get(round_num) or self.select_clients(round_num)

        if not round_dir.exists():
            return {
                'round_completed': False,
                'server_weights_exist': False,
                'client_status': {},
                'completed_clients': [],
                'pending_clients': selected_clients,
            }

        server_weights_exist = self._detect_aggregated_weights(round_dir)

        centralized_resume_epoch = os.environ.get('CENTRALIZED_RESUME_EPOCH', 'false').lower() == 'true'
        eval_only = self._is_eval_only_round(round_num)
        client_status = {}
        completed_clients = []
        pending_clients = []

        for client_id in selected_clients:
            client_dir = round_dir / f"client_{client_id}"
            if eval_only:
                is_completed = self._is_client_eval_logged(client_dir)
            else:
                is_completed = self._is_client_completed(client_dir, client_id, centralized_resume_epoch)
            if is_completed:
                completed_clients.append(client_id)
            else:
                pending_clients.append(client_id)

            client_status[client_id] = {
                'completed': is_completed,
                'client_dir': str(client_dir) if client_dir.exists() else None,
                'checkpoints_exist': (client_dir / "checkpoints").exists() if client_dir.exists() else False,
            }

        if eval_only:
            # Eval-only round: no aggregation, completion = all clients logged step=0.
            round_completed = (
                len(completed_clients) == len(selected_clients)
                and len(selected_clients) > 0
            )
        else:
            round_completed = (
                len(completed_clients) == len(selected_clients)
                and len(selected_clients) > 0
                and server_weights_exist
            )
        return {
            'round_completed': round_completed,
            'server_weights_exist': server_weights_exist,
            'client_status': client_status,
            'completed_clients': completed_clients,
            'pending_clients': pending_clients,
        }

    def _detect_aggregated_weights(self, round_dir: Path) -> bool:
        """Determine whether usable aggregated weights already exist under this round's aggregated/checkpoints/."""
        aggregated_dir = round_dir / "aggregated"
        if not aggregated_dir.exists():
            self.logger.info(f"No aggregated directory found in {round_dir}")
            return False

        ck_dir = aggregated_dir / "checkpoints"
        if not ck_dir.exists():
            self.logger.info(f"No checkpoints directory found in {aggregated_dir}")
            return False

        latest_iter_file = ck_dir / "latest_checkpointed_iteration.txt"
        if not latest_iter_file.exists():
            self.logger.info(f"No latest_checkpointed_iteration.txt found in {ck_dir}")
            return False

        global_step_dirs = [d for d in ck_dir.iterdir()
                            if d.is_dir() and d.name.startswith('global_step_')]
        if global_step_dirs:
            latest_step = max(global_step_dirs, key=lambda x: int(x.name.split('_')[2]))
            actor_dir = latest_step / "actor"
            if actor_dir.exists():
                shard_files = list(actor_dir.glob("model_world_size_*_rank_*.pt"))
                if shard_files:
                    self.logger.info(
                        f"Found aggregated FSDP checkpoint: {len(shard_files)} shard files in {actor_dir}"
                    )
                    return True
                self.logger.info(f"No FSDP shard files found in {actor_dir}")
            else:
                self.logger.info(f"No actor directory found in {latest_step}")
            return False

        # Non-global_step layout
        model_files = []
        for ext in ("*.pt", "*.pth", "*.safetensors", "*.bin"):
            model_files.extend(list(ck_dir.glob(ext)))
        if model_files:
            self.logger.info(f"Found aggregated checkpoint: {len(model_files)} model files")
            return True
        self.logger.info(f"No model files found in {ck_dir}")
        return False

    def _is_client_eval_logged(self, client_dir: Path) -> bool:
        """Eval-only completion check: metrics.json exists and has a step=0 entry.

        Eval-only rounds run total_epochs=0 (only val_before_train), so no checkpoint
        is written — completion has to be detected from the logged metrics instead.
        """
        if not client_dir.exists():
            return False
        metrics_file = client_dir / "json_logs" / "metrics.json"
        if not metrics_file.exists():
            return False
        try:
            import json
            entries = json.loads(metrics_file.read_text())
        except (ValueError, OSError):
            return False
        return any(isinstance(e, dict) and e.get('step') == 0 for e in entries)

    def _is_client_completed(self, client_dir: Path, client_id: int,
                             centralized_resume_epoch: bool) -> bool:
        """Determine whether client_dir already holds a complete training checkpoint."""
        if not client_dir.exists():
            return False
        ck_dir = client_dir / "checkpoints"
        if not ck_dir.exists():
            return False

        global_step_dirs = [d for d in ck_dir.iterdir()
                            if d.is_dir() and d.name.startswith('global_step_')]
        if global_step_dirs:
            latest_step = max(global_step_dirs, key=lambda x: int(x.name.split('_')[2]))
            actor_dir = latest_step / "actor"
            if actor_dir.exists():
                model_files = []
                for ext in ("*.pt", "*.pth", "*.safetensors", "*.bin"):
                    model_files.extend(list(actor_dir.glob(ext)))
                if model_files:
                    if centralized_resume_epoch:
                        self.logger.info(
                            f"Centralized resume epoch mode: client {client_id} has "
                            "checkpoints but will continue training"
                        )
                        return False
                    return True
            return False

        # Non-global_step layout — check direct model files in checkpoints/
        model_files = []
        for ext in ("*.pth", "*.pt", "*.safetensors", "*.bin"):
            model_files.extend(list(ck_dir.glob(ext)))
        if model_files:
            if centralized_resume_epoch:
                self.logger.info(
                    f"Centralized resume epoch mode: client {client_id} has "
                    "checkpoints but will continue training"
                )
                return False
            return True
        return False
