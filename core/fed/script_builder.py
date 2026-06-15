"""Bash script generation for verl-agent client training.

Extracted from FederatedServer. Takes the base training script from
verl-agent and rewrites it with federated-specific env vars, config overrides,
resume paths, and data-partition settings for each (client, round).
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional

from core.fed.config_helpers import get_shuffle_seed


class ScriptBuilder:
    def __init__(self, config: Dict, output_dir: Path, dataset_name: str,
                 total_clients: int, base_script_path: Path,
                 aggregated_models: Dict, checkpoint_manager, logger):
        self.config = config
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.total_clients = total_clients
        self.base_script_path = Path(base_script_path)
        self.aggregated_models = aggregated_models  # shared reference
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger

    # ------------------------------------------------------------------
    def create_client_script(self, client_id: int, round_num: int,
                             epochs: int, model_path: str = None,
                             gpu_id: int = 0) -> str:
        """Create the training script for a single client."""
        client_dir = self.output_dir / f"round_{round_num}" / f"client_{client_id}"
        client_dir.mkdir(parents=True, exist_ok=True)

        centralized_resume_epoch = os.environ.get('CENTRALIZED_RESUME_EPOCH', 'false').lower() == 'true'
        json_logs_dir = client_dir / "json_logs"

        if centralized_resume_epoch:
            json_logs_dir.mkdir(exist_ok=True)
            self.logger.info(
                f"Centralized resume epoch mode: preserving existing json_logs "
                f"for client {client_id} in round {round_num}"
            )
            checkpoints_dir = client_dir / "checkpoints"
            if checkpoints_dir.exists() and model_path is None:
                latest_checkpoint = self.checkpoint_manager._find_latest_checkpoint(checkpoints_dir)
                if latest_checkpoint:
                    model_path = str(latest_checkpoint)
                    self.logger.info(
                        f"Centralized resume epoch mode: found latest checkpoint "
                        f"for client {client_id} in round {round_num}: {model_path}"
                    )
                    self.checkpoint_manager._cleanup_old_global_step_checkpoints(
                        checkpoints_dir, latest_checkpoint
                    )
                else:
                    self.logger.info(
                        f"Centralized resume epoch mode: no checkpoint found for "
                        f"client {client_id} in round {round_num}, will start from scratch"
                    )
        else:
            if json_logs_dir.exists():
                shutil.rmtree(json_logs_dir)
                self.logger.info(
                    f"Cleaned existing json_logs directory for client {client_id} in round {round_num}"
                )
            json_logs_dir.mkdir(exist_ok=True)

        script_path = client_dir / "run_client.sh"
        with open(self.base_script_path, 'r') as f:
            base_script = f.read()

        modified_script = self.modify_script_for_federated(
            base_script, client_id, round_num, epochs, model_path, gpu_id
        )

        with open(script_path, 'w') as f:
            f.write(modified_script)
        os.chmod(script_path, 0o755)
        return str(script_path)

    # ------------------------------------------------------------------
    def _project_root_or_raise(self) -> str:
        """Read project_root from config/paths.yaml. Raise if missing.

        Used to resolve user-supplied relative paths in yaml (e.g. holdout_file)
        to absolute paths before exporting to env vars; subprocess `cd verl_agent_repo`
        otherwise loses the cwd anchor.
        """
        from omegaconf import OmegaConf
        paths_yaml = "./config/paths.yaml"
        if not os.path.exists(paths_yaml):
            raise FileNotFoundError(
                f"[script_builder] config/paths.yaml not found from cwd={os.getcwd()}; "
                f"cannot resolve relative paths in yaml partition.kwargs"
            )
        path_cfg = OmegaConf.to_container(OmegaConf.load(paths_yaml), resolve=True)
        if 'project_root' not in path_cfg:
            raise KeyError("[script_builder] paths.yaml missing 'project_root' field")
        return path_cfg['project_root']

    # ------------------------------------------------------------------
    def _get_partition_strategy_env_vars(self) -> str:
        """Return the env-var exports that correspond to the configured data-sharding strategy."""
        strategy = self.config['federated']['data_sharding']['partition']['strategy']
        kwargs = self.config['federated']['data_sharding']['partition'].get('kwargs', {})
        if strategy == 'preference':
            # New: Dirichlet PreferencePartition uses `omega`; legacy yamls used `tau`.
            # Export both env vars so downstream (verl) can read whichever; partition_strategy.py
            # treats them as aliases (omega takes precedence when both present).
            val = kwargs.get('omega', kwargs.get('tau'))
            if val is None:
                raise KeyError("[script_builder] preference strategy needs kwargs.omega (or legacy kwargs.tau)")
            return f'export OMEGA="{val}"\nexport TAU="{val}"'
        if strategy == 'coverage':
            return f'export SIZE_STD="{kwargs["size_std"]}"'
        if strategy == 'hardness':
            return f'export SUCCESS_STD="{kwargs["success_std"]}"'
        if strategy == 'uniform_single':
            return f'export CL_ID="{kwargs["cl_id"]}"'
        if strategy in ('distractor_disjoint', 'catalog_split'):
            # Env-level heterogeneity for WebShop
            #   distractor_disjoint    -> v4 algo (full-target floor, all clients share goals[500:])
            #   catalog_split -> v5 algo (per-client target floor, uniform 100/cl)
            #     v5 also reads MIN_GOALS_PER_CLIENT (we still bridge the value
            #     even when yaml's federated.data_sharding.min_goals_per_client is set,
            #     because main_ppo_fed's hydra config doesn't include the federated block).
            # See docs/heterogeneity.md
            lines = [
                f'export ENV_DIV="{kwargs.get("env_div", 0.7)}"',
                f'export KEEP_RATIO="{kwargs.get("keep_ratio", 0.7)}"',
            ]
            if strategy == 'catalog_split':
                min_goals = (
                    self.config.get('federated', {})
                    .get('data_sharding', {})
                    .get('min_goals_per_client', 100)
                )
                lines.append(f'export MIN_GOALS_PER_CLIENT="{min_goals}"')
            holdout_file = kwargs.get('holdout_file')
            if holdout_file:
                # Resolve relative path → absolute (subprocess will `cd verl-agent_repo`,
                # relative paths get lost; project_root anchor keeps it stable)
                if not os.path.isabs(holdout_file):
                    project_root = self._project_root_or_raise()
                    holdout_file_abs = os.path.join(project_root, holdout_file)
                else:
                    holdout_file_abs = holdout_file
                # Fail-loud: validate holdout file exists at build time, not at client launch
                if not os.path.exists(holdout_file_abs):
                    raise FileNotFoundError(
                        f"[script_builder] holdout_file not found: {holdout_file_abs}\n"
                        f"  yaml value: {holdout_file}\n"
                        f"  resolved   : {holdout_file_abs}\n"
                        f"  Run `python tools/env_heterogeneity/gen_holdout_webshop.py` to generate."
                    )
                lines.append(f'export HOLDOUT_FILE="{holdout_file_abs}"')
            if kwargs.get('search_return_n'):
                lines.append(f'export WEBSHOP_SEARCH_RETURN_N="{kwargs["search_return_n"]}"')
            return '\n'.join(lines)
        if strategy == 'bm25_variant':
            # Transition-level env heterogeneity (BM25 Reweighting / Field-Subset Index).
            # See docs/heterogeneity.md (Task-level heterogeneity).
            # `variant_pool` yaml kwarg picks between:
            #   - default (omit / 'default'): BM25 Reweighting = extreme k1/b on full fields
            #   - 'fields_only':              Field-Subset Index = field-subset variants only
            lines = [f'export N_VARIANTS="{kwargs.get("N", 4)}"']
            pool = kwargs.get('variant_pool')
            if pool:
                lines.append(f'export BM25_VARIANT_POOL="{pool}"')
            if kwargs.get('search_return_n'):
                lines.append(f'export WEBSHOP_SEARCH_RETURN_N="{kwargs["search_return_n"]}"')
            return '\n'.join(lines)
        if strategy == 'rank_wrapper':
            # Transition-level env heterogeneity (search-engine TYPE swap).
            lines = [f'export N_VARIANTS="{kwargs.get("N", 4)}"']
            if kwargs.get('search_return_n'):
                lines.append(f'export WEBSHOP_SEARCH_RETURN_N="{kwargs["search_return_n"]}"')
            return '\n'.join(lines)
        if strategy == 'lookalike_injection':
            # Transition-level env heterogeneity (Lookalike Injection).
            # See docs/heterogeneity.md
            lines = [
                f'export N_VARIANTS="{kwargs.get("N", 2)}"',
                # PROJECT_ROOT lets partition_strategy resolve relative lookalike file paths.
                f'export PROJECT_ROOT="{self._project_root_or_raise()}"',
            ]
            if kwargs.get('search_return_n'):
                lines.append(f'export WEBSHOP_SEARCH_RETURN_N="{kwargs["search_return_n"]}"')
            return '\n'.join(lines)
        if strategy == 'env_disjoint':
            # Env-level heterogeneity for AlfWorld (scene-disjoint partition).
            # See docs/heterogeneity.md
            #
            # main_ppo_fed runs hydra with `verl/trainer/config/ppo_trainer.yaml`,
            # which doesn't include the `federated` block — so fed_env_manager
            # cannot read partition.kwargs from `config.federated.*` directly.
            # All kwargs flow via env vars (mirrors the WebShop pattern).
            lines = [
                f'export ENV_DIV="{kwargs.get("env_div", 0.7)}"',
                f'export FALLBACK="{kwargs.get("fallback", "skip")}"',
            ]
            holdout_file = kwargs.get('holdout_file')
            if holdout_file:
                if not os.path.isabs(holdout_file):
                    project_root = self._project_root_or_raise()
                    holdout_file_abs = os.path.join(project_root, holdout_file)
                else:
                    holdout_file_abs = holdout_file
                if not os.path.exists(holdout_file_abs):
                    raise FileNotFoundError(
                        f"[script_builder] holdout_file not found: {holdout_file_abs}\n"
                        f"  yaml value: {holdout_file}\n"
                        f"  resolved   : {holdout_file_abs}\n"
                        f"  Run `python tools/env_heterogeneity/gen_holdout_alfworld.py` to generate."
                    )
                lines.append(f'export HOLDOUT_FILE="{holdout_file_abs}"')
            return '\n'.join(lines)
        return ""

    # ------------------------------------------------------------------
    def modify_script_for_federated(self, base_script: str, client_id: int,
                                    round_num: int, epochs: int,
                                    model_path: str = None, gpu_id: int = 0) -> str:
        """Rewrite the base training script so it runs as a federated client."""
        model_config_path = (self.config.get('verl', {}).get('actor_rollout_ref', {})
                             .get('model', {}).get('path', 'Qwen/Qwen2.5-1.5B-Instruct'))
        gpus_per_client = self.config.get('verl', {}).get('trainer', {}).get('n_gpus_per_node', 1)
        base_cuda_device = self.config['federated']['environment'].get('cuda_device', 0)

        if gpus_per_client > 1:
            actual_cuda_device = f"{base_cuda_device + gpu_id}"
            for i in range(1, gpus_per_client):
                actual_cuda_device += f",{base_cuda_device + gpu_id + i}"
            self.logger.info(f"Client {client_id}: Using {gpus_per_client} GPUs: {actual_cuda_device}")
        else:
            actual_cuda_device = base_cuda_device + gpu_id
            self.logger.info(f"Client {client_id}: Using single GPU: {actual_cuda_device}")

        partition_strategy = self.config['federated']['data_sharding']['partition']['strategy']
        actual_strategy = "uniform" if partition_strategy == "uniform_single" else partition_strategy

        shuffle_seed = get_shuffle_seed(self.config, logger=self.logger)
        shuffle_seed_env = f"export SHUFFLE_SEED={shuffle_seed}\n" if shuffle_seed is not None else ""

        env_vars = f"""#!/bin/bash
set -x

# Federated client configuration
export CLIENT_ID={client_id}
export ROUND_NUM={round_num}
export FEDERATED_EPOCHS={epochs}
export FEDERATED_OUTPUT_DIR={self.output_dir}
export CUDA_VISIBLE_DEVICES={actual_cuda_device}

# Data-sharding strategy configuration
export PARTITION_STRATEGY="{actual_strategy}"
{self._get_partition_strategy_env_vars()}

# shuffle_seed configuration
{shuffle_seed_env}
"""
        if model_path:
            env_vars += f"export INITIAL_MODEL_PATH={model_path}\n"

        modified_script = base_script.replace(
            "python3 -m verl.trainer.main_ppo",
            "python3 -m verl.trainer.main_ppo_fed",
        )

        modified_script = re.sub(
            r'actor_rollout_ref\.model\.path=[^\s\\]+',
            f'actor_rollout_ref.model.path={model_config_path}',
            modified_script,
        )
        self.logger.info(f"Client {client_id}: Using model path from config: {model_config_path}")

        modified_script = re.sub(
            r'trainer\.total_epochs=\d+',
            f'trainer.total_epochs={epochs}',
            modified_script,
        )

        modified_script = self._modify_ppo_config(modified_script)
        modified_script = self._modify_vllm_config(modified_script)
        modified_script = self._modify_data_config(modified_script)
        modified_script = self._modify_ref_config(modified_script)
        modified_script = self._modify_trainer_config(modified_script, model_path)

        modified_script = modified_script.replace(
            "trainer.experiment_name='grpo_qwen2.5_1.5b'",
            f"trainer.experiment_name='federated_client_{client_id}_round_{round_num}'",
        )
        modified_script = modified_script.replace(
            "trainer.experiment_name='ppo_qwen2.5_1.5b'",
            f"trainer.experiment_name='federated_client_{client_id}_round_{round_num}'",
        )

        # save_freq — centralized resume epoch uses env, normal mode saves at last epoch
        centralized_resume_epoch = os.environ.get('CENTRALIZED_RESUME_EPOCH', 'false').lower() == 'true'
        epoch_save_freq = int(os.environ.get('EPOCH_SAVE_FREQ', '10'))
        if centralized_resume_epoch:
            modified_script = re.sub(
                r'trainer\.save_freq=-?\d+',
                f'trainer.save_freq={epoch_save_freq}',
                modified_script,
            )
            self.logger.info(f"Centralized resume epoch mode: setting save_freq to {epoch_save_freq}")
        else:
            # epochs=0 → eval-only round (val_before_train only). save_freq=0 is
            # nonsensical (and verl treats nonpositive as "never"); use -1 explicitly.
            save_freq_value = epochs if epochs > 0 else -1
            modified_script = re.sub(
                r'trainer\.save_freq=-?\d+',
                f'trainer.save_freq={save_freq_value}',
                modified_script,
            )
            if epochs == 0:
                # Force val_before_train=True so the round still produces the step=0
                # eval metric we want, regardless of yaml override or upstream edits.
                modified_script = re.sub(
                    r'trainer\.val_before_train=(True|False)',
                    'trainer.val_before_train=True',
                    modified_script,
                )
                self.logger.info(
                    f"Client {client_id} round {round_num}: eval-only (epochs=0), "
                    "forcing val_before_train=True, save_freq=-1"
                )

        train_batch_size = self.config['data_preprocess']['train_data_size']
        val_batch_size = self.config['data_preprocess']['val_data_size']

        modified_script = re.sub(r'train_data_size=\d+',
                                 f'train_data_size={train_batch_size}', modified_script)
        modified_script = re.sub(r'val_data_size=\d+',
                                 f'val_data_size={val_batch_size}', modified_script)
        modified_script = re.sub(r'group_size=\d+',
                                 f'group_size={train_batch_size}', modified_script)

        # Data file path substitutions (both ${project_root} and $project_root forms)
        for src, dst in [
            ("data.train_files=${project_root}/data/verl-agent/text/train.parquet",
             f"data.train_files=${{project_root}}/data/{self.dataset_name}/text/train.parquet"),
            ("data.train_files=$project_root/data/verl-agent/text/train.parquet",
             f"data.train_files=$project_root/data/{self.dataset_name}/text/train.parquet"),
            ("data.val_files=${project_root}/data/verl-agent/text/test.parquet",
             f"data.val_files=${{project_root}}/data/{self.dataset_name}/text/test.parquet"),
            ("data.val_files=$project_root/data/verl-agent/text/test.parquet",
             f"data.val_files=$project_root/data/{self.dataset_name}/text/test.parquet"),
            ("--local_dir ${project_root}/data/verl-agent",
             f"--local_dir ${{project_root}}/data/{self.dataset_name}"),
            ("--local_dir $project_root/data/verl-agent",
             f"--local_dir $project_root/data/{self.dataset_name}"),
            # Skip-check guard in GRPO base script — must point to the same per-task-algo path
            # as the prepare output, else prepare is wrongly skipped (stale parquet → empty DataLoader).
            ("[ ! -f $project_root/data/verl-agent/text/train.parquet ]",
             f"[ ! -f $project_root/data/{self.dataset_name}/text/train.parquet ]"),
            ("[ ! -f ${project_root}/data/verl-agent/text/train.parquet ]",
             f"[ ! -f ${{project_root}}/data/{self.dataset_name}/text/train.parquet ]"),
        ]:
            modified_script = modified_script.replace(src, dst)

        federated_save_config = (
            f"    +trainer.save_dir='{self.output_dir}/round_{round_num}"
            f"/client_{client_id}/checkpoints' \\"
        )

        min_goals_per_client = self.config['federated']['data_sharding'].get('min_goals_per_client', 100)
        data_shard_config = (
            f"    +data.client_id={client_id} \\\n"
            f"    +data.client_num={self.total_clients} \\\n"
            f"    +data.round_num={round_num} \\\n"
            f"    +data.min_goals_per_client={min_goals_per_client} \\"
        )

        partition_config = self._build_partition_config(partition_strategy)

        has_critic = 'critic' in self.config.get('verl', {})
        checkpoint_config = self._build_checkpoint_config(client_id, has_critic, centralized_resume_epoch)

        dataloader_config = "    +data.num_workers=0 \\"
        local_dir_config = f"    +data_preprocess.local_dir='data/{self.dataset_name}' \\"

        env_vars += (
            f"export JSON_LOG_DIR='{self.output_dir}/round_{round_num}"
            f"/client_{client_id}/json_logs'\n"
        )
        json_logger_config = "    trainer.logger=['console','json'] \\"

        resume_config = self._build_resume_config(client_id, round_num, model_path, has_critic)

        # Splice the config into the training command line
        lines = modified_script.split('\n')
        for i, line in enumerate(lines):
            if 'python3 -m verl.trainer.main_ppo_fed' in line:
                config_parts = [data_shard_config, partition_config, federated_save_config,
                                checkpoint_config, dataloader_config, local_dir_config,
                                json_logger_config]
                if resume_config:
                    config_parts.append(resume_config)
                lines[i] = line + '\n' + '\n'.join(config_parts)
                break

        modified_script = env_vars + '\n'.join(lines)
        modified_script = modified_script.replace(
            "trainer.logger=['console']",
            "trainer.logger=['console','json']",
        )
        return modified_script

    # ------------------------------------------------------------------
    def _build_partition_config(self, strategy: str) -> str:
        kwargs = self.config['federated']['data_sharding']['partition'].get('kwargs', {})
        if strategy == 'preference':
            # Dirichlet uses omega; legacy used tau. Pass via +data.tau (alias).
            val = kwargs.get('omega', kwargs.get('tau'))
            if val is None:
                raise KeyError("[script_builder] preference strategy needs kwargs.omega (or legacy kwargs.tau)")
            return (f"    +data.partition_strategy={strategy} \\\n"
                    f"    +data.tau={val} \\")
        if strategy == 'coverage':
            return (f"    +data.partition_strategy={strategy} \\\n"
                    f"    +data.size_std={kwargs['size_std']} \\")
        if strategy == 'hardness':
            return (f"    +data.partition_strategy={strategy} \\\n"
                    f"    +data.success_std={kwargs['success_std']} \\")
        if strategy == 'uniform_single':
            # uniform_single: fix client selection but use uniform partitioning
            return (f"    +data.partition_strategy=uniform \\\n"
                    f"    +data.cl_id={kwargs['cl_id']} \\")
        return f"    +data.partition_strategy={strategy} \\"

    def _build_checkpoint_config(self, client_id: int, has_critic: bool,
                                 centralized_resume_epoch: bool) -> str:
        if centralized_resume_epoch:
            contents = "['model','optimizer','extra']"
            if has_critic:
                self.logger.info(
                    f"Client {client_id}: Centralized resume epoch mode - "
                    "configuring full checkpoint for both actor and critic models"
                )
                return (f"    actor_rollout_ref.actor.checkpoint.contents={contents} \\\n"
                        f"    critic.checkpoint.contents={contents} \\")
            self.logger.info(
                f"Client {client_id}: Centralized resume epoch mode - "
                "configuring full checkpoint for actor model only"
            )
            return f"    actor_rollout_ref.actor.checkpoint.contents={contents} \\"

        if has_critic:
            self.logger.info(
                f"Client {client_id}: Configuring checkpoint for both actor and critic models"
            )
            return ("    actor_rollout_ref.actor.checkpoint.contents=[model] \\\n"
                    "    critic.checkpoint.contents=[model] \\")
        self.logger.info(f"Client {client_id}: Configuring checkpoint for actor model only")
        return "    actor_rollout_ref.actor.checkpoint.contents=[model] \\"

    def _build_resume_config(self, client_id: int, round_num: int,
                             model_path: Optional[str], has_critic: bool) -> str:
        if not model_path:
            self.logger.warning(f"No model_path provided for client {client_id} in round {round_num}")
            return ""

        # If model_path points to a specific sharded file, walk up to the checkpoint dir.
        if 'global_step_' in str(model_path) and 'model_world_size_' in str(model_path):
            checkpoint_dir = Path(model_path).parent.parent
            model_path = str(checkpoint_dir)
            self.logger.info(f"Converted model file path to checkpoint directory: {model_path}")

        base = (f"    +trainer.resume_from_path='{model_path}' \\\n"
                "    ++trainer.resume_mode=resume_path \\")

        if not has_critic:
            self.logger.info(f"Setting resume_from_path for client {client_id}: {model_path}")
            return base

        critic_model_path = None
        round_match = re.search(r'round_(\d+)', str(model_path))
        if round_match:
            critic_model_path = self.aggregated_models.get(f"{int(round_match.group(1))}_critic")

        if not critic_model_path:
            critic_model_path = self.checkpoint_manager._find_critic_model_path(model_path)

        if critic_model_path:
            self.logger.info(
                f"Setting resume_from_path for client {client_id}: "
                f"actor={model_path}, critic={critic_model_path}"
            )
            return base + f"\n    +critic.resume_from_path='{critic_model_path}' \\"

        self.logger.info(
            f"Setting resume_from_path for client {client_id}: "
            f"actor={model_path} (no critic model found)"
        )
        return base

    # ------------------------------------------------------------------
    def _modify_ppo_config(self, script: str) -> str:
        actor_config = self.config.get('verl', {}).get('actor_rollout_ref', {}).get('actor', {})
        script = _sub(script, r'actor_rollout_ref\.actor\.ppo_mini_batch_size=\d+',
                      f'actor_rollout_ref.actor.ppo_mini_batch_size={actor_config.get("ppo_mini_batch_size", 16)}')
        script = _sub(script, r'actor_rollout_ref\.actor\.ppo_micro_batch_size_per_gpu=\d+',
                      f'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={actor_config.get("ppo_micro_batch_size_per_gpu", 8)}')
        script = _sub(script, r'actor_rollout_ref\.actor\.use_kl_loss=(True|False)',
                      f'actor_rollout_ref.actor.use_kl_loss={actor_config.get("use_kl_loss", True)}')
        script = _sub(script, r'actor_rollout_ref\.actor\.kl_loss_coef=[\d.]+',
                      f'actor_rollout_ref.actor.kl_loss_coef={actor_config.get("kl_loss_coef", 0.01)}')
        script = _sub(script, r'actor_rollout_ref\.actor\.kl_loss_type=\w+',
                      f'actor_rollout_ref.actor.kl_loss_type={actor_config.get("kl_loss_type", "low_var_kl")}')
        script = _sub(script, r'actor_rollout_ref\.actor\.use_invalid_action_penalty=(True|False)',
                      f'actor_rollout_ref.actor.use_invalid_action_penalty={actor_config.get("use_invalid_action_penalty", True)}')
        script = _sub(script, r'actor_rollout_ref\.actor\.invalid_action_penalty_coef=[\d.]+',
                      f'actor_rollout_ref.actor.invalid_action_penalty_coef={actor_config.get("invalid_action_penalty_coef", 0.1)}')
        fsdp = actor_config.get('fsdp_config', {})
        script = _sub(script, r'actor_rollout_ref\.actor\.fsdp_config\.param_offload=(True|False)',
                      f'actor_rollout_ref.actor.fsdp_config.param_offload={fsdp.get("param_offload", False)}')
        script = _sub(script, r'actor_rollout_ref\.actor\.fsdp_config\.optimizer_offload=(True|False)',
                      f'actor_rollout_ref.actor.fsdp_config.optimizer_offload={fsdp.get("optimizer_offload", False)}')
        return script

    def _modify_vllm_config(self, script: str) -> str:
        rollout = self.config.get('verl', {}).get('actor_rollout_ref', {}).get('rollout', {})
        script = _sub(script, r'actor_rollout_ref\.rollout\.tensor_model_parallel_size=\d+',
                      f'actor_rollout_ref.rollout.tensor_model_parallel_size={rollout.get("tensor_model_parallel_size", 1)}')
        script = _sub(script, r'actor_rollout_ref\.rollout\.gpu_memory_utilization=[\d.]+',
                      f'actor_rollout_ref.rollout.gpu_memory_utilization={rollout.get("gpu_memory_utilization", 0.6)}')
        script = _sub(script, r'actor_rollout_ref\.rollout\.log_prob_micro_batch_size_per_gpu=\d+',
                      f'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={rollout.get("log_prob_micro_batch_size_per_gpu", 16)}')
        script = _sub(script, r'actor_rollout_ref\.rollout\.enable_chunked_prefill=(True|False)',
                      f'actor_rollout_ref.rollout.enable_chunked_prefill={rollout.get("enable_chunked_prefill", False)}')
        script = _sub(script, r'actor_rollout_ref\.rollout\.enforce_eager=(True|False)',
                      f'actor_rollout_ref.rollout.enforce_eager={rollout.get("enforce_eager", False)}')
        script = _sub(script, r'actor_rollout_ref\.rollout\.free_cache_engine=(True|False)',
                      f'actor_rollout_ref.rollout.free_cache_engine={rollout.get("free_cache_engine", False)}')
        script = _sub(script, r'actor_rollout_ref\.rollout\.prompt_length=\d+',
                      f'actor_rollout_ref.rollout.prompt_length={rollout.get("prompt_length", 4096)}')
        script = _sub(script, r'actor_rollout_ref\.rollout\.max_model_len=\d+',
                      f'actor_rollout_ref.rollout.max_model_len={rollout.get("max_model_len", 4096)}')
        script = _sub(script, r'actor_rollout_ref\.rollout\.response_length=\d+',
                      f'actor_rollout_ref.rollout.response_length={rollout.get("response_length", 512)}')
        engine_kwargs = rollout.get('engine_kwargs', {}) or {}
        vllm_kwargs = engine_kwargs.get('vllm', {}) or {}
        kv_cache_dtype = vllm_kwargs.get('kv_cache_dtype', 'auto')
        script = _sub(script, r'\+?actor_rollout_ref\.rollout\.engine_kwargs\.vllm\.kv_cache_dtype=\w+',
                      f'+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype={kv_cache_dtype}')
        val_kwargs = rollout.get('val_kwargs', {})
        script = _sub(script, r'actor_rollout_ref\.rollout\.val_kwargs\.temperature=[\d.]+',
                      f'actor_rollout_ref.rollout.val_kwargs.temperature={val_kwargs.get("temperature", 0.4)}')
        script = _sub(script, r'actor_rollout_ref\.rollout\.val_kwargs\.do_sample=(True|False)',
                      f'actor_rollout_ref.rollout.val_kwargs.do_sample={val_kwargs.get("do_sample", True)}')
        return script

    def _modify_data_config(self, script: str) -> str:
        data = self.config.get('verl', {}).get('data', {})
        script = _sub(script, r'data\.max_prompt_length=\d+',
                      f'data.max_prompt_length={data.get("max_prompt_length", 4096)}')
        script = _sub(script, r'data\.max_response_length=\d+',
                      f'data.max_response_length={data.get("max_response_length", 512)}')
        return script

    def _modify_ref_config(self, script: str) -> str:
        ref = self.config.get('verl', {}).get('actor_rollout_ref', {}).get('ref', {})
        script = _sub(script, r'actor_rollout_ref\.ref\.log_prob_micro_batch_size_per_gpu=\d+',
                      f'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={ref.get("log_prob_micro_batch_size_per_gpu", 16)}')
        ref_fsdp = ref.get('fsdp_config', {})
        script = _sub(script, r'actor_rollout_ref\.ref\.fsdp_config\.param_offload=(True|False)',
                      f'actor_rollout_ref.ref.fsdp_config.param_offload={ref_fsdp.get("param_offload", True)}')
        return script

    def _modify_trainer_config(self, script: str, model_path: str = None) -> str:
        trainer = self.config.get('verl', {}).get('trainer', {})
        script = _sub(script, r'trainer\.test_freq=\d+',
                      f'trainer.test_freq={trainer.get("test_freq", 3)}')
        script = _sub(script, r'trainer\.save_freq=-?\d+',
                      f'trainer.save_freq={trainer.get("save_freq", -1)}')
        script = _sub(script, r'trainer\.critic_warmup=\d+',
                      f'trainer.critic_warmup={trainer.get("critic_warmup", 0)}')
        script = _sub(script, r'trainer\.n_gpus_per_node=\d+',
                      f'trainer.n_gpus_per_node={trainer.get("n_gpus_per_node", 1)}')
        script = _sub(script, r'trainer\.nnodes=\d+',
                      f'trainer.nnodes={trainer.get("nnodes", 1)}')

        centralized_resume_epoch = os.environ.get('CENTRALIZED_RESUME_EPOCH', 'false').lower() == 'true'
        if centralized_resume_epoch and model_path:
            val_before_train = False
            self.logger.info("Centralized resume epoch mode with checkpoint: setting val_before_train=False")
        else:
            val_before_train = trainer.get('val_before_train', True)
        script = _sub(script, r'trainer\.val_before_train=(True|False)',
                      f'trainer.val_before_train={val_before_train}')
        return script


def _sub(script: str, pattern: str, replacement: str) -> str:
    return re.sub(pattern, replacement, script)
