# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os

import hydra
import ray

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


import debugpy
import os

# Assume you use the RANK environment variable to identify each process
rank = int(os.getenv('RANK', '0'))
port = 5679 + rank  # base port + process ID

# debugpy.listen(('127.0.0.1',port))
# print(f"Process {rank} waiting for debugger to attach on port {port}...")
# debugpy.wait_for_client()

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "true", 
                "NCCL_DEBUG": "WARN", 
                "VLLM_LOGGING_LEVEL": "WARN", 
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                # "RAY_DEBUG":"1"
            }},
            num_cpus=config.ray_init.num_cpus,
            # num_cpus=30,
            address="local"
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        from agent_system.environments import make_envs
        envs, val_envs = make_envs(config)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == 'episode':
            from agent_system.reward_manager.episode import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"

        from agent_system.multi_turn_rollout import TrajectoryCollector
        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        
        # If in inference mode (total_epochs=0), force-save the results
        if config.trainer.total_epochs == 0:
            print("Inference mode: skip training, only run validation and save results...")
            # Force the save parameters
            trainer.config.trainer.save_freq = 1
            trainer.config.trainer.test_freq = 1
            trainer.config.trainer.log_val_generations = 10

            # Run validation - use the fit method but only run validation
            print("Starting validation inference...")
            # Call fit directly; when total_epochs=0 it skips the training loop
            trainer.fit()

            # Manually save the validation results
            print("Saving inference results...")
            import os
            import json
            import pandas as pd
            import numpy as np
            import torch
            
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            # Create the results directory
            results_dir = config.trainer.validation_data_dir
            os.makedirs(results_dir, exist_ok=True)

            # Run the actual inference and save the results
            val_results = []
            if hasattr(trainer, 'val_dataloader') and trainer.val_dataloader is not None:
                print("Running inference with the validation dataloader...")

                # Import the necessary modules
                from verl import DataProto

                for batch_idx, test_data in enumerate(trainer.val_dataloader):
                    print(f"Processing validation batch {batch_idx + 1}...")

                    try:
                        # Convert to DataProto format
                        test_batch = DataProto.from_single_dict(test_data)

                        # Repeat the test batch
                        test_batch = test_batch.repeat(repeat_times=trainer.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

                        # Store the original inputs
                        input_ids = test_batch.batch["input_ids"]
                        input_texts = [trainer.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

                        # Get the original user question
                        raw_prompts = test_batch.non_tensor_batch.get('raw_prompt', input_texts)
                        if isinstance(raw_prompts, np.ndarray):
                            raw_prompts = raw_prompts.tolist()

                        # Prepare the generation batch
                        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                        non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                        if "multi_modal_data" in test_batch.non_tensor_batch:
                            non_tensor_batch_keys_to_pop.append("multi_modal_data")
                        if "raw_prompt" in test_batch.non_tensor_batch:
                            non_tensor_batch_keys_to_pop.append("raw_prompt")
                        if "tools_kwargs" in test_batch.non_tensor_batch:
                            non_tensor_batch_keys_to_pop.append("tools_kwargs")
                        
                        test_gen_batch = test_batch.pop(
                            batch_keys=batch_keys_to_pop,
                            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                        )
                        
                        test_gen_batch.meta_info = {
                            "eos_token_id": trainer.tokenizer.eos_token_id,
                            "pad_token_id": trainer.tokenizer.pad_token_id,
                            "recompute_log_prob": False,
                            "do_sample": trainer.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                            "validate": True,
                        }
                        
                        # Run the agent-environment loop for inference
                        test_output_gen_batch = trainer.traj_collector.multi_turn_loop(
                            gen_batch=test_gen_batch,
                            actor_rollout_wg=trainer.actor_rollout_wg,
                            envs=trainer.val_envs,
                            is_train=False,
                        )
                        
                        # Get the generated outputs
                        output_ids = test_output_gen_batch.batch["responses"]
                        output_texts = [trainer.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

                        # Get the original answers and other information
                        original_answers = test_batch.non_tensor_batch.get('answer', [''] * len(output_texts))
                        data_sources = test_batch.non_tensor_batch.get('data_source', ['text'] * len(output_texts))
                        abilities = test_batch.non_tensor_batch.get('ability', ['agent'] * len(output_texts))
                        extra_infos = test_batch.non_tensor_batch.get('extra_info', [{}] * len(output_texts))

                        # Save the results
                        for i in range(len(input_texts)):
                            # Get the original user question
                            if isinstance(raw_prompts, list) and i < len(raw_prompts):
                                raw_user_question = raw_prompts[i]
                            else:
                                raw_user_question = input_texts[i]

                            val_results.append({
                                'prompt': raw_user_question,  # original user question
                                'conversation_history': input_texts[i],  # full conversation history
                                'answer': original_answers[i] if i < len(original_answers) else '',
                                'data_source': data_sources[i] if i < len(data_sources) else 'text',
                                'ability': abilities[i] if i < len(abilities) else 'agent',
                                'extra_info': extra_infos[i] if i < len(extra_infos) else {},
                                'responses': [output_texts[i]]  # full multi-turn response generated by the model
                            })

                        print(f"Batch {batch_idx + 1} processed, generated {len(output_texts)} inference results")

                    except Exception as e:
                        print(f"Error while processing batch {batch_idx + 1}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                    # Limit the number processed
                    if len(val_results) >= 32:
                        break
            else:
                print("No validation dataloader")

            # Save as JSON
            json_file = os.path.join(results_dir, 'inference_results.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(convert_numpy(val_results), f, ensure_ascii=False, indent=2)
            print(f"Inference results saved to: {json_file}")
            print(f"Processed {len(val_results)} records in total")
        else:
            trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
