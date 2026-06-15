"""
Federated Ray PPO Trainer - subclasses verl-agent's RayPPOTrainer.

A thin subclass that inherits RayPPOTrainer and overrides only the methods
needed to support federated learning (per-client parameter extraction,
post-aggregation parameter updates, and federated checkpoint I/O).
"""

import torch
import numpy as np
import time
import os
import pickle
from typing import Dict, Any, List, Optional

# Inherit verl-agent's RayPPOTrainer.
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class FedRayPPOTrainer(RayPPOTrainer):
    """
    Federated-learning variant of RayPPOTrainer.

    Keeps all of the parent's behavior and adds the parameter-extraction and
    parameter-update hooks that federated training requires.
    """

    def __init__(self, client_id=0, round=None, **kwargs):
        """
        Initialize the federated PPO trainer.

        Args:
            client_id: The federated client ID.
            **kwargs: Arguments forwarded to the parent RayPPOTrainer.
        """
        super().__init__(**kwargs)

        # Federated-learning bookkeeping attributes.
        self.client_id = client_id
        self.round_num = round
        self.fed_metrics = {}
        self.training_completed = False

        print(f"[Fed Client {self.client_id}] FedRayPPOTrainer initialized")

    def fit_federated(self, local_epochs=None):
        """
        Federated training entry point.

        Trains for the requested number of local epochs, then returns the
        model parameters together with the training metrics.

        Args:
            local_epochs: Number of local training epochs; if None, the value
                from the config is used.

        Returns:
            tuple: (sample_size, model_parameters, metrics)
        """
        if local_epochs is not None:
            # Temporarily override the configured epoch count.
            original_epochs = self.config.trainer.total_epochs
            self.config.trainer.total_epochs = local_epochs

        print(
            f"[Fed Client {self.client_id}] Starting federated training for round {self.round_num}"
        )

        # Record the initial model state.
        # initial_params = self.get_model_parameters()
        start_time = time.time()

        try:
            # Run training by delegating to the parent's fit() method.
            super().fit()

            # Collect the results once training has finished.
            final_params = self.get_model_parameters()
            training_time = time.time() - start_time

            # Assemble the training metrics.
            self.fed_metrics = {
                "client_id": self.client_id,
                "round": self.round_num,
                "training_time": training_time,
                "total_steps": self.global_steps,
                "training_completed": True,
                "final_global_step": self.global_steps,
            }

            self.training_completed = True

            print(
                f"[Fed Client {self.client_id}] Training completed for round {self.round_num}"
            )
            print(f"Training metrics: {self.fed_metrics}")

        except Exception as e:
            print(f"[Fed Client {self.client_id}] Training failed: {str(e)}")
            self.fed_metrics = {
                "client_id": self.client_id,
                "round": self.round_num,
                "error": str(e),
                "training_completed": False,
            }
            raise e
        finally:
            # Restore the original config value.
            if local_epochs is not None:
                self.config.trainer.total_epochs = original_epochs

        # Return the information the federated server needs.
        sample_size = self.fed_metrics.get("total_steps", 1)
        model_parameters = final_params
        metrics = self.fed_metrics

        self.round_num += 1

        return sample_size, model_parameters, metrics

    def get_model_parameters(self):
        """
        Extract the model parameters for federated aggregation.

        Returns:
            dict: A dictionary containing all model parameters.

        """
        import ray

        try:
            model_parameters = {}

            # Extract the actor parameters.
            if hasattr(self, "actor_rollout_wg"):
                actor_params = self._extract_worker_parameters(
                    self.actor_rollout_wg, "actor"
                )
                if actor_params:
                    # Apply the `actor.` prefix and move tensors to CPU.
                    for k, v in actor_params.items():
                        if isinstance(v, torch.Tensor):
                            model_parameters[f"actor.{k}"] = v.cpu().detach()
                        else:
                            model_parameters[f"actor.{k}"] = v
                    torch.cuda.empty_cache()
            # Extract the critic parameters (when a critic is in use).
            if self.use_critic and hasattr(self, "critic_wg"):
                critic_params = self._extract_worker_parameters(
                    self.critic_wg, "critic"
                )
                if critic_params:
                    for k, v in critic_params.items():
                        if isinstance(v, torch.Tensor):
                            model_parameters[f"critic.{k}"] = v.cpu().detach()
                        else:
                            model_parameters[f"critic.{k}"] = v
                torch.cuda.empty_cache()

            # Extract the reference policy parameters (when in use).
            # if self.use_reference_policy and hasattr(self, 'ref_policy_wg'):
            #     ref_params = self._extract_worker_parameters(self.ref_policy_wg, 'ref')
            #     if ref_params:
            #         model_parameters['ref'] = ref_params

            # print(f"[Fed Client {self.client_id}] Extracted parameters: {list(model_parameters.keys())}")
            return model_parameters

        except Exception as e:
            print(
                f"[Fed Client {self.client_id}] Parameter extraction failed: {str(e)}"
            )
            return {
                "error": str(e),
                "client_id": self.client_id,
                "round": self.round_num,
            }

    def _extract_worker_parameters(self, worker_group, role_name):
        """
        Extract parameters from a Ray worker group (hardened version).

        Args:
            worker_group: The Ray worker group.
            role_name: The role name (actor / critic / ref).

        Returns:
            dict: A dictionary of model parameters.
        """
        try:
            import ray

            # Map the role to its corresponding weight-getter method name.
            method_map = {
                "actor": "actor_rollout_get_weights",
                "critic": "critic_get_weights",
                "ref": "actor_rollout_get_weights",
            }
            method_name = method_map.get(role_name, None)
            futures = []
            for i, worker in enumerate(worker_group.workers):
                if hasattr(worker, method_name):
                    try:
                        # Pass the role through to the remote call.
                        future = getattr(worker, method_name).remote(role_name)
                        futures.append((future, i))
                    except Exception as e:
                        print(
                            f"Warning: Failed to call {method_name} on worker {i}: {e}"
                        )

            if not futures:
                print(f"Warning: No workers available for {role_name}")
                return {}

            # Wait for every worker to finish.
            results = []
            for future, worker_id in futures:
                try:
                    result = ray.get(future)
                    results.append((result, worker_id))
                except Exception as e:
                    print(f"Warning: Worker {worker_id} failed: {e}")
                    results.append((None, worker_id))

            # Use the first successful result (typically rank 0's weights).
            for result, worker_id in results:
                if result is not None and len(result) > 0:
                    print(
                        f"[Fed Client {self.client_id}] Using parameters from worker {worker_id} with role {role_name}"
                    )

                    # Move the tensors to CPU.
                    cpu_params = {}
                    for key, value in result.items():
                        if isinstance(value, torch.Tensor):
                            cpu_params[key] = value.cpu().detach()
                        else:
                            cpu_params[key] = value

                    return cpu_params

            print(f"Warning: No valid parameters returned from {role_name} workers")
            return {}

        except Exception as e:
            print(f"Warning: Failed to extract {role_name} parameters: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def _restructure_flat_params(self, flat_params):
        """
        Re-nest a flattened parameter dictionary back into a structured form.
        For example:
            {"actor.layer1.weight": tensor, "critic.layer2.bias": tensor}
        becomes:
            {"actor": {"layer1.weight": tensor}, "critic": {"layer2.bias": tensor}}
        """
        structured = {}

        for key, value in flat_params.items():
            if "." in key:
                parts = key.split(".", 1)  # split on the first dot only
                prefix = parts[0]  # actor, critic, ref
                param_name = parts[1]  # the remaining parameter name

                if prefix not in structured:
                    structured[prefix] = {}
                structured[prefix][param_name] = value
            else:
                # No prefix: treat it as a top-level parameter.
                structured[key] = value

        return structured

    def update_model_parameters(self, federated_params, strict=False):
        """
        Update the model parameters (after federated aggregation).

        Args:
            federated_params: The parameters produced by federated aggregation.
            strict: Whether to enforce strict loading.
        """
        try:
            print(
                f"[Fed Client {self.client_id}] Updating model parameters from federation"
            )
            # print(f"Received params keys: {list(federated_params.keys()) if isinstance(federated_params, dict) else 'Not a dict'}")

            # Unwrap the parameters.
            if "model_parameters" in federated_params:
                model_params = federated_params["model_parameters"]
            else:
                model_params = federated_params

            # Handle the flattened parameter layout (keys stored as "actor.xxx").
            if any(key.startswith("actor.") for key in model_params.keys()):
                structured_params = self._restructure_flat_params(model_params)
                if "actor" in structured_params and hasattr(self, "actor_rollout_wg"):
                    print(f"Updating actor parameters...")
                    self._update_worker_parameters(
                        self.actor_rollout_wg, structured_params["actor"], "actor"
                    )
                    # Update the critic parameters.
                if "critic" in structured_params and hasattr(self, "critic_wg"):
                    print(f"Updating critic parameters...")
                    self._update_worker_parameters(
                        self.critic_wg, structured_params["critic"], "critic"
                    )

                # Update the reference policy parameters.
                # if 'ref' in structured_params and hasattr(self, 'ref_policy_wg'):
                #     print(f"Updating ref parameters...")
                #     self._update_worker_parameters(self.ref_policy_wg, structured_params['actor'], 'ref')
            else:
                structured_params = model_params
                print(f"Structured params keys: {list(structured_params.keys())}")
                # the model param may be an orderedDict
                self._update_worker_parameters(
                    self.actor_rollout_wg, structured_params, "actor"
                )
                self._update_worker_parameters(
                    self.critic_wg, structured_params, "critic"
                )
                # self._update_worker_parameters(self.ref_policy_wg, structured_params, 'ref')

                torch.cuda.empty_cache()

            print(f"[Fed Client {self.client_id}] Parameter update completed")

        except Exception as e:
            print(f"[Fed Client {self.client_id}] Parameter update failed: {str(e)}")
            if strict:
                raise e

    def _update_worker_parameters(self, worker_group, params, role_name):
        """
        Update the parameters of a Ray worker group (hardened version).

        Args:
            worker_group: The Ray worker group.
            params: The new parameter dictionary.
            role_name: The role name.
        """
        try:
            import ray

            # Map the role to its corresponding weight-setter method name.
            update_method_map = {
                "actor": "actor_rollout_set_weights",
                "critic": "critic_set_weights",
                "ref": "actor_rollout_set_weights",
            }

            # Pre-process the parameters.
            # processed_params = self._prepare_params_for_fsdp(params, role_name)

            # Look up the update method for this role.
            update_method_name = update_method_map.get(role_name)

            if not update_method_name:
                print(f"Warning: No update method found for role {role_name}")
                return

            print(f"[Fed Client {self.client_id}] Updating {role_name} parameters...")

            # Push the parameters to every worker.
            futures = []
            for i, worker in enumerate(worker_group.workers):
                if hasattr(worker, update_method_name):
                    try:
                        # First attempt: pass strict=False.
                        future = getattr(worker, update_method_name).remote(
                            params, strict=True
                        )
                        futures.append(future)
                        print(
                            f"[Fed Client {self.client_id}] Sent parameters to {role_name} worker {i}"
                        )
                    except TypeError:
                        # If the method doesn't accept a strict argument, fall
                        # back to the plain call signature.
                        try:
                            future = getattr(worker, update_method_name).remote(params)
                            futures.append(future)
                            print(
                                f"[Fed Client {self.client_id}] Sent parameters to {role_name} worker {i} (fallback)"
                            )
                        except Exception as e:
                            print(
                                f"Warning: Failed to send parameters to {role_name} worker {i}: {e}"
                            )
                else:
                    print(
                        f"Warning: Worker {i} doesn't have method {update_method_name}"
                    )

            # Wait for every update to complete.
            if futures:
                try:
                    results = ray.get(futures)
                    success_count = 0
                    for i, result in enumerate(results):
                        if result is not None:
                            success_count += 1
                        else:
                            print(f"Warning: {role_name} worker {i} returned None")

                    print(
                        f"[Fed Client {self.client_id}] Updated {success_count}/{len(futures)} {role_name} workers"
                    )

                except Exception as e:
                    print(f"Warning: Some {role_name} parameter updates failed: {e}")
                    # Try to collect whatever partial results are available.
                    for i, future in enumerate(futures):
                        try:
                            result = ray.get(future)
                            print(
                                f"[Fed Client {self.client_id}] {role_name} worker {i} update: {'Success' if result else 'Failed'}"
                            )
                        except Exception as worker_e:
                            print(
                                f"Warning: {role_name} worker {i} update failed: {worker_e}"
                            )
            else:
                print(f"Warning: No workers updated for {role_name}")

        except Exception as e:
            print(f"Warning: Failed to update {role_name} parameters: {e}")
            import traceback

            traceback.print_exc()

    def _prepare_params_for_fsdp(self, params, role_name):
        """
        Coerce parameters into the format FSDP expects.

        Args:
            params: The original parameter dictionary.
            role_name: The role name.

        Returns:
            dict: The processed parameter dictionary.
        """
        try:
            processed_params = {}

            for key, value in params.items():
                if isinstance(value, torch.Tensor):
                    # Make sure each tensor lives on the correct device.
                    if torch.cuda.is_available():
                        processed_tensor = value.cuda()
                    else:
                        processed_tensor = value.cpu()

                    # Make sure the tensor is contiguous.
                    if not processed_tensor.is_contiguous():
                        processed_tensor = processed_tensor.contiguous()

                    processed_params[key] = processed_tensor
                else:
                    processed_params[key] = value

            return processed_params

        except Exception as e:
            print(f"Warning: Failed to prepare params for FSDP: {e}")
            return params

    def evaluate_federated(self, target_data_split_name="test"):
        """
        Federated evaluation entry point.
        """
        try:
            print(f"[Fed Client {self.client_id}] Starting federated evaluation")

            if hasattr(self, "_validate") and self.val_reward_fn is not None:
                eval_results = self._validate()

                # Post-process the evaluation results and add the required
                # `_total` fields.
                processed_results = {}

                # Assumed validation-set size; adjust to your setup (it can be
                # read from the config or derived from the dataset).
                val_total_samples = 1  # or fetch the real size from your dataset

                for key, value in eval_results.items():
                    processed_results[key] = value

                # Add the required `_total` fields.
                # FederatedScope expects every dataset prefix to have a matching
                # `_total` entry.
                dataset_prefixes = set()
                for key in processed_results.keys():
                    if "/" in key:
                        prefix = key.split("/")[
                            0
                        ]  # e.g. derive "val" from "val/success_rate"
                        dataset_prefixes.add(prefix)

                # Add a `_total` field for each dataset prefix.
                for prefix in dataset_prefixes:
                    total_key = f"{prefix}_total"  # e.g. "val_total"
                    if total_key not in processed_results:
                        processed_results[total_key] = val_total_samples

                print(f"[Fed Client {self.client_id}] Evaluation completed")
                print(f"Processed metrics: {list(processed_results.keys())}")
                return processed_results
            else:
                return {
                    "val_total": 1,  # still provide the total field even without evaluation
                }

        except Exception as e:
            print(f"[Fed Client {self.client_id}] Evaluation failed: {str(e)}")
            return {
                "val_total": 1,  # provide the total field on the error path too
            }

    def save_federated_checkpoint(self, path, additional_info=None):
        """
        Save a federated-learning checkpoint.

        Args:
            path: The destination path.
            additional_info: Optional extra information to store.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            checkpoint = {
                "client_id": self.client_id,
                "round": self.round_num,
                "model_parameters": self.get_model_parameters(),
                "fed_metrics": self.fed_metrics,
                "global_steps": getattr(self, "global_steps", 0),
                "training_completed": self.training_completed,
                "timestamp": time.time(),
            }

            if additional_info:
                checkpoint.update(additional_info)

            with open(path, "wb") as f:
                pickle.dump(checkpoint, f)

            print(f"[Fed Client {self.client_id}] Checkpoint saved to {path}")

        except Exception as e:
            print(f"[Fed Client {self.client_id}] Checkpoint save failed: {str(e)}")

    def load_federated_checkpoint(self, path):
        """
        Load a federated-learning checkpoint.

        Args:
            path: The checkpoint path.

        Returns:
            dict: The checkpoint data.
        """
        try:
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)

            # Restore the trainer state.
            self.round_num = checkpoint.get("round", 0)
            self.fed_metrics = checkpoint.get("fed_metrics", {})
            self.training_completed = checkpoint.get("training_completed", False)

            # Apply the stored model parameters.
            if "model_parameters" in checkpoint:
                self.update_model_parameters(checkpoint["model_parameters"])

            print(f"[Fed Client {self.client_id}] Checkpoint loaded from {path}")
            return checkpoint

        except Exception as e:
            print(f"[Fed Client {self.client_id}] Checkpoint load failed: {str(e)}")
            return None

    def cleanup_resources(self):
        """
        Release resources - make sure every Ray resource is fully freed.
        """
        try:
            import ray

            print(f"[Fed Client {self.client_id}] Starting resource cleanup...")

            # 1. First, shut down all worker groups.
            if hasattr(self, "actor_rollout_wg"):
                self._shutdown_worker_group(self.actor_rollout_wg, "actor_rollout")
            # available_resources = ray.available_resources()
            # print(f"[Fed Client {self.client_id}] Resources after cleanup: {available_resources}")
            if hasattr(self, "critic_wg"):
                self._shutdown_worker_group(self.critic_wg, "critic")
            # available_resources = ray.available_resources()
            # print(f"[Fed Client {self.client_id}] Resources after cleanup: {available_resources}")
            if hasattr(self, "ref_policy_wg"):
                self._shutdown_worker_group(self.ref_policy_wg, "ref_policy")
            # available_resources = ray.available_resources()
            # print(f"[Fed Client {self.client_id}] Resources after cleanup: {available_resources}")
            # 2. Clean up the resource pool manager.
            if hasattr(self, "resource_pool_manager"):
                self._cleanup_resource_pools()

            # 3. Force a garbage-collection pass.
            import gc

            gc.collect()
            available_resources = ray.available_resources()
            print(
                f"[Fed Client {self.client_id}] Resources after cleanup: {available_resources}"
            )
            # 4. Clear the CUDA cache.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 5. Wait a moment to make sure the resources are actually released.
            import time
            time.sleep(2)
            # 4. Verify the GPU resources are still present.

            available_resources = ray.available_resources()
            print(
                f"[Fed Client {self.client_id}] Resources after cleanup: {available_resources}"
            )

            print(f"[Fed Client {self.client_id}] Minimal cleanup completed")
            # print(f"[Fed Client {self.client_id}] Resources cleaned up successfully")

        except Exception as e:
            print(f"[Fed Client {self.client_id}] Resource cleanup failed: {str(e)}")

    def _cleanup_resource_pools(self):
        """Clean up the resource pools."""
        try:
            if hasattr(self.resource_pool_manager, "resource_pool_dict"):
                for (
                    pool_name,
                    resource_pool,
                ) in self.resource_pool_manager.resource_pool_dict.items():
                    print(
                        f"[Fed Client {self.client_id}] Cleaning up resource pool: {pool_name}"
                    )

                    # Tear down every process held by the resource pool.
                    if hasattr(resource_pool, "shutdown"):
                        resource_pool.shutdown()
                    elif hasattr(resource_pool, "cleanup"):
                        resource_pool.cleanup()
                    elif hasattr(resource_pool, "processes"):
                        # Kill every process manually.
                        for process in resource_pool.processes:
                            try:
                                import ray

                                ray.kill(process, no_restart=True)
                            except Exception as e:
                                print(f"Warning: Failed to kill process {process}: {e}")

                # Empty the resource-pool dictionary.
                self.resource_pool_manager.resource_pool_dict.clear()

        except Exception as e:
            print(f"Warning: Failed to cleanup resource pools: {e}")

    def _shutdown_worker_group(self, worker_group, name):
        """
        Shut down a worker group (enhanced version).
        """
        try:
            print(f"[Fed Client {self.client_id}] Shutting down {name} worker group...")

            # Method 1: call the group's shutdown() method.
            if hasattr(worker_group, "shutdown"):
                worker_group.shutdown()
                print(f"[Fed Client {self.client_id}] Called shutdown() on {name}")

            # Method 2: kill every worker manually.
            if hasattr(worker_group, "workers"):
                import ray

                for i, worker in enumerate(worker_group.workers):
                    try:
                        ray.kill(worker, no_restart=True)
                        print(f"[Fed Client {self.client_id}] Killed {name} worker {i}")
                    except Exception as e:
                        print(f"Warning: Failed to kill {name} worker {i}: {e}")

            # Method 3: clear the worker group's internal state.
            if hasattr(worker_group, "clear"):
                worker_group.clear()

            print(
                f"[Fed Client {self.client_id}] Shutdown {name} worker group completed"
            )

        except Exception as e:
            print(f"Warning: Failed to shutdown {name} worker group: {e}")

    def get_training_metrics(self):
        """
        Return the training metrics.

        Returns:
            dict: The training metrics.
        """
        return self.fed_metrics.copy()

    def reset_for_next_round(self):
        """
        Reset the trainer state for the next training round.
        """
        self.training_completed = False
        self.fed_metrics = {}

        # Reset the training step counter.
        if hasattr(self, "global_steps"):
            self.global_steps = 0

        print(f"[Fed Client {self.client_id}] Reset for round {self.round_num}")


# Factory function used for FederatedScope registration.
def create_fed_ray_ppo_trainer(client_id=0, **verl_kwargs):
    """
    Factory function that constructs a federated Ray PPO trainer.

    Args:
        client_id: The federated client ID.
        **verl_kwargs: Arguments forwarded to verl's RayPPOTrainer.

    Returns:
        FedRayPPOTrainer: The federated trainer instance.
    """
    return FedRayPPOTrainer(client_id=client_id, **verl_kwargs)
