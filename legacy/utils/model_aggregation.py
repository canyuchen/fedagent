#!/usr/bin/env python3
"""Model-weight aggregation utility for FedAgent.

Implements the server-side weight-aggregation step of FedAgent (the
Federated Agent RL method): after each round, the selected clients'
locally RL-trained policies are combined into one global policy. The
default and paper-reported aggregator is FedAvg (uniform / weighted
parameter averaging), and every experiment config under config/ uses it.
FedProx does NOT change this server-side step: it adds a proximal term to
each CLIENT's local objective during training (see verl dp_actor.update_policy,
actor.fedprox_mu), while the server still aggregates by plain FedAvg.

Models are stored as VERL FSDP-sharded checkpoints, so this module also
handles loading, merging, and re-sharding those per-rank shard files.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime
import copy
import os
import shutil

class ModelAggregator:
    """Aggregator for model weights."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def load_fsdp_model(self, model_path, device="cpu"):
        """Load an FSDP model and recombine its parameters, keeping the DTensor format."""
        model_path = Path(model_path)
        
        self.logger.info(f"Loading FSDP model: {model_path}")
        
        # Initialize the distributed environment (if needed).
        try:
            import torch.distributed as dist
            import os
            if not dist.is_initialized():
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                os.environ['RANK'] = '0'
                os.environ['WORLD_SIZE'] = '1'
                # Use the gloo backend because we are loading on CPU.
                dist.init_process_group(backend='gloo')
        except Exception as e:
            self.logger.warning(f"Failed to initialize distributed: {e}")
        
        # Find the rank files.
        rank_files = [f for f in os.listdir(model_path) if f.startswith("model_world_size_") and f.endswith(".pt")]
        rank_files.sort()
        
        if not rank_files:
            raise FileNotFoundError(f"No rank files found in: {model_path}")
        
        self.logger.info(f"Found {len(rank_files)} rank files")
        
        # Load all rank files.
        rank_data = []
        for rank_file in rank_files:
            self.logger.info(f"Loading {rank_file}...")
            rank_path = os.path.join(model_path, rank_file)
            rank_state = torch.load(rank_path, map_location=device, weights_only=False)
            rank_data.append(rank_state)
        
        # Recombine the parameters, keeping the DTensor format.
        self.logger.info("Recombining parameters...")
        combined_state = {}

        # Get the names of all parameters.
        all_keys = set()
        for rank_state in rank_data:
            all_keys.update(rank_state.keys())
        
        self.logger.info(f"Total parameters: {len(all_keys)}")
        
        for key in sorted(all_keys):
            # Collect this parameter's shards across all ranks.
            param_shards = []
            for rank_state in rank_data:
                if key in rank_state:
                    param = rank_state[key]
                    # Keep the DTensor format; do not convert to a plain tensor.
                    param_shards.append(param)

            if not param_shards:
                continue

            # Decide how to combine based on the parameter type.
            if "down_proj.weight" in key:
                # Down projection: concatenate along the first dimension.
                combined_state[key] = torch.cat(param_shards, dim=0)
            elif "gate_proj.weight" in key or "up_proj.weight" in key:
                # Gate / up projection: concatenate along the first dimension.
                combined_state[key] = torch.cat(param_shards, dim=0)
            elif "q_proj.weight" in key or "k_proj.weight" in key or "v_proj.weight" in key:
                # Q / K / V projection: concatenate along the first dimension.
                combined_state[key] = torch.cat(param_shards, dim=0)
            elif "o_proj.weight" in key:
                # Output projection: concatenate along the first dimension.
                combined_state[key] = torch.cat(param_shards, dim=0)
            elif any(x in key for x in ["input_layernorm", "post_attention_layernorm", "norm"]):
                # LayerNorm: concatenate along the first dimension.
                combined_state[key] = torch.cat(param_shards, dim=0)
            elif "bias" in key:
                # Bias: concatenate along the first dimension.
                combined_state[key] = torch.cat(param_shards, dim=0)
            elif "embed_tokens.weight" in key:
                # Embedding layer: concatenate along the first dimension.
                combined_state[key] = torch.cat(param_shards, dim=0)
            elif "lm_head.weight" in key:
                # Language-model head: concatenate along the second dimension.
                combined_state[key] = torch.cat(param_shards, dim=1)
            else:
                # Default: concatenate along the first dimension.
                combined_state[key] = torch.cat(param_shards, dim=0)
        
        return combined_state
    
    def average_models(self, model_states: List[Dict[str, torch.Tensor]], weights: Optional[List[float]] = None):
        """Average the parameters of multiple models."""
        self.logger.info("Averaging model parameters...")

        if not model_states:
            raise ValueError("No model states provided for averaging")

        # Use the first model as the baseline.
        averaged_state = {}
        base_state = model_states[0]

        # Get all weights.
        if weights is None:
            weights = [1.0 / len(model_states)] * len(model_states)

        # Initialize the aggregated weights.
        for key, value in base_state.items():
            averaged_state[key] = value.clone() * weights[0]

        # Aggregate the remaining models.
        for i, state_dict in enumerate(model_states[1:], 1):
            weight = weights[i] if i < len(weights) else weights[0]
            
            for key, value in state_dict.items():
                if key in averaged_state:
                    averaged_state[key] += value * weight
                else:
                    self.logger.warning(f"Key {key} not found in base model")
        
        return averaged_state
    
    def reshard_model(self, state_dict: Dict[str, torch.Tensor], world_size: int = 2, output_dir: Path = None):
        """Re-shard a full (un-sharded) state dict back into VERL's FSDP per-rank shard files.

        The primary path wraps each per-rank tensor as a torch DTensor (the format
        VERL expects when resuming an FSDP checkpoint). If DTensor construction
        fails (e.g. no CUDA device mesh available), it falls back to saving plain
        per-rank tensors instead -- so 'real DTensors' are produced only on the
        primary path, not the fallback.
        """
        self.logger.info(f"Resharding model to {world_size} GPUs using VERL FSDP format...")
        
        if output_dir is None:
            raise ValueError("Output directory must be specified")
        
        # Create the output directory.
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save the shard files using VERL's FSDP format.
        self.logger.info("Saving shard files using VERL FSDP format...")

        # Re-shard the parameters.
        rank_states = [{} for _ in range(world_size)]

        try:
            # Import the DTensor-related modules.
            from torch.distributed.tensor import DTensor, DeviceMesh, Shard
            import torch.distributed as dist

            # Initialize the distributed environment.
            if not dist.is_initialized():
                import os
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                os.environ['RANK'] = '0'
                os.environ['WORLD_SIZE'] = '1'
                dist.init_process_group(backend='gloo')
            
            # Create the device mesh.
            device_mesh = DeviceMesh("cuda", list(range(world_size)))
            
            for key, param in state_dict.items():
                self.logger.debug(f"Resharding parameter: {key} - shape: {param.shape}")
                
                # Make sure the parameter shape is correct.
                if len(param.shape) == 0:
                    # Scalar parameter: copy it directly to every rank.
                    for rank in range(world_size):
                        rank_states[rank][key] = param.clone()
                    continue
                
                # Decide how to shard based on the parameter type.
                if "down_proj.weight" in key:
                    # Down projection: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        # If the first dimension is smaller than world_size, copy directly to every rank.
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                            
                elif "gate_proj.weight" in key or "up_proj.weight" in key:
                    # Gate / up projection: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                            
                elif "q_proj.weight" in key or "k_proj.weight" in key or "v_proj.weight" in key:
                    # Q / K / V projection: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                            
                elif "o_proj.weight" in key:
                    # Output projection: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                            
                elif any(x in key for x in ["input_layernorm", "post_attention_layernorm", "norm"]):
                    # LayerNorm: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                            
                elif "bias" in key:
                    # Bias: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                            
                elif "embed_tokens.weight" in key:
                    # Embedding layer: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                            
                elif "lm_head.weight" in key:
                    # Language-model head: shard along the second dimension.
                    if param.shape[1] >= world_size:
                        chunk_size = param.shape[1] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[1]
                            rank_states[rank][key] = param[:, start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                else:
                    # Default: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
            
            # Build the DTensor-sharded state dicts.
            for rank in range(world_size):
                rank_file = f"model_world_size_{world_size}_rank_{rank}.pt"
                rank_path = output_path / rank_file

                # Build the DTensor-sharded state dict.
                dtensor_state_dict = {}
                for key, tensor in rank_states[rank].items():
                    # Convert the plain tensor into a DTensor. The placement MUST match
                    # the dimension this parameter was sliced along above: lm_head.weight
                    # is sharded along dim 1 (column slice, param[:, start:end]); every
                    # other parameter is sharded along dim 0. A mismatch here mislabels
                    # the shard and corrupts the weight when the DTensor is later gathered
                    # or resharded (e.g. on FSDP resume for an untied lm_head).
                    if len(tensor.shape) > 0:  # Non-scalar parameter.
                        shard_dim = 1 if "lm_head.weight" in key else 0
                        dtensor = DTensor.from_local(
                            tensor,
                            device_mesh,
                            [Shard(shard_dim)]
                        )
                        dtensor_state_dict[key] = dtensor
                    else:  # Scalar parameter.
                        dtensor_state_dict[key] = tensor

                # Save the DTensor state dict.
                torch.save(dtensor_state_dict, rank_path)
                self.logger.info(f"Saved {rank_file} - parameters: {len(dtensor_state_dict)}")
                
        except Exception as e:
            self.logger.warning(f"Failed to create DTensor shards: {e}, falling back to standard save")
            # Fall back to the standard save path; rank_states must be populated first.
            for key, param in state_dict.items():
                self.logger.debug(f"Resharding parameter: {key} - shape: {param.shape}")
                
                # Make sure the parameter shape is correct.
                if len(param.shape) == 0:
                    # Scalar parameter: copy it directly to every rank.
                    for rank in range(world_size):
                        rank_states[rank][key] = param.clone()
                    continue
                
                # Decide how to shard based on the parameter type.
                if "down_proj.weight" in key:
                    # Down projection: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        # If the first dimension is smaller than world_size, copy directly to every rank.
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                            
                elif "gate_proj.weight" in key or "up_proj.weight" in key:
                    # Gate / up projection: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                            
                elif "q_proj.weight" in key or "k_proj.weight" in key or "v_proj.weight" in key:
                    # Q / K / V projection: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                            
                elif "o_proj.weight" in key:
                    # Output projection: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                            
                elif any(x in key for x in ["input_layernorm", "post_attention_layernorm", "norm"]):
                    # LayerNorm: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                            
                elif "bias" in key:
                    # Bias: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                            
                elif "embed_tokens.weight" in key:
                    # Embedding layer: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                            
                elif "lm_head.weight" in key:
                    # Language-model head: shard along the second dimension.
                    if param.shape[1] >= world_size:
                        chunk_size = param.shape[1] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[1]
                            rank_states[rank][key] = param[:, start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
                else:
                    # Default: shard along the first dimension.
                    if param.shape[0] >= world_size:
                        chunk_size = param.shape[0] // world_size
                        for rank in range(world_size):
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size if rank < world_size - 1 else param.shape[0]
                            rank_states[rank][key] = param[start_idx:end_idx].clone()
                    else:
                        for rank in range(world_size):
                            rank_states[rank][key] = param.clone()
            
            # Save the shard files.
            for rank in range(world_size):
                rank_file = f"model_world_size_{world_size}_rank_{rank}.pt"
                rank_path = output_path / rank_file
                torch.save(rank_states[rank], rank_path)
                self.logger.info(f"Saved {rank_file} - parameters: {len(rank_states[rank])}")
        
        self.logger.info(f"✅ Aggregated model saved to: {output_path}")
        return output_path
    
    def fedavg_aggregation(self, model_paths: List[str], 
                          output_path: str,
                          expected_global_step: int = 0,
                          weights: Optional[List[float]] = None,
                          model_type: str = "actor",
                          n_gpus_per_node: int = 1) -> str:
        """FedAvg aggregation method - aggregate using FSDP shards.

        Args:
            model_paths: List of client directory paths (containing FSDP shards).
            output_path: Output path.
            expected_global_step: Expected global step number.
            weights: List of weights (optional; defaults to equal weights).
            model_type: Model type ("actor" or "critic").
            n_gpus_per_node: Number of GPUs per node.

        Returns:
            Path to the aggregated model.
        """
        if not model_paths:
            self.logger.error("No model paths provided for aggregation")
            return None
        
        self.logger.info(f"Starting FedAvg aggregation with {len(model_paths)} {model_type} clients using FSDP shards")
        
        # Create the output directory structure.
        output_path = Path(output_path)
        output_dir = output_path.parent
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        global_step_dir = checkpoints_dir / f"global_step_{expected_global_step}"
        global_step_dir.mkdir(exist_ok=True)
        
        model_dir = global_step_dir / model_type
        model_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Load the FSDP models from all clients.
            self.logger.info("Loading FSDP models from all clients...")
            model_states = []
            for i, model_path in enumerate(model_paths):
                try:
                    self.logger.info(f"Loading model {i+1}/{len(model_paths)}: {model_path}")
                    state_dict = self.load_fsdp_model(model_path)
                    model_states.append(state_dict)
                    self.logger.info(f"Successfully loaded model from {model_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load model from {model_path}: {str(e)}")
                    continue
            
            if not model_states:
                self.logger.error("No valid models loaded for aggregation")
                return None
            
            # 2. Average the model parameters.
            self.logger.info(f"Averaging {len(model_states)} models...")
            averaged_state = self.average_models(model_states, weights)

            # 3. Re-shard and save.
            self.logger.info(f"Resharding aggregated model to {n_gpus_per_node} GPUs...")
            aggregated_path = self.reshard_model(averaged_state, n_gpus_per_node, model_dir)

            # 4. Copy the configuration files.
            self._copy_config_files(model_paths[0], model_dir)

            # 5. Create the latest_checkpointed_iteration.txt file.
            latest_iteration_file = model_dir.parent.parent / "latest_checkpointed_iteration.txt"
            with open(latest_iteration_file, 'w') as f:
                f.write(str(expected_global_step))

            self.logger.info(f"✅ FedAvg aggregation completed successfully: {aggregated_path}")
            # Return the global_step directory rather than model_dir, since VERL expects the global_step_X directory.
            return str(global_step_dir)
            
        except Exception as e:
            self.logger.error(f"FedAvg aggregation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def average_rank_states(self, rank_states: List[Dict[str, torch.Tensor]], weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """Average a set of same-rank shard state dicts."""
        if not rank_states:
            raise ValueError("No rank states provided for averaging")
        
        if len(rank_states) == 1:
            return rank_states[0]
        
        # Get all parameter keys.
        all_keys = set()
        for rank_state in rank_states:
            all_keys.update(rank_state.keys())

        # Set the weights.
        if weights is None:
            weights = [1.0 / len(rank_states)] * len(rank_states)
        else:
            # Normalize the weights.
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

        # Compute a weighted average for each parameter.
        averaged_state = {}
        for key in all_keys:
            # Collect this parameter's value across all ranks.
            param_values = []
            for rank_state in rank_states:
                if key in rank_state:
                    param_values.append(rank_state[key])

            if not param_values:
                self.logger.warning(f"Parameter {key} not found in any rank state")
                continue

            # Check that all parameter shapes match.
            first_shape = param_values[0].shape
            for i, param in enumerate(param_values):
                if param.shape != first_shape:
                    self.logger.warning(f"Shape mismatch for {key}: {param.shape} vs {first_shape}")
                    continue

            # Compute the weighted average.
            if len(param_values) == 1:
                averaged_state[key] = param_values[0].clone()
            else:
                # Initialize the weighted sum.
                weighted_sum = param_values[0] * weights[0]
                for i in range(1, len(param_values)):
                    weighted_sum += param_values[i] * weights[i]
                averaged_state[key] = weighted_sum
        
        return averaged_state
    
    def direct_shard_aggregation(self, model_paths: List[str], 
                                output_path: str,
                                expected_global_step: int = 0,
                                weights: Optional[List[float]] = None,
                                model_type: str = "actor",
                                n_gpus_per_node: int = 2) -> str:
        """Aggregate same-rank shards directly, avoiding the complex recombine-then-reshard path."""
        self.logger.info(f"Starting direct shard aggregation with {len(model_paths)} {model_type} clients")
        
        # Create the output directory structure.
        output_path = Path(output_path)
        output_dir = output_path.parent
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        global_step_dir = checkpoints_dir / f"global_step_{expected_global_step}"
        global_step_dir.mkdir(exist_ok=True)
        
        model_dir = global_step_dir / model_type
        model_dir.mkdir(exist_ok=True)
        
        try:
            # Use the first client's list of rank files as the reference.
            first_model_path = Path(model_paths[0])
            rank_files = [f for f in os.listdir(first_model_path) if f.startswith("model_world_size_") and f.endswith(".pt")]
            rank_files.sort()
            
            if not rank_files:
                raise FileNotFoundError(f"No rank files found in: {first_model_path}")
            
            self.logger.info(f"Found {len(rank_files)} rank files to aggregate")
            
            # Aggregate each rank file.
            for rank_file in rank_files:
                self.logger.info(f"Aggregating {rank_file}...")

                # Collect this rank file from every client.
                rank_states = []
                for model_path in model_paths:
                    rank_path = Path(model_path) / rank_file
                    if rank_path.exists():
                        rank_state = torch.load(rank_path, weights_only=False)
                        rank_states.append(rank_state)
                    else:
                        self.logger.warning(f"Rank file {rank_file} not found in {model_path}")
                
                if not rank_states:
                    raise RuntimeError(f"No valid rank states found for {rank_file}")
                
                # Average the same-rank shards.
                averaged_rank_state = self.average_rank_states(rank_states, weights)

                # Save the aggregated rank file.
                output_rank_path = model_dir / rank_file
                torch.save(averaged_rank_state, output_rank_path)
                self.logger.info(f"Saved aggregated {rank_file} - parameters: {len(averaged_rank_state)}")
            
            # Copy the configuration files.
            self._copy_config_files(model_paths[0], model_dir)

            # Create the latest_checkpointed_iteration.txt file.
            latest_iteration_file = model_dir.parent.parent / "latest_checkpointed_iteration.txt"
            with open(latest_iteration_file, 'w') as f:
                f.write(str(expected_global_step))

            self.logger.info(f"✅ Direct shard aggregation completed successfully: {model_dir}")
            return str(global_step_dir)
            
        except Exception as e:
            self.logger.error(f"Direct shard aggregation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _copy_config_files(self, source_model_path: str, target_dir: Path):
        """Copy the configuration files into the target directory."""
        source_dir = Path(source_model_path)
        config_files = [
            "config.json", "tokenizer_config.json", "tokenizer.json", 
            "vocab.json", "merges.txt", "special_tokens_map.json", 
            "added_tokens.json", "generation_config.json"
        ]
        
        self.logger.info("Copying configuration files...")
        for config_file in config_files:
            src = source_dir / config_file
            dst = target_dir / config_file
            if src.exists():
                shutil.copy2(src, dst)
                self.logger.info(f"Copied {config_file}")
            else:
                self.logger.warning(f"Configuration file not found: {src}")
    
    def _single_gpu_aggregation(self, model_paths: List[str], model_dir: Path, 
                               expected_global_step: int, weights: Optional[List[float]], 
                               model_type: str) -> str:
        """Single-GPU aggregation."""
        # Load the first model as the baseline.
        base_checkpoint = torch.load(model_paths[0], map_location='cpu', weights_only=False)
        base_state_dict = base_checkpoint.get('model', base_checkpoint)

        # Initialize the aggregated weights.
        aggregated_state = {}
        for key, value in base_state_dict.items():
            if weights is None:
                # Equal-weight aggregation.
                aggregated_state[key] = value.clone() * (1.0 / len(model_paths))
            else:
                # Weighted aggregation.
                aggregated_state[key] = value.clone() * weights[0]

        # Aggregate the remaining models.
        for i, model_path in enumerate(model_paths[1:], 1):
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                state_dict = checkpoint.get('model', checkpoint)
                
                weight = 1.0 / len(model_paths) if weights is None else weights[i]
                
                for key, value in state_dict.items():
                    if key in aggregated_state:
                        aggregated_state[key] += value * weight
                    else:
                        self.logger.warning(f"Key {key} not found in base model")
                        
            except Exception as e:
                self.logger.error(f"Failed to load model {model_path}: {str(e)}")
                continue
        
        # Save the single-GPU model.
        model_file = model_dir / "model_world_size_1_rank_0.pt"
        torch.save(aggregated_state, model_file)
        self.logger.info(f"Saved single GPU model: {model_file}")
        
        # Create the latest_checkpointed_iteration.txt file.
        latest_iteration_file = model_dir.parent.parent / "latest_checkpointed_iteration.txt"
        with open(latest_iteration_file, 'w') as f:
            f.write(str(expected_global_step))
        
        return str(model_file)
    
    def _multi_gpu_fsdp_aggregation(self, model_paths: List[str], model_dir: Path,
                                   expected_global_step: int, weights: Optional[List[float]],
                                   model_type: str, n_gpus_per_node: int) -> str:
        """Multi-GPU FSDP aggregation."""
        # Use torchrun to invoke the create_fsdp_shards.py script.
        import subprocess
        import sys
        
        script_path = Path(__file__).parent.parent / "tools" / "aggregation" / "create_fsdp_shards.py"
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={n_gpus_per_node}",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=12358",
            str(script_path),
            "--client_dirs"] + model_paths + [
            "--output_dir", str(model_dir),
            "--n_gpus_per_node", str(n_gpus_per_node)
        ]
        
        try:
            self.logger.info(f"Running FSDP aggregation: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                self.logger.info("FSDP aggregation completed successfully")
                
                # Create the latest_checkpointed_iteration.txt file.
                latest_iteration_file = model_dir.parent.parent / "latest_checkpointed_iteration.txt"
                with open(latest_iteration_file, 'w') as f:
                    f.write(str(expected_global_step))
                
                # Return the path of the first shard file.
                result_file = model_dir / f"model_world_size_{n_gpus_per_node}_rank_0.pt"
                return str(result_file)
            else:
                self.logger.error(f"FSDP aggregation failed: {result.stderr}")
                # Fall back to single-GPU aggregation.
                self.logger.warning("Falling back to single GPU aggregation")
                return self._single_gpu_aggregation(model_paths, model_dir, expected_global_step, weights, model_type)
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout during FSDP aggregation, falling back to single GPU")
            return self._single_gpu_aggregation(model_paths, model_dir, expected_global_step, weights, model_type)
        except Exception as e:
            self.logger.error(f"Error during FSDP aggregation: {e}, falling back to single GPU")
            return self._single_gpu_aggregation(model_paths, model_dir, expected_global_step, weights, model_type)

    def aggregate_verl_models(self, client_results: List[Dict[str, Any]], 
                            output_dir: Path, n_gpus_per_node: int = 1) -> Dict[str, str]:
        """Aggregate verl-agent models - using the FSDP aggregation method.

        Args:
            client_results: List of per-client training results.
            output_dir: Output directory.
            n_gpus_per_node: Number of GPUs per node.

        Returns:
            Dictionary mapping component name to the aggregated model path.
        """
        self.logger.info(f"Aggregating verl models for {len(client_results)} clients using FSDP aggregation")
        
        # Check the number of clients.
        expected_clients = len(client_results)
        if expected_clients == 0:
            self.logger.error("No client results provided for aggregation")
            return {}

        # Collect every client's model path (using the FSDP shard directory directly).
        model_paths = []
        client_ids = []
        global_steps = []

        for client_result in client_results:
            if not client_result.get('success'):
                continue

            # model_path is now the client directory itself.
            client_dir = Path(client_result['model_path'])
            client_id = client_result['client_id']

            # Find the FSDP shard directory.
            fsdp_dir = self._find_fsdp_model_dir(client_dir, client_id)
            if fsdp_dir:
                # Check the global step number.
                global_step = self._get_global_step_from_path(str(fsdp_dir))
                if global_step is None:
                    self.logger.error(f"Could not determine global step for client {client_id}")
                    return {}
                
                model_paths.append(str(fsdp_dir))
                client_ids.append(client_id)
                global_steps.append(global_step)
                self.logger.info(f"Found FSDP model directory for client {client_id}: {fsdp_dir} (global_step: {global_step})")
            else:
                self.logger.warning(f"No FSDP model directory found for client {client_id}")
        
        # Check the number of clients to aggregate.
        if len(model_paths) < expected_clients:
            error_msg = f"Insufficient models for aggregation: {len(model_paths)}/{expected_clients} clients have valid models. Expected {expected_clients} clients but only {len(model_paths)} clients provided valid models."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Check that the global step numbers are consistent.
        if len(set(global_steps)) > 1:
            self.logger.error(f"Inconsistent global steps among clients: {global_steps}")
            return {}

        # The aggregated model always uses global_step_0, marking the start of a new training round.
        expected_global_step = 0
        self.logger.info(f"All clients have consistent global_step: {global_steps[0]}, will aggregate to global_step_0 (new round starting point)")
        self.logger.info(f"Model paths for aggregation: {model_paths}")
        
        if not model_paths:
            error_msg = "No models found for aggregation. All clients failed to provide valid models."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Aggregate the models.
        aggregated_models = {}

        # Aggregate the actor model (the primary model) - using direct shard aggregation.
        output_path = output_dir / "aggregated_actor_model.pth"
        aggregated_path = self.direct_shard_aggregation(model_paths, str(output_path), expected_global_step, model_type="actor", n_gpus_per_node=n_gpus_per_node)
        if aggregated_path:
            aggregated_models['actor'] = aggregated_path
            self.logger.info(f"Aggregated actor model: {aggregated_path}")
        
        # Check whether there are critic models to aggregate.
        critic_model_paths = []
        for client_result in client_results:
            if not client_result.get('success'):
                continue

            # client_result['model_path'] is returned by
            # checkpoint_manager.find_client_model_path and is already the client's
            # main directory (e.g. .../round_1/client_14).
            # Previously this was treated as a file path and walked back via
            # .parent.parent.parent.parent, which went one level too high and caused
            # _find_fsdp_critic_dir to miss the critic, so it was skipped during aggregation.
            client_dir = Path(client_result['model_path'])
            client_id = client_result['client_id']

            # Find the critic model directory.
            critic_dir = self._find_fsdp_critic_dir(client_dir, client_id)
            if critic_dir:
                critic_model_paths.append(str(critic_dir))
                self.logger.info(f"Found critic FSDP directory for client {client_id}: {critic_dir}")
            else:
                self.logger.warning(f"No critic FSDP directory found for client {client_id} in directory: {client_dir}")
        
        # If critic models were found, aggregate them - using direct shard aggregation.
        if critic_model_paths and len(critic_model_paths) >= len(model_paths) * 0.8:  # At least 80% of clients have a critic model.
            critic_output_path = output_dir / "aggregated_critic_model.pth"
            critic_aggregated_path = self.direct_shard_aggregation(critic_model_paths, str(critic_output_path), expected_global_step, model_type="critic", n_gpus_per_node=n_gpus_per_node)
            if critic_aggregated_path:
                aggregated_models['critic'] = critic_aggregated_path
                self.logger.info(f"Aggregated critic model: {critic_aggregated_path}")
        else:
            self.logger.info("No critic models found or insufficient critic models for aggregation")
        
        return aggregated_models
    
    def _find_fsdp_model_dir(self, client_dir: Path, client_id: int) -> Optional[Path]:
        """Find the client's FSDP model directory."""
        # If client_dir is already the actor directory, check it directly.
        if client_dir.name == "actor":
            fsdp_files = list(client_dir.glob("model_world_size_*_rank_*.pt"))
            if fsdp_files:
                self.logger.info(f"Found FSDP sharded files for client {client_id}: {fsdp_files}")
                return client_dir
        
        # Inspect the global_step structure under the checkpoints directory.
        checkpoints_dir = client_dir / "checkpoints"
        if checkpoints_dir.exists():
            # Find the global_step directories.
            global_step_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith('global_step_')]
            if global_step_dirs:
                # Sort by step number and take the most recent one.
                latest_step_dir = max(global_step_dirs, key=lambda x: int(x.name.split('_')[2]))
                
                # Find the FSDP shard files under the actor directory.
                actor_dir = latest_step_dir / "actor"
                if actor_dir.exists():
                    # Check whether FSDP shard files are present.
                    fsdp_files = list(actor_dir.glob("model_world_size_*_rank_*.pt"))
                    if fsdp_files:
                        self.logger.info(f"Found FSDP sharded files for client {client_id}: {fsdp_files}")
                        return actor_dir
        
        return None
    
    def _find_fsdp_critic_dir(self, client_dir: Path, client_id: int) -> Optional[Path]:
        """Find the client's FSDP critic model directory."""
        # Inspect the global_step structure under the checkpoints directory.
        checkpoints_dir = client_dir / "checkpoints"
        if checkpoints_dir.exists():
            # Find the global_step directories.
            global_step_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith('global_step_')]
            if global_step_dirs:
                # Sort by step number and take the most recent one.
                latest_step_dir = max(global_step_dirs, key=lambda x: int(x.name.split('_')[2]))
                
                # Find the FSDP shard files under the critic directory.
                critic_dir = latest_step_dir / "critic"
                if critic_dir.exists():
                    # Check whether FSDP shard files are present.
                    fsdp_files = list(critic_dir.glob("model_world_size_*_rank_*.pt"))
                    if fsdp_files:
                        self.logger.info(f"Found FSDP sharded critic files for client {client_id}: {fsdp_files}")
                        return critic_dir
        
        return None
    
    def _find_verl_model(self, client_dir: Path, client_id: int) -> Optional[str]:
        """Find the verl-agent model file, supporting FSDP shard files."""
        # Inspect the global_step structure under the checkpoints directory.
        checkpoints_dir = client_dir / "checkpoints"
        if checkpoints_dir.exists():
            # Find the global_step directories.
            global_step_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith('global_step_')]
            if global_step_dirs:
                # Sort by step number and take the most recent one.
                latest_step_dir = max(global_step_dirs, key=lambda x: int(x.name.split('_')[2]))
                
                # Find the model files under the actor directory.
                actor_dir = latest_step_dir / "actor"
                if actor_dir.exists():
                    # First check whether FSDP shard files are present.
                    fsdp_files = list(actor_dir.glob("model_world_size_*_rank_*.pt"))
                    if fsdp_files:
                        # FSDP shard files exist and must be merged.
                        self.logger.info(f"Found FSDP sharded files for client {client_id}: {fsdp_files}")
                        merged_model_path = self._merge_fsdp_shards(fsdp_files, actor_dir, client_id)
                        if merged_model_path:
                            return str(merged_model_path)
                    
                    # If there are no FSDP shard files, look for a single model file.
                    model_files = []
                    for ext in ["*.pt", "*.pth", "*.safetensors", "*.bin"]:
                        model_files.extend(list(actor_dir.glob(ext)))
                    
                    if model_files:
                        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                        return str(latest_model)
        
        # If no global_step structure was found, look for a model file directly.
        model_files = []
        for ext in ["*.pth", "*.pt", "*.safetensors", "*.bin"]:
            model_files.extend(list(client_dir.glob(ext)))

        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            return str(latest_model)

        return None

    def _find_all_fsdp_shards(self, client_dir: Path, client_id: int) -> List[str]:
        """Find all of the client's FSDP shard files."""
        # Inspect the global_step structure under the checkpoints directory.
        checkpoints_dir = client_dir / "checkpoints"
        if checkpoints_dir.exists():
            # Find the global_step directories.
            global_step_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith('global_step_')]
            if global_step_dirs:
                # Sort by step number and take the most recent one.
                latest_step_dir = max(global_step_dirs, key=lambda x: int(x.name.split('_')[2]))
                
                # Find the FSDP shard files under the actor directory.
                actor_dir = latest_step_dir / "actor"
                if actor_dir.exists():
                    fsdp_files = list(actor_dir.glob("model_world_size_*_rank_*.pt"))
                    if fsdp_files:
                        # Sort by rank.
                        fsdp_files.sort(key=lambda x: int(x.name.split('_rank_')[1].split('.')[0]))
                        return [str(f) for f in fsdp_files]
        
        return []
    
    def _merge_fsdp_shards(self, fsdp_files: List[Path], actor_dir: Path, client_id: int) -> Optional[str]:
        """
        Merge FSDP shard files.

        Args:
            fsdp_files: List of FSDP shard files.
            actor_dir: Path to the actor directory.
            client_id: Client ID.

        Returns:
            Path to the merged model file.
        """
        try:
            # Parse the world_size and rank information.
            shard_info = []
            for file_path in fsdp_files:
                filename = file_path.name
                # Parse the model_world_size_X_rank_Y.pt format.
                parts = filename.replace('.pt', '').split('_')
                if len(parts) >= 6 and parts[0] == 'model' and parts[1] == 'world' and parts[2] == 'size':
                    world_size = int(parts[3])
                    rank = int(parts[5])
                    shard_info.append((rank, world_size, file_path))
            
            if not shard_info:
                self.logger.error(f"Could not parse FSDP shard information from files: {fsdp_files}")
                return None
            
            # Sort by rank.
            shard_info.sort(key=lambda x: x[0])

            # Check that world_size is consistent.
            world_sizes = set(info[1] for info in shard_info)
            if len(world_sizes) > 1:
                self.logger.error(f"Inconsistent world_size in FSDP shards: {world_sizes}")
                return None
            
            world_size = list(world_sizes)[0]
            expected_ranks = set(range(world_size))
            actual_ranks = set(info[0] for info in shard_info)
            
            if expected_ranks != actual_ranks:
                self.logger.error(f"Missing FSDP shards: expected ranks {expected_ranks}, got {actual_ranks}")
                return None
            
            self.logger.info(f"Merging {len(shard_info)} FSDP shards for client {client_id} (world_size={world_size})")
            
            # Load all shards.
            shard_state_dicts = []
            for rank, world_size, file_path in shard_info:
                try:
                    # Use weights_only=False to load the FSDP shard files.
                    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                    state_dict = checkpoint.get('model', checkpoint)
                    shard_state_dicts.append((rank, state_dict))
                    self.logger.info(f"Loaded shard rank {rank} from {file_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load shard {file_path}: {str(e)}")
                    return None
            
            # Merge the state dicts.
            # Inspect the DTensor's local_tensor shape to decide whether this is real parameter sharding.
            sample_key = list(shard_state_dicts[0][1].keys())[0]
            sample_dtensor = shard_state_dicts[0][1][sample_key]

            # Check whether it is a DTensor whose placements include Shard.
            is_dtensor_sharding = False
            if hasattr(sample_dtensor, 'to_local') and hasattr(sample_dtensor, 'placements'):
                # Check whether placements include Shard (str(p) returns the 'S(0)' format).
                placements = sample_dtensor.placements
                if any('S(' in str(p) for p in placements):
                    is_dtensor_sharding = True
                    self.logger.info(f"Detected DTensor with sharding placements: {placements}")
            
            if is_dtensor_sharding:
                # Real parameter sharding: merge the DTensors' local_tensors.
                self.logger.info(f"Detected DTensor parameter sharding for client {client_id}, merging local tensors")
                merged_state_dict = {}

                for key in shard_state_dicts[0][1].keys():
                    # Collect the local_tensor from every shard.
                    local_tensors = []
                    for _, state_dict in shard_state_dicts:
                        if key in state_dict:
                            local_tensor = state_dict[key].to_local()
                            local_tensors.append(local_tensor)

                    if local_tensors:
                        if len(local_tensors) == 1:
                            # Only one shard: use it directly.
                            merged_state_dict[key] = local_tensors[0]
                        else:
                            # Merge multiple shards.
                            try:
                                # Try concatenating along the first dimension.
                                merged_tensor = torch.cat(local_tensors, dim=0)
                                merged_state_dict[key] = merged_tensor
                            except Exception as e:
                                self.logger.warning(f"Failed to concatenate {key}: {e}, using first shard")
                                merged_state_dict[key] = local_tensors[0]
                
                self.logger.info(f"Merged DTensor local tensors from {len(shard_state_dicts)} shards for client {client_id}")
            else:
                # Data-parallel or non-DTensor: average the parameters.
                self.logger.info(f"Detected data parallel or non-DTensor shards for client {client_id}, averaging parameters")
                merged_state_dict = {}

                for key in shard_state_dicts[0][1].keys():
                    # Collect this parameter's value across all shards.
                    param_values = []
                    for _, state_dict in shard_state_dicts:
                        if key in state_dict:
                            param_values.append(state_dict[key])

                    if param_values:
                        # Average the parameter.
                        if len(param_values) == 1:
                            merged_state_dict[key] = param_values[0]
                        else:
                            # Compute the mean.
                            avg_param = sum(param_values) / len(param_values)
                            merged_state_dict[key] = avg_param
                
                self.logger.info(f"Averaged parameters from {len(shard_state_dicts)} shards for client {client_id}")
            
            # Save the merged model.
            merged_file = actor_dir / f"merged_model_client_{client_id}.pt"
            torch.save(merged_state_dict, merged_file)
            
            self.logger.info(f"Successfully merged FSDP shards for client {client_id}: {merged_file}")
            return str(merged_file)
            
        except Exception as e:
            self.logger.error(f"Failed to merge FSDP shards for client {client_id}: {str(e)}")
            return None
    def _find_verl_critic_model(self, client_dir: Path, client_id: int) -> Optional[str]:
        """Find the verl-agent critic model file, supporting FSDP shard files."""
        # Inspect the global_step structure under the checkpoints directory.
        checkpoints_dir = client_dir / "checkpoints"
        if checkpoints_dir.exists():
            # Find the global_step directories.
            global_step_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith('global_step_')]
            if global_step_dirs:
                # Sort by step number and take the most recent one.
                latest_step_dir = max(global_step_dirs, key=lambda x: int(x.name.split('_')[2]))
                
                # Find the model files under the critic directory.
                critic_dir = latest_step_dir / "critic"
                if critic_dir.exists():
                    # First check whether FSDP shard files are present.
                    fsdp_files = list(critic_dir.glob("model_world_size_*_rank_*.pt"))
                    if fsdp_files:
                        # FSDP shard files exist and must be merged.
                        self.logger.info(f"Found FSDP sharded critic files for client {client_id}: {fsdp_files}")
                        merged_model_path = self._merge_fsdp_shards(fsdp_files, critic_dir, client_id)
                        if merged_model_path:
                            return str(merged_model_path)
                    
                    # If there are no FSDP shard files, look for a single model file.
                    model_files = []
                    for ext in ["*.pt", "*.pth", "*.safetensors", "*.bin"]:
                        model_files.extend(list(critic_dir.glob(ext)))
                    
                    if model_files:
                        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                        return str(latest_model)
        
        # If no global_step structure was found, look for a critic model file directly.
        critic_files = []
        for ext in ["*critic*.pth", "*critic*.pt", "*critic*.safetensors", "*critic*.bin"]:
            critic_files.extend(list(client_dir.glob(ext)))
        
        if critic_files:
            latest_critic = max(critic_files, key=lambda x: x.stat().st_mtime)
            return str(latest_critic)
        
        return None
    
    def _get_global_step_from_path(self, model_path: str) -> Optional[int]:
        """Extract the global step number from the model path."""
        try:
            path = Path(model_path)
            # Extract global_step_X from the path.
            for part in path.parts:
                if part.startswith('global_step_'):
                    return int(part.split('_')[2])
            return None
        except (ValueError, IndexError):
            return None


def aggregate_round_models(round_num: int, 
                         client_results: List[Dict[str, Any]], 
                         output_dir: Path,
                         aggregation_method: str = 'fedavg',
                         n_gpus_per_node: int = 1,
                         **kwargs) -> Dict[str, str]:
    """Aggregate the models from one training round.

    Args:
        round_num: Round number.
        client_results: Per-client training results.
        output_dir: Output directory.
        aggregation_method: Aggregation method ('fedavg', 'fedprox').
        **kwargs: Additional parameters.

    Returns:
        Dictionary mapping component name to the aggregated model path.
    """
    aggregator = ModelAggregator()

    # Create the per-round aggregation directory.
    round_aggregate_dir = output_dir / f"round_{round_num}" / "aggregated"
    round_aggregate_dir.mkdir(parents=True, exist_ok=True)

    # Run the aggregation.
    if aggregation_method in ('fedavg', 'fedprox'):
        # Server-side aggregation is plain uniform FedAvg for BOTH methods. FedProx
        # differs from FedAvg only in the CLIENT's local objective: it adds the
        # proximal term (mu/2)||w - w^t||^2 during local training (see verl
        # dp_actor.update_policy, driven by actor.fedprox_mu / the FEDPROX_MU env
        # var that script_builder exports). There is no separate server-side FedProx
        # step -- the round aggregate is the uniform average of the client models in
        # both cases.
        aggregated_models = aggregator.aggregate_verl_models(client_results, round_aggregate_dir, n_gpus_per_node)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
    
    # Save the aggregation metadata.
    aggregation_info = {
        'round_num': round_num,
        'method': aggregation_method,
        'timestamp': datetime.now().isoformat(),
        'num_clients': len(client_results),
        'successful_clients': len([r for r in client_results if r.get('success')]),
        'aggregated_models': aggregated_models,
        'kwargs': kwargs
    }
    
    info_file = round_aggregate_dir / "aggregation_info.json"
    with open(info_file, 'w') as f:
        json.dump(aggregation_info, f, indent=2, default=str)
    
    return aggregated_models


if __name__ == "__main__":
    # Smoke-test the aggregation functionality.
    from utils.colored_logging import setup_colored_logging

    # Set up colored logging.
    logger = setup_colored_logging(logging.INFO)

    # Mock test.
    test_models = [
        "/path/to/model1.pth",
        "/path/to/model2.pth",
        "/path/to/model3.pth"
    ]
    
    aggregator = ModelAggregator()
    result = aggregator.fedavg_aggregation(test_models, "/tmp/aggregated_model.pth")
    print(f"Aggregation result: {result}") 