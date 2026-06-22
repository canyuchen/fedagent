#!/usr/bin/env python3
"""Script that loads client shards directly via the FSDP API and aggregates them.

This script is meant to be launched with torchrun; each rank is responsible for
loading the matching client shard and aggregating it.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, DeviceMesh, Shard
import argparse
import os
import sys
from pathlib import Path
import logging
import glob

# Configure logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_client_fsdp_shards(client_dirs: list, rank: int, world_size: int):
    """Load the FSDP shards from all clients."""
    client_shards = []

    for client_dir in client_dirs:
        client_path = Path(client_dir)
        # Find this client's FSDP shard files.
        shard_pattern = client_path / "**/model_world_size_*_rank_*.pt"
        shard_files = glob.glob(str(shard_pattern), recursive=True)

        # Find the shard file for the matching rank.
        rank_shard = None
        for shard_file in shard_files:
            if f"_rank_{rank}.pt" in shard_file:
                rank_shard = shard_file
                break
        
        if rank_shard:
            logger.info(f"Rank {rank}: Loading client shard from {rank_shard}")
            shard_state = torch.load(rank_shard, map_location='cpu', weights_only=False)
            client_shards.append(shard_state)
        else:
            logger.warning(f"Rank {rank}: No shard found for client {client_dir}")
    
    return client_shards

def aggregate_shards(client_shards: list, rank: int):
    """Aggregate the same-rank shards across multiple clients."""
    if not client_shards:
        return {}

    if len(client_shards) == 1:
        return client_shards[0]

    # Get the keys of all parameters.
    all_keys = set()
    for shard in client_shards:
        all_keys.update(shard.keys())

    aggregated_shard = {}

    for key in all_keys:
        # Collect this parameter for this rank across all clients.
        param_values = []
        for shard in client_shards:
            if key in shard:
                param_values.append(shard[key])

        if not param_values:
            continue

        if len(param_values) == 1:
            aggregated_shard[key] = param_values[0]
        else:
            # FedAvg parameter averaging: take the unweighted (uniform 1/k) mean of
            # this parameter across the k = len(param_values) participating clients
            # for this round. The paper aggregates with FedAvg; with the default
            # M=2 clients per round, k=2 so each client contributes 1/2. NOTE: this
            # FSDP path is always unweighted -- it ignores any per-client weights
            # (model_aggregation._multi_gpu_fsdp_aggregation does not pass --weights
            # to this script), which matches FedAvg's default equal weighting.
            # DTensors are averaged on their local shards (.to_local()) so the sharded
            # layout is preserved; plain tensors are averaged directly.
            if isinstance(param_values[0], DTensor):
                # DTensor parameter: average the local_tensor.
                local_tensors = [p.to_local() for p in param_values]
                avg_local = sum(local_tensors) / len(local_tensors)
                # Preserve the DTensor structure.
                aggregated_shard[key] = DTensor.from_local(
                    avg_local,
                    param_values[0].device_mesh,
                    param_values[0].placements
                )
            elif isinstance(param_values[0], torch.Tensor):
                # Plain tensor parameter.
                aggregated_shard[key] = sum(param_values) / len(param_values)
            else:
                # Non-tensor parameter: take the first one.
                aggregated_shard[key] = param_values[0]
    
    logger.info(f"Rank {rank}: Aggregated {len(aggregated_shard)} parameters from {len(client_shards)} clients")
    return aggregated_shard

def create_fsdp_shards(client_dirs: list, output_dir: Path, n_gpus_per_node: int):
    """Load client shards directly via the FSDP API and aggregate them."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if world_size != n_gpus_per_node:
        logger.error(f"World size ({world_size}) does not match n_gpus_per_node ({n_gpus_per_node}). Exiting.")
        return

    # Initialize the distributed environment.
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    logger.info(f"Rank {rank}/{world_size} initialized process group.")

    # Load the matching-rank shard from every client.
    logger.info(f"Rank {rank}: Loading client shards from {len(client_dirs)} clients")
    client_shards = load_client_fsdp_shards(client_dirs, rank, world_size)

    if not client_shards:
        logger.error(f"Rank {rank}: No client shards found!")
        return

    # Aggregate the shards.
    logger.info(f"Rank {rank}: Aggregating {len(client_shards)} client shards")
    aggregated_shard = aggregate_shards(client_shards, rank)

    # Save the aggregated shard.
    output_file = output_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
    torch.save(aggregated_shard, output_file)

    # Compute the shard size.
    shard_size_gb = 0
    for key, param in aggregated_shard.items():
        if isinstance(param, DTensor):
            shard_size_gb += param.to_local().numel() * param.to_local().element_size() / (1024**3)
        elif isinstance(param, torch.Tensor):
            shard_size_gb += param.numel() * param.element_size() / (1024**3)
        else:
            shard_size_gb += sys.getsizeof(param) / (1024**3)
    
    logger.info(f"Rank {rank}: Saved aggregated shard to {output_file} (size: {shard_size_gb:.2f} GB)")

    # Synchronize all processes.
    dist.barrier()

    # Only print the summary on rank 0.
    if rank == 0:
        logger.info(f"All {world_size} aggregated FSDP shards created successfully in {output_dir}")
        total_size = 0
        for i in range(world_size):
            shard_file = output_dir / f"model_world_size_{world_size}_rank_{i}.pt"
            if shard_file.exists():
                size_gb = shard_file.stat().st_size / (1024**3)
                total_size += size_gb
                logger.info(f"  Rank {i}: {shard_file.name} ({size_gb:.2f} GB)")
        logger.info(f"Total size: {total_size:.2f} GB")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate client shards directly via the FSDP API")
    parser.add_argument("--client_dirs", type=str, nargs='+', required=True, help="List of client directories")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--n_gpus_per_node", type=int, required=True, help="Number of GPUs per node")
    args = parser.parse_args()

    # Create the output directory.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the aggregated shards.
    create_fsdp_shards(args.client_dirs, output_dir, args.n_gpus_per_node)