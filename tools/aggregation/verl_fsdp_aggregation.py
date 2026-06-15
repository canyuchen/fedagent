#!/usr/bin/env python3
"""Re-aggregate models using VERL's FSDP scheme."""

import os
import sys
import logging
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Repository root (this file lives at tools/aggregation/). Put it on the Python
# path so `import utils...` resolves, and use it to build the example paths below
# relative to the checkout rather than an absolute machine path.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.model_aggregation import ModelAggregator

def create_dummy_fsdp_model(model_config, world_size=2):
    """Create a temporary FSDP model used for saving."""
    from transformers import AutoConfig, AutoModelForCausalLM

    # Load the model configuration.
    config = AutoConfig.from_pretrained(model_config)

    # Create the model.
    model = AutoModelForCausalLM.from_config(config)

    # Initialize FSDP.
    from torch.distributed.device_mesh import init_device_mesh
    device_mesh = init_device_mesh("cuda", (world_size,))

    from torch.distributed.fsdp import FSDP
    from torch.distributed.fsdp.api import ShardingStrategy

    model = FSDP(
        model,
        device_mesh=device_mesh,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
    )
    
    return model

def verl_fsdp_aggregation():
    """Re-aggregate models using VERL's FSDP scheme."""

    # Configure logging.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        logger.info("=" * 60)
        logger.info("Re-aggregating models using VERL's FSDP scheme")
        logger.info("=" * 60)

        # Initialize the distributed environment.
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        logger.info(f"Rank {rank}/{world_size} starting aggregation")

        # Example run directory, built relative to the repository checkout.
        run_name = (
            "fed_Qwen2.5-1.5B-Instruct_webshop_grpo_Qwen2.5-1.5B-Instruct"
            "_total-100_cl-per-rd-2_rd-70_ep-per-cl-1_min-goals-per-cl-100_p-uniform"
        )
        output_dir = REPO_ROOT / "output" / "test" / run_name

        # Mock client results.
        client_results = [
            {
                'client_id': 81,
                'success': True,
                'model_path': str(output_dir / "round_1" / "client_81")
            },
            {
                'client_id': 14,
                'success': True,
                'model_path': str(output_dir / "round_1" / "client_14")
            }
        ]

        # Remove any existing aggregated model.
        aggregated_dir = output_dir / "round_1" / "aggregated"
        if aggregated_dir.exists():
            import shutil
            shutil.rmtree(aggregated_dir)
            logger.info(f"removed the existing aggregated directory: {aggregated_dir}")

        # Aggregate using ModelAggregator.
        aggregator = ModelAggregator()

        # Run the aggregation.
        logger.info("starting aggregation...")
        aggregated_models = aggregator.aggregate_verl_models(
            round_num=1,
            client_results=client_results,
            output_dir=output_dir,
            aggregation_method='fedavg',
            n_gpus_per_node=2
        )
        
        logger.info(f"aggregation complete: {aggregated_models}")

        # Validate the format of the aggregated model.
        if 'actor' in aggregated_models:
            actor_path = Path(aggregated_models['actor'])
            logger.info(f"validating the aggregated actor model: {actor_path}")

            # Check the shard files.
            shard_files = list(actor_path.glob("model_world_size_*_rank_*.pt"))
            logger.info(f"found {len(shard_files)} shard files")

            if shard_files:
                # Check the format of the first shard file.
                checkpoint = torch.load(shard_files[0], weights_only=False)
                first_key = list(checkpoint.keys())[0]
                first_tensor = checkpoint[first_key]
                logger.info(f"type of the first tensor: {type(first_tensor)}")

                if hasattr(first_tensor, 'device_mesh'):
                    logger.info("✅ shard file is in the DTensor format")
                else:
                    logger.warning("⚠️ shard file is not in the DTensor format")

        logger.info("=" * 60)
        logger.info("🎉 VERL FSDP aggregation complete!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ aggregation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Launch via torchrun.
    import subprocess
    import sys
    
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--master_port=29500",
        __file__
    ]
    
    subprocess.run(cmd)