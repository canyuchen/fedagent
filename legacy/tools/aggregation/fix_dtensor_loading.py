#!/usr/bin/env python3
"""Fix DTensor loading issues by initializing the distributed environment correctly."""

import os
import sys
import logging
from pathlib import Path
import torch
import torch.distributed as dist

# Vestigial sys.path insert. NOTE: Path(__file__).parent is this file's own
# directory (tools/aggregation/), not the repository root, so this does NOT put
# the repo on the import path. It is harmless because this script only imports
# stdlib/torch and never imports repo packages (e.g. `utils.*`). The real repo
# root is REPO_ROOT below (parents[2]), which is used solely to build the example
# checkpoint paths in test_dtensor_loading(), not for imports.
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Repository root (this file lives at tools/aggregation/), used to build the
# example paths below relative to the checkout rather than an absolute machine path.
REPO_ROOT = Path(__file__).resolve().parents[2]

def init_distributed():
    """Initialize the distributed environment."""
    if not dist.is_initialized():
        # Set the environment variables.
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

        # Initialize the process group.
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

def load_dtensor_checkpoint(checkpoint_path):
    """Load a DTensor checkpoint correctly."""
    try:
        # Initialize the distributed environment.
        init_distributed()

        # Load the checkpoint.
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

        # Check the type of the first tensor.
        first_key = list(checkpoint.keys())[0]
        first_tensor = checkpoint[first_key]

        print(f"type of the first tensor: {type(first_tensor)}")

        if hasattr(first_tensor, 'device_mesh'):
            print("✅ is in the DTensor format")
            return checkpoint
        else:
            print("⚠️ is not in the DTensor format")
            return checkpoint

    except Exception as e:
        print(f"❌ load failed: {e}")
        return None

def test_dtensor_loading():
    """Test DTensor loading."""

    # Configure logging.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        logger.info("=" * 60)
        logger.info("Testing DTensor loading")
        logger.info("=" * 60)

        # Example run directory, built relative to the repository checkout.
        run_name = (
            "fed_Qwen2.5-1.5B-Instruct_webshop_grpo_Qwen2.5-1.5B-Instruct"
            "_total-100_cl-per-rd-2_rd-70_ep-per-cl-1_min-goals-per-cl-100_p-uniform"
        )
        run_dir = REPO_ROOT / "output" / "test" / run_name

        # Test the original shard file.
        original_path = str(
            run_dir / "round_1" / "client_81" / "checkpoints"
            / "global_step_1" / "actor" / "model_world_size_2_rank_0.pt"
        )

        logger.info(f"testing the original shard file: {original_path}")
        checkpoint = load_dtensor_checkpoint(original_path)

        if checkpoint:
            logger.info("✅ original shard file loaded successfully")
        else:
            logger.error("❌ original shard file failed to load")

        # Test the aggregated shard file.
        aggregated_path = str(
            run_dir / "round_1" / "aggregated" / "checkpoints"
            / "global_step_0" / "actor" / "model_world_size_2_rank_0.pt"
        )

        if os.path.exists(aggregated_path):
            logger.info(f"testing the aggregated shard file: {aggregated_path}")
            checkpoint = load_dtensor_checkpoint(aggregated_path)

            if checkpoint:
                logger.info("✅ aggregated shard file loaded successfully")
            else:
                logger.error("❌ aggregated shard file failed to load")
        else:
            logger.warning("aggregated shard file does not exist")

        logger.info("=" * 60)
        logger.info("🎉 DTensor loading test complete!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dtensor_loading()