#!/usr/bin/env python3
"""
Test the ALFWorld federated-learning data sharding functionality.

Run from the repository root (so ./config/paths.yaml resolves):
    python tests/heterogenous/test_alfworld_fed.py
"""

import os
import sys
import ray
from omegaconf import OmegaConf

# Resolve the vendored verl-agent root via config/paths.yaml (run from repo root),
# then put it on the import path so `agent_system...` resolves.
path_cfg = OmegaConf.load("./config/paths.yaml")
verl_agent_root = os.path.join(path_cfg.project_root, "third_party/verl-agent")
sys.path.append(verl_agent_root)

from agent_system.environments.env_package.alfworld.envs import build_alfworld_envs

def test_alfworld_federated_slicing():
    """Test the ALFWorld federated-learning data sharding functionality."""

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Path to the ALFWorld configuration file
    alf_config_path = os.path.join(
        verl_agent_root,
        "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
    )

    print("=" * 60)
    print("Testing ALFWorld federated-learning data sharding functionality")
    print("=" * 60)

    # Test parameters
    test_cases = [
        {
            "name": "Non-federated mode - train",
            "client_id": None,
            "client_num": None,
            "is_train": True
        },
        {
            "name": "Non-federated mode - validation",
            "client_id": None,
            "client_num": None,
            "is_train": False
        },
        {
            "name": "Federated mode - Client 1/3 - train",
            "client_id": 1,
            "client_num": 3,
            "is_train": True
        },
        {
            "name": "Federated mode - Client 2/3 - train",
            "client_id": 2,
            "client_num": 3,
            "is_train": True
        },
        {
            "name": "Federated mode - Client 3/3 - train",
            "client_id": 3,
            "client_num": 3,
            "is_train": True
        },
        {
            "name": "Federated mode - Client 1/3 - validation",
            "client_id": 1,
            "client_num": 3,
            "is_train": False
        }
    ]

    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print("-" * 40)

        try:
            # Create the environments
            envs = build_alfworld_envs(
                alf_config_path=alf_config_path,
                seed=42,
                env_num=2,
                group_n=1,
                is_train=test_case['is_train'],
                env_kwargs={},
                client_id=test_case['client_id'],
                client_num=test_case['client_num'],
                min_goals_per_client=50,  # Use a smaller minimum count for testing
                val_batch_size=100        # Use a smaller validation set size for testing
            )

            print(f"[OK] Environment created successfully")
            print(f"   Environment type: {type(envs).__name__}")
            print(f"   Number of processes: {envs.num_processes}")
            print(f"   Multi-modal: {envs.multi_modal}")

            # Test environment reset
            try:
                obs, image_obs, infos = envs.reset()
                print(f"[OK] Environment reset successfully")
                print(f"   Number of observations: {len(obs)}")
                print(f"   Image observations: {'yes' if image_obs else 'no'}")
                print(f"   Number of infos: {len(infos)}")
            except Exception as e:
                print(f"[FAIL] Environment reset failed: {e}")

            # Close the environments
            envs.close()
            print(f"[OK] Environment closed successfully")

        except Exception as e:
            print(f"[FAIL] Environment creation failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)

if __name__ == "__main__":
    test_alfworld_federated_slicing()
