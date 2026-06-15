#!/usr/bin/env python3
"""Check whether the aggregated parameters are the average of the client models.

Usage:
1. Run directly (using the default paths):
   python check_aggregation.py

2. Use command-line arguments:
   python check_aggregation.py --aggregated-dir /path/to/aggregated/checkpoints/global_step_0/actor --client-dirs /path/to/client1/checkpoints/global_step_1/actor /path/to/client2/checkpoints/global_step_1/actor

3. Specify the base directory:
   python check_aggregation.py --base-dir /path/to/your/round_directory

4. Custom check (called from code):
   from check_aggregation import compare_aggregation
   result = compare_aggregation(aggregated_dir, client_dirs)

Examples:
    # Command-line usage
    python check_aggregation.py --aggregated-dir /path/to/aggregated/checkpoints/global_step_0/actor --client-dirs /path/to/client_1/checkpoints/global_step_1/actor /path/to/client_2/checkpoints/global_step_1/actor

    # Usage from code
    aggregated_dir = Path("/path/to/aggregated/checkpoints/global_step_0/actor")
    client_dirs = [
        Path("/path/to/client_1/checkpoints/global_step_1/actor"),
        Path("/path/to/client_2/checkpoints/global_step_1/actor")
    ]
    result = compare_aggregation(aggregated_dir, client_dirs)
"""

import torch
import os
from pathlib import Path
import numpy as np

# Repository root (this file lives at tools/aggregation/). The default --base-dir
# below is built relative to the checkout rather than an absolute machine path.
REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUN_NAME = (
    "fed_Qwen2.5-1.5B-Instruct_webshop_grpo_Qwen2.5-1.5B-Instruct"
    "_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform"
)
_DEFAULT_BASE_DIR = str(REPO_ROOT / "output" / "test" / _DEFAULT_RUN_NAME / "round_1")

def load_model_shards(model_dir):
    """Load the model shard files."""
    model_dir = Path(model_dir)

    # Find all shard files.
    shard_files = list(model_dir.glob("model_world_size_*_rank_*.pt"))
    shard_files.sort()

    if not shard_files:
        print(f"❌ No shard files found in {model_dir}")
        return None

    print(f"Found {len(shard_files)} shard files in {model_dir}")

    # Load all shards.
    all_shards = {}
    for shard_file in shard_files:
        print(f"Loading {shard_file.name}...")
        shard_data = torch.load(shard_file, map_location='cpu', weights_only=False)
        all_shards[shard_file.name] = shard_data

    return all_shards

def calculate_client_differences(client_dirs):
    """Compute the differences between the client models."""
    if len(client_dirs) < 2:
        return {'max_diff': 0.0, 'mean_diff': 0.0}

    # Load all client models.
    client_shards_list = []
    for client_dir in client_dirs:
        client_shards = load_model_shards(client_dir)
        if client_shards:
            client_shards_list.append(client_shards)

    if len(client_shards_list) < 2:
        return {'max_diff': 0.0, 'mean_diff': 0.0}

    # Compute the differences between clients.
    all_diffs = []

    # Use the first client's shard files as the reference.
    first_client_shards = client_shards_list[0]

    for shard_name in first_client_shards.keys():
        # Check whether all clients have the same shard.
        client_shards = []
        for client_shards_dict in client_shards_list:
            if shard_name in client_shards_dict:
                client_shards.append(client_shards_dict[shard_name])

        if len(client_shards) != len(client_dirs):
            continue

        # Get all parameter keys.
        all_keys = set()
        for shard in client_shards:
            all_keys.update(shard.keys())

        # Compute the difference between each pair of clients.
        for i in range(len(client_shards)):
            for j in range(i + 1, len(client_shards)):
                for param_name in all_keys:
                    if param_name in client_shards[i] and param_name in client_shards[j]:
                        param1 = client_shards[i][param_name]
                        param2 = client_shards[j][param_name]
                        
                        if param1.shape == param2.shape:
                            diff = torch.abs(param1 - param2)
                            max_diff = diff.max().item()
                            all_diffs.append(max_diff)
    
    if all_diffs:
        return {
            'max_diff': max(all_diffs),
            'mean_diff': np.mean(all_diffs)
        }
    else:
        return {'max_diff': 0.0, 'mean_diff': 0.0}

def calculate_aggregated_vs_client_differences(aggregated_dir, client_dirs):
    """Compute the difference between the aggregated model and each client model."""
    # Load the aggregated model.
    aggregated_shards = load_model_shards(aggregated_dir)
    if not aggregated_shards:
        return []

    client_diffs = []

    for client_dir in client_dirs:
        # Load the client model.
        client_shards = load_model_shards(client_dir)
        if not client_shards:
            client_diffs.append({'max_diff': 0.0, 'mean_diff': 0.0, 'param_count': 0})
            continue

        all_diffs = []
        param_count = 0

        # Use the aggregated model's shard files as the reference.
        for shard_name in aggregated_shards.keys():
            if shard_name in client_shards:
                aggregated_shard = aggregated_shards[shard_name]
                client_shard = client_shards[shard_name]

                # Get all parameter keys.
                all_keys = set()
                all_keys.update(aggregated_shard.keys())
                all_keys.update(client_shard.keys())

                # Compute the difference for each parameter.
                for param_name in all_keys:
                    if param_name in aggregated_shard and param_name in client_shard:
                        agg_param = aggregated_shard[param_name]
                        client_param = client_shard[param_name]
                        
                        if agg_param.shape == client_param.shape:
                            diff = torch.abs(agg_param - client_param)
                            max_diff = diff.max().item()
                            all_diffs.append(max_diff)
                            param_count += 1
        
        if all_diffs:
            client_diffs.append({
                'max_diff': max(all_diffs),
                'mean_diff': np.mean(all_diffs),
                'param_count': param_count,
                'diff_values': all_diffs
            })
        else:
            client_diffs.append({'max_diff': 0.0, 'mean_diff': 0.0, 'param_count': 0, 'diff_values': []})
    
    return client_diffs

def compare_aggregation(aggregated_dir, client_dirs):
    """Compare the aggregated model against the client models."""
    print("=" * 80)
    print("Checking whether the aggregated parameters are the average of the client models")
    print("=" * 80)

    # Used to accumulate difference statistics.
    diff_stats = {
        'total_params': 0,
        'perfect_match': 0,
        'small_diff': 0,
        'large_diff': 0,
        'max_diff_overall': 0.0,
        'mean_diff_overall': 0.0,
        'diff_values': []
    }
    
    # Load the aggregated model.
    print(f"\n1. Loading the aggregated model: {aggregated_dir}")
    aggregated_shards = load_model_shards(aggregated_dir)
    if not aggregated_shards:
        return False

    # Load the client models.
    client_shards_list = []
    for i, client_dir in enumerate(client_dirs):
        print(f"\n{i+2}. Loading the client model: {client_dir}")
        client_shards = load_model_shards(client_dir)
        if client_shards:
            client_shards_list.append(client_shards)
        else:
            print(f"❌ Failed to load client model from {client_dir}")
            return False
    
    if len(client_shards_list) != len(client_dirs):
        print(f"❌ Only loaded {len(client_shards_list)} out of {len(client_dirs)} client models")
        return False
    
    print(f"\n✅ Successfully loaded aggregated model and {len(client_shards_list)} client models")
    
    # Compare each shard file.
    all_correct = True

    for shard_name in aggregated_shards.keys():
        print(f"\n--- Checking shard: {shard_name} ---")

        aggregated_shard = aggregated_shards[shard_name]

        # Check whether all clients have the same shard.
        client_shards = []
        for client_shards_dict in client_shards_list:
            if shard_name in client_shards_dict:
                client_shards.append(client_shards_dict[shard_name])
            else:
                print(f"❌ Client model missing shard {shard_name}")
                all_correct = False
                break
        
        if len(client_shards) != len(client_dirs):
            continue
        
        # Compute the average of the client models.
        print(f"Computing the average of {len(client_shards)} client models...")

        # Get all parameter keys.
        all_keys = set()
        for shard in client_shards:
            all_keys.update(shard.keys())

        print(f"The shard contains {len(all_keys)} parameters")

        # Compare each parameter.
        param_correct = True
        for param_name in sorted(all_keys):
            if param_name not in aggregated_shard:
                print(f"❌ Parameter {param_name} missing in aggregated model")
                param_correct = False
                continue

            # Collect the client parameters.
            client_params = []
            for shard in client_shards:
                if param_name in shard:
                    client_params.append(shard[param_name])
                else:
                    print(f"❌ Parameter {param_name} missing in client model")
                    param_correct = False
                    break
            
            if len(client_params) != len(client_dirs):
                continue
            
            # Compute the average.
            if len(client_params) == 1:
                expected_param = client_params[0]
            else:
                expected_param = torch.stack(client_params).mean(dim=0)

            aggregated_param = aggregated_shard[param_name]

            # Compare the parameter.
            if aggregated_param.shape != expected_param.shape:
                print(f"❌ Shape mismatch for {param_name}: aggregated {aggregated_param.shape} vs expected {expected_param.shape}")
                param_correct = False
                continue

            # Compute the difference.
            diff = torch.abs(aggregated_param - expected_param)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Update the statistics.
            diff_stats['total_params'] += 1
            diff_stats['diff_values'].append(max_diff)
            diff_stats['max_diff_overall'] = max(diff_stats['max_diff_overall'], max_diff)
            diff_stats['mean_diff_overall'] += mean_diff

            # Set the tolerance.
            tolerance = 1e-6
            if max_diff > tolerance:
                print(f"❌ Parameter {param_name} differs: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
                param_correct = False
                if max_diff > 1e-3:
                    diff_stats['large_diff'] += 1
                else:
                    diff_stats['small_diff'] += 1
            else:
                print(f"✅ Parameter {param_name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
                if max_diff == 0.0:
                    diff_stats['perfect_match'] += 1
                else:
                    diff_stats['small_diff'] += 1
        
        if param_correct:
            print(f"✅ Shard {shard_name} aggregated correctly")
        else:
            print(f"❌ Shard {shard_name} aggregated incorrectly")
            all_correct = False

    # Compute the final statistics.
    if diff_stats['total_params'] > 0:
        diff_stats['mean_diff_overall'] /= diff_stats['total_params']

        # Compute the difference distribution.
        diff_values = np.array(diff_stats['diff_values'])
        diff_percentiles = {
            '50th': np.percentile(diff_values, 50),
            '90th': np.percentile(diff_values, 90),
            '95th': np.percentile(diff_values, 95),
            '99th': np.percentile(diff_values, 99)
        }
        
        # Print the difference-statistics report.
        print("\n" + "=" * 80)
        print("📊 Aggregated-parameter difference statistics report")
        print("=" * 80)
        print(f"Total parameter count: {diff_stats['total_params']}")
        print(f"Perfect match (diff=0): {diff_stats['perfect_match']} ({diff_stats['perfect_match']/diff_stats['total_params']*100:.1f}%)")
        print(f"Small difference (<=1e-6): {diff_stats['small_diff']} ({diff_stats['small_diff']/diff_stats['total_params']*100:.1f}%)")
        print(f"Large difference (>1e-3): {diff_stats['large_diff']} ({diff_stats['large_diff']/diff_stats['total_params']*100:.1f}%)")
        print()
        print("Difference distribution:")
        print(f"  Max difference: {diff_stats['max_diff_overall']:.2e}")
        print(f"  Mean difference: {diff_stats['mean_diff_overall']:.2e}")
        print(f"  Median difference: {diff_percentiles['50th']:.2e}")
        print(f"  90th percentile: {diff_percentiles['90th']:.2e}")
        print(f"  95th percentile: {diff_percentiles['95th']:.2e}")
        print(f"  99th percentile: {diff_percentiles['99th']:.2e}")

        # Show the difference between the aggregated model and each client model.
        print("\n" + "-" * 60)
        print("🔍 Aggregated model vs. each client model")
        print("-" * 60)

        # Compute the difference between the aggregated model and each client.
        client_individual_diffs = calculate_aggregated_vs_client_differences(aggregated_dir, client_dirs)

        for i, (client_dir, client_stats) in enumerate(zip(client_dirs, client_individual_diffs)):
            print(f"\nClient {i+1} ({client_dir.name}):")
            print(f"  Max difference: {client_stats['max_diff']:.2e}")
            print(f"  Mean difference: {client_stats['mean_diff']:.2e}")
            print(f"  Parameter count: {client_stats['param_count']}")

            # Show the difference distribution.
            if 'diff_values' in client_stats and client_stats['diff_values']:
                diff_values = np.array(client_stats['diff_values'])
                print(f"  Difference distribution:")
                print(f"    Median difference: {np.percentile(diff_values, 50):.2e}")
                print(f"    90th percentile: {np.percentile(diff_values, 90):.2e}")
                print(f"    95th percentile: {np.percentile(diff_values, 95):.2e}")
                print(f"    99th percentile: {np.percentile(diff_values, 99):.2e}")

                # Show the difference range.
                print(f"  Difference range:")
                print(f"    Min difference: {np.min(diff_values):.2e}")
                print(f"    Max difference: {np.max(diff_values):.2e}")
                print(f"    Std dev: {np.std(diff_values):.2e}")

        # Show the differences between the client models (for comparison).
        print("\n" + "-" * 60)
        print("🔍 Differences between the client models")
        print("-" * 60)
        if len(client_dirs) >= 2:
            client_diff_stats = calculate_client_differences(client_dirs)
            print(f"Max difference between client models: {client_diff_stats['max_diff']:.2e}")
            print(f"Mean difference between client models: {client_diff_stats['mean_diff']:.2e}")
            print(f"Difference between the aggregated model and the client average: {diff_stats['mean_diff_overall']:.2e}")

            if diff_stats['mean_diff_overall'] < client_diff_stats['mean_diff']:
                print("✅ The aggregated model successfully reduced the inter-client differences")
            else:
                print("⚠️  The aggregated model's difference is large; the aggregation algorithm may need to be checked")
    
    return all_correct

def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Check whether the aggregated parameters are the average of the client models')
    parser.add_argument('--aggregated-dir', type=str,
                       help='Path to the aggregated model directory')
    parser.add_argument('--client-dirs', type=str, nargs='+',
                       help='List of client model directory paths')
    parser.add_argument('--base-dir', type=str,
                       default=_DEFAULT_BASE_DIR,
                       help='Base directory path (used for the default paths)')

    args = parser.parse_args()

    # Set up the paths.
    if args.aggregated_dir and args.client_dirs:
        # Use the command-line arguments.
        aggregated_dir = Path(args.aggregated_dir)
        client_dirs = [Path(d) for d in args.client_dirs]
    else:
        # Use the default paths.
        base_dir = Path(args.base_dir)
        aggregated_dir = base_dir / "aggregated" / "checkpoints" / "global_step_0" / "actor"
        client_dirs = [
            base_dir / "client_14" / "checkpoints" / "global_step_3" / "actor",
            base_dir / "client_81" / "checkpoints" / "global_step_3" / "actor"
        ]

    print(f"Aggregated model directory: {aggregated_dir}")
    print(f"Client model directories:")
    for i, client_dir in enumerate(client_dirs):
        print(f"  {i+1}. {client_dir}")

    # Check whether the directories exist.
    if not aggregated_dir.exists():
        print(f"❌ Aggregated model directory does not exist: {aggregated_dir}")
        return

    for client_dir in client_dirs:
        if not client_dir.exists():
            print(f"❌ Client model directory does not exist: {client_dir}")
            return

    # Run the comparison.
    result = compare_aggregation(aggregated_dir, client_dirs)

    print("\n" + "=" * 80)
    if result:
        print("✅ Aggregated-parameter check passed: the aggregated model is indeed the average of the client models")
    else:
        print("❌ Aggregated-parameter check failed: the aggregated model does not match the average of the client models")
    print("=" * 80)

if __name__ == "__main__":
    main()