#!/usr/bin/env python3
"""
Preference Partition Strategy Simulation

Sweeps the preference-heterogeneity knob to visualise non-IID over the
product-category marginal (PreferencePartition, Dirichlet construction).

NAMING: the paper calls this knob omega (the Dirichlet spread fraction;
larger omega -> higher heterogeneity). This script and the underlying
preference_partition() still use the LEGACY keyword 'tau' for the same
value -- partition_strategy.py aliases omega=tau when omega is unset. Do
NOT confuse this 'tau' with the paper's task-descriptor tau, which is an
unrelated concept; here 'tau' == omega.

Supports multiple datasets: webshop (~6410 train samples) and alfworld
(3553 samples).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
import sys
from omegaconf import OmegaConf

path_cfg = OmegaConf.load("./config/paths.yaml")
print(path_cfg.config.root)

# Add the verl-agent path to import partition_strategy
sys.path.append(os.path.join(path_cfg.project_root, 'third_party/verl-agent'))
from agent_system.environments.partition_strategy import (
    preference_partition, visualize_client_category_distribution
)

def create_synthetic_data(dataset: str = 'webshop') -> List[Dict[str, Any]]:
    """
    Create synthetic data for simulation based on specified dataset

    Args:
        dataset: Dataset type ('webshop' or 'alfworld')

    Returns:
        List of synthetic data samples
    """
    if dataset == 'webshop':
        # WebShop dataset: 6410 samples
        categories = ['beauty', 'electronics', 'fashion', 'garden', 'grocery']
        category_counts = {
            'beauty': 198,
            'electronics': 180,
            'fashion': 5012,  # Will be sampled down to 1002 (20%)
            'garden': 905,
            'grocery': 115
        }
        # Apply fashion sampling (20% as in real data)
        category_counts['fashion'] = int(category_counts['fashion'] * 0.2)  # 1002 samples

    elif dataset == 'alfworld':
        # ALFWorld dataset: 3553 samples (real distribution)
        categories = ['pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                     'pick_and_place_simple', 'pick_two_obj_and_place',
                     'pick_cool_then_place_in_recep', 'look_at_obj_in_light']
        category_counts = {
            'pick_clean_then_place_in_recep': 650,
            'pick_heat_then_place_in_recep': 459,
            'pick_and_place_simple': 790,
            'pick_two_obj_and_place': 813,
            'pick_cool_then_place_in_recep': 533,
            'look_at_obj_in_light': 308
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose 'webshop' or 'alfworld'")

    # Create synthetic data with real category distribution
    data = []
    sample_id = 0

    for category, count in category_counts.items():
        for i in range(count):
            data.append({
                'id': sample_id,
                'category': category,
                'text': f'Sample {sample_id} from {category}',
                'features': np.random.randn(10).tolist()
            })
            sample_id += 1

    # Shuffle the data
    np.random.shuffle(data)

    return data

def simulate_preference_partition(
    data: List[Dict[str, Any]],
    dataset: str = 'webshop',
    client_num: int = 100,
    min_samples_per_client: int = 100,
    tau_values: List[float] = [0.1, 0.5, 0.9],
    save_dir: str = os.path.join(path_cfg.project_root, 'output/heterogenous/preference')
) -> None:
    """
    Simulate preference partition with different tau values

    Args:
        data: Synthetic data
        client_num: Number of clients
        min_samples_per_client: Minimum samples per client
        tau_values: List of preference-heterogeneity values to sweep. The
            keyword is the LEGACY name 'tau'; the paper's symbol for this
            knob is omega (Dirichlet spread fraction, in (0,1); larger ->
            more heterogeneous). Endpoints used in the paper: omega=0.01
            (near-uniform) and omega=0.99 (extreme). Passed straight through
            to preference_partition(tau=...), which aliases it to omega.
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)

    print("="*80)
    print("PREFERENCE PARTITION SIMULATION")
    print("="*80)

    for tau in tau_values:
        print(f"\nSimulating preference partition with tau={tau}")

        # Simulate for each client
        client_data = {}
        for client_id in range(client_num):
            client_data[client_id] = preference_partition(
                data=data,
                client_id=client_id,
                client_num=client_num,
                min_samples_per_client=min_samples_per_client,
                start_idx=0,
                tau=tau,
                data_type='generic',
                fashion_sample_ratio=1.0  # Avoid double-sampling fashion
            )

        # Calculate statistics
        category_stats = {}
        for client_id in range(client_num):
            category_counts = {}
            for item in client_data[client_id]:
                cat = item['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            category_stats[client_id] = category_counts

        # Create visualization
        save_path = os.path.join(save_dir, f'preference_tau_{tau}.png')
        visualize_client_category_distribution(
            data=data,
            client_num=client_num,
            min_samples_per_client=min_samples_per_client,
            strategy='preference',
            category_key='category',
            start_idx=0,
            tau=tau,
            save_path=save_path,
            fashion_sample_ratio=1.0  # Avoid double-sampling fashion
        )

        # Print statistics
        print(f"Tau={tau} Statistics:")
        for client_id in range(min(3, client_num)):  # Show first 3 clients
            total = sum(category_stats[client_id].values())
            print(f"  Client {client_id}: {total} samples")
            for cat, count in category_stats[client_id].items():
                if count > 0:
                    print(f"    {cat}: {count} ({count/total*100:.1f}%)")

def create_preference_comparison_plot(
    data: List[Dict[str, Any]],
    dataset: str = 'webshop',
    client_num: int = 10,
    min_samples_per_client: int = 500,
    tau_values: List[float] = [0.1, 0.3, 0.5],
    save_dir: str = os.path.join(path_cfg.project_root, 'output/heterogenous/preference')
) -> None:
    """
    Create comparison plot for different tau values
    """
    os.makedirs(save_dir, exist_ok=True)

    # Collect statistics for each tau value
    tau_stats = {}

    for tau in tau_values:
        print(f"Collecting statistics for tau={tau}")

        client_sizes = []
        category_diversity = []

        for client_id in range(client_num):
            client_data = preference_partition(
                data=data,
                client_id=client_id,
                client_num=client_num,
                min_samples_per_client=min_samples_per_client,
                start_idx=0,
                tau=tau,
                data_type='generic',
                fashion_sample_ratio=1.0  # Avoid double-sampling fashion
            )

            client_sizes.append(len(client_data))

            # Calculate category diversity (number of unique categories)
            categories = set(item['category'] for item in client_data)
            category_diversity.append(len(categories))

        tau_stats[tau] = {
            'sizes': client_sizes,
            'diversity': category_diversity,
            'mean_size': np.mean(client_sizes),
            'std_size': np.std(client_sizes),
            'mean_diversity': np.mean(category_diversity),
            'std_diversity': np.std(category_diversity)
        }

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Client sizes for different tau values
    ax1 = axes[0, 0]
    for tau, stats in tau_stats.items():
        ax1.hist(stats['sizes'], alpha=0.7, label=f'tau={tau}', bins=10)
    ax1.set_xlabel('Client Size')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Client Size Distribution by Tau')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Category diversity for different tau values
    ax2 = axes[0, 1]
    for tau, stats in tau_stats.items():
        ax2.hist(stats['diversity'], alpha=0.7, label=f'tau={tau}', bins=10)
    ax2.set_xlabel('Number of Categories per Client')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Category Diversity by Tau')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mean vs Std for client sizes
    ax3 = axes[1, 0]
    taus = list(tau_stats.keys())
    means = [tau_stats[tau]['mean_size'] for tau in taus]
    stds = [tau_stats[tau]['std_size'] for tau in taus]

    ax3.plot(taus, means, 'o-', label='Mean Size', linewidth=2, markersize=8)
    ax3.plot(taus, stds, 's-', label='Std Size', linewidth=2, markersize=8)
    ax3.set_xlabel('Tau Value')
    ax3.set_ylabel('Client Size')
    ax3.set_title('Client Size Statistics vs Tau')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Category diversity vs tau
    ax4 = axes[1, 1]
    diversity_means = [tau_stats[tau]['mean_diversity'] for tau in taus]
    diversity_stds = [tau_stats[tau]['std_diversity'] for tau in taus]

    ax4.plot(taus, diversity_means, 'o-', label='Mean Diversity', linewidth=2, markersize=8)
    ax4.plot(taus, diversity_stds, 's-', label='Std Diversity', linewidth=2, markersize=8)
    ax4.set_xlabel('Tau Value')
    ax4.set_ylabel('Category Diversity')
    ax4.set_title('Category Diversity vs Tau')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'preference_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Preference comparison plot saved to: {os.path.join(save_dir, 'preference_comparison.png')}")

def main():
    """
    Main function to run preference partition simulation with command line arguments
    """
    parser = argparse.ArgumentParser(description='Preference Partition Strategy Simulation')
    parser.add_argument('--dataset', type=str, choices=['webshop', 'alfworld'],
                       default='webshop', help='Dataset to use (default: webshop)')
    parser.add_argument('--client_num', type=int, default=100,
                       help='Number of clients (default: 100)')
    parser.add_argument('--min_samples', type=int, default=100,
                       help='Minimum samples per client (default: 100)')
    parser.add_argument('--tau_values', type=float, nargs='+', default=[0.1, 0.5, 0.9],
                       help='Tau values to test (default: [0.1, 0.5, 0.9])')

    args = parser.parse_args()

    print("Starting Preference Partition Simulation")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Client number: {args.client_num}")
    print(f"Min samples per client: {args.min_samples}")
    print(f"Tau values: {args.tau_values}")

    # Create synthetic data
    print("\nCreating synthetic data...")
    data = create_synthetic_data(dataset=args.dataset)
    print(f"Created {len(data)} synthetic samples")
    print("Category distribution:")
    category_counts = {}
    for item in data:
        cat = item['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} samples")

    # Set save directory based on dataset
    save_dir = os.path.join(path_cfg.project_root, f'output/heterogenous/{args.dataset}/preference')

    # Run preference partition simulation
    simulate_preference_partition(
        data=data,
        dataset=args.dataset,
        client_num=args.client_num,
        min_samples_per_client=args.min_samples,
        tau_values=args.tau_values,
        save_dir=save_dir
    )

    # Create comparison plots
    create_preference_comparison_plot(
        data=data,
        dataset=args.dataset,
        client_num=args.client_num,
        min_samples_per_client=args.min_samples,
        tau_values=args.tau_values,
        save_dir=save_dir
    )

    print("\n" + "="*80)
    print("Preference partition simulation completed!")
    print(f"Results saved in '{args.dataset}/preference/' directory")
    print("="*80)

if __name__ == "__main__":
    main()
