#!/usr/bin/env python3
"""
Hardness Partition Strategy Simulation

Sweeps the hardness-heterogeneity knob to visualise how per-client
task-difficulty (success-rate) mixes spread out (HardnessPartition).

NAMING: the paper calls this 'Hardness' and controls it via the symbol
xi' (xi-prime). NOTE: 'hardness' is NOT a misspelling -- it is just the
lowercased paper term 'Hardness'. Despite its name, 'success_std' is NOT a
standard deviation -- it is the Beta concentration, i.e. the paper's xi'
DIRECTLY (same value, same direction): large success_std == large xi' ==
near-uniform; small success_std == EXTREME heterogeneity. Paper sweep
endpoints: success_std=256 (xi'=256, near-uniform) and success_std=1
(xi'=1, extreme). 'success_std' is the runtime config key, passed down as
hardness_partition(success_std=...).

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
    hardness_partition, visualize_hardness_distribution
)

def create_synthetic_data(dataset: str = 'webshop') -> List[Dict[str, Any]]:
    """
    Create synthetic data for simulation based on real WebShop or ALFWorld categories
    
    Args:
        dataset: Name of the dataset ('webshop' or 'alfworld')
    
    Returns:
        List of synthetic data samples
    """
    data = []
    sample_id = 0
    
    if dataset == 'webshop':
        # Real WebShop categories and their distribution
        category_counts = {
            'beauty': 198,
            'electronics': 180, 
            'fashion': 5012,  # Will be sampled down to 1002 (20%)
            'garden': 905,
            'grocery': 115
        }
        # Apply fashion sampling (20% as in real data)
        category_counts['fashion'] = int(category_counts['fashion'] * 0.2)  # 1002 samples
        categories = list(category_counts.keys())
    elif dataset == 'alfworld':
        # Real ALFWorld task types and their distribution
        category_counts = {
            'pick_clean_then_place_in_recep': 650,
            'pick_heat_then_place_in_recep': 459,
            'pick_and_place_simple': 790,
            'pick_two_obj_and_place': 813,
            'pick_cool_then_place_in_recep': 533,
            'look_at_obj_in_light': 308
        }
        categories = list(category_counts.keys())
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
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

def create_synthetic_trajectories(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create synthetic trajectory data for hardness simulation
    
    Args:
        data: Original data samples
    
    Returns:
        Synthetic trajectory data
    """
    trajectories = []
    
    for i, item in enumerate(data):
        # Create synthetic task_id from item id
        task_id = f"synthetic_task_{item['id']}"
        
        # Assign a synthetic per-item success probability. NOTE: the checks
        # below key on the literal substrings 'category_0'/'category_1',
        # which the real category names produced by create_synthetic_data()
        # (beauty/electronics/... or the ALFWorld task types) never contain.
        # In practice every item therefore falls through to the 0.3 branch,
        # i.e. a uniform synthetic success rate. This placeholder logic is
        # only used by create_hardness_comparison_plot()'s synthetic path; it
        # is NOT used for the real-trajectory hardness results.
        category = item['category']
        if 'category_0' in category:
            success_prob = 0.8  # (dead branch: no real category matches)
        elif 'category_1' in category:
            success_prob = 0.6  # (dead branch: no real category matches)
        else:
            success_prob = 0.3  # the branch actually taken for all items
        
        success = np.random.random() < success_prob
        
        trajectory = {
            "task_info": {
                "task_id": task_id
            },
            "traj_info": {
                "success": success
            }
        }
        trajectories.append(trajectory)
    
    return {"trajectories": trajectories}

def simulate_hardness_partition(
    data: List[Dict[str, Any]],
    dataset: str = 'webshop',
    client_num: int = 10,
    min_samples_per_client: int = 500,
    success_std_values: List[float] = [0.05, 0.15, 0.25],
    save_dir: str = os.path.join(path_cfg.project_root, 'output/heterogenous/hardness')
) -> None:
    """
    Simulate hardness partition with different success_std values
    
    Args:
        data: Synthetic data
        dataset: Name of the dataset ('webshop' or 'alfworld')
        client_num: Number of clients
        min_samples_per_client: Minimum samples per client
        success_std_values: List of hardness-heterogeneity values to sweep.
            'success_std' is misleadingly named: it is NOT a standard
            deviation but the Beta concentration, i.e. the paper's Hardness
            symbol xi' DIRECTLY (large success_std -> large xi' -> near-uniform;
            small success_std -> extreme heterogeneity). Paper endpoints: 256
            (near-uniform) and 1 (extreme).
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*80)
    print("HARDNESS PARTITION SIMULATION")
    print("="*80)
    
    # Use real trajectory data instead of synthetic
    if dataset == 'webshop':
        trajectories_file = os.path.join(path_cfg.project_root, 'output/inference/all_trajectories.json')
    elif dataset == 'alfworld':
        trajectories_file = os.path.join(path_cfg.project_root, 'output/inference/all_trajectories_alfworld.json')
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Load and analyze real trajectory data
    with open(trajectories_file, 'r') as f:
        trajectories_data = json.load(f)
    
    success_count = sum(1 for traj in trajectories_data['trajectories'] if traj['traj_info']['success'])
    total_count = len(trajectories_data['trajectories'])
    success_rate = success_count / total_count * 100
    
    print(f"Using real {dataset.upper()} trajectory data:")
    print(f"  Total trajectories: {total_count}")
    print(f"  Successful trajectories: {success_count}")
    print(f"  Success rate: {success_rate:.1f}%")
    
    for success_std in success_std_values:
        print(f"\nSimulating hardness partition with success_std={success_std}")
        
        # Simulate for each client
        client_data = {}
        for client_id in range(client_num):
            client_data[client_id] = hardness_partition(
                data=data,
                client_id=client_id,
                client_num=client_num,
                min_samples_per_client=min_samples_per_client,
                start_idx=0,
                trajectories_file=trajectories_file,
                success_std=success_std,
                data_type=dataset
            )
        
        # Calculate statistics
        client_sizes = [len(client_data[client_id]) for client_id in range(client_num)]
        
        # Create visualization
        save_path = os.path.join(save_dir, f'hardness_success_std_{success_std}.png')
        if dataset == 'alfworld':
            from agent_system.environments.partition_strategy import visualize_hardness_distribution_alfworld
            visualize_hardness_distribution_alfworld(
                data=data,
                client_num=client_num,
                min_samples_per_client=min_samples_per_client,
                start_idx=0,
                success_std=success_std,
                trajectories_file=trajectories_file,
                save_path=save_path
            )
        else:
            visualize_hardness_distribution(
                data=data,
                client_num=client_num,
                min_samples_per_client=min_samples_per_client,
                start_idx=0,
                success_std=success_std,
                trajectories_file=trajectories_file,
                save_path=save_path
            )
        
        # Print statistics
        print(f"Success_std={success_std} Statistics:")
        print(f"  Mean client size: {np.mean(client_sizes):.1f}")
        print(f"  Std client size: {np.std(client_sizes):.1f}")
        print(f"  Min client size: {np.min(client_sizes)}")
        print(f"  Max client size: {np.max(client_sizes)}")
        print(f"  Client sizes: {client_sizes}")

def create_hardness_comparison_plot(
    data: List[Dict[str, Any]],
    client_num: int = 10,
    min_samples_per_client: int = 500,
    success_std_values: List[float] = [0.05, 0.15, 0.25],
    save_dir: str = os.path.join(path_cfg.project_root, 'output/heterogenous/hardness')
) -> None:
    """
    Create comparison plot for different success_std values
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create synthetic trajectory data
    trajectories_data = create_synthetic_trajectories(data)
    trajectories_file = os.path.join(save_dir, 'synthetic_trajectories.json')
    with open(trajectories_file, 'w') as f:
        json.dump(trajectories_data, f, indent=2)
    
    # Collect statistics for each success_std value
    success_std_stats = {}
    
    for success_std in success_std_values:
        print(f"Collecting statistics for success_std={success_std}")
        
        client_sizes = []
        success_rates = []
        
        for client_id in range(client_num):
            client_data = hardness_partition(
                data=data,
                client_id=client_id,
                client_num=client_num,
                min_samples_per_client=min_samples_per_client,
                start_idx=0,
                trajectories_file=trajectories_file,
                success_std=success_std
            )
            
            client_sizes.append(len(client_data))
            
            # Calculate success rate for this client
            # Count how many items have success=True in trajectories
            client_success_count = 0
            for item in client_data:
                task_id = f"synthetic_task_{item['id']}"
                for traj in trajectories_data['trajectories']:
                    if traj['task_info']['task_id'] == task_id:
                        if traj['traj_info']['success']:
                            client_success_count += 1
                        break
            
            success_rate = client_success_count / len(client_data) if len(client_data) > 0 else 0
            success_rates.append(success_rate)
        
        success_std_stats[success_std] = {
            'sizes': client_sizes,
            'success_rates': success_rates,
            'mean_size': np.mean(client_sizes),
            'std_size': np.std(client_sizes),
            'mean_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates),
            'min_size': np.min(client_sizes),
            'max_size': np.max(client_sizes)
        }
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Client sizes for different success_std values
    ax1 = axes[0, 0]
    for success_std, stats in success_std_stats.items():
        ax1.hist(stats['sizes'], alpha=0.7, label=f'success_std={success_std}', bins=10)
    ax1.set_xlabel('Client Size', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=20, fontweight='bold')
    ax1.set_title('')
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot 2: Success rates for different success_std values
    ax2 = axes[0, 1]
    for success_std, stats in success_std_stats.items():
        # success_rates already in [0,1]
        ax2.hist(stats['success_rates'], alpha=0.7, label=f'success_std={success_std}', bins=10)
    ax2.set_xlabel('Success Rate', fontsize=15, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=15, fontweight='bold')
    ax2.set_title('')
    ax2.legend(fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot 3: Mean vs Std for client sizes
    ax3 = axes[1, 0]
    success_stds = list(success_std_stats.keys())
    means = [success_std_stats[std]['mean_size'] for std in success_stds]
    stds = [success_std_stats[std]['std_size'] for std in success_stds]
    
    ax3.plot(success_stds, means, 'o-', label='Mean Size', linewidth=2, markersize=8)
    ax3.plot(success_stds, stds, 's-', label='Std Size', linewidth=2, markersize=8)
    ax3.set_xlabel('Success Std Parameter', fontsize=20, fontweight='bold')
    ax3.set_ylabel('Client Size', fontsize=20, fontweight='bold')
    ax3.set_title('')
    ax3.legend(fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot 4: Success rate vs success_std
    ax4 = axes[1, 1]
    success_rate_means = [success_std_stats[std]['mean_success_rate'] for std in success_stds]
    success_rate_stds = [success_std_stats[std]['std_success_rate'] for std in success_stds]
    
    ax4.plot(success_stds, success_rate_means, 'o-', label='Mean Success Rate', linewidth=2, markersize=8)
    ax4.plot(success_stds, success_rate_stds, 's-', label='Std Success Rate', linewidth=2, markersize=8)
    ax4.set_xlabel('Success Std Parameter', fontsize=20, fontweight='bold')
    ax4.set_ylabel('Success Rate', fontsize=20, fontweight='bold')
    ax4.set_title('')
    ax4.legend(fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    ax4.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    comparison_png = os.path.join(save_dir, 'hardness_comparison.png')
    comparison_pdf = os.path.join(save_dir, 'hardness_comparison.pdf')
    plt.savefig(comparison_png, dpi=300, bbox_inches='tight')
    plt.savefig(comparison_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"Hardness comparison plot saved to: {comparison_png}")
    print(f"Hardness comparison PDF saved to: {comparison_pdf}")

def main():
    """
    Main function to run hardness partition simulation with command-line arguments
    """
    parser = argparse.ArgumentParser(description="Run Hardness Partition Simulation for different datasets.")
    parser.add_argument('--dataset', type=str, default='webshop', choices=['webshop', 'alfworld'],
                        help='Dataset to use for simulation (webshop or alfworld)')
    parser.add_argument('--client_num', type=int, default=100,
                        help='Number of clients for the simulation')
    parser.add_argument('--min_samples', type=int, default=100,
                        help='Minimum samples per client')
    parser.add_argument('--success_std', nargs='+', type=float, default=[1, 256],
                        help='List of success_std values for heterogeneity')
    
    args = parser.parse_args()

    print(f"Starting Hardness Partition Simulation for {args.dataset.upper()} Dataset")
    print("="*80)
    
    # Create synthetic data
    print(f"Creating synthetic data for {args.dataset}...")
    data = create_synthetic_data(dataset=args.dataset)
    print(f"Created {len(data)} synthetic samples for {args.dataset}")
    print("Category distribution:")
    category_counts = {}
    for item in data:
        cat = item['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} samples ({count/len(data)*100:.1f}%)")
    
    # Determine save directory based on dataset
    base_save_dir = os.path.join(path_cfg.project_root, 'output/heterogenous')
    dataset_save_dir = os.path.join(base_save_dir, args.dataset, 'hardness')
    os.makedirs(dataset_save_dir, exist_ok=True)
    
    # Run hardness partition simulation
    simulate_hardness_partition(
        data=data,
        dataset=args.dataset,
        client_num=args.client_num,
        min_samples_per_client=args.min_samples,
        success_std_values=args.success_std,
        save_dir=dataset_save_dir
    )
    
    # Create comparison plots
    create_hardness_comparison_plot(
        data=data,
        client_num=args.client_num,
        min_samples_per_client=args.min_samples,
        success_std_values=args.success_std,
        save_dir=dataset_save_dir
    )
    
    print("\n" + "="*80)
    print(f"Hardness partition simulation for {args.dataset.upper()} completed!")
    print(f"Results saved in '{dataset_save_dir}' directory")
    print("="*80)

if __name__ == "__main__":
    main()