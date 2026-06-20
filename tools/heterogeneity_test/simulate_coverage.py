#!/usr/bin/env python3
"""
Coverage Partition Strategy Simulation

Sweeps the coverage-heterogeneity knob to visualise how per-client pool
sizes spread out (CoveragePartition).

NAMING: the paper controls Coverage via the symbol xi (a Beta concentration;
LARGER xi = LOWER variance = MORE UNIFORM). Despite its name, the config key
'size_std' is NOT a standard deviation -- it IS that Beta concentration and
maps to xi DIRECTLY (same value, same direction): large size_std == large xi
== near-uniform, small size_std == small xi == EXTREME heterogeneity. Paper
sweep endpoints: size_std=256 (xi=256, near-uniform) and size_std=1 (xi=1,
extreme). 'size_std' is the runtime config key, passed down as
coverage_partition(dispersion_s=size_std).

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
    coverage_partition, visualize_coverage_sample_distribution
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
        # ALFWorld dataset: 3553 samples
        categories = ['pick_and_place', 'stack_objects', 'put_in_receptacle', 'clean', 'heat']
        category_counts = {
            'pick_and_place': 1200,
            'stack_objects': 800,
            'put_in_receptacle': 1000,
            'clean': 400,
            'heat': 153
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

def simulate_coverage_partition(
    data: List[Dict[str, Any]],
    dataset: str = 'webshop',
    client_num: int = 10,
    min_samples_per_client: int = 500,
    size_std_values: List[float] = [5, 15, 25],
    save_dir: str = os.path.join(path_cfg.project_root, 'output/heterogenous/coverage')
) -> None:
    """
    Simulate coverage partition with different size_std values
    
    Args:
        data: Synthetic data
        client_num: Number of clients
        min_samples_per_client: Minimum samples per client
        size_std_values: List of coverage-heterogeneity values to sweep.
            'size_std' is misleadingly named: it is NOT a standard deviation
            but the Beta concentration, i.e. the paper's Coverage symbol xi
            DIRECTLY (large size_std -> large xi -> near-uniform; small
            size_std -> extreme heterogeneity). Paper endpoints: 256
            (near-uniform) and 1 (extreme). Passed through as
            coverage_partition(dispersion_s=size_std).
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*80)
    print("COVERAGE PARTITION SIMULATION")
    print("="*80)
    
    for size_std in size_std_values:
        print(f"\nSimulating coverage partition with size_std={size_std}")
        
        # Simulate for each client
        client_data = {}
        for client_id in range(client_num):
            client_data[client_id] = coverage_partition(
                data=data,
                client_id=client_id,
                client_num=client_num,
                min_samples_per_client=min_samples_per_client,
                start_idx=0,
                dispersion_s=size_std
            )
        
        # Calculate statistics
        client_sizes = [len(client_data[client_id]) for client_id in range(client_num)]
        
        # Create visualization
        save_path = os.path.join(save_dir, f'coverage_size_std_{size_std}.png')
        visualize_coverage_sample_distribution(
            data=data,
            client_num=client_num,
            min_samples_per_client=min_samples_per_client,
            start_idx=0,
            dispersion_s=size_std,
            save_path=save_path
        )
        
        # Print statistics
        print(f"Size_std={size_std} Statistics:")
        print(f"  Mean client size: {np.mean(client_sizes):.1f}")
        print(f"  Std client size: {np.std(client_sizes):.1f}")
        print(f"  Min client size: {np.min(client_sizes)}")
        print(f"  Max client size: {np.max(client_sizes)}")
        print(f"  Client sizes: {client_sizes}")

def create_coverage_comparison_plot(
    data: List[Dict[str, Any]],
    dataset: str = 'webshop',
    client_num: int = 10,
    min_samples_per_client: int = 500,
    size_std_values: List[float] = [5, 15, 25],
    save_dir: str = os.path.join(path_cfg.project_root, 'output/heterogenous/coverage')
) -> None:
    """
    Create comparison plot for different size_std values
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect statistics for each size_std value
    size_std_stats = {}
    
    for size_std in size_std_values:
        print(f"Collecting statistics for size_std={size_std}")
        
        client_sizes = []
        overlap_ratios = []
        
        for client_id in range(client_num):
            client_data = coverage_partition(
                data=data,
                client_id=client_id,
                client_num=client_num,
                min_samples_per_client=min_samples_per_client,
                start_idx=0,
                dispersion_s=size_std
            )
            
            client_sizes.append(len(client_data))
        
        # Calculate overlap statistics
        # For coverage partition, we can estimate overlap by looking at total assignments vs unique samples
        total_assignments = sum(client_sizes)
        unique_samples = len(data)
        estimated_overlap = total_assignments / unique_samples if unique_samples > 0 else 1.0
        overlap_ratios.append(estimated_overlap)
        
        size_std_stats[size_std] = {
            'sizes': client_sizes,
            'overlap_ratio': estimated_overlap,
            'mean_size': np.mean(client_sizes),
            'std_size': np.std(client_sizes),
            'min_size': np.min(client_sizes),
            'max_size': np.max(client_sizes),
            'total_assignments': total_assignments
        }
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Client sizes for different size_std values
    ax1 = axes[0, 0]
    for size_std, stats in size_std_stats.items():
        ax1.hist(stats['sizes'], alpha=0.7, label=f'size_std={size_std}', bins=10)
    ax1.set_xlabel('Client Size', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=20, fontweight='bold')
    ax1.set_title('Client Size Distribution by Size Std', fontsize=22, fontweight='bold')
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)  # Set y-axis display range
    ax1.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot 2: Box plot comparison
    ax2 = axes[0, 1]
    sizes_data = [stats['sizes'] for stats in size_std_stats.values()]
    size_std_labels = [f'std={std}' for std in size_std_stats.keys()]
    ax2.boxplot(sizes_data, labels=size_std_labels)
    ax2.set_xlabel('Size Std Value', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Client Size', fontsize=20, fontweight='bold')
    ax2.set_title('Client Size Box Plot Comparison', fontsize=22, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1000)  # Set y-axis display range
    ax2.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot 3: Mean vs Std for client sizes
    ax3 = axes[1, 0]
    size_stds = list(size_std_stats.keys())
    means = [size_std_stats[std]['mean_size'] for std in size_stds]
    stds = [size_std_stats[std]['std_size'] for std in size_stds]
    
    ax3.plot(size_stds, means, 'o-', label='Mean Size', linewidth=2, markersize=8)
    ax3.plot(size_stds, stds, 's-', label='Std Size', linewidth=2, markersize=8)
    ax3.set_xlabel('Size Std Parameter', fontsize=20, fontweight='bold')
    ax3.set_ylabel('Client Size', fontsize=20, fontweight='bold')
    ax3.set_title('Client Size Statistics vs Size Std', fontsize=22, fontweight='bold')
    ax3.legend(fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1000)  # Set y-axis display range
    ax3.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot 4: Coverage efficiency (total assignments vs unique samples)
    ax4 = axes[1, 1]
    total_assignments = [size_std_stats[std]['total_assignments'] for std in size_stds]
    unique_samples = len(data)
    efficiency = [unique_samples / total for total in total_assignments]
    
    ax4.plot(size_stds, efficiency, 'o-', label='Coverage Efficiency', linewidth=2, markersize=8)
    ax4.set_xlabel('Size Std Parameter', fontsize=20, fontweight='bold')
    ax4.set_ylabel('Coverage Efficiency (unique/total)', fontsize=20, fontweight='bold')
    ax4.set_title('Coverage Efficiency vs Size Std', fontsize=22, fontweight='bold')
    ax4.legend(fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'coverage_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Coverage comparison plot saved to: {os.path.join(save_dir, 'coverage_comparison.png')}")

def main():
    """
    Main function to run coverage partition simulation with command line arguments
    """
    parser = argparse.ArgumentParser(description='Coverage Partition Strategy Simulation')
    parser.add_argument('--dataset', type=str, choices=['webshop', 'alfworld'], 
                       default='webshop', help='Dataset to use (default: webshop)')
    parser.add_argument('--client_num', type=int, default=100, 
                       help='Number of clients (default: 100)')
    parser.add_argument('--min_samples', type=int, default=80, 
                       help='Minimum samples per client (default: 80)')
    parser.add_argument('--size_std', type=float, nargs='+', default=[1, 256], 
                       help='Size standard deviation values (default: [1, 256])')
    
    args = parser.parse_args()
    
    print("Starting Coverage Partition Simulation")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Client number: {args.client_num}")
    print(f"Min samples per client: {args.min_samples}")
    print(f"Size std values: {args.size_std}")
    
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
    save_dir = os.path.join(path_cfg.project_root, f'output/heterogenous/{args.dataset}/coverage')
    
    # Run coverage partition simulation
    simulate_coverage_partition(
        data=data,
        dataset=args.dataset,
        client_num=args.client_num,
        min_samples_per_client=args.min_samples,
        size_std_values=args.size_std,
        save_dir=save_dir
    )
    
    # Create comparison plots
    create_coverage_comparison_plot(
        data=data,
        dataset=args.dataset,
        client_num=args.client_num,
        min_samples_per_client=args.min_samples,
        size_std_values=args.size_std,
        save_dir=save_dir
    )
    
    print("\n" + "="*80)
    print("Coverage partition simulation completed!")
    print(f"Results saved in '{args.dataset}/coverage/' directory")
    print("="*80)

if __name__ == "__main__":
    main()