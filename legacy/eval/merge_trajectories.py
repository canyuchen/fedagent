#!/usr/bin/env python3
"""
Script for merging trajectory files.

Combines every individual trajectory file produced during evaluation into a
single JSON file (with metadata) for easier downstream analysis.
"""

import json
import os
import glob
import sys
from datetime import datetime


def merge_trajectories(trajectory_dir, output_file):
    """
    Merge trajectory files.

    Args:
        trajectory_dir: directory containing the per-episode trajectory files.
        output_file: path of the merged JSON file to write.
    """
    # Find all trajectory files.
    trajectory_files = glob.glob(os.path.join(trajectory_dir, 'trajectory_*.json'))
    print(f'Found {len(trajectory_files)} trajectory file(s)')

    all_trajectories = []
    for file_path in trajectory_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                trajectory = json.load(f)
                all_trajectories.append(trajectory)
        except Exception as e:
            print(f'Error reading file {file_path}: {e}')

    # Save the merged result.
    summary = {
        'metadata': {
            'total_trajectories': len(all_trajectories),
            'generated_at': datetime.now().isoformat(),
            'source_directory': trajectory_dir
        },
        'trajectories': all_trajectories
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'Trajectory merge complete. {len(all_trajectories)} trajectory(ies) saved to: {output_file}')


def main():
    """Entry point."""
    if len(sys.argv) != 3:
        print("Usage: python merge_trajectories.py <trajectory_dir> <output_file>")
        print("Example: python merge_trajectories.py ./trajectories ./all_trajectories.json")
        sys.exit(1)

    trajectory_dir = sys.argv[1]
    output_file = sys.argv[2]

    # Make sure the output directory exists.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    merge_trajectories(trajectory_dir, output_file)


if __name__ == "__main__":
    main()
