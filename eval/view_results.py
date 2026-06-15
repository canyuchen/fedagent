#!/usr/bin/env python3
"""
Script for viewing and converting evaluation/inference results.

Supports converting parquet result files into JSON so they can be inspected
easily, either for a single file or for every parquet file in a directory.
"""

import pandas as pd
import json
import argparse
import os
import numpy as np
from pathlib import Path

def view_parquet_as_json(parquet_path, output_json_path=None, max_rows=None, pretty_print=True):
    """
    Convert a parquet file to JSON for viewing.

    Args:
        parquet_path: path to the parquet file.
        output_json_path: optional path to write the JSON output to.
        max_rows: maximum number of rows to display (optional; None = all rows).
        pretty_print: whether to pretty-print the JSON output.
    """
    try:
        # Read the parquet file.
        df = pd.read_parquet(parquet_path)

        print(f"File: {parquet_path}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("-" * 50)

        # Limit the number of displayed rows.
        if max_rows and len(df) > max_rows:
            df_display = df.head(max_rows)
            print(f"Showing first {max_rows} row(s) (out of {len(df)} total)")
        else:
            df_display = df

        # Convert to JSON, handling numpy arrays.
        json_data = df_display.to_dict('records')

        # Handle numpy arrays and other non-serializable objects.
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        json_data = convert_numpy(json_data)

        if output_json_path:
            # Write to a JSON file.
            with open(output_json_path, 'w', encoding='utf-8') as f:
                if pretty_print:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                else:
                    json.dump(json_data, f, ensure_ascii=False)
            print(f"JSON results saved to: {output_json_path}")
        else:
            # Print directly to the console.
            if pretty_print:
                print(json.dumps(json_data, ensure_ascii=False, indent=2))
            else:
                print(json.dumps(json_data, ensure_ascii=False))

    except Exception as e:
        print(f"Error: {e}")

def view_inference_results(results_dir, max_rows=5, output_json_path=None):
    """
    View every parquet file in an evaluation/inference results directory.

    Args:
        results_dir: directory containing the result files.
        max_rows: maximum number of rows to display per file.
        output_json_path: optional path to write the combined JSON output to.
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Directory does not exist: {results_dir}")
        return

    parquet_files = list(results_path.glob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {results_dir}")
        return

    print(f"Found {len(parquet_files)} parquet file(s):")
    for i, file_path in enumerate(parquet_files, 1):
        print(f"{i}. {file_path.name}")

    print("\n" + "="*60)

    # Inspect each file.
    all_results = {}
    for file_path in parquet_files:
        print(f"\nProcessing: {file_path.name}")
        print("-" * 40)

        try:
            df = pd.read_parquet(file_path)
            print(f"Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            # Show the first few rows.
            if len(df) > 0:
                print(f"\nFirst {min(max_rows, len(df))} row(s):")
                sample_data = df.head(max_rows).to_dict('records')

                # Handle numpy arrays.
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {key: convert_numpy(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    else:
                        return obj

                sample_data = convert_numpy(sample_data)
                print(json.dumps(sample_data, ensure_ascii=False, indent=2))

                # Store into the combined results.
                full_data = df.to_dict('records') if len(df) <= 100 else df.head(100).to_dict('records')
                full_data = convert_numpy(full_data)

                all_results[file_path.stem] = {
                    'file_path': str(file_path),
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'sample_data': sample_data,
                    'full_data': full_data
                }
            else:
                print("File is empty")

        except Exception as e:
            print(f"Error processing file {file_path.name}: {e}")

    # Save the combined JSON results.
    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nCombined JSON results saved to: {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description='View evaluation/inference results')
    parser.add_argument('--file', '-f', help='path to a parquet file')
    parser.add_argument('--dir', '-d', help='path to a results directory')
    parser.add_argument('--output', '-o', help='path to write the JSON output to')
    parser.add_argument('--max-rows', '-m', type=int, default=5, help='maximum number of rows to display')
    parser.add_argument('--no-pretty', action='store_true', help='do not pretty-print the JSON output')

    args = parser.parse_args()

    if args.file:
        # View a single file.
        view_parquet_as_json(
            args.file,
            args.output,
            args.max_rows,
            not args.no_pretty
        )
    elif args.dir:
        # View every file in the directory.
        view_inference_results(
            args.dir,
            args.max_rows,
            args.output
        )
    else:
        # No file or directory was provided.
        print("Please specify a file or directory path")
        print("Usage:")
        print("  python view_results.py --file path/to/file.parquet")
        print("  python view_results.py --dir path/to/results/")
        print("  python view_results.py --dir path/to/results/ --output results.json")

if __name__ == "__main__":
    main()
