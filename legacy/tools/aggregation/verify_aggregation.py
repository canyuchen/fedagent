#!/usr/bin/env python3
"""Verify that the federated aggregated model is the parameter average of two client models."""

import torch
import os
import sys
from pathlib import Path
import numpy as np

# Repository root (this file lives at tools/aggregation/). The example model
# paths in main() are built relative to the checkout rather than an absolute
# machine path.
REPO_ROOT = Path(__file__).resolve().parents[2]

def load_model_weights(model_path):
    """Load the model weights."""
    try:
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            state_dict = torch.load(model_path, map_location='cpu')
            print(f"Model loaded successfully. Keys: {len(state_dict.keys())}")
            return state_dict
        else:
            print(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def compare_models(aggregated_path, client1_path, client2_path, tolerance=1e-6):
    """Compare the aggregated model against the average of two client models."""

    print("=" * 80)
    print("Federated-learning model aggregation verification")
    print("=" * 80)

    # Load the models.
    print("\n1. Loading models...")
    aggregated_model = load_model_weights(aggregated_path)
    client1_model = load_model_weights(client1_path)
    client2_model = load_model_weights(client2_path)

    if aggregated_model is None or client1_model is None or client2_model is None:
        print("❌ Could not load all models; verification failed")
        return False, None, None, None

    # First check whether there are any differences between the client models.
    print("\n1.5. Checking whether the client models differ...")
    client_differences = []
    client_max_diff = 0.0
    client_max_diff_key = None

    for key in client1_model.keys():
        if key in client2_model:
            client1_weight = client1_model[key]
            client2_weight = client2_model[key]

            if client1_weight.shape != client2_weight.shape:
                print(f"❌ Client model shapes differ: {key} - client1: {client1_weight.shape}, client2: {client2_weight.shape}")
                return False, None, None, None

            # Compute the difference.
            diff = torch.abs(client1_weight - client2_weight)
            max_diff_tensor = torch.max(diff)
            mean_diff = torch.mean(diff)

            if max_diff_tensor.item() > client_max_diff:
                client_max_diff = max_diff_tensor.item()
                client_max_diff_key = key

            client_differences.append({
                'key': key,
                'max_diff': max_diff_tensor.item(),
                'mean_diff': mean_diff.item(),
                'shape': client1_weight.shape
            })

    print(f"Max difference between client models: {client_max_diff:.2e} (key: {client_max_diff_key})")

    if client_max_diff < 1e-8:
        print("⚠️  Warning: the client models are almost identical! This may mean:")
        print("   - both clients used the same model")
        print("   - training did not produce enough divergence")
        print("   - there is a problem with model saving")
    else:
        print("✅ The client models differ significantly; training is healthy")

    # Show the top 5 largest differences between the client models.
    client_differences.sort(key=lambda x: x['max_diff'], reverse=True)
    print(f"\nTop 5 largest differences between client models:")
    for i, diff in enumerate(client_differences[:5]):
        print(f"  {i+1}. {diff['key']:50s} | max diff: {diff['max_diff']:.2e} | mean diff: {diff['mean_diff']:.2e}")

    # Summarize the client-model differences.
    client_max_diffs = [d['max_diff'] for d in client_differences]
    print(f"\nClient-model difference statistics:")
    print(f"  Total parameter count: {len(client_differences)}")
    print(f"  Max difference: {max(client_max_diffs):.2e}")
    print(f"  Mean of max differences: {np.mean(client_max_diffs):.2e}")
    print(f"  Number of parameters with diff > 1e-6: {sum(1 for d in client_max_diffs if d > 1e-6)}")
    print(f"  Number of parameters with diff > 1e-5: {sum(1 for d in client_max_diffs if d > 1e-5)}")
    print(f"  Number of parameters with diff > 1e-4: {sum(1 for d in client_max_diffs if d > 1e-4)}")

    if client_max_diff < 1e-8:
        print("\n❌ The client models differ too little to run a meaningful federated-learning verification")
        return False, None, None, None

    # Check whether the model keys match.
    print("\n2. Checking the model structure...")
    aggregated_keys = set(aggregated_model.keys())
    client1_keys = set(client1_model.keys())
    client2_keys = set(client2_model.keys())
    
    if aggregated_keys != client1_keys or aggregated_keys != client2_keys:
        print("❌ Model structures do not match")
        print(f"Aggregated keys: {len(aggregated_keys)}")
        print(f"Client1 keys: {len(client1_keys)}")
        print(f"Client2 keys: {len(client2_keys)}")
        
        # Show the differences.
        missing_in_client1 = aggregated_keys - client1_keys
        missing_in_client2 = aggregated_keys - client2_keys
        extra_in_client1 = client1_keys - aggregated_keys
        extra_in_client2 = client2_keys - aggregated_keys
        
        if missing_in_client1:
            print(f"Missing in client1: {missing_in_client1}")
        if missing_in_client2:
            print(f"Missing in client2: {missing_in_client2}")
        if extra_in_client1:
            print(f"Extra in client1: {extra_in_client1}")
        if extra_in_client2:
            print(f"Extra in client2: {extra_in_client2}")
        
        return False, None, None, None
    
    print("✅ Model structures match")

    # Compute the average of the client models.
    print("\n3. Computing the client-model average...")
    averaged_model = {}
    
    for key in aggregated_keys:
        if key in client1_model and key in client2_model:
            # Compute the average of the two client models.
            client1_weight = client1_model[key]
            client2_weight = client2_model[key]

            # Make sure the shapes match.
            if client1_weight.shape != client2_weight.shape:
                print(f"❌ Shapes do not match: {key} - client1: {client1_weight.shape}, client2: {client2_weight.shape}")
                return False, None, None, None

            # Compute the average.
            averaged_weight = (client1_weight + client2_weight) / 2.0
            averaged_model[key] = averaged_weight
        else:
            print(f"❌ Key {key} is missing from the client models")
            return False, None, None, None

    print("✅ Client-model average computed")

    # Compare the aggregated model against the averaged model.
    print("\n4. Comparing the aggregated model against the averaged model...")
    differences = []
    max_diff = 0.0
    max_diff_key = None
    
    for key in aggregated_keys:
        aggregated_weight = aggregated_model[key]
        averaged_weight = averaged_model[key]
        
        # Compute the difference.
        diff = torch.abs(aggregated_weight - averaged_weight)
        max_diff_tensor = torch.max(diff)
        mean_diff = torch.mean(diff)

        if max_diff_tensor.item() > max_diff:
            max_diff = max_diff_tensor.item()
            max_diff_key = key
        
        differences.append({
            'key': key,
            'max_diff': max_diff_tensor.item(),
            'mean_diff': mean_diff.item(),
            'shape': aggregated_weight.shape
        })
    
    # Check whether it is within tolerance.
    print(f"\n5. Verification result:")
    print(f"Max difference: {max_diff:.2e} (key: {max_diff_key})")
    print(f"Tolerance threshold: {tolerance:.2e}")

    if max_diff <= tolerance:
        print("✅ The aggregated model matches the client-model average!")
        success = True
    else:
        print("❌ The aggregated model does not match the client-model average!")
        success = False

    # Show the top 10 largest differences.
    print(f"\n6. Top 10 largest differences:")
    differences.sort(key=lambda x: x['max_diff'], reverse=True)
    for i, diff in enumerate(differences[:10]):
        print(f"  {i+1:2d}. {diff['key']:50s} | max diff: {diff['max_diff']:.2e} | mean diff: {diff['mean_diff']:.2e} | shape: {diff['shape']}")

    # Statistics.
    print(f"\n7. Statistics:")
    max_diffs = [d['max_diff'] for d in differences]
    mean_diffs = [d['mean_diff'] for d in differences]

    print(f"  Total parameter count: {len(differences)}")
    print(f"  Max difference: {max(max_diffs):.2e}")
    print(f"  Mean of max differences: {np.mean(max_diffs):.2e}")
    print(f"  Mean difference: {np.mean(mean_diffs):.2e}")
    print(f"  Number of parameters with diff > 1e-6: {sum(1 for d in max_diffs if d > 1e-6)}")
    print(f"  Number of parameters with diff > 1e-5: {sum(1 for d in max_diffs if d > 1e-5)}")
    print(f"  Number of parameters with diff > 1e-4: {sum(1 for d in max_diffs if d > 1e-4)}")
    
    return success, aggregated_model, client1_model, client2_model

def main():
    """Entry point."""
    # Example run directory, built relative to the repository checkout.
    run_name = (
        "fed_Llama-3.2-1B-Instruct_webshop_grpo_Llama-3.2-1B-Instruct"
        "_total-100_cl-per-rd-2_rd-70_ep-per-cl-3_min-goals-per-cl-100_p-uniform"
    )
    round_dir = REPO_ROOT / "output" / "test" / run_name / "round_1"

    # Model paths.
    aggregated_path = str(round_dir / "aggregated" / "checkpoints" / "global_step_0" / "actor" / "model_world_size_1_rank_0.pt")

    client1_path = str(round_dir / "client_14" / "checkpoints" / "global_step_3" / "actor" / "model_world_size_1_rank_0.pt")

    client2_path = str(round_dir / "client_81" / "checkpoints" / "global_step_3" / "actor" / "model_world_size_1_rank_0.pt")

    # Check whether the files exist.
    print("Checking whether the files exist:")
    for name, path in [("aggregated model", aggregated_path), ("client 14", client1_path), ("client 81", client2_path)]:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"  {name}: {'✅' if exists else '❌'} {path}")
        if exists:
            print(f"    file size: {size / (1024*1024):.2f} MB")

    print()

    # Run the verification.
    success, aggregated_model, client1_model, client2_model = compare_models(aggregated_path, client1_path, client2_path, tolerance=1e-6)

    if success:
        print("\n🎉 Verification passed: the aggregated model is indeed the average of the two client models!")

        # Additional check: differences between the aggregated model and the original client models.
        print("\n" + "=" * 80)
        print("Difference analysis: aggregated model vs. original client models")
        print("=" * 80)

        # Get the model keys.
        aggregated_keys = set(aggregated_model.keys())

        # Compare the aggregated model against client 1.
        print("\n8. Aggregated model vs. client 14 differences:")
        agg_vs_client1_diffs = []
        agg_vs_client1_max = 0.0
        agg_vs_client1_max_key = None
        
        for key in aggregated_keys:
            agg_weight = aggregated_model[key]
            client1_weight = client1_model[key]
            
            diff = torch.abs(agg_weight - client1_weight)
            max_diff_tensor = torch.max(diff)
            mean_diff = torch.mean(diff)
            
            if max_diff_tensor.item() > agg_vs_client1_max:
                agg_vs_client1_max = max_diff_tensor.item()
                agg_vs_client1_max_key = key
            
            agg_vs_client1_diffs.append({
                'key': key,
                'max_diff': max_diff_tensor.item(),
                'mean_diff': mean_diff.item(),
                'shape': agg_weight.shape
            })
        
        print(f"  Max difference: {agg_vs_client1_max:.2e} (key: {agg_vs_client1_max_key})")

        # Compare the aggregated model against client 2.
        print("\n9. Aggregated model vs. client 81 differences:")
        agg_vs_client2_diffs = []
        agg_vs_client2_max = 0.0
        agg_vs_client2_max_key = None
        
        for key in aggregated_keys:
            agg_weight = aggregated_model[key]
            client2_weight = client2_model[key]
            
            diff = torch.abs(agg_weight - client2_weight)
            max_diff_tensor = torch.max(diff)
            mean_diff = torch.mean(diff)
            
            if max_diff_tensor.item() > agg_vs_client2_max:
                agg_vs_client2_max = max_diff_tensor.item()
                agg_vs_client2_max_key = key
            
            agg_vs_client2_diffs.append({
                'key': key,
                'max_diff': max_diff_tensor.item(),
                'mean_diff': mean_diff.item(),
                'shape': agg_weight.shape
            })
        
        print(f"  Max difference: {agg_vs_client2_max:.2e} (key: {agg_vs_client2_max_key})")

        # Show the top 5 largest differences between the aggregated model and the client models.
        agg_vs_client1_diffs.sort(key=lambda x: x['max_diff'], reverse=True)
        print(f"\nAggregated model vs. client 14, top 5 largest differences:")
        for i, diff in enumerate(agg_vs_client1_diffs[:5]):
            print(f"  {i+1}. {diff['key']:50s} | max diff: {diff['max_diff']:.2e} | mean diff: {diff['mean_diff']:.2e}")

        agg_vs_client2_diffs.sort(key=lambda x: x['max_diff'], reverse=True)
        print(f"\nAggregated model vs. client 81, top 5 largest differences:")
        for i, diff in enumerate(agg_vs_client2_diffs[:5]):
            print(f"  {i+1}. {diff['key']:50s} | max diff: {diff['max_diff']:.2e} | mean diff: {diff['mean_diff']:.2e}")

        # Summarize the differences between the aggregated model and the client models.
        agg_vs_client1_max_diffs = [d['max_diff'] for d in agg_vs_client1_diffs]
        agg_vs_client2_max_diffs = [d['max_diff'] for d in agg_vs_client2_diffs]

        print(f"\nAggregated-model difference statistics:")
        print(f"  vs client 14:")
        print(f"    Max difference: {max(agg_vs_client1_max_diffs):.2e}")
        print(f"    Mean of max differences: {np.mean(agg_vs_client1_max_diffs):.2e}")
        print(f"    Number of parameters with diff > 1e-6: {sum(1 for d in agg_vs_client1_max_diffs if d > 1e-6)}")
        print(f"    Number of parameters with diff > 1e-5: {sum(1 for d in agg_vs_client1_max_diffs if d > 1e-5)}")
        print(f"    Number of parameters with diff > 1e-4: {sum(1 for d in agg_vs_client1_max_diffs if d > 1e-4)}")

        print(f"  vs client 81:")
        print(f"    Max difference: {max(agg_vs_client2_max_diffs):.2e}")
        print(f"    Mean of max differences: {np.mean(agg_vs_client2_max_diffs):.2e}")
        print(f"    Number of parameters with diff > 1e-6: {sum(1 for d in agg_vs_client2_max_diffs if d > 1e-6)}")
        print(f"    Number of parameters with diff > 1e-5: {sum(1 for d in agg_vs_client2_max_diffs if d > 1e-5)}")
        print(f"    Number of parameters with diff > 1e-4: {sum(1 for d in agg_vs_client2_max_diffs if d > 1e-4)}")

        # Verify that the aggregated model lies between the two client models.
        print(f"\n10. Verifying that the aggregated model lies between the two client models:")
        balanced = True
        for i, (client1_diff, client2_diff) in enumerate(zip(agg_vs_client1_max_diffs, agg_vs_client2_max_diffs)):
            # The aggregated model should differ roughly equally from both clients (since it is the average).
            ratio = client1_diff / client2_diff if client2_diff > 0 else 1.0
            if ratio < 0.1 or ratio > 10.0:  # If the difference ratio is too large, something may be wrong.
                if i < 10:  # Only show the first 10 anomalies.
                    print(f"  ⚠️  parameter {i} has an abnormal difference ratio: {ratio:.2f}")
                balanced = False
        
        if balanced:
            print("  ✅ The aggregated model's difference ratio against the two clients is reasonable, indicating balanced aggregation")
        else:
            print("  ⚠️  The aggregated model's difference ratio against the two clients is abnormal")
        
        sys.exit(0)
    else:
        print("\n⚠️  Verification failed: the aggregated model does not match the client-model average!")
        sys.exit(1)

if __name__ == "__main__":
    main()