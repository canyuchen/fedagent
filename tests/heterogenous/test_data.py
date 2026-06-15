#!/usr/bin/env python3
"""
Test script to verify data generation
"""

import numpy as np

def create_synthetic_data(n_samples: int = 6410):
    """
    Create synthetic data for simulation based on real WebShop categories
    """
    # Real WebShop categories and their distribution
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
    
    print("Category counts after processing:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}")
    
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

if __name__ == "__main__":
    data = create_synthetic_data()
    
    # Count actual distribution
    category_counts = {}
    for item in data:
        cat = item['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nActual generated category distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} samples")
    print(f"Total samples: {len(data)}")