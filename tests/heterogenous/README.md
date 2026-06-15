# Partition Strategy Simulations

This directory contains simulation scripts for different partition strategies in federated learning.

## Files Overview

### Individual Simulation Scripts

1. **`simulate_preference.py`** - Preference Partition Simulation
   - Tests different tau values (0.1, 0.3, 0.5) for preference heterogeneity
   - Generates preference-partition distribution visualizations
   - Saves results in `preference/` directory

2. **`simulate_coverage.py`** - Coverage Partition Simulation  
   - Tests different size_std values (5, 15, 25) for coverage dispersion
   - Generates coverage distribution visualizations
   - Saves results in `coverage/` directory

3. **`simulate_hardness.py`** - Hardness Partition Simulation
   - Tests different success_std values (0.05, 0.15, 0.25) for difficulty distribution
   - Generates hardness distribution visualizations
   - Saves results in `hardness/` directory



## Usage



### Run Individual Simulations
```bash
# Preference partition simulation
python simulate_preference.py --dataset webshop

# Coverage partition simulation  
python simulate_coverage.py --dataset webshop

# Hardness partition simulation
python simulate_hardness.py --dataset webshop


```



## Notes

- All simulations use synthetic data based on real WebShop categories:
  - beauty: 198 samples
  - electronics: 180 samples  
  - fashion: 1002 samples (20% of original 5012)
  - garden: 905 samples
  - grocery: 115 samples
  - Total: 2,400 samples
- 100 clients with minimum 100 samples per client
- Results are saved as high-resolution PNG files
- Each simulation is independent and can be run separately