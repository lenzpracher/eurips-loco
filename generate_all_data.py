#!/usr/bin/env python3
"""
Generate all datasets for neural operator experiments.
"""

import torch
import os
import sys
import argparse
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data import generate_data, get_data_info

def get_default_configs() -> Dict[str, Dict[str, Any]]:
    """Get default configurations for each equation type."""
    return {
        'burgers': {
            'n_train': 1000,
            'n_test': 200,
            'N': 512,
            'T': 2.5,
            'dt': 1e-4,
            'nu': 0.01,
            'input_length': 1,
            'output_length': 1,
            'time_gap': 10  # Predict 10*dt timesteps ahead
        },
        'kdv': {
            'n_train': 1000,
            'n_test': 200,
            'N': 128,
            'T': 4.0,
            'dt': 1e-3,
            'a': 6.0,
            'b': 1.0,
            'input_length': 1,
            'output_length': 1,
            'time_gap': 10  # Predict 10*dt timesteps ahead
        },
        'ns2d': {
            'n_train': 100,  # Smaller for 2D due to computational cost
            'n_test': 20,
            'N': 64,
            'T': 10.0,
            'dt': 1e-4,  # Simulation timestep
            'visc': 1e-3,  # Use 'visc' instead of 'nu' for NS solver
            'input_length': 10,  # Multi-timestep for 2D NS
            'output_length': 1,
            'time_gap': 1000  # Sample every 0.1 time units (1000*dt)
        }
    }

def generate_single_dataset(equation: str, config: Dict[str, Any], data_path: str, device: str):
    """Generate dataset for a single equation type."""
    print(f"\n{'='*60}")
    print(f"Generating {equation.upper()} dataset")
    print(f"{'='*60}")
    
    # Get equation info
    info = get_data_info(equation)
    print(f"Description: {info['description']}")
    print(f"Spatial dimension: {info['spatial_dim']}D")
    print(f"Configuration: {config}")
    
    # Generate data
    train_dataset, test_dataset = generate_data(
        equation=equation,
        data_path=data_path,
        device=device,
        **config
    )
    
    print(f"✓ Generated {equation} dataset:")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    print(f"  - Data saved to: {data_path}")
    
    return train_dataset, test_dataset

def main():
    parser = argparse.ArgumentParser(description='Generate all neural operator datasets')
    parser.add_argument('--data_path', type=str, default='data', 
                       help='Directory to save datasets (default: data)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for generation (default: auto-detect)')
    parser.add_argument('--equations', type=str, default='all',
                       help='Equations to generate: burgers,kdv,ns2d or all (default: all)')
    parser.add_argument('--small', action='store_true',
                       help='Generate smaller datasets for testing')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("Neural Operator Data Generation")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_path}")
    
    # Create data directory
    os.makedirs(args.data_path, exist_ok=True)
    
    # Get configurations
    configs = get_default_configs()
    
    # Modify for small datasets if requested
    if args.small:
        print("Using small dataset sizes for testing")
        for eq_config in configs.values():
            eq_config['n_train'] = min(100, eq_config['n_train'])
            eq_config['n_test'] = min(20, eq_config['n_test'])
    
    # Determine which equations to generate
    if args.equations.lower() == 'all':
        equations = list(configs.keys())
    else:
        equations = [eq.strip().lower() for eq in args.equations.split(',')]
        invalid = [eq for eq in equations if eq not in configs]
        if invalid:
            print(f"Error: Unknown equations: {invalid}")
            print(f"Available: {list(configs.keys())}")
            return
    
    print(f"Generating datasets for: {equations}")
    
    # Generate datasets
    results = {}
    total_train_samples = 0
    total_test_samples = 0
    
    try:
        for equation in equations:
            config = configs[equation]
            train_ds, test_ds = generate_single_dataset(
                equation, config, args.data_path, device
            )
            results[equation] = {
                'train_size': len(train_ds),
                'test_size': len(test_ds),
                'config': config
            }
            total_train_samples += len(train_ds)
            total_test_samples += len(test_ds)
        
        # Summary
        print(f"\n{'='*60}")
        print("DATA GENERATION COMPLETE")
        print(f"{'='*60}")
        
        for equation, info in results.items():
            print(f"{equation.upper():>8}: {info['train_size']:>6} train, {info['test_size']:>4} test samples")
        
        print(f"{'TOTAL':>8}: {total_train_samples:>6} train, {total_test_samples:>4} test samples")
        print(f"\nAll datasets saved to: {os.path.abspath(args.data_path)}")
        
    except KeyboardInterrupt:
        print("\nData generation interrupted by user")
    except Exception as e:
        print(f"\nError during data generation: {e}")
        raise

if __name__ == "__main__":
    main()