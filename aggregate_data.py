#!/usr/bin/env python3
"""
Data Aggregation Script

Aggregates thousands of individual run_*.pt files into efficient single files
for fast loading. This creates the aggregated datasets used by the main experiments.

Usage:
    # Aggregate all datasets
    python aggregate_data.py --experiment all
    # Aggregate specific dataset
    python aggregate_data.py --experiment burgers
    python aggregate_data.py --experiment kdv
"""
import argparse
import glob
import os

import torch
from tqdm import tqdm


def aggregate_burgers_data(data_dir="data/Burgers", output_file=None):
    """
    Aggregate all Burgers run files into a single file
    Args:
        data_dir: Directory containing run_*.pt files
        output_file: Output file path (default: {data_dir}/burgers_aggregated.pt)
    """
    run_files = sorted(glob.glob(os.path.join(data_dir, "run_*.pt")))

    if not run_files:
        print(f"No run files found in {data_dir}")
        return

    print(f"Found {len(run_files)} run files to aggregate")

    if output_file is None:
        output_file = os.path.join(data_dir, "burgers_data.pt")

    # Load first file to get dimensions
    first_data = torch.load(run_files[0], map_location='cpu')
    time_steps, spatial_points = first_data['snapshots'].shape
    num_runs = len(run_files)

    print(f"Dimensions: {num_runs} runs × {time_steps} timesteps × {spatial_points} spatial points")

    # Preallocate tensors
    all_snapshots = torch.zeros((num_runs, time_steps, spatial_points), dtype=torch.float32)
    all_times = torch.zeros((num_runs, time_steps), dtype=torch.float32)
    all_energy = torch.zeros((num_runs, time_steps), dtype=torch.float32)
    all_run_ids = torch.zeros(num_runs, dtype=torch.long)

    # Load and aggregate all data
    print("Aggregating data...")
    for i, file_path in enumerate(tqdm(run_files)):
        try:
            data = torch.load(file_path, map_location='cpu')

            if data.get('status') != 'success':
                print(f"Warning: {file_path} has status '{data.get('status')}'")
                continue

            all_snapshots[i] = data['snapshots']
            all_times[i] = data['times']
            all_energy[i] = data['energy_t']
            all_run_ids[i] = data['run_id']

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    # Save aggregated data
    aggregated_data = {
        'snapshots': all_snapshots,  # [num_runs, time_steps, spatial_points]
        'times': all_times,         # [num_runs, time_steps]
        'energy_t': all_energy,     # [num_runs, time_steps]
        'run_ids': all_run_ids,     # [num_runs]
        'num_runs': num_runs,
        'time_steps': time_steps,
        'spatial_points': spatial_points,
        'original_files': len(run_files)
    }

    print(f"Saving aggregated data to {output_file}")
    torch.save(aggregated_data, output_file)

    # Print size comparison
    original_size = sum(os.path.getsize(f) for f in run_files[:100])  # Sample first 100
    original_size_est = original_size * len(run_files) / 100  # Estimate total
    new_size = os.path.getsize(output_file)

    print("\nSize comparison:")
    print(f"  Estimated original: {original_size_est/1e9:.2f} GB")
    print(f"  Aggregated file: {new_size/1e9:.2f} GB")
    print(f"  Reduction: {(1 - new_size/original_size_est)*100:.1f}%")

    print(f"Successfully aggregated {len(run_files)} files into {output_file}")

def aggregate_kdv_data(data_dir="data/KdV", output_file=None):
    """Aggregate KdV simulation files"""
    run_files = sorted(glob.glob(os.path.join(data_dir, "simulation_run_*.pt")))

    if not run_files:
        print(f"No KdV simulation files found in {data_dir}")
        return

    if output_file is None:
        output_file = os.path.join(data_dir, "kdv_data.pt")

    print(f"Found {len(run_files)} KdV files to aggregate")

    # Load first file to get dimensions
    first_data = torch.load(run_files[0], map_location='cpu')
    time_steps, spatial_points = first_data['snapshots'].shape
    num_runs = len(run_files)

    print(f"Dimensions: {num_runs} runs × {time_steps} timesteps × 1 channels × {spatial_points} spatial points")

    # Preallocate tensors with channel dimension to match old format
    all_snapshots = torch.zeros((num_runs, time_steps, 1, spatial_points), dtype=torch.float32)
    all_times = torch.zeros((num_runs, time_steps), dtype=torch.float32)
    all_mass = torch.zeros((num_runs, time_steps), dtype=torch.float32)
    all_momentum = torch.zeros((num_runs, time_steps), dtype=torch.float32)
    all_energy = torch.zeros((num_runs, time_steps), dtype=torch.float32)
    all_run_ids = torch.zeros(num_runs, dtype=torch.long)

    # Load and aggregate all data
    print("Aggregating KdV data...")
    for i, file_path in enumerate(tqdm(run_files)):
        try:
            data = torch.load(file_path, map_location='cpu')

            if data.get('status') != 'success':
                print(f"Warning: {file_path} has status '{data.get('status')}'")
                continue

            all_snapshots[i] = data['snapshots'].unsqueeze(1)  # Add channel dimension
            all_times[i] = data['times']
            all_mass[i] = data.get('mass_t', torch.zeros(time_steps))
            all_momentum[i] = data.get('momentum_t', torch.zeros(time_steps))
            all_energy[i] = data.get('energy_t', torch.zeros(time_steps))
            all_run_ids[i] = data['run_id']

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    # Save aggregated data
    aggregated_data = {
        'snapshots': all_snapshots,  # [num_runs, time_steps, spatial_points]
        'times': all_times,         # [num_runs, time_steps]
        'mass_t': all_mass,         # [num_runs, time_steps] - Conservation law
        'momentum_t': all_momentum, # [num_runs, time_steps] - Conservation law
        'energy_t': all_energy,     # [num_runs, time_steps] - Conservation law
        'run_ids': all_run_ids,     # [num_runs]
        'num_runs': num_runs,
        'time_steps': time_steps,
        'spatial_points': spatial_points,
        'original_files': len(run_files)
    }

    print(f"Saving aggregated KdV data to {output_file}")
    torch.save(aggregated_data, output_file)

    # Print size comparison
    original_size = sum(os.path.getsize(f) for f in run_files)
    new_size = os.path.getsize(output_file)

    print("\nSize comparison:")
    print(f"  Original files: {original_size/1e6:.2f} MB")
    print(f"  Aggregated file: {new_size/1e6:.2f} MB")
    print(f"  Reduction: {(1 - new_size/original_size)*100:.1f}%")

    print(f"Successfully aggregated {len(run_files)} files into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Aggregate individual run files into efficient single files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--experiment', choices=['burgers', 'kdv', 'all'], default='all',
                      help='Which experiment data to aggregate')
    parser.add_argument('--data-dir', help='Data directory (optional)')
    parser.add_argument('--output-file', help='Output file path (optional)')

    args = parser.parse_args()

    print("Neural Operator Data Aggregation")
    print("="*50)

    success = True

    if args.experiment in ['burgers', 'all']:
        print("\nAggregating Burgers data...")
        data_dir = args.data_dir or "data/Burgers"
        try:
            aggregate_burgers_data(data_dir, args.output_file)
        except Exception as e:
            print(f"Burgers aggregation failed: {e}")
            success = False

    if args.experiment in ['kdv', 'all']:
        print("\nAggregating KdV data...")
        data_dir = args.data_dir or "data/KdV"
        try:
            aggregate_kdv_data(data_dir, args.output_file)
        except Exception as e:
            print(f"KdV aggregation failed: {e}")
            success = False

    if success:
        print("\nData aggregation completed successfully!")
        print("The aggregated files can now be used for efficient training.")
    else:
        print("\nSome aggregation operations failed!")

    print("="*50)
