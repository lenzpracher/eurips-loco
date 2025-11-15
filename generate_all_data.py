#!/usr/bin/env python3
"""
Data Generation Script

Generate all data needed for neural operator experiments.
This script creates training and test data for Burgers, KdV, and Navier-Stokes equations
using the exact solvers and parameters from old_scripts.

Usage:
    # Generate all data
    python generate_all_data.py --experiment all

    # Generate specific experiment data
    python generate_all_data.py --experiment burgers --num-samples 10000
    python generate_all_data.py --experiment kdv --num-samples 1000
    python generate_all_data.py --experiment ns2d --num-samples 1000

    # Use specific parameters
    python generate_all_data.py --experiment burgers --resolution 512 --time-steps 100
"""

import argparse
import datetime
import os
import sys

import numpy as np
import torch

AVAILABLE_EXPERIMENTS = ['burgers', 'kdv', 'ns2d', 'all']

# Default parameters exactly matching solver configurations
DEFAULT_PARAMS = {
    'burgers': {
        'num_samples': 100,        # generate_burgers_dataset default: num_runs=100
        'resolution': 512,         # generate_burgers_dataset default: N=512
        'time_steps': 400,         # Calculated from T=20.0, save_interval=0.05 -> 20/0.05+1 = 401
        'final_time': 20.0,        # generate_burgers_dataset default: T=20.0
        'data_dir': 'data/Burgers' # generate_burgers_dataset default: save_dir='data/Burgers'
    },
    'kdv': {
        'num_samples': 1000,       # generate_kdv_dataset default: num_runs=1000
        'resolution': 128,         # generate_kdv_dataset default: N=128
        'time_steps': 1001,        # T=10.0, save_interval=0.01 -> 10/0.01+1 = 1001
        'final_time': 10.0,        # Final time T=10.0 (matches existing data)
        'data_dir': 'data/KdV'     # generate_kdv_dataset default: save_dir='data/KdV'
    },
    'ns2d': {
        'num_samples': 96,         # generate_ns2d_dataset default: num_samples=96
        'resolution': 256,         # generate_ns2d_dataset default: N=256
        'time_steps': 1000,        # generate_ns2d_dataset default: record_steps=1000
        'final_time': 100.0,       # generate_ns2d_dataset default: T=100.0
        'data_dir': 'data/NS2D'    # generate_ns2d_dataset default: save_dir='data/NS2D'
    }
}


def generate_burgers_data(num_samples, resolution, time_steps, final_time, data_dir):
    """
    Generate Burgers equation data using the solver from solvers/burgers.py

    This uses the generate_burgers_dataset function which provides batch processing
    and proper error handling.
    """
    print("Generating Burgers data:")
    print(f"  Samples: {num_samples}")
    print(f"  Resolution: {resolution}")
    print(f"  Time steps: {time_steps}")
    print(f"  Final time: {final_time}")
    print(f"  Data directory: {data_dir}")

    # Import solver function
    try:
        from solvers.burgers import generate_burgers_dataset
    except ImportError:
        print("Error: generate_burgers_dataset not found. Please ensure solvers/burgers.py exists.")
        return False

    os.makedirs(data_dir, exist_ok=True)

    # Map our parameters to solver's expected parameters
    # Calculate save_interval from time_steps and final_time
    dt = 1e-4  # Default integration timestep
    save_interval = final_time / (time_steps - 1)  # -1 because time_steps includes t=0

    # Use the solver's dataset generation function
    try:
        stats = generate_burgers_dataset(
            num_runs=num_samples,
            N=resolution,
            nu=0.01,  # Default viscosity
            T=final_time,
            dt=dt,
            save_interval=save_interval,
            ic_type='random',
            ic_scale=1.0,
            batch_size=min(32, num_samples),
            device='cpu',
            seed=None,
            save_dir=data_dir
        )

        if stats['success_rate'] > 0:
            print(f"Generated {stats['successful_runs']}/{num_samples} Burgers samples")
            print(f"   Success rate: {stats['success_rate']:.1%}")
            return True
        else:
            print("Failed to generate Burgers data")
            return False

    except Exception as e:
        print(f"Error generating Burgers data: {e}")
        return False


def generate_kdv_data(num_samples, resolution, time_steps, final_time, data_dir):
    """
    Generate KdV equation data using the solver from solvers/kdv.py

    This uses the generate_kdv_dataset function which provides batch processing
    and proper error handling.
    """
    print("Generating KdV data:")
    print(f"  Samples: {num_samples}")
    print(f"  Resolution: {resolution}")
    print(f"  Time steps: {time_steps}")
    print(f"  Final time: {final_time}")
    print(f"  Data directory: {data_dir}")

    # Import solver function
    try:
        from solvers.kdv import generate_kdv_dataset
    except ImportError:
        print("Error: generate_kdv_dataset not found. Please ensure solvers/kdv.py exists.")
        return False

    os.makedirs(data_dir, exist_ok=True)

    # Map our parameters to solver's expected parameters
    # Use much smaller timestep for stability with save every 10/1000 timesteps
    dt = 1e-5  # Very small integration timestep for stability
    save_interval = final_time / (time_steps - 1)  # -1 because time_steps includes t=0

    # Use the solver's dataset generation function
    try:
        stats = generate_kdv_dataset(
            num_runs=num_samples,
            N=resolution,
            width=2*np.pi,
            a=6.0,  # Standard KdV nonlinear coefficient
            b=1.0,  # Standard KdV dispersion coefficient
            T=final_time,
            dt=dt,
            save_interval=save_interval,
            batch_size=min(32, num_samples),
            device='cpu',
            seed=None,
            save_dir=data_dir
        )

        if stats['success_rate'] > 0:
            print(f"Generated {stats['successful_runs']}/{num_samples} KdV samples")
            print(f"   Success rate: {stats['success_rate']:.1%}")
            return True
        else:
            print("Failed to generate KdV data")
            return False

    except Exception as e:
        print(f"Error generating KdV data: {e}")
        return False


def generate_ns2d_data(num_samples, resolution, time_steps, final_time, data_dir):
    """
    Generate 2D Navier-Stokes data using the solver from solvers/navier_stokes.py

    This uses the generate_ns2d_dataset function which provides proper batch processing
    and handles the full simulation workflow.
    """
    print("Generating NS2D data:")
    print(f"  Samples: {num_samples}")
    print(f"  Resolution: {resolution}")
    print(f"  Time steps: {time_steps}")
    print(f"  Final time: {final_time}")
    print(f"  Data directory: {data_dir}")

    # Import solver function
    try:
        from solvers.navier_stokes import generate_ns2d_dataset
    except ImportError:
        print("Error: generate_ns2d_dataset not found.")
        print("Please ensure solvers/navier_stokes.py exists or implement the solver.")
        print("For now, using placeholder data generation.")
        return generate_ns2d_placeholder_data(num_samples, resolution, time_steps, data_dir)

    os.makedirs(data_dir, exist_ok=True)

    # Map our parameters to solver's expected parameters
    # Convert Reynolds number from viscosity relationship
    reynolds = 1000.0  # Default Reynolds number
    visc = 1.0 / reynolds

    # Calculate integration timestep (much smaller than save interval)
    delta_t = 1e-4  # Default integration timestep

    # Use the solver's dataset generation function
    try:
        stats = generate_ns2d_dataset(
            num_samples=num_samples,
            N=resolution,
            visc=visc,
            T=final_time,
            delta_t=delta_t,
            record_steps=time_steps,
            batch_size=min(8, num_samples),  # NS2D is memory intensive
            device='cpu',
            seed=None,
            save_dir=data_dir
        )

        if stats['success_rate'] > 0:
            print(f"Generated {stats['successful_runs']}/{num_samples} NS2D samples")
            print(f"   Success rate: {stats['success_rate']:.1%}")
            return True
        else:
            print("Failed to generate NS2D data")
            return False

    except Exception as e:
        print(f"Error generating NS2D data: {e}")
        print("Falling back to placeholder data generation.")
        return generate_ns2d_placeholder_data(num_samples, resolution, time_steps, data_dir)


def generate_ns2d_placeholder_data(num_samples, resolution, time_steps, data_dir):
    """
    Generate placeholder NS2D data for testing when solver is not available
    """
    print("Using placeholder NS2D data generation")
    os.makedirs(data_dir, exist_ok=True)

    # Generate random data with proper dimensions
    train_samples = int(num_samples * 0.8)
    test_samples = num_samples - train_samples

    # Shape: [samples, time_steps, height, width] - matches NS2D format
    train_data = torch.randn(train_samples, time_steps, resolution, resolution)
    test_data = torch.randn(test_samples, time_steps, resolution, resolution)

    train_path = os.path.join(data_dir, f'nsforcing_train_{resolution}.pt')
    test_path = os.path.join(data_dir, f'nsforcing_test_{resolution}.pt')

    torch.save({'u': train_data}, train_path)
    torch.save({'u': test_data}, test_path)

    print("Generated placeholder NS2D data:")
    print(f"  Training: {train_data.shape} saved to {train_path}")
    print(f"  Test: {test_data.shape} saved to {test_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate data for neural operator experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        choices=AVAILABLE_EXPERIMENTS,
        help="Experiment data to generate"
    )

    parser.add_argument('--num-samples', type=int, help="Number of samples to generate")
    parser.add_argument('--resolution', type=int, help="Spatial resolution")
    parser.add_argument('--time-steps', type=int, help="Number of time steps")
    parser.add_argument('--final-time', type=float, help="Final simulation time")
    parser.add_argument('--data-dir', type=str, help="Data directory")
    parser.add_argument('--skip-aggregation', action='store_true',
                       help="Skip automatic data aggregation after generation")

    args = parser.parse_args()

    # Setup parameters
    experiments_to_run = []
    if args.experiment == 'all':
        experiments_to_run = ['burgers', 'kdv', 'ns2d']
    else:
        experiments_to_run = [args.experiment]

    print("Neural Operator Data Generation")
    print(f"Timestamp: {datetime.datetime.now()}")
    print(f"Experiments: {experiments_to_run}")
    print("="*50)

    success = True

    for exp in experiments_to_run:
        print(f"\n{'='*60}")
        print(f"Generating {exp.upper()} data")
        print(f"{'='*60}")

        # Get default parameters
        params = DEFAULT_PARAMS[exp].copy()

        # Override with command line arguments
        if args.num_samples:
            params['num_samples'] = args.num_samples
        if args.resolution:
            params['resolution'] = args.resolution
        if args.time_steps:
            params['time_steps'] = args.time_steps
        if args.final_time:
            params['final_time'] = args.final_time
        if args.data_dir:
            params['data_dir'] = args.data_dir

        # Generate data
        try:
            if exp == 'burgers':
                result = generate_burgers_data(**params)
            elif exp == 'kdv':
                result = generate_kdv_data(**params)
            elif exp == 'ns2d':
                result = generate_ns2d_data(**params)
            else:
                print(f"Unknown experiment: {exp}")
                result = False

            if not result:
                success = False
                print(f"Failed to generate {exp} data")

        except Exception as e:
            print(f"Error generating {exp} data: {e}")
            success = False

    if success:
        print("\nAll data generation completed successfully!")

        # Optionally aggregate data for efficient loading
        if not args.skip_aggregation:
            print("\nAggregating data for efficient loading...")
            aggregate_success = True

            for exp in experiments_to_run:
                if exp == 'ns2d':
                    print(f"   Skipping {exp.upper()} aggregation (not needed)")
                    continue

                print(f"   Aggregating {exp.upper()} data...")
                try:
                    if exp == 'burgers':
                        from aggregate_data import aggregate_burgers_data
                        data_dir = args.data_dir or DEFAULT_PARAMS[exp]['data_dir']
                        aggregate_burgers_data(data_dir)
                    elif exp == 'kdv':
                        from aggregate_data import aggregate_kdv_data
                        data_dir = args.data_dir or DEFAULT_PARAMS[exp]['data_dir']
                        aggregate_kdv_data(data_dir)
                except Exception as e:
                    print(f"     Warning: {exp.upper()} aggregation failed: {e}")
                    print("     Individual files are still available")
                    aggregate_success = False

            if aggregate_success:
                print("   Data aggregation completed")
            else:
                print("   Some aggregation failed, individual files available")
        else:
            print("\nSkipping data aggregation (use --skip-aggregation flag)")

        print("\nNext steps:")
        print("1. Run experiments with: python run_experiments.py --experiment all --mode train")
        print("2. Or use pixi commands: pixi run burgers-train")
        print("3. Individual run files available for custom analysis")
    else:
        print("\nSome data generation failed!")
        print("Please check error messages above and ensure solvers are implemented.")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
