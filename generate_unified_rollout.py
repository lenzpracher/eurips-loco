#!/usr/bin/env python3
"""
Unified Rollout Analysis Script

Creates a comprehensive 1×3 subplot layout showing rollout loss progression for all three experiments:
- Column 1: Burgers 1D (MSE rollout loss)
- Column 2: KdV 1D (MSE rollout loss)
- Column 3: Navier-Stokes 2D (H1 rollout loss)

Each subplot shows mean ± error bands for all models (LOCO, Hybrid, FNO) across rollout steps.
"""

import argparse
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch

# Color scheme for models (consistent across all experiments)
MODEL_COLORS = {
    'LOCO': '#1f77b4',    # Blue
    'Hybrid': '#ff7f0e',  # Orange
    'FNO': '#2ca02c'      # Green
}

# Experiment configurations
EXPERIMENTS = {
    'Burgers': {
        'name': 'Burgers',
        'script': 'burgers_1d.py',
        'loss_type': 'MSE',
        'rollout_steps': 35,
        'num_samples': 100,
        'checkpoints_dir': 'checkpoints/Burgers',
        'plots_dir': 'plots/Burgers',
        'ylim': (1e-6, 1e-3),
        'time_per_step': 0.5
    },
    'KdV': {
        'name': 'KdV',
        'script': 'kdv_1d.py',
        'loss_type': 'MSE',
        'rollout_steps': 99,
        'num_samples': 100,
        'checkpoints_dir': 'checkpoints/KdV',
        'plots_dir': 'plots/KdV',
        'ylim': (1e-5, 1e-1),
        'time_per_step': 0.01
    },
    'NS2D': {
        'name': 'Navier-Stokes',
        'script': 'navier_stokes_2d.py',
        'loss_type': 'H1',
        'rollout_steps': 20,
        'num_samples': 40,
        'checkpoints_dir': 'checkpoints/NS2D_paper_runs',
        'plots_dir': 'plots/NS2D',
        'ylim': (1e-4, 1e-1),
        'time_per_step': 0.1
    }
}


def check_experiment_readiness(exp_key):
    """Check if experiment has trained models ready for rollout analysis."""
    config = EXPERIMENTS[exp_key]
    checkpoints_dir = config['checkpoints_dir']

    if not os.path.exists(checkpoints_dir):
        return False, f"Checkpoints directory not found: {checkpoints_dir}"

    # Check for model checkpoints
    expected_models = ['LOCO', 'Hybrid', 'FNO']

    if exp_key == 'NS2D':
        # For NS2D, check paper runs structure
        found_models = []
        for model in expected_models:
            model_dir = os.path.join(checkpoints_dir, model)
            if os.path.exists(model_dir):
                # Check if there are run directories with checkpoints
                run_dirs = [d for d in os.listdir(model_dir) if d.startswith('run_')]
                if run_dirs:
                    # Check if at least one run has a checkpoint
                    for run_dir in run_dirs:
                        run_path = os.path.join(model_dir, run_dir)
                        checkpoint_file = os.path.join(run_path, f'{model}_best.pth')
                        if os.path.exists(checkpoint_file):
                            found_models.append(model)
                            break

        if not found_models:
            return False, "No trained models found in paper runs directories"

        missing = set(expected_models) - set(found_models)
        if missing:
            return True, f"Partial models available: {found_models}, missing: {list(missing)}"
        else:
            return True, f"All models available: {found_models}"

    else:
        # For 1D experiments, check for individual model checkpoints
        found_models = []
        for model in expected_models:
            model_file = os.path.join(checkpoints_dir, f'{model.lower()}_model.pt')
            if os.path.exists(model_file):
                found_models.append(model)

        if not found_models:
            return False, "No trained model checkpoints found"

        missing = set(expected_models) - set(found_models)
        if missing:
            return True, f"Partial models available: {found_models}, missing: {list(missing)}"
        else:
            return True, f"All models available: {found_models}"


def run_rollout_analysis(exp_key, gpu_id=0, temp_dir=None):
    """Run rollout analysis for a specific experiment and return the rollout data."""
    config = EXPERIMENTS[exp_key]

    print(f"Running rollout analysis for {config['name']}...")
    print(f"  Loss type: {config['loss_type']}")
    print(f"  Rollout steps: {config['rollout_steps']}")
    print(f"  Samples: {config['num_samples']}")

    # Construct command based on experiment type
    if exp_key in ['Burgers', 'KdV']:
        # 1D experiments
        cmd = [
            'pixi', 'run', 'python', config['script'],
            '--mode', 'rollout-loss',
            '--gpu', str(gpu_id),
            '--rollout-steps', str(config['rollout_steps']),
            '--num-samples', str(config['num_samples'])
        ]
    else:
        # NS2D experiment
        cmd = [
            'pixi', 'run', 'python', config['script'],
            '--mode', 'rollout-loss',
            '--gpu', str(gpu_id),
            '--rollout-steps', str(config['rollout_steps']),
            '--num-samples', str(config['num_samples'])
        ]

    try:
        # Run the rollout analysis
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 minute timeout

        if result.returncode != 0:
            print(f"Error running rollout analysis for {config['name']}:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return None

        print(f"Rollout analysis completed for {config['name']}")

        # Try to load the generated rollout data
        rollout_data = load_rollout_data(exp_key)
        return rollout_data

    except subprocess.TimeoutExpired:
        print(f"Timeout running rollout analysis for {config['name']}")
        return None
    except Exception as e:
        print(f"Error running rollout analysis for {config['name']}: {e}")
        return None


def load_rollout_data(exp_key):
    """Load rollout data from the plots directory after analysis."""
    config = EXPERIMENTS[exp_key]
    config['plots_dir']

    # Look for rollout analysis files
    if exp_key in ['Burgers', 'KdV']:
        # 1D experiments save rollout data as PDF plots
        # We need to extract data from the analysis functions directly
        # For now, return None and implement direct function calls
        return None
    else:
        # NS2D experiment - look for specific rollout analysis files
        return None


def extract_rollout_data_directly(exp_key, gpu_id=0):
    """Extract rollout data by calling analysis functions directly."""
    config = EXPERIMENTS[exp_key]
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    try:
        if exp_key == 'NS2D':
            # NS2D experiment - use modified analysis function
            from utils2D.analysis import analyze_rollout_loss_paper_runs
            rollout_stats = analyze_rollout_loss_paper_runs(
                device=device,
                rollout_steps=config['rollout_steps'],
                num_samples=config['num_samples'],
                h1_only=True,  # Only H1 loss for NS2D
                return_data=True
            )

            # Convert to unified format
            if rollout_stats:
                rollout_data = {}
                for model_name, stats in rollout_stats.items():
                    rollout_data[model_name] = {
                        'mean': stats['h1_mean'].tolist(),
                        'p10': stats['h1_p10'].tolist(),
                        'p90': stats['h1_p90'].tolist(),
                        'steps': config['rollout_steps']
                    }
                return rollout_data
            else:
                return None
        else:
            # For 1D experiments, use the actual analyze_rollout_loss_1d function
            from utils1D.utils import analyze_rollout_loss_1d

            # Call the analysis function which returns rollout_stats
            rollout_stats = analyze_rollout_loss_1d(
                device=device,
                rollout_steps=config['rollout_steps'],
                num_samples=config['num_samples'],
                data_dir='data',
                checkpoints_dir=config['checkpoints_dir'],
                plots_dir=config['plots_dir'],
                equation=config['name']
            )

            # Convert to unified format
            if rollout_stats:
                rollout_data = {}
                for model_name, stats in rollout_stats.items():
                    # Convert MSE statistics to unified format
                    rollout_data[model_name] = {
                        'mean': stats['mse_mean'].tolist(),
                        'p10': stats['mse_p10'].tolist(),
                        'p90': stats['mse_p90'].tolist(),
                        'steps': config['rollout_steps']
                    }
                return rollout_data
            else:
                print(f"No rollout data returned for {config['name']}")
                return None

    except Exception as e:
        print(f"Error extracting rollout data for {config['name']}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_mock_rollout_data(exp_key):
    """Create mock rollout data for testing the plotting functionality."""
    config = EXPERIMENTS[exp_key]
    steps = config['rollout_steps']

    # Create realistic rollout loss patterns
    models = ['LOCO', 'Hybrid', 'FNO']
    rollout_data = {}

    for model in models:
        # Simulate rollout loss degradation
        base_loss = {
            'LOCO': 1e-5 if config['loss_type'] == 'MSE' else 1e-2,
            'Hybrid': 8e-6 if config['loss_type'] == 'MSE' else 8e-3,
            'FNO': 2e-5 if config['loss_type'] == 'MSE' else 2e-2
        }[model]

        # Exponential degradation with some noise
        x = np.arange(steps)
        mean_loss = base_loss * np.exp(x * 0.1)  # Exponential growth

        # Add some variability
        std_mult = 0.3
        mean_loss * std_mult

        rollout_data[model] = {
            'mean': mean_loss.tolist(),
            'p10': (mean_loss * 0.7).tolist(),  # Mock 10th percentile
            'p90': (mean_loss * 1.3).tolist(),  # Mock 90th percentile
            'steps': steps
        }

    return rollout_data


def plot_rollout_subplot(ax, data, experiment_name, loss_type, time_per_step, ylim=None):
    """Plot rollout curves for all models in a single subplot."""
    if data is None:
        ax.text(0.5, 0.5, f'No data available\nfor {experiment_name}',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{experiment_name}', fontsize=20, fontweight='bold')
        return

    # Plot each model
    for model_name in ['LOCO', 'Hybrid', 'FNO']:
        if model_name not in data:
            continue

        model_data = data[model_name]
        color = MODEL_COLORS[model_name]

        mean_losses = model_data['mean']
        p10_values = model_data['p10']
        p90_values = model_data['p90']
        time_values = [i * time_per_step for i in range(len(mean_losses))]

        # Convert to numpy for easier manipulation
        np.array(mean_losses)
        p10_array = np.array(p10_values)
        p90_array = np.array(p90_values)

        # Use true percentile bounds
        lower_bound = np.maximum(p10_array, 1e-10)  # Avoid negative values
        upper_bound = p90_array

        # Plot mean line
        ax.plot(time_values, mean_losses, label=model_name, color=color, linewidth=2, marker='o', markersize=3)
        # Fill error band
        ax.fill_between(time_values, lower_bound, upper_bound, alpha=0.2, color=color)

    # Configure subplot
    ax.set_title(f'{experiment_name}', fontsize=20, fontweight='bold')
    ax.set_xlabel('Time', fontsize=18)
    ax.set_ylabel(f'{loss_type} Loss', fontsize=18)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18)

    # Set y-axis limits if specified
    if ylim:
        ax.set_ylim(ylim)


def generate_unified_rollout_plot(gpu_id=0, save_path=None, use_mock_data=False):
    """Generate the unified 1×3 subplot layout with rollout analysis."""
    print("=== Unified Rollout Analysis ===")

    # Check experiment readiness
    exp_status = {}
    for exp_key in ['Burgers', 'KdV', 'NS2D']:
        ready, message = check_experiment_readiness(exp_key)
        exp_status[exp_key] = (ready, message)
        print(f"{EXPERIMENTS[exp_key]['name']}: {message}")

    if not any(status[0] for status in exp_status.values()):
        print("\nNo experiments are ready for rollout analysis!")
        print("Please train models first using:")
        print("  pixi run python burgers_1d.py --mode parallel-train")
        print("  pixi run python kdv_1d.py --mode parallel-train")
        print("  pixi run python navier_stokes_2d.py --mode parallel-train")
        return None

    # Generate or load rollout data
    all_rollout_data = {}
    for exp_key in ['Burgers', 'KdV', 'NS2D']:
        if exp_status[exp_key][0]:  # If experiment is ready
            if use_mock_data:
                print(f"Using mock data for {EXPERIMENTS[exp_key]['name']}")
                all_rollout_data[exp_key] = create_mock_rollout_data(exp_key)
            else:
                print(f"Extracting rollout data for {EXPERIMENTS[exp_key]['name']}...")
                all_rollout_data[exp_key] = extract_rollout_data_directly(exp_key, gpu_id)
        else:
            print(f"Skipping {EXPERIMENTS[exp_key]['name']} - not ready")
            all_rollout_data[exp_key] = None

    # Create figure with 1×3 subplot layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define experiments in order for columns
    exp_order = ['Burgers', 'KdV', 'NS2D']

    # Plot each subplot
    for col_idx, exp_key in enumerate(exp_order):
        ax = axes[col_idx]
        config = EXPERIMENTS[exp_key]
        data = all_rollout_data[exp_key]

        plot_rollout_subplot(
            ax, data, config['name'], config['loss_type'], config['time_per_step'], config['ylim']
        )

    # Adjust layout
    plt.tight_layout()

    # Save figure
    if save_path is None:
        save_path = 'plots/unified_rollout.pdf'

    # Ensure plots directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\nUnified rollout plot saved to: {save_path}")

    # Show plot
    plt.show()

    # Print summary
    print("\n" + "="*80)
    print("ROLLOUT ANALYSIS SUMMARY")
    print("="*80)

    available_experiments = [exp for exp, data in all_rollout_data.items() if data is not None]
    for exp_key in available_experiments:
        config = EXPERIMENTS[exp_key]
        data = all_rollout_data[exp_key]

        print(f"\n{config['name']} ({config['loss_type']} Loss):")
        print(f"  Rollout steps: {config['rollout_steps']}")
        print(f"  Samples: {config['num_samples']}")

        # Print final step losses
        for model in ['LOCO', 'Hybrid', 'FNO']:
            if model in data:
                final_loss = data[model]['mean'][-1]
                print(f"  {model}: Final step loss = {final_loss:.2e}")

    print("="*80)

    return fig, all_rollout_data


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Generate unified rollout analysis plot")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--save-path', type=str, default='plots/unified_rollout.pdf',
                       help='Path to save the plot (default: plots/unified_rollout.pdf)')
    parser.add_argument('--check-readiness', action='store_true',
                       help='Check experiment readiness without running analysis')
    parser.add_argument('--mock-data', action='store_true',
                       help='Use mock data for testing plot functionality')

    args = parser.parse_args()

    if args.check_readiness:
        # Check readiness of all experiments
        print("Checking experiment readiness...")
        print("=" * 60)

        all_ready = True
        for exp_key in ['Burgers', 'KdV', 'NS2D']:
            config = EXPERIMENTS[exp_key]
            ready, message = check_experiment_readiness(exp_key)
            status = "Ready" if ready else "Not Ready"
            print(f"{config['name']:<20}: {status}")
            print(f"  {message}")
            print()
            if not ready:
                all_ready = False

        if all_ready:
            print("🎉 All experiments are ready for rollout analysis!")
        else:
            print("Some experiments need training before rollout analysis.")
            print("\nTo train missing models:")
            print("  pixi run python burgers_1d.py --mode parallel-train")
            print("  pixi run python kdv_1d.py --mode parallel-train")
            print("  pixi run python navier_stokes_2d.py --mode parallel-train")
        return

    # Generate the unified rollout plot
    try:
        fig, data = generate_unified_rollout_plot(
            gpu_id=args.gpu,
            save_path=args.save_path,
            use_mock_data=args.mock_data
        )

        if fig is not None:
            print("Unified rollout analysis completed successfully!")
        else:
            print("No data was available for rollout analysis.")

    except Exception as e:
        print(f"Error generating rollout analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
