#!/usr/bin/env python3
"""
Unified Loss Plotting Script

Creates a comprehensive 2×3 subplot grid combining loss plots from all three experiments:
- Columns: Burgers 1D, KdV 1D, Navier-Stokes 2D
- Rows: Training Loss, Validation Loss

Each cell contains the parallel training results with mean ± error bands for all models.
"""

import argparse
import os

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
        'checkpoints_dir': 'checkpoints/Burgers',
        'data_file': 'aggregated_losses.pt',
        'train_ylim': (1e-6, 5e-4),
        'val_ylim': (1e-6, 5e-4)
    },
    'KdV': {
        'name': 'KdV',
        'checkpoints_dir': 'checkpoints/KdV',
        'data_file': 'aggregated_losses.pt',
        'train_ylim': (2e-5, 1e-3),
        'val_ylim': (2e-5, 1e-3)
    },
    'NS2D': {
        'name': 'Navier-Stokes',
        'checkpoints_dir': 'checkpoints/NS2D_paper_runs',
        'data_file': 'paper_results.pt',  # Different format for 2D
        'train_ylim': (1e-3, 1e-1),
        'val_ylim': (1e-3, 1e-1)
    }
}


def load_burgers_kdv_data(experiment_key):
    """Load data from Burgers or KdV experiments (1D format)."""
    config = EXPERIMENTS[experiment_key]
    data_path = os.path.join(config['checkpoints_dir'], config['data_file'])

    if not os.path.exists(data_path):
        print(f"Warning: Data file not found: {data_path}")
        return None

    try:
        data = torch.load(data_path, weights_only=False)
        # Data should already have clean model names
        return data
    except Exception as e:
        print(f"Error loading {experiment_key} data: {e}")
        return None


def load_ns2d_data():
    """Load data from Navier-Stokes 2D paper runs (different format)."""
    config = EXPERIMENTS['NS2D']
    base_dir = config['checkpoints_dir']

    if not os.path.exists(base_dir):
        print(f"Warning: NS2D paper runs directory not found: {base_dir}")
        return None

    # Try to aggregate from individual run directories
    paper_models = ['LOCO', 'Hybrid', 'FNO']
    aggregated_data = {}

    for model_name in paper_models:
        model_dir = os.path.join(base_dir, model_name)
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory not found: {model_dir}")
            continue

        # Find run directories
        run_dirs = [d for d in os.listdir(model_dir)
                   if os.path.isdir(os.path.join(model_dir, d)) and d.startswith('run_')]

        if not run_dirs:
            print(f"Warning: No run directories found for {model_name}")
            continue

        all_train_losses = []
        all_val_losses = []

        for run_dir in sorted(run_dirs):
            run_path = os.path.join(model_dir, run_dir)

            # Look for loss files
            loss_files = [f for f in os.listdir(run_path) if f.endswith('.pt') and 'loss' in f.lower()]

            if loss_files:
                try:
                    loss_file = os.path.join(run_path, loss_files[0])
                    data = torch.load(loss_file, weights_only=False, map_location='cpu')

                    if isinstance(data, dict):
                        train_losses = data.get('train_losses', data.get('train', []))
                        val_losses = data.get('val_losses', data.get('val', []))

                        if train_losses and val_losses:
                            all_train_losses.append(train_losses)
                            all_val_losses.append(val_losses)
                except Exception as e:
                    print(f"Warning: Failed to load {loss_files[0]} for {run_dir}: {e}")

        if all_train_losses and all_val_losses:
            # Convert to numpy arrays and compute statistics
            min_len = min(len(losses) for losses in all_train_losses)
            train_array = np.array([losses[:min_len] for losses in all_train_losses])
            val_array = np.array([losses[:min_len] for losses in all_val_losses])

            # Create runs data structure to match 1D format
            runs_data = [{'train': train_losses, 'val': val_losses}
                        for train_losses, val_losses in zip(all_train_losses, all_val_losses)]

            aggregated_data[model_name] = {
                'runs': runs_data,  # Add individual runs for true percentile calculation
                'mean_train': np.mean(train_array, axis=0).tolist(),
                'std_train': np.std(train_array, axis=0).tolist(),
                'mean_val': np.mean(val_array, axis=0).tolist(),
                'std_val': np.std(val_array, axis=0).tolist(),
                'num_runs': len(all_train_losses)
            }

    return aggregated_data if aggregated_data else None


def plot_loss_subplot(ax, data, loss_type, experiment_name, ylim=None, row_idx=0, col_idx=0):
    """Plot loss curves for all models in a single subplot."""
    if data is None:
        ax.text(0.5, 0.5, f'No data available\nfor {experiment_name}',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{experiment_name}\n{loss_type}', fontsize=14, fontweight='bold')
        return

    # Plot each model
    for model_name in ['LOCO', 'Hybrid', 'FNO']:
        if model_name not in data:
            continue

        model_data = data[model_name]
        color = MODEL_COLORS[model_name]

        if loss_type == 'Training Loss':
            mean_losses = model_data['mean_train']
            model_data['std_train']
        else:  # Validation Loss
            mean_losses = model_data['mean_val']
            model_data['std_val']

        epochs = range(len(mean_losses))

        # Calculate true percentiles from individual runs
        # Extract individual loss trajectories based on loss type
        loss_key = 'train' if loss_type == 'Training Loss' else 'val'
        individual_losses = [run[loss_key] for run in model_data['runs']]
        # Convert to numpy array for percentile calculation
        min_len = min(len(loss_traj) for loss_traj in individual_losses)
        losses_array = np.array([loss_traj[:min_len] for loss_traj in individual_losses])
        # Calculate true 10th-90th percentiles
        p10_values = np.percentile(losses_array, 10, axis=0)
        p90_values = np.percentile(losses_array, 90, axis=0)

        # Plot mean line
        ax.plot(epochs, mean_losses, label=model_name, color=color, linewidth=2)
        # Fill error band
        ax.fill_between(epochs, p10_values, p90_values, alpha=0.2, color=color)

    # Configure subplot
    # Only show experiment names on the top row
    if row_idx == 0:
        ax.set_title(experiment_name, fontsize=20, fontweight='bold')

    # Add row labels on the leftmost column
    if col_idx == 0:
        ax.set_ylabel(f'{loss_type}\nL2 Loss', fontsize=18)
    else:
        if experiment_name == "Navier-Stokes":
            ax.set_ylabel('H1 Loss', fontsize=18)
        else:
            ax.set_ylabel('L2 Loss', fontsize=18)

    # Only show x-axis label on bottom row
    if row_idx == 1:
        ax.set_xlabel('Epoch', fontsize=18)

    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18)

    # Set y-axis limits if specified
    if ylim:
        ax.set_ylim(ylim)


def create_summary_table(all_data):
    """Create a summary table with final loss values."""
    summary_data = {}

    for exp_key, data in all_data.items():
        if data is None:
            continue

        exp_name = EXPERIMENTS[exp_key]['name']
        summary_data[exp_name] = {}

        for model_name in ['LOCO', 'Hybrid', 'FNO']:
            if model_name not in data:
                summary_data[exp_name][model_name] = {'train': 'N/A', 'val': 'N/A'}
                continue

            model_data = data[model_name]
            final_train = model_data['mean_train'][-1] if model_data['mean_train'] else 0
            final_val = model_data['mean_val'][-1] if model_data['mean_val'] else 0

            summary_data[exp_name][model_name] = {
                'train': f'{final_train:.2e}',
                'val': f'{final_val:.2e}'
            }

    return summary_data


def generate_unified_plot(save_path=None, show_summary=True):
    """Generate the unified 2×3 subplot grid with all experiments."""
    # Load data from all experiments
    print("Loading data from all experiments...")
    all_data = {
        'Burgers': load_burgers_kdv_data('Burgers'),
        'KdV': load_burgers_kdv_data('KdV'),
        'NS2D': load_ns2d_data()
    }

    # Check which experiments have data available
    available_experiments = [exp for exp, data in all_data.items() if data is not None]
    print(f"Available experiments: {available_experiments}")

    # Create figure with 2×3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Define experiments in order for columns
    exp_order = ['Burgers', 'KdV', 'NS2D']
    loss_types = ['Training Loss', 'Validation Loss']

    # Plot each subplot
    for row_idx, loss_type in enumerate(loss_types):
        for col_idx, exp_key in enumerate(exp_order):
            ax = axes[row_idx, col_idx]
            config = EXPERIMENTS[exp_key]
            data = all_data[exp_key]

            # Choose appropriate y-limits
            ylim = config['train_ylim'] if loss_type == 'Training Loss' else config['val_ylim']

            plot_loss_subplot(ax, data, loss_type, config['name'], ylim, row_idx, col_idx)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])  # Leave space for suptitle and summary

    # Add summary table if requested
    if show_summary:
        summary_data = create_summary_table(all_data)

        # Create text summary below the plots
        summary_text = "\nFinal Loss Values Summary:\n"
        summary_text += "=" * 80 + "\n"

        # Create status line for available data
        available_status = []
        if all_data['Burgers'] is not None:
            available_status.append("Burgers: Ready")
        else:
            available_status.append("Burgers: Not Ready")

        if all_data['KdV'] is not None:
            available_status.append("KdV: Ready")
        else:
            available_status.append("KdV: Not Ready")

        if all_data['NS2D'] is not None:
            available_status.append("Navier-Stokes: Ready")
        else:
            available_status.append("Navier-Stokes: Training in progress")

        summary_text += "Data Availability: " + " | ".join(available_status) + "\n\n"

        summary_text += f"{'Model':<8} | {'Burgers':<20} | {'KdV':<20} | {'Navier-Stokes':<20}\n"
        summary_text += f"{'':8} | {'Train / Val':<20} | {'Train / Val':<20} | {'Train / Val':<20}\n"
        summary_text += "-" * 80 + "\n"

        for model in ['LOCO', 'Hybrid', 'FNO']:
            line = f"{model:<8} |"
            for exp_name in ['Burgers', 'KdV', 'Navier-Stokes']:
                if exp_name in summary_data and model in summary_data[exp_name]:
                    train_val = summary_data[exp_name][model]
                    line += f" {train_val['train']} / {train_val['val']:<10} |"
                else:
                    line += f" {'N/A / N/A':<20} |"
            summary_text += line + "\n"

        summary_text += "=" * 80
        print(summary_text)

    # Save figure
    if save_path is None:
        save_path = 'plots/unified_losses.pdf'

    # Ensure plots directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\nUnified plot saved to: {save_path}")

    # Show plot
    plt.show()

    return fig, all_data


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Generate unified neural operator results plot")
    parser.add_argument('--save-path', type=str, default='plots/unified_losses.pdf',
                       help='Path to save the plot (default: plots/unified_neural_operator_results.pdf)')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip printing summary table')
    parser.add_argument('--check-data', action='store_true',
                       help='Check which data files are available without plotting')

    args = parser.parse_args()

    if args.check_data:
        # Check data availability
        print("Checking data availability...")
        print("=" * 60)

        for exp_key, config in EXPERIMENTS.items():
            data_path = os.path.join(config['checkpoints_dir'], config['data_file'])
            status = "Available" if os.path.exists(data_path) else "Missing"
            print(f"{config['name']:<20}: {status}")
            print(f"  Expected path: {data_path}")

            if exp_key == 'NS2D' and not os.path.exists(data_path) and os.path.exists(config['checkpoints_dir']):
                    run_dirs = []
                    for model in ['LOCO', 'Hybrid', 'FNO']:
                        model_dir = os.path.join(config['checkpoints_dir'], model)
                        if os.path.exists(model_dir):
                            runs = [d for d in os.listdir(model_dir) if d.startswith('run_')]
                            if runs:
                                run_dirs.extend([f"{model}/{r}" for r in runs])
                    if run_dirs:
                        print(f"  Found paper runs: {len(run_dirs)} run directories")
                    else:
                        print("  No paper run directories found")
            print()

        print("To generate training data, run:")
        print("  python burgers_1d.py --mode parallel-train")
        print("  python kdv_1d.py --mode parallel-train")
        print("  python navier_stokes_2d.py --mode parallel-train")
        return

    # Generate the unified plot
    try:
        fig, data = generate_unified_plot(
            save_path=args.save_path,
            show_summary=not args.no_summary
        )
        print("Unified plot generation completed successfully!")

    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
