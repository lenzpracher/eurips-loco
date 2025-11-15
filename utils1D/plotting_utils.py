import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def plot_losses(losses_dict, title_suffix="", save_path=None, figsize=(15, 5)):
    """
    Plot training losses for multiple models.

    Args:
        losses_dict: Dictionary with model names as keys and loss arrays as values
        title_suffix: Optional suffix to add to the title
        save_path: Path to save the plot
        figsize: Figure size tuple
    """
    # Check if we have parallel results with both train and val data
    has_parallel_results = any(
        isinstance(losses, dict) and 'mean_train' in losses and 'mean_val' in losses
        for losses in losses_dict.values() if losses is not None
    )

    if has_parallel_results:
        # Create side-by-side subplots for training and validation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        for model_name, losses in losses_dict.items():
            if losses is not None and isinstance(losses, dict):
                # Plot training losses
                if 'mean_train' in losses:
                    mean_train = losses['mean_train']
                    losses['std_train']
                    epochs = range(len(mean_train))

                    # Calculate true percentiles from individual runs
                    # Extract individual training loss trajectories
                    individual_losses = [run['train'] for run in losses['runs']]
                    # Convert to numpy array for percentile calculation
                    min_len = min(len(loss_traj) for loss_traj in individual_losses)
                    losses_array = np.array([loss_traj[:min_len] for loss_traj in individual_losses])
                    # Calculate true 10th-90th percentiles
                    p10_values = np.percentile(losses_array, 10, axis=0)
                    p90_values = np.percentile(losses_array, 90, axis=0)

                    ax1.plot(epochs, mean_train, label=f'{model_name}')
                    ax1.fill_between(epochs, p10_values, p90_values, alpha=0.2)


                # Plot validation losses
                if 'mean_val' in losses:
                    mean_val = losses['mean_val']
                    losses['std_val']
                    epochs = range(len(mean_val))

                    # Calculate true percentiles from individual runs
                    # Extract individual validation loss trajectories
                    individual_losses = [run['val'] for run in losses['runs']]
                    # Convert to numpy array for percentile calculation
                    min_len = min(len(loss_traj) for loss_traj in individual_losses)
                    losses_array = np.array([loss_traj[:min_len] for loss_traj in individual_losses])
                    # Calculate true 10th-90th percentiles
                    p10_values = np.percentile(losses_array, 10, axis=0)
                    p90_values = np.percentile(losses_array, 90, axis=0)

                    ax2.plot(epochs, mean_val, label=f'{model_name}')
                    ax2.fill_between(epochs, p10_values, p90_values, alpha=0.2)


        # Configure training loss subplot
        train_title = 'Training Loss Over Epochs (Mean + 10th-90th Percentile)'
        if title_suffix:
            train_title += f' - {title_suffix}'
        ax1.set_title(train_title)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('L2 Loss')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True)
        if title_suffix == "Burgers":
            ax1.set_ylim(1e-6, 0.5*1e-3)
        if title_suffix == "KdV":
            ax1.set_ylim(7e-5, 2e-3)

        # Configure validation loss subplot
        val_title = 'Validation Loss Over Epochs (Mean + 10th-90th Percentile)'
        if title_suffix:
            val_title += f' - {title_suffix}'
        ax2.set_title(val_title)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('L2 Loss')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True)

        if title_suffix == "Burgers":
            ax2.set_ylim(1e-6, 0.5*1e-3)
        if title_suffix == "KdV":
            ax2.set_ylim(9e-5, 2e-3)
        plt.tight_layout()

    else:
        # Single plot for simple cases
        plt.figure(figsize=figsize)

        for model_name, losses in losses_dict.items():
            if losses is not None:
                # Handle both simple loss arrays and parallel training aggregated results
                if isinstance(losses, dict):
                    # Parallel training aggregated results with mean and std (single type)
                    if 'mean_val' in losses:
                        mean_losses = losses['mean_val']
                        losses['std_val']
                        epochs = range(len(mean_losses))

                        # Plot mean line
                        plt.plot(epochs, mean_losses, label=f'{model_name}')

                        # Add error band using true percentiles
                        # Extract individual validation loss trajectories
                        individual_losses = [run['val'] for run in losses['runs']]
                        min_len = min(len(loss_traj) for loss_traj in individual_losses)
                        losses_array = np.array([loss_traj[:min_len] for loss_traj in individual_losses])
                        p10_values = np.percentile(losses_array, 10, axis=0)
                        p90_values = np.percentile(losses_array, 90, axis=0)
                        plt.fill_between(epochs, p10_values, p90_values, alpha=0.2)
                    else:
                        # Handle regular dict format (train/val losses)
                        if 'val' in losses:
                            plt.plot(losses['val'], label=f'{model_name}')
                        elif 'train' in losses:
                            plt.plot(losses['train'], label=f'{model_name}')
                else:
                    # Simple loss array
                    plt.plot(losses, label=f'{model_name}')

        title = 'Training Loss Over Epochs'
        if title_suffix:
            title += f' ({title_suffix})'

        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('L2 Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_spacetime_diagram(ax, data, title, vmin=None, vmax=None, colormap='viridis',
                          equation='Burgers', time_scale=1.0):
    """
    Plot a spacetime diagram on the given axis with proper physical units.

    Args:
        ax: Matplotlib axis to plot on
        data: 2D array with shape (time, spatial)
        title: Title for the plot
        vmin, vmax: Color scale limits
        colormap: Colormap to use
        equation: Equation type ('Burgers' or 'KdV') for proper scaling
        time_scale: Physical time per rollout step (already includes prediction_gap)

    Returns:
        Image object for colorbar creation
    """
    kwargs = {'aspect': 'auto', 'origin': 'lower', 'cmap': colormap}
    if vmin is not None:
        kwargs['vmin'] = vmin
    if vmax is not None:
        kwargs['vmax'] = vmax

    im = ax.imshow(data.T, **kwargs)

    # Set proper axis labels and ticks with physical units
    time_steps, spatial_points = data.shape

    # Time axis scaling (time_scale already includes prediction_gap)
    physical_time_per_step = time_scale  # Physical time per rollout step
    time_steps * physical_time_per_step
    time_ticks = np.linspace(0, time_steps-1, 6)  # 6 time ticks
    time_labels = [f'{t * physical_time_per_step:.1f}' for t in time_ticks]
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels)

    # Spatial axis scaling
    if equation.lower() == 'kdv':
        # KdV: space from -π to π
        spatial_ticks = np.linspace(0, spatial_points-1, 7)  # 7 spatial ticks
        spatial_labels = [f'{np.pi * (2*t/(spatial_points-1) - 1):.1f}' for t in spatial_ticks]
        ax.set_ylabel('Space', fontsize=18)
    else:  # Burgers
        # Burgers: space from 0 to 2π
        [0, 2*np.pi]
        spatial_ticks = np.linspace(0, spatial_points-1, 7)  # 7 spatial ticks
        spatial_labels = [f'{2*np.pi * t/(spatial_points-1):.1f}' for t in spatial_ticks]
        ax.set_ylabel('Space', fontsize=18)

    ax.set_yticks(spatial_ticks)
    ax.set_yticklabels(spatial_labels)

    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Time', fontsize=18)

    return im

def plot_rollout_losses(rollout_losses, save_path=None, figsize=(10, 5), ylim=(1e-6, 1)):
    """
    Plot rollout losses for multiple models over prediction timesteps.

    Args:
        rollout_losses: Dictionary with model names as keys and loss arrays as values
        save_path: Path to save the plot
        figsize: Figure size tuple
        ylim: Y-axis limits tuple
    """
    plt.figure(figsize=figsize)

    # Gather all loss values (after clipping) to determine lower bound
    all_vals = []
    for model_name, losses in rollout_losses.items():
        if losses is None or len(losses) == 0:
            continue

        # Clip losses into (0,1) for visibility on the fixed axis range
        loss_arr = np.asarray(losses, dtype=float)
        clipped_losses = np.clip(loss_arr, 1e-8, 0.999)
        plt.plot(clipped_losses, label=f'{model_name} Rollout Loss')
        all_vals.extend(clipped_losses.flatten())

    if len(all_vals) == 0:
        print("Warning: No rollout loss data available to plot.")
        return

    all_vals = np.asarray(all_vals)
    all_vals = all_vals[all_vals > 0]
    # Determine lower limit, ensure it's positive and not zero for log scale
    ymin = max(all_vals.min() / 10, 1e-8)
    ymax = 1.0  # Fixed upper bound as requested

    plt.title('Rollout L2 Loss Over Prediction Timesteps')
    plt.xlabel('Prediction Step')
    plt.ylabel('L2 Loss')
    plt.yscale('log')
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def generate_rollout(model, x0, num_steps, model_type='sno'):
    """
    Generate an autoregressive rollout for a given model.

    Args:
        model: The model to use for prediction
        x0: Initial condition tensor
        num_steps: Number of steps to predict
        model_type: Type of model ('sno', 'fno', 'mlp')

    Returns:
        Numpy array of predicted trajectory
    """
    model.eval()
    predictions = []
    current_x = x0.clone()

    # Adjust input format based on model type
    is_2d = current_x.dim() == 4 # Shape [B, H, W, C]
    if model_type == 'fno':
        # FNO expects [batch, channels, spatial_dims...]
        if is_2d:
            # Permute [B, H, W, C] to [B, C, H, W]
            current_x = current_x.permute(0, 3, 1, 2)
        else:
            # Transpose [B, S, C] to [B, C, S]
            current_x = current_x.transpose(1, 2)

    elif model_type == 'mlp':
        current_x = current_x.squeeze(-1)

    with torch.no_grad():
        for _ in range(num_steps):
            pred = model(current_x)

            # Convert prediction to standard format for storage
            if model_type == 'fno':
                 if is_2d:
                    # Permute back to [B, H, W, C]
                    plot_pred = pred.permute(0, 2, 3, 1)
                 else:
                    # Transpose back to [B, S, C]
                    plot_pred = pred.transpose(1, 2)
            elif model_type == 'mlp':
                plot_pred = pred.unsqueeze(-1)
            else:
                plot_pred = pred

            predictions.append(plot_pred.squeeze(0).cpu().numpy())
            current_x = pred.detach()

    return np.array(predictions)

def plot_three_panel_spacetime(true_trajectory, predictions, title_prefix="", save_path=None,
                               figsize=(20, 5), colormap_kwargs=None):
    """
    Create a three-panel spacetime plot: Ground Truth, Prediction, and Absolute Error.

    Args:
        true_trajectory: Ground truth data array
        predictions: Model predictions array
        title_prefix: Prefix for the plot title
        save_path: Path to save the plot
        figsize: Figure size tuple
        colormap_kwargs: Dictionary of keyword arguments for colormap
    """
    if colormap_kwargs is None:
        colormap_kwargs = {}

    # Use constrained layout to properly handle colorbar spacing
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    plot_spacetime_diagram(axes[0], true_trajectory, 'Ground Truth', **colormap_kwargs)
    plot_spacetime_diagram(axes[1], predictions, f'{title_prefix} Prediction', **colormap_kwargs)
    im = plot_spacetime_diagram(axes[2], np.abs(true_trajectory - predictions), 'Absolute Error', **colormap_kwargs)

    # Create colorbar with proper positioning
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Value', rotation=270, labelpad=15)

    plt.suptitle(f'Spacetime Evolution: {title_prefix}', fontsize=14)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.show()

def evaluate_rollout_loss(model, test_dataset, device, model_type='sno', num_rollout_steps=40,
                         generate_plot=False, title_prefix="", plots_dir=None):
    """
    Evaluate a single rollout trajectory and optionally generate plots.

    Args:
        model: Model to evaluate
        test_dataset: Test dataset
        device: Device to run on
        model_type: Type of model
        num_rollout_steps: Number of rollout steps
        generate_plot: Whether to generate spacetime plot
        title_prefix: Prefix for plot titles
        plots_dir: Directory to save plots

    Returns:
        Per-step losses array or None if trajectory too short
    """
    model.eval()
    test_case_idx = np.random.randint(0, len(test_dataset))

    # Get initial condition
    if hasattr(test_dataset[test_case_idx], 'keys'):
        x0 = test_dataset[test_case_idx]['x'].unsqueeze(0).to(device)
    else:
        x0 = test_dataset[test_case_idx].unsqueeze(0).to(device)

    # Get ground truth trajectory
    run_idx, start_time_idx = test_dataset.data_map[test_case_idx]

    # Get snapshots from aggregated data
    snapshots = test_dataset.data['snapshots'][run_idx]

    gap = test_dataset.prediction_gap
    max_idx = snapshots.shape[0]
    max_possible_steps = (max_idx - start_time_idx - 1) // gap
    num_steps = min(num_rollout_steps, max_possible_steps)

    if num_steps < 2:
        return None

    # Generate predictions
    predictions = generate_rollout(model, x0, num_steps, model_type)

    # Get ground truth
    true_trajectory = []
    for i in range(num_steps):
        true_idx = start_time_idx + (i + 1) * gap
        if true_idx < max_idx:
            if snapshots.ndim == 3:  # KdV format: [time, channels, spatial]
                true_slice = snapshots[true_idx].transpose(0, 1)
            else:  # Burgers/CH format: [time, spatial]
                true_slice = snapshots[true_idx].unsqueeze(-1)
            true_trajectory.append(true_slice.numpy())
        else:
            break

    if len(true_trajectory) == 0:
        return None

    predictions = np.array(predictions[:len(true_trajectory)])
    true_trajectory = np.array(true_trajectory)

    # Squeeze channel dimension if needed, but ensure we keep at least 2D for spacetime plots
    if predictions.ndim == 3 and predictions.shape[-1] == 1:
        predictions = predictions.squeeze(-1)
    if true_trajectory.ndim == 3 and true_trajectory.shape[-1] == 1:
        true_trajectory = true_trajectory.squeeze(-1)

    # Check if we have valid 2D data for spacetime plotting
    if predictions.ndim < 2 or true_trajectory.ndim < 2:
        print(f"Warning: Invalid data dimensions for spacetime plot. predictions: {predictions.shape}, true: {true_trajectory.shape}")
        return None

    # Check if we have enough time steps
    if predictions.shape[0] < 2 or true_trajectory.shape[0] < 2:
        print(f"Warning: Not enough time steps for spacetime plot. predictions: {predictions.shape[0]}, true: {true_trajectory.shape[0]}")
        return None

    # Calculate step losses
    if predictions.ndim == 2 and true_trajectory.ndim == 2:
        step_losses = np.mean((predictions - true_trajectory)**2, axis=1)
    else:
        step_losses = np.mean((predictions - true_trajectory)**2, axis=(1, 2))

    # Generate plot if requested
    if generate_plot and true_trajectory.ndim >= 2 and true_trajectory.shape[0] >= 2:
        save_path = None
        if plots_dir:
            save_path = os.path.join(plots_dir, f'{title_prefix.replace(" ", "_").lower()}_spacetime.png')

        plot_three_panel_spacetime(
            true_trajectory, predictions,
            title_prefix=title_prefix,
            save_path=save_path
        )

    return step_losses

def create_combined_model_comparison_plot(models_dict, test_dataset, device,
                                        num_rollout_steps=40, plots_dir=None,
                                        title_suffix="comparison", sample_id=100, equation='Burgers'):
    """
    Create a single combined plot showing all models vs ground truth with difference plots.

    Layout:
    Row 1: Ground Truth, Model1, Model2, Model3
    Row 2: --, Diff1 (GT-M1), Diff2 (GT-M2), Diff3 (GT-M3)

    Args:
        models_dict: Dictionary of models {name: (model, model_type)}
        test_dataset: Test dataset
        device: Device to run on
        num_rollout_steps: Number of steps to predict (use max available)
        plots_dir: Directory to save the plots
        title_suffix: Suffix for the filename
    """
    print("--- Generating Combined Model Comparison Plot ---")

    # Use the specified sample ID for consistent comparison
    test_case_idx = sample_id

    # --- Get Ground Truth ---
    x0_base = test_dataset[test_case_idx]['x'].unsqueeze(0).to(device)
    run_idx, start_time_idx = test_dataset.data_map[test_case_idx]

    print(f"Using run {run_idx}, starting from time step {start_time_idx} for all models")

    # Get snapshots from aggregated data
    snapshots = test_dataset.data['snapshots'][run_idx]

    if snapshots is None:
        print("Warning: Could not find snapshots for selected trajectory. Aborting.")
        return

    gap = test_dataset.prediction_gap
    max_idx = snapshots.shape[0]
    max_possible_steps = (max_idx - start_time_idx - 1) // gap
    num_steps = min(num_rollout_steps, max_possible_steps)

    if num_steps < 2:
        print("Warning: Not enough steps for a valid rollout. Aborting.")
        return

    # Generate ground truth trajectory
    true_trajectory = []
    for step in range(num_steps):
        true_idx = start_time_idx + (step + 1) * gap
        if snapshots.ndim == 3:
            true_slice = snapshots[true_idx].transpose(0, 1)
        else:
            true_slice = snapshots[true_idx].unsqueeze(-1)
        true_trajectory.append(true_slice.numpy())

    true_trajectory = np.array(true_trajectory).squeeze()

    # --- Generate Model Predictions ---
    print(f"Generating {num_steps}-step rollouts for all models from the same initial condition...")
    model_predictions = {}

    # Store the exact initial condition for verification
    x0_initial = x0_base.clone().detach()

    for name, (model, model_type) in models_dict.items():
        # Use fresh copy of the initial condition for each model
        x0 = x0_initial.clone()
        print(f"  Generating rollout for {name}...")
        predictions = generate_rollout(model, x0, num_steps, model_type).squeeze()
        model_predictions[name] = predictions

    # --- Create the Combined Plot ---
    num_models = len(models_dict)
    ncols = max(4, num_models + 1)  # At least 4 columns, or accommodate all models + GT
    nrows = 2  # Row 1: GT + Models, Row 2: Differences

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             constrained_layout=True)

    # Ensure axes is always 2D
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    # Determine physical time scaling based on equation (already includes prediction_gap)
    if equation.lower() == 'kdv':
        time_scale = 0.01   # KdV: 0.01 time units per rollout step
    else:  # Burgers
        time_scale = 0.5   # Burgers: 0.5 time units per rollout step

    # Set common color scale
    all_data = [true_trajectory] + list(model_predictions.values())
    vmin = min(data.min() for data in all_data)
    vmax = max(data.max() for data in all_data)

    # Row 1: Ground Truth + Model Predictions
    plot_spacetime_diagram(axes[0, 0], true_trajectory, 'Ground Truth',
                          vmin=vmin, vmax=vmax, equation=equation,
                          time_scale=time_scale)

    # Enforce consistent model ordering: Hybrid, LOCO, FNO
    preferred_order = ['Hybrid', 'LOCO', 'FNO']
    available_models = set(models_dict.keys())
    model_names = [name for name in preferred_order if name in available_models]
    # Add any additional models that weren't in preferred order
    model_names.extend([name for name in models_dict if name not in model_names])
    for i, name in enumerate(model_names):
        col_idx = i + 1
        if col_idx < ncols:
            plot_spacetime_diagram(axes[0, col_idx], model_predictions[name],
                                   name, vmin=vmin, vmax=vmax, equation=equation,
                                   time_scale=time_scale)

    # Row 2: Difference Plots (Ground Truth - Model)
    axes[1, 0].set_visible(False)  # Empty space under ground truth

    # Calculate difference color scale
    diff_data = []
    for name in model_names:
        diff = true_trajectory - model_predictions[name]
        diff_data.append(diff)

    if diff_data:
        diff_vmin = min(diff.min() for diff in diff_data)
        diff_vmax = max(diff.max() for diff in diff_data)
        diff_vabs = max(abs(diff_vmin), abs(diff_vmax))
        diff_vmin, diff_vmax = -diff_vabs, diff_vabs  # Symmetric around zero

        for i, name in enumerate(model_names):
            col_idx = i + 1
            if col_idx < ncols:
                diff = true_trajectory - model_predictions[name]
                plot_spacetime_diagram(axes[1, col_idx], diff,
                                       f'GT - {name}',
                                       vmin=diff_vmin, vmax=diff_vmax,
                                       colormap='RdBu_r', equation=equation,
                                       time_scale=time_scale)

    # Hide unused subplots
    for i in range(num_models + 1, ncols):
        axes[0, i].set_visible(False)
        axes[1, i].set_visible(False)

    # Remove suptitle for cleaner look
    # fig.suptitle(f'Model Comparison - {num_steps} Step Rollout (Run {run_idx}, Start {start_time_idx})',
    #              fontsize=16)

    if plots_dir:
        save_path = os.path.join(plots_dir, f'combined_{title_suffix}.pdf')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', format='pdf')
        print(f"Saved combined comparison plot to {save_path}")

    plt.show()

def create_model_comparison_plot(models_dict, test_dataset, device, num_trajectories=5,
                               num_rollout_steps=40, plots_dir=None):
    """
    Generates a grid of spacetime plots to compare model rollouts against ground truth.

    Args:
        models_dict: Dictionary of models to compare, including model_type and path
        test_dataset: Test dataset
        device: Device to run on
        num_trajectories: Number of random trajectories to plot
        num_rollout_steps: Number of steps to predict in the rollout
        plots_dir: Directory to save the plots
    """
    print(f"--- Generating {num_trajectories} Model Comparison Plots ---")

    num_models = len(models_dict)
    num_plots = num_models + 1

    # Create an adaptive grid (2 columns)
    ncols = 2
    nrows = int(np.ceil(num_plots / ncols))

    for i in range(num_trajectories):
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                                 squeeze=False, constrained_layout=True)
        axes = axes.flatten()

        test_case_idx = np.random.randint(0, len(test_dataset))

        # --- Get Ground Truth ---
        x0_base = test_dataset[test_case_idx]['x'].unsqueeze(0).to(device)
        run_idx, start_time_idx = test_dataset.data_map[test_case_idx]

        # Get snapshots from aggregated data
        snapshots = test_dataset.data['snapshots'][run_idx]

        if snapshots is None:
            print(f"Warning: Could not find snapshots for trajectory {i}. Skipping.")
            continue

        gap = test_dataset.prediction_gap
        max_idx = snapshots.shape[0]
        max_possible_steps = (max_idx - start_time_idx - 1) // gap
        num_steps = min(num_rollout_steps, max_possible_steps)

        if num_steps < 2:
            print(f"Warning: Not enough steps for a valid rollout on trajectory {i}. Skipping.")
            continue

        true_trajectory = []
        for step in range(num_steps):
            true_idx = start_time_idx + (step + 1) * gap
            if snapshots.ndim == 3:
                true_slice = snapshots[true_idx].transpose(0, 1)
            else:
                true_slice = snapshots[true_idx].unsqueeze(-1)
            true_trajectory.append(true_slice.numpy())

        true_trajectory = np.array(true_trajectory).squeeze()

        # --- Plot Ground Truth ---
        vmin, vmax = true_trajectory.min(), true_trajectory.max()
        plot_spacetime_diagram(axes[0], true_trajectory, 'Ground Truth', vmin=vmin, vmax=vmax)

        # --- Plot Model Predictions ---
        for j, (name, model_info) in enumerate(models_dict.items()):
            ax_idx = j + 1
            model, model_type = model_info[0], model_info[1]

            x0 = x0_base.clone()
            predictions = generate_rollout(model, x0, num_steps, model_type).squeeze()

            plot_spacetime_diagram(axes[ax_idx], predictions, f'{name} Prediction', vmin=vmin, vmax=vmax)

        # Hide any unused subplots
        for j in range(num_plots, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f'Model Comparison on Trajectory {i+1}', fontsize=16)

        if plots_dir:
            save_path = os.path.join(plots_dir, f'model_comparison_traj_{i+1}.png')
            plt.savefig(save_path, dpi=150)
        plt.show()

def plot_data_trajectories(dataset, num_trajectories=2, plots_dir=None, title_suffix=""):
    """
    Plot sample data trajectories from the dataset.

    Args:
        dataset: Dataset to sample from
        num_trajectories: Number of trajectories to plot
        plots_dir: Directory to save plots
        title_suffix: Suffix for the plot title
    """
    if len(dataset) == 0:
        print("Dataset is empty. Cannot plot trajectories.")
        return

    fig, axes = plt.subplots(1, num_trajectories, figsize=(8 * num_trajectories, 6))
    if num_trajectories == 1:
        axes = [axes]

    # Get unique file indices from the dataset map
    file_indices = sorted({item[0] for item in dataset.data_map})

    if len(file_indices) < num_trajectories:
        print(f"Warning: Requested {num_trajectories} plots, but only {len(file_indices)} files found.")
        num_trajectories = len(file_indices)

    plot_file_indices = np.random.choice(file_indices, size=num_trajectories, replace=False)

    plot_idx = 0  # Track actual plotted trajectories
    for _i, file_idx in enumerate(plot_file_indices):
        if plot_idx >= num_trajectories:
            break

        print(f"Plotting trajectory from run: {file_idx}")

        # Get snapshots from aggregated data
        snapshots = dataset.data['snapshots'][file_idx]

        # Convert to numpy and handle different data formats
        trajectory = snapshots.numpy()

        # Ensure we have 2D data [time, spatial] for plotting
        if trajectory.ndim == 3:
            # Remove channel dimension if present (e.g., KdV format)
            trajectory = trajectory.squeeze()

        # Skip if still not 2D or if we don't have enough time steps
        if trajectory.ndim != 2 or trajectory.shape[0] < 2:
            print(f"Warning: Skipping trajectory from {os.path.basename(file_path)} - invalid shape {trajectory.shape}")
            continue

        ax = axes[plot_idx] if num_trajectories > 1 else axes[0]
        im = ax.imshow(trajectory.T, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Sample Trajectory from\n{os.path.basename(file_path)}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Spatial Coordinate")
        fig.colorbar(im, ax=ax)

        plot_idx += 1

    # Hide any unused subplots
    if num_trajectories > 1:
        for idx in range(plot_idx, num_trajectories):
            axes[idx].set_visible(False)

    plt.tight_layout()

    save_path = None
    if plots_dir:
        save_path = os.path.join(plots_dir, f'data_trajectories{title_suffix}.png')
        plt.savefig(save_path)

    plt.show()

def average_rollout_losses(models_dict, test_dataset, device, num_trajectories=100,
                          num_rollout_steps=40, plots_dir=None):
    """
    Computes and plots the average L2 rollout loss for multiple models.

    Args:
        models_dict: Dictionary of models to evaluate
        test_dataset: Test dataset
        device: Device to run on
        num_trajectories: Number of trajectories to average over
        num_rollout_steps: Number of steps in each rollout
        plots_dir: Directory to save the plot
    """
    print(f"--- Averaging Rollout Losses Over {num_trajectories} Trajectories ---")

    all_model_losses = {name: [] for name in models_dict}

    for i in tqdm(range(num_trajectories), desc="Averaging Rollouts"):
        test_case_idx = np.random.randint(0, len(test_dataset))
        x0_base = test_dataset[test_case_idx]['x'].unsqueeze(0).to(device)

        run_idx, start_time_idx = test_dataset.data_map[test_case_idx]

        # Get snapshots from aggregated data
        snapshots = test_dataset.data['snapshots'][run_idx]

        if snapshots is None:
            continue

        gap = test_dataset.prediction_gap
        max_idx = snapshots.shape[0]
        max_possible_steps = (max_idx - start_time_idx - 1) // gap
        num_steps = min(num_rollout_steps, max_possible_steps)

        if num_steps < 2:
            continue

        true_trajectory = []
        for step in range(num_steps):
            true_idx = start_time_idx + (step + 1) * gap
            if snapshots.ndim == 3:
                true_slice = snapshots[true_idx].transpose(0, 1)
            else:
                true_slice = snapshots[true_idx].unsqueeze(-1)
            true_trajectory.append(true_slice)

        true_trajectory_t = torch.stack(true_trajectory).to(device)

        for name, model_info in models_dict.items():
            model, model_type = model_info[0], model_info[1]
            x0 = x0_base.clone()

            predictions = generate_rollout(model, x0, num_steps, model_type)
            predictions_t = torch.from_numpy(predictions).to(device)

            # Reshape tensors for loss calculation
            true_reshaped = true_trajectory_t.squeeze().view(num_steps, -1)
            pred_reshaped = predictions_t.squeeze().view(num_steps, -1)

            # Per-step L2 loss
            per_step_loss = torch.norm(true_reshaped - pred_reshaped, p=2, dim=1) / torch.norm(true_reshaped, p=2, dim=1)

            # Check if the rollout became unstable (NaN)
            if np.isnan(per_step_loss.cpu().numpy()).any():
                print(f"  - Warning: Trajectory {i+1}/{num_trajectories} for model {name} became unstable (NaN) and will be excluded from averaging.")
                continue # Skip this trajectory for this model

            all_model_losses[name].append(per_step_loss.cpu().numpy())

    # Average the losses across all successful trajectories
    avg_losses = {}
    for name, losses in all_model_losses.items():
        if not losses:
            continue  # Skip models with no valid rollouts

        # Determine the maximum rollout length across trajectories for this model
        max_len = max(len(l) for l in losses)

        # Pad each trajectory to max_len with NaN so we can compute nan-safe means
        padded_losses = np.full((len(losses), max_len), np.nan, dtype=float)
        for i, l in enumerate(losses):
            padded_losses[i, :len(l)] = l

        # Compute the nan-mean along trajectories dimension, ignoring missing values
        avg_loss = np.nanmean(padded_losses, axis=0)

        # Remove any trailing NaNs (in case all trajectories were shorter than max_len at tail)
        valid_mask = ~np.isnan(avg_loss)
        if valid_mask.sum() <= 1:
            print(f"  - Warning: Not enough valid rollout steps for model '{name}' to compute meaningful average.")
            continue

        avg_losses[name] = avg_loss[valid_mask]

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    save_path = os.path.join(plots_dir, 'averaged_rollout_losses.png') if plots_dir else None
    plot_rollout_losses(avg_losses, save_path=save_path)
    print("Averaged rollout loss plot has been generated.")

    # --- Optional diagnostic printing ---
    print_rollout_diagnostics(avg_losses)

    return avg_losses

def make_rollout_comparison_gif(
    models_dict,
    test_dataset,
    device,
    trajectory_idx=None,
    num_steps=100,
    plots_dir=None,
    duration_sec=10,
    gif_name="rollout_comparison.gif"
):
    """
    Creates a GIF comparing model rollouts against ground truth for a 1D trajectory.

    Args:
        models_dict: Dictionary of models to compare, including model_type.
                     Format: {name: (model, type)}
        test_dataset: Test dataset to sample from.
        device: PyTorch device.
        trajectory_idx: Index of the trajectory to use. If None, a random one is chosen.
        num_steps: Number of steps for the rollout.
        plots_dir: Directory to save the GIF.
        duration_sec: Desired duration of the output GIF in seconds.
        gif_name: Filename for the saved GIF.
    """
    print("--- Generating Rollout Comparison GIF ---")

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, gif_name)
    else:
        save_path = gif_name

    # --- 1. Select trajectory and load data ---
    if trajectory_idx is None:
        trajectory_idx = np.random.randint(0, len(test_dataset))

    print(f"Using trajectory index: {trajectory_idx}")

    # Get initial condition
    x0 = test_dataset[trajectory_idx]['x'].unsqueeze(0).to(device)

    # Get ground truth trajectory
    run_idx, start_time_idx = test_dataset.data_map[trajectory_idx]

    # Get snapshots from aggregated data
    snapshots = test_dataset.data['snapshots'][run_idx]

    if snapshots is None:
        print(f"Warning: Could not find snapshots for trajectory {trajectory_idx}. Aborting.")
        return

    gap = test_dataset.prediction_gap
    max_idx = snapshots.shape[0]
    max_possible_steps = (max_idx - start_time_idx - 1) // gap
    actual_num_steps = min(num_steps, max_possible_steps)

    if actual_num_steps < 2:
        print(f"Warning: Not enough steps ({actual_num_steps}) for a valid rollout. Aborting.")
        return

    true_trajectory = []
    for step in range(actual_num_steps):
        true_idx = start_time_idx + (step + 1) * gap
        if snapshots.ndim == 3: # [time, channels, spatial]
            true_slice = snapshots[true_idx].transpose(0, 1) # -> [time, spatial, channels]
        else: # [time, spatial]
            true_slice = snapshots[true_idx].unsqueeze(-1) # -> [time, spatial, channels]
        true_trajectory.append(true_slice)

    true_trajectory = torch.stack(true_trajectory).squeeze().cpu().numpy()

    # --- 2. Initialize model states for rollout ---
    model_states = {}
    for name, (model, *_) in models_dict.items():
        model.eval()
        model_states[name] = x0.clone()

    # --- 3. Generate frames for GIF ---
    frames = []

    # Determine consistent y-axis limits for the plot
    vmin = true_trajectory.min()
    vmax = true_trajectory.max()

    with torch.no_grad():
        for t in tqdm(range(actual_num_steps), desc="Generating GIF frames"):
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot ground truth for current timestep
            ax.plot(true_trajectory[t], label='Ground Truth', color='black', linewidth=2)

            # Plot each model's prediction
            for name, (_model, _model_type, *_) in models_dict.items():
                current_state_np = model_states[name].squeeze().cpu().numpy()
                ax.plot(current_state_np, label=name, linestyle='--')

            ax.set_title(f'Rollout at Timestep {t+1}')
            ax.set_xlabel('Spatial Coordinate')
            ax.set_ylabel('Value')
            ax.set_ylim(vmin * 1.1, vmax * 1.1)
            ax.legend()
            ax.grid(True)

            # --- Capture frame ---
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy().reshape(h, w, 4)
            frames.append(buf[:, :, :3])
            plt.close(fig)

            # --- Update model states for next step ---
            for name, (model, model_type, *_) in models_dict.items():
                current_state = model_states[name]

                # Prepare input format for the model
                if model_type == 'fno':
                    # FNO expects [batch, channels, height, width]
                    inp = current_state.transpose(1, 2).unsqueeze(-1)
                elif model_type == 'mlp':
                    inp = current_state.squeeze(-1)
                else: # 'sno'
                    inp = current_state

                # Get prediction
                pred = model(inp)

                # Convert prediction back to standard format [batch, spatial, channels]
                if model_type == 'fno':
                    # Squeeze dummy height and transpose back
                    next_state = pred.squeeze(-1).transpose(1, 2)
                elif model_type == 'mlp':
                    next_state = pred.unsqueeze(-1)
                else:
                    next_state = pred

                model_states[name] = next_state.detach()

    # --- 4. Save GIF ---
    if not frames:
        print("No frames were generated. Cannot create GIF.")
        return

    fps = len(frames) / duration_sec
    print(f"Saving GIF with {len(frames)} frames to {save_path} (fps={fps:.1f})...")
    imageio.mimsave(save_path, frames, fps=fps)
    print("GIF saved successfully.")


# =============================================================================
# Diagnostic Utilities
# =============================================================================

def print_rollout_diagnostics(avg_losses):
    """Prints simple diagnostics for averaged rollout losses.

    Args:
        avg_losses: dict mapping model name to 1-D numpy array of per-step losses.
    """
    if not avg_losses:
        print("No averaged rollout losses available for diagnostics.")
        return

    print("\n--- Rollout Diagnostics ---")
    for name, losses in avg_losses.items():
        n_steps = len(losses)
        first_loss = losses[0]
        final_loss = losses[-1]
        mean_loss = losses.mean()
        max_loss = losses.max()
        print(f"{name}: steps={n_steps}, first={first_loss:.3e}, final={final_loss:.3e}, "
              f"mean={mean_loss:.3e}, max={max_loss:.3e}")
    print("---------------------------\n")
