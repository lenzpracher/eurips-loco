"""
1D Plotting Utilities for Neural Operators

This module provides plotting utilities for 1D PDE experiments.
Extracted from old_scripts/Burgers_KdV/ for clean organization.

Functions:
- plot_losses: Plot training/validation losses
- plot_spacetime_diagram: Plot spacetime evolution
- plot_rollout_losses: Plot rollout losses comparison
- generate_rollout: Generate model rollout predictions
- plot_three_panel_spacetime: Three-panel spacetime comparison
- evaluate_rollout_loss: Evaluate single rollout trajectory
- create_model_comparison_plot: Compare multiple models
- plot_data_trajectories: Plot sample data trajectories
- average_rollout_losses: Average rollout losses over multiple trajectories
- make_rollout_comparison_gif: Create comparison GIF
"""

import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_losses(train_losses, val_losses, title="Training Progress", save_path=None):
    """
    Plot training and validation losses



    Args:
        train_losses: List/array of training losses
        val_losses: List/array of validation losses
        title: Plot title
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'o-', label='Training Loss', alpha=0.7)
    ax.plot(epochs, val_losses, 's-', label='Validation Loss', alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_spacetime_diagram(ax, data, title, vmin=None, vmax=None, **kwargs):
    """
    Plot a spacetime diagram on given axes



    Args:
        ax: Matplotlib axes object
        data: 2D array [time, space] to plot
        title: Title for the plot
        vmin, vmax: Color scale limits
        **kwargs: Additional arguments for imshow

    Returns:
        Image object for colorbar
    """
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    im = ax.imshow(data.T, aspect='auto', origin='lower',
                   vmin=vmin, vmax=vmax, cmap='viridis', **kwargs)
    ax.set_xlabel('Time')
    ax.set_ylabel('Space')
    ax.set_title(title)

    return im


def plot_rollout_losses(losses_dict, title="Rollout Losses", save_path=None):
    """
    Plot rollout losses for multiple models



    Args:
        losses_dict: Dictionary of model_name -> losses array
        title: Plot title
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for model_name, losses in losses_dict.items():
        steps = range(1, len(losses) + 1)
        ax.plot(steps, losses, 'o-', label=model_name, alpha=0.8, markersize=4)

    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('Relative L2 Error')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def generate_rollout(model, x0, num_steps, model_type='loco'):
    """
    Generate rollout predictions from a model



    Args:
        model: Neural operator model
        x0: Initial condition [batch, spatial, channels]
        num_steps: Number of rollout steps
        model_type: 'loco', 'hybrid', or 'fno' for proper tensor handling

    Returns:
        Numpy array of predictions [steps, spatial, channels]
    """
    model.eval()
    predictions = []
    current_x = x0.clone()

    with torch.no_grad():
        for _ in range(num_steps):
            # Handle different model input formats
            if model_type == 'fno':
                # FNO expects [batch, channels, spatial]
                if current_x.dim() == 4:
                    # Already in correct format [B, C, H, W]
                    model_input = current_x
                else:
                    # Convert from [batch, spatial, channels] to [batch, channels, spatial]
                    model_input = current_x.transpose(1, 2)
            elif model_type == 'mlp':
                # MLP expects [batch, spatial] without channels
                model_input = current_x.squeeze(-1)
            else:  # loco, hybrid
                # Standard format [batch, spatial, channels]
                model_input = current_x

            pred = model(model_input)

            # Convert prediction back to standard format [batch, spatial, channels]
            if model_type == 'fno':
                if pred.dim() == 4:
                    # 2D FNO output [B, C, H, W] -> [B, H, W, C]
                    plot_pred = pred.permute(0, 2, 3, 1)
                else:
                    # 1D FNO output [B, C, S] -> [B, S, C]
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


def evaluate_rollout_loss(model, test_dataset, device, model_type='loco', num_rollout_steps=40,
                         generate_plot=False, title_prefix="", plots_dir=None):
    """
    Evaluate a single rollout trajectory and optionally generate plots.



    Args:
        model: Model to evaluate
        test_dataset: Test dataset
        device: Device to run on
        model_type: Type of model ('loco', 'hybrid', 'fno')
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
    file_idx, start_time_idx = test_dataset.data_map[test_case_idx]
    file_path = test_dataset.files[file_idx]

    # Load snapshots (handle different data formats)
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    if 'snapshots' in data:
        snapshots = data['snapshots']
    elif 'results' in data and len(data['results']) > 0:
        snapshots = data['results'][0]['snapshots']
    else:
        print(f"Warning: Could not find snapshots in {file_path}")
        return None

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
            else:  # Burgers format: [time, spatial]
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


def create_model_comparison_plot(models_dict, test_dataset, device, num_trajectories=5,
                               num_rollout_steps=40, plots_dir=None):
    """
    Generates a grid of spacetime plots to compare model rollouts against ground truth.



    Args:
        models_dict: Dictionary of models to compare, format: {name: (model, model_type)}
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
        file_idx, start_time_idx = test_dataset.data_map[test_case_idx]
        file_path = test_dataset.files[file_idx]

        data = torch.load(file_path, map_location='cpu', weights_only=False)
        snapshots = data.get('snapshots', data.get('results', [{}])[0].get('snapshots'))

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

        file_path = dataset.files[file_idx]
        print(f"Plotting trajectory from: {file_path}")

        data = torch.load(file_path, map_location='cpu', weights_only=False)

        # Handle different data formats
        if 'snapshots' in data:
            snapshots = data['snapshots']
        elif 'results' in data and len(data['results']) > 0:
            snapshots = data['results'][0]['snapshots']
        else:
            print(f"Warning: No snapshots found in {file_path}")
            continue

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
        models_dict: Dictionary of models to evaluate, format: {name: (model, model_type)}
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

        file_idx, start_time_idx = test_dataset.data_map[test_case_idx]
        file_path = test_dataset.files[file_idx]

        data = torch.load(file_path, map_location='cpu', weights_only=False)
        snapshots = data.get('snapshots', data.get('results', [{}])[0].get('snapshots'))

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
        models_dict: Dictionary of models to compare, format: {name: (model, model_type)}
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
    file_idx, start_time_idx = test_dataset.data_map[trajectory_idx]
    file_path = test_dataset.files[file_idx]

    data = torch.load(file_path, map_location='cpu', weights_only=False)
    snapshots = data.get('snapshots', data.get('results', [{}])[0].get('snapshots'))

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
            true_slice = snapshots[true_idx].transpose(0, 1) # -> [spatial, channels]
        else: # [time, spatial]
            true_slice = snapshots[true_idx].unsqueeze(-1) # -> [spatial, channels]
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
                    # FNO expects [batch, channels, spatial]
                    inp = current_state.transpose(1, 2)
                elif model_type == 'mlp':
                    inp = current_state.squeeze(-1)
                else: # 'loco', 'hybrid'
                    inp = current_state

                # Get prediction
                pred = model(inp)

                # Convert prediction back to standard format [batch, spatial, channels]
                if model_type == 'fno':
                    # Transpose back to [batch, spatial, channels]
                    next_state = pred.transpose(1, 2)
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


def print_rollout_diagnostics(avg_losses):
    """
    Prints simple diagnostics for averaged rollout losses.



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
