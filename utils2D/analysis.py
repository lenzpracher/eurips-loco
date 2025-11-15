"""
Analysis utilities for 2D neural operator experiments.

This module contains functions for analyzing model performance, rollout losses,
branch contributions, and other specialized analysis tasks.
"""

import os
import types

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def map_old_keys_to_new_2d(state_dict, model_type='LOCO'):
    """
    Map old state dict keys to new naming conventions for 2D models.
    The saved models use 'sno_conv' while current models expect 'loco_conv'.

    Args:
        state_dict: The loaded state dict from a saved model
        model_type: Type of model ('LOCO', 'Hybrid', 'FNO')

    Returns:
        Updated state dict with corrected keys
    """
    new_state_dict = {}

    for old_key, value in state_dict.items():
        new_key = old_key

        # Handle the main naming changes for 2D models
        if model_type in ['LOCO', 'Hybrid'] and 'sno_conv.weight' in old_key:
                new_key = old_key.replace('sno_conv.weight', 'loco_conv.weight')

        # No changes needed for FNO models - they don't use sno_conv

        new_state_dict[new_key] = value

    return new_state_dict

# Import custom 2-D models (simplified version)
from neuralop import H1Loss

# Import neuraloperator library components
from neuralop.models import FNO as NeuralOpFNO

from models2D import (
    LOCO,
    # NFNO2D,  # Not available in simplified version
    # FMLP,    # Not available in simplified version
)

# from .models import initialize_models, load_trained_models  # Not available in simplified version


def analyze_output_errors(device, rollout_steps=5, num_samples=10):
    """
    Analyzes the power spectrum of the rollout error for each model, all at training resolution.
    """
    # Import constants locally to avoid circular imports
    from moNS2D import DATA_DIR, PLOTS_DIR, TRAINING_RESOLUTION

    from .datasets import NS2DDataset

    # Use rollout plots directory if set
    plots_dir = os.environ.get('ROLLOUT_PLOTS_DIR', PLOTS_DIR)

    print(f"--- Analyzing output errors on {device} (rollout_steps={rollout_steps}, num_samples={num_samples}) ---")
    os.makedirs(plots_dir, exist_ok=True)

    # Use training resolution for all analysis
    test_dataset = NS2DDataset(DATA_DIR, split='test', target_resolution=TRAINING_RESOLUTION)
    loaded_models = load_trained_models(device, training_resolution=TRAINING_RESOLUTION)

    if not loaded_models:
        print("No models were loaded. Aborting analysis.")
        return

    # Initialize dictionaries to store accumulated power spectra at training resolution
    power_spectra = {name: torch.zeros(TRAINING_RESOLUTION, TRAINING_RESOLUTION, device=device) for name in loaded_models}
    gt_power_spectrum = torch.zeros(TRAINING_RESOLUTION, TRAINING_RESOLUTION, device=device)
    error_count = 0

    with torch.no_grad():
        for _i in tqdm(range(num_samples), desc="Analyzing samples"):
            sim_idx = np.random.randint(0, test_dataset.num_sims)
            max_start_t = test_dataset.num_samples_per_sim - rollout_steps
            if max_start_t <= 0:
                print(f"Skipping sim {sim_idx} as it's too short for rollout.")
                continue

            start_t = np.random.randint(0, max_start_t)

            gt_trajectory = test_dataset.data[sim_idx].to(device)

            x_initial_sequence = gt_trajectory[start_t : start_t + test_dataset.n_input_timesteps]
            x_initial_sequence = x_initial_sequence.permute(1, 2, 0)

            model_states = {name: x_initial_sequence.clone() for name in loaded_models}

            for step in range(rollout_steps):
                gt_timestep_idx = start_t + test_dataset.n_input_timesteps + step
                gt_frame = gt_trajectory[gt_timestep_idx]

                # Accumulate ground truth power spectrum
                fft_gt = torch.fft.fftshift(torch.fft.fft2(gt_frame, norm='ortho')).abs()**2
                gt_power_spectrum += fft_gt

                updated_states = {}
                for name, info in loaded_models.items():
                    current_state = model_states[name]
                    model = info['model']
                    model_type = info['type']

                    inp = current_state.unsqueeze(0)

                    if model_type == 'mlp':
                        # Downsampling handled by dataset
                        pass
                    elif model_type in ['fno', 'loco', 'nfno', 'fmlp', 'bno', 'ano', 'aano', 'cno', 'dno', 'zeno', 'cosno', 'kosno', 'ampsno', 'channel_hybrid']:
                        inp = inp.permute(0, 3, 1, 2)

                    out = model(inp)

                    if model_type in ['fno', 'sno', 'nfno', 'fmlp', 'bno', 'ano', 'aano', 'cno', 'dno', 'zeno', 'cosno', 'kosno', 'channel_nfno_sno']:
                        next_frame_pred = out.permute(0, 2, 3, 1).squeeze(0)
                    else: # MLP
                        next_frame_pred = out.squeeze(0)
                        # NO upsampling, all analysis at training resolution

                    error = next_frame_pred.squeeze(-1) - gt_frame

                    fft_error = torch.fft.fftshift(torch.fft.fft2(error, norm='ortho')).abs()**2

                    power_spectra[name] += fft_error

                    new_sequence = torch.cat([current_state[..., 1:], next_frame_pred], dim=-1)
                    updated_states[name] = new_sequence

                model_states = updated_states
                error_count += 1

    if error_count == 0:
        print("No samples were processed. Aborting plotting.")
        return

    print("\n--- Plotting Average Ground Truth Power Spectrum ---")
    avg_gt_ps = gt_power_spectrum / error_count

    plt.figure(figsize=(8, 7))
    plt.imshow(torch.log10(avg_gt_ps + 1e-12).cpu().numpy(), cmap='viridis', origin='lower')
    plt.colorbar(label='Log10(Power)')
    plt.title(f'Average GT Power Spectrum ({TRAINING_RESOLUTION}x{TRAINING_RESOLUTION})')
    plt.xlabel('Frequency (kx)')
    plt.ylabel('Frequency (ky)')

    plot_path = os.path.join(plots_dir, 'ground_truth_power_spectrum.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved ground truth spectrum plot for to {plot_path}")

    print("\n--- Plotting Average Error Power Spectra ---")
    for name, ps_sum in power_spectra.items():
        avg_ps = ps_sum / error_count

        plt.figure(figsize=(8, 7))
        plt.imshow(torch.log10(avg_ps + 1e-12).cpu().numpy(), cmap='viridis', origin='lower')
        plt.colorbar(label='Log10(Power)')
        plt.title(f'{name} - Average Error Power Spectrum (log scale)')
        plt.xlabel('Frequency (kx)')
        plt.ylabel('Frequency (ky)')

        plot_path = os.path.join(plots_dir, f'{name}_error_power_spectrum.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved error spectrum plot for {name} to {plot_path}")

    print("Analysis complete.")


def analyze_rollout_loss(device, rollout_steps=50, num_samples=10):
    """
    Analyzes and plots the rollout loss over time for each model.
    Creates two subplots: rollout MSE loss and rollout H1 loss comparison.
    Uses the same evaluation format as verified to be consistent with training.
    """
    # Import constants locally to avoid circular imports
    from moNS2D import DATA_DIR, PLOTS_DIR, TRAINING_RESOLUTION

    from .datasets import NS2DDataset

    # Use rollout plots directory if set
    plots_dir = os.environ.get('ROLLOUT_PLOTS_DIR', PLOTS_DIR)

    print(f"--- Analyzing rollout loss on {device} (rollout_steps={rollout_steps}, num_samples={num_samples}) ---")
    os.makedirs(plots_dir, exist_ok=True)

    # Set random seed for reproducible sample selection (same as training)
    np.random.seed(42)

    # Use same subset mode as training for consistency
    test_dataset = NS2DDataset(DATA_DIR, split='test', target_resolution=TRAINING_RESOLUTION,
                              subset_mode=True, subset_size=200)
    loaded_models = load_trained_models(device, training_resolution=TRAINING_RESOLUTION)

    if not loaded_models:
        print("No models were loaded. Aborting analysis.")
        return

    # Use both MSE and H1 loss for this analysis
    mse_loss_fn = nn.MSELoss()
    h1_loss_fn = H1Loss(d=2)

    # Store cumulative losses: {model_name: [step1_loss, step2_loss, ...]}
    rollout_mse_losses = {name: np.zeros(rollout_steps) for name in loaded_models}
    rollout_h1_losses = {name: np.zeros(rollout_steps) for name in loaded_models}

    # Track successful samples for averaging
    successful_samples = 0

    with torch.no_grad():
        for _i in tqdm(range(num_samples), desc="Analyzing samples"):
            sim_idx = np.random.randint(0, test_dataset.num_sims)

            # Ensure we have enough future timesteps for the full rollout
            max_start_t = test_dataset.num_samples_per_sim - test_dataset.n_input_timesteps - rollout_steps
            if max_start_t <= 0:
                continue # Skip sim if it's too short for full rollout

            start_t = np.random.randint(0, max_start_t)
            gt_trajectory = test_dataset.data[sim_idx].to(device)

            # Initial sequence construction - same as verification script
            x_initial_sequence = gt_trajectory[start_t : start_t + test_dataset.n_input_timesteps]
            x_initial_sequence = x_initial_sequence.permute(1, 2, 0)  # [H, W, C]

            # Initialize states for all models
            model_states = {name: x_initial_sequence.clone() for name in loaded_models}

            # Track if this sample completed successfully for all models
            sample_successful = True

            for step in range(rollout_steps):
                gt_timestep_idx = start_t + test_dataset.n_input_timesteps + step
                if gt_timestep_idx >= gt_trajectory.shape[0]:
                    sample_successful = False
                    break

                gt_frame = gt_trajectory[gt_timestep_idx]

                for name, info in loaded_models.items():
                    try:
                        current_state = model_states[name]
                        model = info['model']
                        model_type = info['type']

                        # Prepare input - same as verification script
                        inp = current_state.unsqueeze(0)  # [1, H, W, C]

                        if model_type in ['fno', 'loco', 'nfno', 'fmlp', 'bno', 'ano', 'aano', 'cno', 'dno', 'zeno', 'paper_hybrid', 'paper_fno', 'asno', 'paper_asno', 'basno', 'cosno', 'kosno', 'ampsno', 'channel_hybrid']:
                            inp = inp.permute(0, 3, 1, 2)  # [1, C, H, W]

                        # Forward pass
                        out = model(inp)

                        # Loss computation - EXACTLY like verification script
                        if model_type in ['fno', 'loco', 'nfno', 'fmlp', 'bno', 'ano', 'aano', 'cno', 'dno', 'zeno', 'paper_hybrid', 'paper_fno', 'asno', 'paper_asno', 'basno', 'cosno', 'kosno', 'ampsno', 'channel_hybrid']:
                            # Keep output in [1, C, H, W], expand gt to match
                            gt_frame_batch = gt_frame.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

                            # Calculate losses in the same format as training: [B, C, H, W]
                            mse_loss = mse_loss_fn(out, gt_frame_batch)
                            h1_loss = h1_loss_fn(out, gt_frame_batch)

                            rollout_mse_losses[name][step] += mse_loss.item()
                            rollout_h1_losses[name][step] += h1_loss.item()

                            # For sequence update, permute back to [H, W, C] format
                            next_frame_pred = out.permute(0, 2, 3, 1).squeeze(0)  # [H, W, C]

                        else:  # MLP models
                            # MLP models use different format
                            next_frame_pred = out.squeeze(0)  # [H, W, C] or similar

                            # MSE loss for MLP
                            mse_loss = mse_loss_fn(next_frame_pred.squeeze(), gt_frame)
                            rollout_mse_losses[name][step] += mse_loss.item()

                            # H1 loss for MLP - match training format
                            pred_for_h1 = next_frame_pred.unsqueeze(0)  # [1, H, W, C]
                            gt_for_h1 = gt_frame.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
                            pred_h1_batch = pred_for_h1.permute(0, 3, 1, 2)  # [1, C, H, W]
                            gt_h1_batch = gt_for_h1.permute(0, 3, 1, 2)      # [1, 1, H, W]
                            h1_loss = h1_loss_fn(pred_h1_batch, gt_h1_batch)
                            rollout_h1_losses[name][step] += h1_loss.item()

                        # Update state for next step - handle variable channel sizes
                        if next_frame_pred.dim() == 3:  # [H, W, C]
                            new_sequence = torch.cat([current_state[..., 1:], next_frame_pred], dim=-1)
                        elif next_frame_pred.dim() == 2:  # [H, W]
                            next_frame_pred = next_frame_pred.unsqueeze(-1)  # [H, W, 1]
                            new_sequence = torch.cat([current_state[..., 1:], next_frame_pred], dim=-1)
                        else:
                            raise ValueError(f"Unexpected prediction shape: {next_frame_pred.shape}")

                        model_states[name] = new_sequence

                    except Exception as e:
                        print(f"Error processing {name} at step {step}: {e}")
                        sample_successful = False
                        break

                if not sample_successful:
                    break

            if sample_successful:
                successful_samples += 1

    if successful_samples == 0:
        print("No samples were successfully processed. Aborting analysis.")
        return

    print(f"Successfully processed {successful_samples}/{num_samples} samples")

    # Average the losses over the successful samples
    for name in rollout_mse_losses:
        if successful_samples > 0:
            rollout_mse_losses[name] /= successful_samples
            rollout_h1_losses[name] /= successful_samples


def analyze_rollout_loss_paper_runs(device, rollout_steps=50, num_samples=10, h1_only=False, return_data=False):
    """
    Analyzes rollout loss for multiple parallel-trained runs with mean and standard deviation.
    Loads models from the paper runs directory and computes statistics across runs.

    Args:
        device: Device to run analysis on
        rollout_steps: Number of rollout steps to analyze
        num_samples: Number of samples to use for analysis
        h1_only: If True, only compute H1 loss (skip MSE for efficiency)
        return_data: If True, return rollout data instead of just plotting
    """
    # Import constants from simplified script instead of old moNS2D
    try:
        import navier_stokes_2d as ns2d
        DATA_DIR = ns2d.DATA_DIR
        TRAINING_RESOLUTION = ns2d.TRAINING_RESOLUTION
        CHECKPOINTS_DIR = ns2d.CHECKPOINTS_DIR
    except ImportError:
        # Fallback values if import fails
        DATA_DIR = 'data/NS2D'
        TRAINING_RESOLUTION = 64
        CHECKPOINTS_DIR = 'checkpoints/NS2D'

    from .data import NS2DDataset

    # Use paper-specific plots directory for paper runs
    plots_dir = os.environ.get('ROLLOUT_PLOTS_DIR', os.path.join('plots', 'moNS2D_paper'))

    print(f"--- Analyzing rollout loss for paper runs on {device} (rollout_steps={rollout_steps}, num_samples={num_samples}) ---")
    os.makedirs(plots_dir, exist_ok=True)

    # Set random seed for reproducible sample selection
    np.random.seed(42)

    # Load test dataset
    test_dataset = NS2DDataset(DATA_DIR, split='test', target_resolution=TRAINING_RESOLUTION,
                              subset_mode=True, subset_size=200)

    # Paper runs directory (same pattern as parallel training)
    paper_runs_dir = CHECKPOINTS_DIR + '_paper_runs'
    if not os.path.exists(paper_runs_dir):
        print(f"Paper runs directory not found: {paper_runs_dir}")
        print("Please run --mode parallel-train-runs first to generate paper training data.")
        return

    # Find model directories
    paper_models = ['LOCO', 'Hybrid', 'FNO']
    model_runs_data = {}

    for model_name in paper_models:
        model_dir = os.path.join(paper_runs_dir, model_name)
        if not os.path.exists(model_dir):
            print(f"Model directory not found: {model_dir}")
            continue

        # Find run directories
        run_dirs = [d for d in os.listdir(model_dir)
                   if os.path.isdir(os.path.join(model_dir, d)) and d.startswith('run_')]

        if not run_dirs:
            print(f"No run directories found for {model_name}")
            continue

        print(f"Found {len(run_dirs)} runs for {model_name}")
        model_runs_data[model_name] = {
            'run_dirs': sorted(run_dirs),
            'model_dir': model_dir
        }

    if not model_runs_data:
        print("No valid model runs found")
        return

    # Initialize loss functions based on mode
    if not h1_only:
        mse_loss_fn = nn.MSELoss()
    h1_loss_fn = H1Loss(d=2)

    # Store results for each model and run
    all_results = {}

    for model_name, model_data in model_runs_data.items():
        print(f"\n🔄 Processing {model_name}...")
        all_results[model_name] = {}

        for run_dir in model_data['run_dirs']:
            run_path = os.path.join(model_data['model_dir'], run_dir)
            checkpoint_path = os.path.join(run_path, f'{model_name}_best.pth')

            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found: {checkpoint_path}")
                continue

            print(f"  Loading {run_dir}...")

            # Load model for this run
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model = initialize_single_model(model_name, device, TRAINING_RESOLUTION)

                # Apply compatibility mapping for old naming conventions
                compatible_state_dict = map_old_keys_to_new_2d(checkpoint['model_state_dict'], model_name)
                model.load_state_dict(compatible_state_dict)
                model.eval()
            except Exception as e:
                print(f"  Failed to load {run_dir}: {e}")
                continue

            # Run rollout analysis for this model
            if not h1_only:
                rollout_mse_losses = np.zeros(rollout_steps)
            rollout_h1_losses = np.zeros(rollout_steps)
            successful_samples = 0

            with torch.no_grad():
                for _i in tqdm(range(num_samples), desc=f"  {run_dir} samples", leave=False):
                    sim_idx = np.random.randint(0, test_dataset.num_sims)

                    # Ensure we have enough future timesteps
                    max_start_t = test_dataset.num_samples_per_sim - test_dataset.n_input_timesteps - rollout_steps
                    if max_start_t <= 0:
                        continue

                    start_t = np.random.randint(0, max_start_t)
                    gt_trajectory = test_dataset.data[sim_idx].to(device)

                    # Initial sequence
                    x_initial_sequence = gt_trajectory[start_t : start_t + test_dataset.n_input_timesteps]
                    x_initial_sequence = x_initial_sequence.permute(1, 2, 0)  # [H, W, C]
                    current_state = x_initial_sequence.clone()

                    sample_successful = True

                    for step in range(rollout_steps):
                        try:
                            gt_timestep_idx = start_t + test_dataset.n_input_timesteps + step
                            gt_frame = gt_trajectory[gt_timestep_idx]

                            # Model prediction
                            inp = current_state.unsqueeze(0).permute(0, 3, 1, 2)  # [1, C, H, W]
                            out = model(inp)
                            next_frame_pred = out.permute(0, 2, 3, 1).squeeze(0)  # [H, W, C] or [H, W, 1]

                            if next_frame_pred.dim() == 2:
                                next_frame_pred = next_frame_pred.unsqueeze(-1)

                            pred_channel = next_frame_pred[..., 0]  # Use first channel

                            # Compute losses based on mode
                            if not h1_only:
                                mse_loss = mse_loss_fn(pred_channel, gt_frame)
                                rollout_mse_losses[step] += mse_loss.item()

                            # H1 loss
                            pred_for_h1 = pred_channel.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                            gt_for_h1 = gt_frame.unsqueeze(0).unsqueeze(0)        # [1, 1, H, W]
                            h1_loss = h1_loss_fn(pred_for_h1, gt_for_h1)
                            rollout_h1_losses[step] += h1_loss.item()

                            # Update state
                            new_sequence = torch.cat([current_state[..., 1:], next_frame_pred], dim=-1)
                            current_state = new_sequence

                        except Exception as e:
                            print(f"    Error at step {step}: {e}")
                            sample_successful = False
                            break

                    if sample_successful:
                        successful_samples += 1

            if successful_samples > 0:
                if not h1_only:
                    rollout_mse_losses /= successful_samples
                rollout_h1_losses /= successful_samples

                result_data = {
                    'h1': rollout_h1_losses.copy(),
                    'successful_samples': successful_samples
                }
                if not h1_only:
                    result_data['mse'] = rollout_mse_losses.copy()

                all_results[model_name][run_dir] = result_data
                print(f"  {run_dir}: {successful_samples}/{num_samples} samples successful")
            else:
                print(f"  {run_dir}: No successful samples")

    # Compute statistics across runs
    model_stats = {}
    for model_name, runs_data in all_results.items():
        if not runs_data:
            continue

        # Collect data from all runs
        h1_data = np.array([data['h1'] for data in runs_data.values()])

        # Compute mean and percentiles for H1
        h1_mean = np.mean(h1_data, axis=0)
        h1_p10 = np.percentile(h1_data, 10, axis=0)
        h1_p90 = np.percentile(h1_data, 90, axis=0)

        model_stats[model_name] = {
            'h1_mean': h1_mean,
            'h1_p10': h1_p10,
            'h1_p90': h1_p90,
            'num_runs': len(runs_data)
        }

        # Add MSE data if available
        if not h1_only and all('mse' in data for data in runs_data.values()):
            mse_data = np.array([data['mse'] for data in runs_data.values()])
            mse_mean = np.mean(mse_data, axis=0)
            mse_p10 = np.percentile(mse_data, 10, axis=0)
            mse_p90 = np.percentile(mse_data, 90, axis=0)

            model_stats[model_name].update({
                'mse_mean': mse_mean,
                'mse_p10': mse_p10,
                'mse_p90': mse_p90
            })

        print(f"{model_name}: Aggregated {len(runs_data)} runs")

    if not model_stats:
        print("No valid results to plot")
        return None if return_data else None

    # Return data if requested
    if return_data:
        return model_stats

    # Create plots
    if h1_only:
        # Single H1 plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
        plot_titles = ['H1 Rollout Loss']
    else:
        # Both MSE and H1 plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        axes = [ax1, ax2]
        plot_titles = ['MSE Rollout Loss', 'H1 Rollout Loss']

    # Color scheme
    colors = {'LOCO': '#1f77b4', 'Hybrid': '#ff7f0e', 'FNO': '#2ca02c'}

    steps = np.arange(rollout_steps)

    # Plot based on available data
    for _plot_idx, (ax, title) in enumerate(zip(axes, plot_titles)):
        loss_type = 'mse' if 'MSE' in title else 'h1'

        for model_name, stats in model_stats.items():
            color = colors.get(model_name, '#d62728')

            # Check if this loss type is available
            mean_key = f'{loss_type}_mean'
            p10_key = f'{loss_type}_p10'
            p90_key = f'{loss_type}_p90'

            if mean_key in stats:
                mean = stats[mean_key]
                p10 = stats[p10_key]
                p90 = stats[p90_key]

                ax.plot(steps, mean, 'o-', label=f"{model_name}",
                       color=color, markersize=3, alpha=0.8)
                ax.fill_between(steps, p10, p90, alpha=0.2, color=color)

        ax.set_title(f'Rollout {title} (Mean + 10th-90th Percentile)')
        ax.set_xlabel('Rollout Step')
        ax.set_ylabel(f'{title} (Log Scale)')
        ax.set_yscale('log')
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save plot
    suffix = '_h1_only' if h1_only else ''
    plot_filename = os.path.join(plots_dir, f'rollout_loss_paper_runs_comparison{suffix}.pdf')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

    print("\nPaper runs rollout analysis complete!")
    print(f"Plot saved: {plot_filename}")

    # Print summary statistics
    print(f"\n{'='*70}")
    print("PAPER RUNS ROLLOUT LOSS SUMMARY")
    print(f"{'='*120}")
    print(f"{'Model':<10} | {'Runs':<5} | {'Step 0 H1':<35} | {'Step 5 H1':<35} | {'Step 10 H1':<35}")
    print(f"{'-'*10} | {'-'*5} | {'-'*35} | {'-'*35} | {'-'*35}")

    for model_name, stats in model_stats.items():
        h1_mean = stats['h1_mean']
        h1_p10 = stats['h1_p10']
        h1_p90 = stats['h1_p90']
        num_runs = stats['num_runs']

        # Format with percentile ranges (same as 1D analysis)
        step0 = f"{h1_mean[0]:.4f}[{h1_p10[0]:.4f}-{h1_p90[0]:.4f}]" if len(h1_mean) > 0 else "N/A"
        step5 = f"{h1_mean[5]:.4f}[{h1_p10[5]:.4f}-{h1_p90[5]:.4f}]" if len(h1_mean) > 5 else "N/A"
        step10 = f"{h1_mean[10]:.4f}[{h1_p10[10]:.4f}-{h1_p90[10]:.4f}]" if len(h1_mean) > 10 else "N/A"

        print(f"{model_name:<10} | {num_runs:<5} | {step0:<35} | {step5:<35} | {step10:<35}")

    print(f"{'='*120}")


def initialize_single_model(model_name, device, training_resolution):
    """Helper function to initialize a single model for rollout analysis."""
    # Import from navier_stokes_2d to use the same initialization function
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from navier_stokes_2d import initialize_models

    models_dict = initialize_models(device, training_resolution)

    if model_name in models_dict:
        return models_dict[model_name]['model']
    else:
        raise ValueError(f"Unknown model: {model_name}")




def run_mode_analysis(gpu_id=0, epochs=25):
    """
    Analyzes model performance by varying the number of modes.
    Trains LOCO, Hybrid, FNO, and FMLP models with different mode counts for a fixed number of epochs
    and plots the final validation H1 loss against the number of modes.
    """
    # Import constants locally to avoid circular imports
    from moNS2D import (
        CHANNELS,
        DATA_DIR,
        GLOBAL_BATCH_SIZE,
        HIDDEN_CHANNELS,
        LEARNING_RATE,
        N_OUTPUT_TIMESTEPS,
        NUM_BLOCKS,
        PAPER_TRAINING_SAMPLES,
        PLOTS_DIR,
        TRAINING_RESOLUTION,
    )
    from torch.utils.data import DataLoader

    # Use rollout plots directory if set
    plots_dir = os.environ.get('ROLLOUT_PLOTS_DIR', PLOTS_DIR)

    os.makedirs(plots_dir, exist_ok=True)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running Mode Analysis on {device} ---")

    # --- Data Loading ---
    train_dataset = NS2DDataset(DATA_DIR, split='train', target_resolution=TRAINING_RESOLUTION)
    test_dataset = NS2DDataset(DATA_DIR, split='test', target_resolution=TRAINING_RESOLUTION)
    train_loader = DataLoader(train_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=False)

    # --- Analysis Parameters ---
    modes_to_test = [4, 8, 12, 16, 20, 24, 32]
    analysis_epochs = epochs
    models_to_analyze = ["LOCO", "Hybrid", "FNO_legacy", "FMLP"]
    results = {model_name: [] for model_name in models_to_analyze}
    loss_fn = H1Loss(d=2)
    max_steps = PAPER_TRAINING_SAMPLES // GLOBAL_BATCH_SIZE

    for model_name in models_to_analyze:
        print(f"\n===== Analyzing {model_name} =====")
        for modes in modes_to_test:
            print(f"  --- Testing with {modes}x{modes} modes ---")

            # --- Model Initialization ---
            if model_name == "LOCO":
                model = LOCO(
                    in_channels=CHANNELS, out_channels=N_OUTPUT_TIMESTEPS,
                    hidden_channels=HIDDEN_CHANNELS, num_blocks=NUM_BLOCKS,
                    modes_x=modes, modes_y=modes).to(device)
                model_type = 'loco'
            elif model_name == "Hybrid":
                 model = NFNO2D(
                    in_channels=CHANNELS, out_channels=N_OUTPUT_TIMESTEPS,
                    hidden_channels=HIDDEN_CHANNELS, modes_x=modes, modes_y=modes,
                    n_layers=NUM_BLOCKS, use_sno=True).to(device)
                 model_type = 'nfno'
            elif model_name == "FNO_legacy":
                model = NeuralOpFNO(
                    n_modes=(modes, modes), in_channels=CHANNELS, out_channels=N_OUTPUT_TIMESTEPS,
                    hidden_channels=HIDDEN_CHANNELS, n_layers=NUM_BLOCKS, use_norm=True).to(device)
                model_type = 'fno'
            elif model_name == "FMLP":
                 model = FMLP(
                    in_channels=CHANNELS, out_channels=N_OUTPUT_TIMESTEPS,
                    hidden_channels=HIDDEN_CHANNELS * 4, modes_x=modes, modes_y=modes,
                    num_layers=3).to(device)
                 model_type = 'fmlp'
            else:
                continue

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # --- Training Loop ---
            from .training import evaluate_model, train_model
            for _epoch in range(analysis_epochs):
                train_model(model, train_loader, optimizer, loss_fn, device, model_type, rank=0,
                            model_name_for_tqdm=f"{model_name}-{modes} modes", max_steps_per_epoch=max_steps)

            # --- Evaluation ---
            val_loss = evaluate_model(model, test_loader, loss_fn, device, model_type)
            print(f"  --> Final Validation H1 Loss for {modes} modes: {val_loss:.6f}")
            results[model_name].append(val_loss)

            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Plotting ---
    plt.figure(figsize=(12, 8))
    for model_name, losses in results.items():
        if losses:
            plt.plot(modes_to_test, losses, 'o-', label=model_name)

    plt.title(f'Final H1 Loss vs. Number of Modes (after {analysis_epochs} epochs)')
    plt.xlabel('Number of Modes (per dimension)')
    plt.ylabel('Final Validation H1 Loss')
    plt.yscale('log')
    plt.xticks(modes_to_test)
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    save_path = os.path.join(plots_dir, 'mode_analysis_loss_vs_modes.png')
    plt.savefig(save_path, dpi=300)
    print(f"Mode analysis plot saved to {save_path}")


def analyze_hybrid_branches(gpu_id=0, num_samples=50):
    """
    Analyzes the relative contributions of the LOCO vs. FNO branches in the Hybrid model.
    Uses hooks to capture the output of each branch and calculates statistics on their contributions.
    """
    # Import constants locally to avoid circular imports
    from moNS2D import CHECKPOINTS_DIR, DATA_DIR, GLOBAL_BATCH_SIZE, TRAINING_RESOLUTION

    from .datasets import NS2DDataset

    # Use rollout checkpoints directory if set
    checkpoints_dir = os.environ.get('ROLLOUT_CHECKPOINTS_DIR', CHECKPOINTS_DIR)

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print(f"NFNO-LOCO Branch Analysis on {device}")
    print("=" * 60)

    # --- Load Dataset and Model ---
    from torch.utils.data import DataLoader
    dataset = NS2DDataset(DATA_DIR, split='test', target_resolution=TRAINING_RESOLUTION)
    models_dict = initialize_models(device, training_resolution=TRAINING_RESOLUTION)

    hybrid_model = models_dict['Hybrid']['model']
    ckpt_path = os.path.join(checkpoints_dir, 'Hybrid_best.pth')

    if not os.path.exists(ckpt_path):
        print(f"Error: Hybrid checkpoint not found at {ckpt_path}")
        print("Please train the Hybrid model first.")
        return

    try:
        hybrid_model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
    except KeyError:
        hybrid_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    hybrid_model.to(device)
    hybrid_model.eval()

    # --- Setup Hooks ---
    branch_outputs = {'sno': [], 'fno': []}

    def make_hook(branch_name):
        def hook(module, input, output):
            branch_outputs[branch_name].append(output.detach())
        return hook

    hooks = []
    for block in hybrid_model.blocks:
        if hasattr(block, 'use_sno') and block.use_sno:
            hooks.append(block.sno_conv.register_forward_hook(make_hook('sno')))
            hooks.append(block.fourier.register_forward_hook(make_hook('fno')))

    if not hooks:
        print("Error: Could not register hooks. Check model architecture (NFNO2D and FNOBlock2D).")
        return

    # --- Collect Statistics ---
    sno_stats = {'magnitudes': [], 'energies': [], 'variances': []}
    fno_stats = {'magnitudes': [], 'energies': [], 'variances': []}

    print(f"Analyzing {num_samples} samples...")

    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Analyzing samples"):
            sample = dataset[i]
            x = sample['x'].unsqueeze(0).to(device).permute(0, 3, 1, 2) # [B,H,W,C] -> [B,C,H,W]

            branch_outputs['sno'].clear()
            branch_outputs['fno'].clear()
            _ = hybrid_model(x)

            if branch_outputs['sno'] and branch_outputs['fno']:
                sno_out = torch.cat(branch_outputs['sno'], dim=1) # Concat over blocks
                fno_out = torch.cat(branch_outputs['fno'], dim=1)

                sno_stats['magnitudes'].append(torch.norm(sno_out, p='fro', dim=(-2,-1)).mean().item())
                fno_stats['magnitudes'].append(torch.norm(fno_out, p='fro', dim=(-2,-1)).mean().item())

                sno_stats['energies'].append((sno_out ** 2).mean().item())
                fno_stats['energies'].append((fno_out ** 2).mean().item())

                sno_stats['variances'].append(sno_out.var().item())
                fno_stats['variances'].append(fno_out.var().item())

    for hook in hooks:
        hook.remove()

    if not sno_stats['magnitudes']:
        print("Warning: No branch outputs were captured.")
        return

    # --- Summarize and Plot ---
    def summarize(stats, name):
        print(f"\n{name} Branch Statistics (avg over samples):")
        mag_mean, mag_std = np.mean(stats['magnitudes']), np.std(stats['magnitudes'])
        eng_mean, eng_std = np.mean(stats['energies']), np.std(stats['energies'])
        var_mean, var_std = np.mean(stats['variances']), np.std(stats['variances'])
        print(f"  Magnitude : {mag_mean:.4f} ± {mag_std:.4f}")
        print(f"  Energy    : {eng_mean:.4f} ± {eng_std:.4f}")
        print(f"  Variance  : {var_mean:.4f} ± {var_std:.4f}")
        return {'mag': mag_mean, 'eng': eng_mean, 'var': var_mean}

    sno_summary = summarize(sno_stats, "LOCO")
    fno_summary = summarize(fno_stats, "FNO")

    # --- Relative Contributions ---
    print("\n" + "=" * 50)
    print("RELATIVE CONTRIBUTIONS")
    print("=" * 50)

    labels = ['Magnitude', 'Energy', 'Variance']
    sno_vals = [sno_summary['mag'], sno_summary['eng'], sno_summary['var']]
    fno_vals = [fno_summary['mag'], fno_summary['eng'], fno_summary['var']]

    total_vals = [s + f if (s + f) > 0 else 1 for s, f in zip(sno_vals, fno_vals)]
    sno_pct = [(s / t) * 100 for s, t in zip(sno_vals, total_vals)]
    fno_pct = [(f / t) * 100 for f, t in zip(fno_vals, total_vals)]

    for i, label in enumerate(labels):
        print(f"\nBy Output {label}:")
        print(f"  LOCO branch: {sno_pct[i]:.1f}%")
        print(f"  FNO branch: {fno_pct[i]:.1f}%")

    # --- Interpretation ---
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)

    sno_wins = sum(1 for s_pct in sno_pct if s_pct > 50)
    dominant_branch = "LOCO" if sno_wins > 1 else "FNO" if sno_wins < 2 else "BALANCED"

    print(f"Dominant branch: {dominant_branch}")
    if dominant_branch == "LOCO":
        print("INTERPRETATION: The LOCO branch appears to be the primary contributor.")
        print("This suggests that the model is relying more on the local, non-linear spectral operations, which can be effective for capturing complex, multi-scale interactions in turbulence.")
    elif dominant_branch == "FNO":
        print("INTERPRETATION: The FNO branch appears to be the primary contributor.")
        print("This suggests that the model is relying more on the global linear operations in the Fourier domain. This is the expected behavior for standard FNOs and is effective at capturing large-scale wave dynamics.")
    else:
        print("INTERPRETATION: Both branches contribute significantly.")
        print("This suggests the hybrid approach is genuinely beneficial, with both local spectral non-linearities (LOCO) and global linear updates (FNO) playing a key role in modeling the Navier-Stokes dynamics.")

    # --- Ablation Study ---
    print("\n" + "=" * 50)
    print("ABLATION STUDY: VALIDATION LOSS")
    print("=" * 50)

    loss_fn = H1Loss(d=2)
    test_loader = DataLoader(dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=False)

    print("Evaluating full NFNO-LOCO model...")
    from .training import evaluate_model
    original_val_loss = evaluate_model(hybrid_model, test_loader, loss_fn, device, model_type='nfno')
    print(f"  > Full Model Validation H1 Loss: {original_val_loss:.6f}")

    # Store original forward methods before monkey-patching
    original_sno_forwards = [block.sno_conv.forward for block in hybrid_model.blocks]
    original_fno_forwards = [block.fourier.forward for block in hybrid_model.blocks]

    try:
        # 1. Ablate LOCO branch (FNO-only)
        print("\nEvaluating with LOCO branch disabled (FNO-only)...")
        def sno_off_forward(self, x):
            return torch.zeros_like(x)

        for block in hybrid_model.blocks:
            block.sno_conv.forward = types.MethodType(sno_off_forward, block.sno_conv)

        fno_only_val_loss = evaluate_model(hybrid_model, test_loader, loss_fn, device, model_type='nfno')
        print(f"  > FNO-Only Validation H1 Loss: {fno_only_val_loss:.6f}")

        # 2. Ablate FNO branch (LOCO-only)
        print("\nEvaluating with FNO branch disabled (LOCO-only)...")
        def fno_off_forward(self, x):
            return torch.zeros_like(x)

        # Restore LOCO and disable FNO
        for i, block in enumerate(hybrid_model.blocks):
            block.sno_conv.forward = original_sno_forwards[i]
            block.fourier.forward = types.MethodType(fno_off_forward, block.fourier)

        sno_only_val_loss = evaluate_model(hybrid_model, test_loader, loss_fn, device, model_type='nfno')
        print(f"  > LOCO-Only Validation H1 Loss: {sno_only_val_loss:.6f}")

    finally:
        # Restore all original methods
        for i, block in enumerate(hybrid_model.blocks):
            block.sno_conv.forward = original_sno_forwards[i]
            block.fourier.forward = original_fno_forwards[i]
        print("\nOriginal model methods have been restored.")

    # --- Ablation Interpretation ---
    print("\n" + "=" * 50)
    print("ABLATION INTERPRETATION")
    print("=" * 50)
    print(f"  - Original Model Loss : {original_val_loss:.6f}")
    print(f"  - LOCO-Only Loss       : {sno_only_val_loss:.6f} (lower is better)")
    print(f"  - FNO-Only Loss       : {fno_only_val_loss:.6f} (lower is better)")

    sno_impact = sno_only_val_loss - original_val_loss
    fno_impact = fno_only_val_loss - original_val_loss

    print(f"\nImpact of removing the FNO branch (LOCO-only vs Original): {fno_impact:+.6f}")
    print(f"Impact of removing the LOCO branch (FNO-only vs Original): {sno_impact:+.6f}")

    if abs(fno_impact) > abs(sno_impact):
        print("\nConclusion: The FNO branch appears to be more critical for this task.")
        print("Its removal causes a larger change in validation error.")
    else:
        print("\nConclusion: The LOCO branch appears to be more critical for this task.")
        print("Its removal causes a larger change in validation error.")

    if sno_only_val_loss < original_val_loss or fno_only_val_loss < original_val_loss:
        print("\nInteresting find! One of the ablated models performs better than the full model.")
        print("This could suggest that a simpler, non-hybrid architecture might be superior for this problem.")

    print("=" * 60)




class EnergyAnomalyDetector(nn.Module):
    """
    Energy-based anomaly detector for Hybrid activations with high capacity.

    This model takes concatenated LOCO and FNO branch activations and outputs
    a scalar energy score. Lower energy indicates normal behavior, higher
    energy indicates anomalous behavior.

    Uses a deeper, wider architecture to leverage available compute.
    """

    def __init__(self, input_dim, hidden_dims=None, dropout=0.1):
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final energy output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform for better training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)

    def energy(self, x):
        """Compute energy score. Lower = normal, higher = anomalous."""
        return self.forward(x).squeeze(-1)


def analyze_activation_anomalies_energy(device, baseline_samples=500, rollout_steps=16, num_rollout_sequences=50,
                                       energy_epochs=100, pos_weight=3.0, model_name='Hybrid'):
    """
    Energy-based activation anomaly analysis for Hybrid model.

    Replaces the CPU-intensive Isolation Forest approach with a GPU-friendly energy-based model
    that focuses specifically on LOCO and FNO branch activations.

    Args:
        device: Device to run analysis on
        baseline_samples: Number of samples for training energy model (default: 500)
        rollout_steps: Maximum number of rollout steps to analyze (default: 16)
        num_rollout_sequences: Number of rollout sequences to analyze (default: 50)
        energy_epochs: Number of epochs to train energy model (default: 100)
        pos_weight: Weight for positive examples in loss (default: 3.0)
        model_name: Name of the model to analyze (must be 'Hybrid')
    """
    # Import constants locally to avoid circular imports
    from moNS2D import (
        DATA_DIR,
        GLOBAL_BATCH_SIZE,
        HIDDEN_CHANNELS,
        NUM_BLOCKS,
        PLOTS_DIR,
        TRAINING_RESOLUTION,
    )
    from torch.utils.data import DataLoader

    from .datasets import NS2DDataset

    # Use rollout plots directory if set
    plots_dir = os.environ.get('ROLLOUT_PLOTS_DIR', PLOTS_DIR)
    checkpoints_dir = os.environ.get('ROLLOUT_CHECKPOINTS_DIR', PLOTS_DIR.replace('plots', 'checkpoints'))

    print("=" * 70)
    print(f"ENERGY-BASED ACTIVATION ANOMALY ANALYSIS FOR {model_name}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Baseline samples: {baseline_samples}")
    print(f"Rollout steps: {rollout_steps}")
    print(f"Rollout sequences: {num_rollout_sequences}")
    print(f"Energy training epochs: {energy_epochs}")
    print(f"Positive sample weight: {pos_weight}")
    print("=" * 70)

    if model_name != 'Hybrid':
        print(f"Error: Analysis designed for Hybrid model, got {model_name}")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # Load dataset and model
    test_dataset = NS2DDataset(DATA_DIR, split='test', target_resolution=TRAINING_RESOLUTION)
    models_dict = initialize_models(device, training_resolution=TRAINING_RESOLUTION)

    model = models_dict[model_name]['model']
    model_type = models_dict[model_name]['type']

    # Load trained model
    ckpt_path = os.path.join(checkpoints_dir, f'{model_name}_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"Error: {model_name} checkpoint not found at {ckpt_path}")
        print(f"Please train the {model_name} model first.")
        return

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Apply compatibility mapping for old naming conventions
        compatible_state_dict = map_old_keys_to_new_2d(state_dict, model_name)
        model.load_state_dict(compatible_state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.to(device)
    model.eval()

    print(f"Model {model_name} loaded successfully")

    # Calculate expected feature dimension
    # Hybrid splits hidden_channels in half for each branch: hidden_channels/2 per branch per layer
    # Total: (hidden_channels/2 + hidden_channels/2) * num_layers = hidden_channels * num_layers
    feature_dim = HIDDEN_CHANNELS * NUM_BLOCKS
    print(f"Expected feature dimension: {feature_dim} (hidden_channels={HIDDEN_CHANNELS} * num_blocks={NUM_BLOCKS})")

    # Setup focused activation collection for LOCO/FNO branches
    sno_activations = []
    fno_activations = []
    hooks = []

    def make_sno_hook():
        def hook(module, input, output):
            # Flatten and store LOCO activations
            sno_activations.append(output.detach().flatten(start_dim=1))  # [batch, features]
        return hook

    def make_fno_hook():
        def hook(module, input, output):
            # Flatten and store FNO activations
            fno_activations.append(output.detach().flatten(start_dim=1))  # [batch, features]
        return hook

    # Register hooks only on LOCO and FNO branches
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'use_sno') and block.use_sno:
            if hasattr(block, 'sno_conv'):
                hooks.append(block.sno_conv.register_forward_hook(make_sno_hook()))
            if hasattr(block, 'fourier'):
                hooks.append(block.fourier.register_forward_hook(make_fno_hook()))
        else:
            print(f"Warning: Block {i} does not have LOCO/FNO branches")

    if not hooks:
        print("Error: No LOCO/FNO hooks registered")
        return

    print(f"Registered {len(hooks)} hooks on LOCO/FNO branches")

    # Phase 1: Generate positive and negative training data
    print("\nPhase 1: Generating training data...")

    positive_features = []
    negative_features = []

    from moNS2D import custom_collate_fn
    data_loader = DataLoader(test_dataset, batch_size=min(8, GLOBAL_BATCH_SIZE), shuffle=True, collate_fn=custom_collate_fn)

    collected_positive = 0
    collected_negative = 0
    target_samples = baseline_samples // 2  # Half positive, half negative

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Collecting training data"):
            if collected_positive >= target_samples and collected_negative >= target_samples:
                break

            x = data['x'].to(device)
            if model_type in ['loco', 'nfno', 'fno', 'fmlp', 'bno', 'ano', 'aano', 'cno', 'dno', 'zeno', 'cosno', 'kosno', 'ampsno', 'channel_hybrid']:
                x = x.permute(0, 3, 1, 2)

            batch_size = x.shape[0]

            # Generate positive examples (real data)
            if collected_positive < target_samples:
                sno_activations.clear()
                fno_activations.clear()
                _ = model(x)

                if sno_activations and fno_activations:
                    # Concatenate LOCO and FNO activations across all layers
                    sno_concat = torch.cat(sno_activations, dim=1)  # [batch, sno_features]
                    fno_concat = torch.cat(fno_activations, dim=1)  # [batch, fno_features]
                    combined_features = torch.cat([sno_concat, fno_concat], dim=1)  # [batch, total_features]

                    for i in range(min(batch_size, target_samples - collected_positive)):
                        positive_features.append(combined_features[i].cpu())
                    collected_positive += min(batch_size, target_samples - collected_positive)

            # Generate negative examples (noisy data)
            if collected_negative < target_samples:
                # Create noisy version of input
                noise = torch.randn_like(x) * 0.5  # Scale noise appropriately
                x_noisy = x + noise

                sno_activations.clear()
                fno_activations.clear()
                _ = model(x_noisy)

                if sno_activations and fno_activations:
                    # Concatenate LOCO and FNO activations across all layers
                    sno_concat = torch.cat(sno_activations, dim=1)
                    fno_concat = torch.cat(fno_activations, dim=1)
                    combined_features = torch.cat([sno_concat, fno_concat], dim=1)

                    for i in range(min(batch_size, target_samples - collected_negative)):
                        negative_features.append(combined_features[i].cpu())
                    collected_negative += min(batch_size, target_samples - collected_negative)

    print(f"Collected {collected_positive} positive and {collected_negative} negative training samples")

    if not positive_features or not negative_features:
        print("Error: Failed to collect training data")
        return

    # Prepare training data
    positive_tensor = torch.stack(positive_features)
    negative_tensor = torch.stack(negative_features)
    actual_feature_dim = positive_tensor.shape[1]

    print(f"Actual feature dimension: {actual_feature_dim}")

    # Create labels (0 for positive/normal, 1 for negative/anomalous)
    pos_labels = torch.zeros(len(positive_features))
    neg_labels = torch.ones(len(negative_features))

    train_features = torch.cat([positive_tensor, negative_tensor], dim=0)
    train_labels = torch.cat([pos_labels, neg_labels], dim=0)

    # Shuffle training data
    perm = torch.randperm(len(train_features))
    train_features = train_features[perm]
    train_labels = train_labels[perm]

    print(f"Training data shape: {train_features.shape}, Labels shape: {train_labels.shape}")

    # Phase 2: Train energy model
    print("\nPhase 2: Training energy model...")

    energy_model = EnergyAnomalyDetector(input_dim=actual_feature_dim).to(device)
    optimizer = torch.optim.Adam(energy_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Weighted MSE loss (focus more on positive examples)
    pos_weight_tensor = torch.tensor([1.0, pos_weight]).to(device)  # [normal_weight, anomaly_weight]

    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    energy_model.train()

    train_losses = []
    pos_energies = []
    neg_energies = []

    for epoch in tqdm(range(energy_epochs), desc="Training energy model"):
        epoch_loss = 0
        epoch_pos_energy = 0
        epoch_neg_energy = 0
        pos_count = 0
        neg_count = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            energy_scores = energy_model.energy(batch_features)

            # Compute weighted loss
            loss_weights = pos_weight_tensor[batch_labels.long()]
            losses = ((energy_scores - batch_labels) ** 2) * loss_weights
            loss = losses.mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Track energy distributions
            pos_mask = batch_labels == 0
            neg_mask = batch_labels == 1

            if pos_mask.any():
                epoch_pos_energy += energy_scores[pos_mask].mean().item()
                pos_count += 1
            if neg_mask.any():
                epoch_neg_energy += energy_scores[neg_mask].mean().item()
                neg_count += 1

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        avg_pos_energy = epoch_pos_energy / max(pos_count, 1)
        avg_neg_energy = epoch_neg_energy / max(neg_count, 1)

        train_losses.append(avg_loss)
        pos_energies.append(avg_pos_energy)
        neg_energies.append(avg_neg_energy)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Pos Energy={avg_pos_energy:.4f}, Neg Energy={avg_neg_energy:.4f}")

    energy_model.eval()

    print("Energy model training completed")
    print(f"Final: Positive Energy={pos_energies[-1]:.4f}, Negative Energy={neg_energies[-1]:.4f}")

    # Phase 3: Convergence analysis and visualization
    print("\n📈 Phase 3: Analyzing convergence...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Training loss
    ax1 = axes[0, 0]
    ax1.plot(train_losses)
    ax1.set_title('Energy Model Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Energy separation
    ax2 = axes[0, 1]
    ax2.plot(pos_energies, label='Positive (Normal)', color='blue')
    ax2.plot(neg_energies, label='Negative (Anomalous)', color='red')
    ax2.set_title('Energy Score Convergence')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Energy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Final energy distributions
    ax3 = axes[1, 0]
    with torch.no_grad():
        final_pos_scores = energy_model.energy(positive_tensor.to(device)).cpu().numpy()
        final_neg_scores = energy_model.energy(negative_tensor.to(device)).cpu().numpy()

    ax3.hist(final_pos_scores, bins=30, alpha=0.7, label='Positive (Normal)', color='blue')
    ax3.hist(final_neg_scores, bins=30, alpha=0.7, label='Negative (Anomalous)', color='red')
    ax3.set_title('Final Energy Score Distributions')
    ax3.set_xlabel('Energy Score')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Separation metric over training
    ax4 = axes[1, 1]
    separation = [abs(neg - pos) for pos, neg in zip(pos_energies, neg_energies)]
    ax4.plot(separation, color='green')
    ax4.set_title('Energy Separation Over Training')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('|Negative Energy - Positive Energy|')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    convergence_plot_path = os.path.join(plots_dir, f'{model_name}_energy_convergence.png')
    plt.savefig(convergence_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved convergence analysis: {convergence_plot_path}")

    # Convergence metrics
    final_separation = abs(neg_energies[-1] - pos_energies[-1])
    print("\n🔍 Convergence Analysis:")
    print(f"Final energy separation: {final_separation:.4f}")
    print(f"Positive energy std: {np.std(final_pos_scores):.4f}")
    print(f"Negative energy std: {np.std(final_neg_scores):.4f}")

    if final_separation > 0.5:
        print("Good energy separation achieved - model converged well")
    else:
        print("Weak energy separation - model may need more training")

    # Save the trained energy model in checkpoints directory
    energy_model_path = os.path.join(checkpoints_dir, f'{model_name}_energy_detector.pth')

    energy_model_data = {
        'model_state_dict': energy_model.state_dict(),
        'input_dim': actual_feature_dim,
        'training_params': {
            'baseline_samples': baseline_samples,
            'energy_epochs': energy_epochs,
            'pos_weight': pos_weight,
            'final_separation': final_separation,
            'pos_energy_mean': pos_energies[-1],
            'neg_energy_mean': neg_energies[-1]
        },
        'convergence_data': {
            'train_losses': train_losses,
            'pos_energies': pos_energies,
            'neg_energies': neg_energies
        }
    }
    torch.save(energy_model_data, energy_model_path)
    print(f"Saved trained energy model: {energy_model_path}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print("=" * 70)
    print("Energy-based activation anomaly analysis completed")
    print(f"Energy model saved to: {energy_model_path}")
    print("=" * 70)


def analyze_layer_specific_attribution(energy_model, layer_activations, device):
    """
    Layer-specific gradient-based attribution analysis.

    Uses gradients to identify which layers and branches
    contribute most to anomaly detection.

    Args:
        energy_model: Trained EnergyAnomalyDetector
        layer_activations: Dict of {layer_name: activation_tensor}
        device: Device to run analysis on

    Returns:
        dict: Layer-specific attribution scores and analysis
    """
    # Prepare layer activations for gradient computation
    layer_names = sorted(layer_activations.keys())
    # These are intermediate tensors from the main model's graph.
    # They should already be on the correct device and part of the graph.
    layer_tensors = [layer_activations[name] for name in layer_names]

    # Ensure the input tensors for the energy model are part of the graph
    combined_activations = torch.cat(layer_tensors, dim=1)

    # Set energy model to eval mode, as we are not training it.
    energy_model.eval()

    # Forward pass through the energy model
    # The graph from the main model flows into this computation.
    energy_scores = energy_model.energy(combined_activations)

    # Use the sum of energy scores as the scalar value for gradient computation
    total_energy = energy_scores.sum()

    # Directly compute gradients of the energy score w.r.t. each layer's activation
    try:
        layer_gradients = torch.autograd.grad(
            outputs=total_energy,
            inputs=layer_tensors,
            allow_unused=True
        )
    except RuntimeError as e:
        raise RuntimeError(f"Gradient computation failed with torch.autograd.grad: {e}") from e

    # Analyze layer-specific contributions using the computed gradients
    layer_contributions = {}
    for i, layer_name in enumerate(layer_names):
        grad = layer_gradients[i]
        if grad is not None:
            # Use gradient-based attribution - take absolute value and mean
            layer_attribution = torch.abs(grad).mean().item()
        else:
            # This can happen if a layer's output is not used in the energy model's computation
            layer_attribution = 0.0
        layer_contributions[layer_name] = layer_attribution

    # Calculate total attribution and percentages
    total_attribution = sum(layer_contributions.values())
    layer_percentages = {
        layer: (contrib / total_attribution * 100) if total_attribution > 0 else 0
        for layer, contrib in layer_contributions.items()
    }

    # Separate LOCO vs FNO contributions
    sno_total = sum(contrib for layer, contrib in layer_contributions.items() if 'sno' in layer)
    fno_total = sum(contrib for layer, contrib in layer_contributions.items() if 'fno' in layer)
    branch_total = sno_total + fno_total

    return {
        'layer_contributions': layer_contributions,
        'layer_percentages': layer_percentages,
        'sno_contribution': (sno_total / branch_total * 100) if branch_total > 0 else 0,
        'fno_contribution': (fno_total / branch_total * 100) if branch_total > 0 else 0,
        'gradient_based': True,
        'energy_score': energy_scores.mean().item()
    }


def analyze_activation_rollout_energy(device, rollout_steps=16, num_rollout_sequences=50, model_name='Hybrid'):
    """
    Rollout analysis using the trained energy model to identify which activations
    cause anomalies during model rollout progression.

    This function loads the trained energy model and performs gradient-based attribution
    analysis on progressive rollout steps to identify:
    1. Which rollout steps show highest anomaly energy
    2. Which LOCO vs FNO branch activations contribute most to anomalies at each step
    3. Visualization of energy progression and attribution over rollout steps

    Args:
        device: Device to run analysis on
        rollout_steps: Number of rollout steps to analyze (default: 16)
        num_rollout_sequences: Number of rollout sequences to analyze (default: 50)
        model_name: Name of the model to analyze (must be 'Hybrid')
    """
    from moNS2D import (
        DATA_DIR,
        GLOBAL_BATCH_SIZE,
        HIDDEN_CHANNELS,
        NUM_BLOCKS,
        PLOTS_DIR,
        TRAINING_RESOLUTION,
    )
    from torch.utils.data import DataLoader

    from .datasets import NS2DDataset

    # Use rollout plots directory if set
    plots_dir = os.environ.get('ROLLOUT_PLOTS_DIR', PLOTS_DIR)
    checkpoints_dir = os.environ.get('ROLLOUT_CHECKPOINTS_DIR', PLOTS_DIR.replace('plots', 'checkpoints'))

    print("=" * 70)
    print(f"ROLLOUT ENERGY ANALYSIS WITH GRADIENT ATTRIBUTION FOR {model_name}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Rollout steps: {rollout_steps}")
    print(f"Rollout sequences: {num_rollout_sequences}")
    print("=" * 70)

    if model_name != 'Hybrid':
        print(f"Error: Analysis designed for Hybrid model, got {model_name}")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # Load trained energy model
    energy_model_path = os.path.join(checkpoints_dir, f'{model_name}_energy_detector.pth')
    if not os.path.exists(energy_model_path):
        print(f"Error: Energy model not found at {energy_model_path}")
        print("Please run --mode analyze-activations first to train the energy model.")
        return

    # Load dataset and model
    test_dataset = NS2DDataset(DATA_DIR, split='test', target_resolution=TRAINING_RESOLUTION)
    models_dict = initialize_models(device, training_resolution=TRAINING_RESOLUTION)

    model = models_dict[model_name]['model']
    model_type = models_dict[model_name]['type']

    # Load trained Hybrid model
    ckpt_path = os.path.join(checkpoints_dir, f'{model_name}_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"Error: {model_name} checkpoint not found at {ckpt_path}")
        return

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Apply compatibility mapping for old naming conventions
        compatible_state_dict = map_old_keys_to_new_2d(state_dict, model_name)
        model.load_state_dict(compatible_state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.to(device)
    model.eval()

    # Load trained energy model
    try:
        energy_checkpoint = torch.load(energy_model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in energy_checkpoint:
            energy_state_dict = energy_checkpoint['model_state_dict']
            feature_dim = energy_checkpoint.get('input_dim', HIDDEN_CHANNELS * NUM_BLOCKS)
        else:
            energy_state_dict = energy_checkpoint
            feature_dim = HIDDEN_CHANNELS * NUM_BLOCKS  # fallback

        energy_model = EnergyAnomalyDetector(input_dim=feature_dim).to(device)
        energy_model.load_state_dict(energy_state_dict)
        energy_model.eval()

        # Enable gradients for attribution (keep model frozen but gradients flowing)
        for param in energy_model.parameters():
            param.requires_grad_(True)

        # Force gradients to be enabled properly by re-initializing if needed
        energy_model.zero_grad()  # Clear any existing gradients

    except Exception as e:
        print(f"Error loading energy model: {e}")
        return

    print(f"Model {model_name} and energy detector loaded successfully")

    # Setup layer-specific activation collection
    layer_activations = {}  # {layer_name: [activations]}
    hooks = []

    def make_layer_hook(layer_name):
        def hook(module, input, output):
            if layer_name not in layer_activations:
                layer_activations[layer_name] = []
            # Keep gradients for attribution analysis - don't detach
            layer_activations[layer_name].append(output.flatten(start_dim=1))
        return hook

    # Register hooks on both LOCO and FNO branches with layer-specific names
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'use_sno') and block.use_sno:
            if hasattr(block, 'sno_conv'):
                layer_name = f'sno_block_{i}'
                hooks.append(block.sno_conv.register_forward_hook(make_layer_hook(layer_name)))
            if hasattr(block, 'fourier'):
                layer_name = f'fno_block_{i}'
                hooks.append(block.fourier.register_forward_hook(make_layer_hook(layer_name)))

    if not hooks:
        print("Error: No LOCO/FNO hooks registered")
        return

    print(f"Registered {len(hooks)} hooks on LOCO/FNO branches")

    # Phase 1: Collect rollout data with energy analysis
    print("\nPhase 1: Analyzing rollout energy progression...")

    rollout_energies = []  # Energy at each rollout step
    rollout_attributions = []  # Attribution at each rollout step
    step_predictions = []  # Store predictions for each step

    from moNS2D import custom_collate_fn
    data_loader = DataLoader(test_dataset, batch_size=min(4, GLOBAL_BATCH_SIZE), shuffle=True, collate_fn=custom_collate_fn)

    sequences_processed = 0

    # The main loop should not be under torch.no_grad()
    for data in tqdm(data_loader, desc="Collecting rollout energy data"):
        if sequences_processed >= num_rollout_sequences:
            break

        x_initial = data['x'].to(device)
        y_target = data['y'].to(device)

        if model_type in ['loco', 'nfno', 'fno', 'fmlp', 'bno', 'ano', 'aano', 'cno', 'dno', 'zeno', 'cosno', 'kosno', 'ampsno', 'channel_hybrid']:
            x_initial = x_initial.permute(0, 3, 1, 2)
            y_target = y_target.permute(0, 3, 1, 2)

        batch_size = x_initial.shape[0]

        for sample_idx in range(batch_size):
            if sequences_processed >= num_rollout_sequences:
                break

            # Initialize the state for the sequence
            current_state_for_next_step = x_initial[sample_idx:sample_idx+1]
            sequence_energies = []
            sequence_attributions = []
            sequence_predictions = []

            for _step in range(rollout_steps):
                # Clear activations from the previous step
                for layer_name in layer_activations:
                    layer_activations[layer_name].clear()

                # Prepare input for the current step, ensuring it's a leaf that requires grad
                current_state = current_state_for_next_step.detach().requires_grad_(True)

                # --- Forward pass with gradients enabled ---
                # This builds the computation graph for the current step
                prediction = model(current_state)

                # Hooks have now captured the intermediate activations, which are part of the graph.

                # --- Attribution and Energy Analysis ---
                if all(len(activations) > 0 for activations in layer_activations.values()):
                    layer_tensors_for_analysis = {
                        name: lst[0] for name, lst in layer_activations.items()
                    }

                    # Perform attribution. This must be done while the graph exists.
                    attribution_result = analyze_layer_specific_attribution(
                        energy_model,
                        layer_tensors_for_analysis,
                        device
                    )
                    sequence_attributions.append(attribution_result)

                    # Energy score can be calculated without gradients
                    with torch.no_grad():
                        all_activations_detached = torch.cat(
                            [t.detach() for t in layer_tensors_for_analysis.values()], dim=1
                        )
                        energy_score = energy_model.energy(all_activations_detached).item()
                        sequence_energies.append(energy_score)
                else:
                    # If no activations were captured, append placeholders
                    sequence_attributions.append({'gradient_based': False, 'error': 'No activations captured'})
                    sequence_energies.append(0.0)

                # --- Prepare for the next step ---
                # We use torch.no_grad() here to prevent the computation graph from extending
                # across rollout steps, which would cause a memory leak.
                with torch.no_grad():
                    sequence_predictions.append(prediction.detach().cpu())
                    prediction_squeezed = prediction.detach().squeeze(1)
                    prediction_timestep = prediction_squeezed.unsqueeze(1)
                    current_state_for_next_step = torch.cat([current_state_for_next_step[:, 1:, :, :], prediction_timestep], dim=1)

            rollout_energies.append(sequence_energies)
            rollout_attributions.append(sequence_attributions)
            step_predictions.append(sequence_predictions)
            sequences_processed += 1

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print(f"Analyzed {sequences_processed} rollout sequences")

    # Phase 2: Analysis and visualization
    print("\n📈 Phase 2: Visualizing rollout energy and attribution analysis...")

    # Convert to numpy for easier analysis
    rollout_energies = np.array(rollout_energies)  # [sequences, steps]

    # Calculate statistics
    mean_energies = np.mean(rollout_energies, axis=0)
    std_energies = np.std(rollout_energies, axis=0)

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Energy progression over rollout steps
    ax1 = axes[0, 0]
    steps = np.arange(rollout_steps)
    ax1.plot(steps, mean_energies, 'b-', linewidth=2, label='Mean Energy')
    ax1.fill_between(steps, mean_energies - std_energies, mean_energies + std_energies,
                     alpha=0.3, color='blue', label='±1 Std Dev')
    ax1.set_title('Energy Score Progression During Rollout')
    ax1.set_xlabel('Rollout Step')
    ax1.set_ylabel('Anomaly Energy Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Energy distribution at different rollout steps
    ax2 = axes[0, 1]
    steps_to_plot = [0, rollout_steps//4, rollout_steps//2, 3*rollout_steps//4, rollout_steps-1]
    for step_idx in steps_to_plot:
        if step_idx < rollout_steps:
            ax2.hist(rollout_energies[:, step_idx], bins=20, alpha=0.6,
                    label=f'Step {step_idx}', density=True)
    ax2.set_title('Energy Score Distributions at Different Steps')
    ax2.set_xlabel('Energy Score')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: LOCO vs FNO attribution over time
    ax3 = axes[1, 0]
    sno_contributions = []
    fno_contributions = []

    for step in range(rollout_steps):
        step_sno = []
        step_fno = []
        for seq_idx in range(len(rollout_attributions)):
            if step < len(rollout_attributions[seq_idx]):
                step_sno.append(rollout_attributions[seq_idx][step]['sno_contribution'])
                step_fno.append(rollout_attributions[seq_idx][step]['fno_contribution'])

        sno_contributions.append(np.mean(step_sno) if step_sno else 0)
        fno_contributions.append(np.mean(step_fno) if step_fno else 0)

    ax3.plot(steps, sno_contributions, 'r-', linewidth=2, label='LOCO Branch', marker='o')
    ax3.plot(steps, fno_contributions, 'g-', linewidth=2, label='FNO Branch', marker='s')
    ax3.set_title('Branch Attribution During Rollout')
    ax3.set_xlabel('Rollout Step')
    ax3.set_ylabel('Attribution Contribution (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)

    # Plot 4: Anomaly detection over rollout steps
    ax4 = axes[1, 1]
    # Define threshold for anomaly detection (e.g., mean + 1.5 * std of first step)
    anomaly_threshold = np.mean(rollout_energies[:, 0]) + 1.5 * np.std(rollout_energies[:, 0])
    anomaly_counts = np.sum(rollout_energies > anomaly_threshold, axis=0)
    anomaly_rates = anomaly_counts / sequences_processed * 100

    ax4.bar(steps, anomaly_rates, alpha=0.7, color='orange')
    ax4.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='20% Threshold')
    ax4.set_title('Anomaly Detection Rate Over Rollout Steps')
    ax4.set_xlabel('Rollout Step')
    ax4.set_ylabel('Anomaly Rate (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save comprehensive analysis plot
    analysis_plot_path = os.path.join(plots_dir, f'{model_name}_rollout_energy_analysis.png')
    plt.savefig(analysis_plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved rollout energy analysis plot to: {analysis_plot_path}")

    # Phase 3: Summary statistics and insights
    print("\n📋 Phase 3: Analysis Summary...")

    print(f"Initial Energy (Step 0): {mean_energies[0]:.4f} ± {std_energies[0]:.4f}")
    print(f"Final Energy (Step {rollout_steps-1}): {mean_energies[-1]:.4f} ± {std_energies[-1]:.4f}")
    print(f"Energy Change: {mean_energies[-1] - mean_energies[0]:+.4f}")

    # Find step with highest energy increase (only if more than 1 step)
    if rollout_steps > 1:
        energy_diffs = np.diff(mean_energies)
        max_increase_step = np.argmax(energy_diffs)
        print(f"Largest energy increase at step: {max_increase_step} → {max_increase_step+1}")
        print(f"Energy increase magnitude: {energy_diffs[max_increase_step]:.4f}")
    else:
        print("Single rollout step - no energy progression to analyze")

    # Find step with highest anomaly rate
    max_anomaly_step = np.argmax(anomaly_rates)
    print(f"Highest anomaly rate at step: {max_anomaly_step} ({anomaly_rates[max_anomaly_step]:.1f}%)")

    # Branch attribution summary
    mean_sno_contribution = np.mean(sno_contributions)
    mean_fno_contribution = np.mean(fno_contributions)
    print(f"Average LOCO branch contribution: {mean_sno_contribution:.1f}%")
    print(f"Average FNO branch contribution: {mean_fno_contribution:.1f}%")

    if mean_sno_contribution > mean_fno_contribution:
        print("🔍 LOCO branch shows higher average attribution to anomalies")
    else:
        print("🔍 FNO branch shows higher average attribution to anomalies")

    # Layer-specific attribution summary (from final rollout step)
    if rollout_attributions and len(rollout_attributions[0]) > 0:
        final_step_attributions = rollout_attributions[0][-1]  # Final step of first sequence
        if 'layer_percentages' in final_step_attributions:
            print("\nLayer-specific attribution breakdown (final rollout step):")
            for layer_name, percentage in sorted(final_step_attributions['layer_percentages'].items()):
                print(f"  {layer_name}: {percentage:.1f}%")

            gradient_method = final_step_attributions.get('gradient_based', False)
            print(f"\nAttribution method: {'Gradient-based' if gradient_method else 'Activation magnitude (fallback)'}")

    print("=" * 70)
    print("Rollout energy analysis with layer-specific gradient attribution completed")
    print(f"Analysis plot saved to: {analysis_plot_path}")
    print("=" * 70)
