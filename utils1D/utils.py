import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def map_old_keys_to_new(state_dict, model_type='LOCO'):
    """
    Map old state dict keys to new naming conventions with backwards compatibility.

    This function handles multiple levels of backwards compatibility:
    1. 'sno_conv.weight' → 'spectral_weights' for 1D LOCO blocks
    2. 'sno_block.' → 'loco_block.' for Hybrid models
    3. Model type inference for space-time modes:
       - 'sno' → 'loco' (old SNO models should be loaded as LOCO)
       - 'fsno' → 'hybrid' (old FSNO models should be loaded as Hybrid)

    Args:
        state_dict: The loaded state dict from a saved model
        model_type: Type of model ('LOCO', 'Hybrid', 'FNO', 'sno', 'fsno')

    Returns:
        Updated state dict with corrected keys
    """
    new_state_dict = {}

    # Handle backwards compatibility for space-time modes model type conversion
    effective_model_type = model_type
    if model_type == 'sno':
        effective_model_type = 'LOCO'
        print("Converting 'sno' model type to 'LOCO'")
    elif model_type == 'fsno':
        effective_model_type = 'Hybrid'
        print("Converting 'fsno' model type to 'Hybrid'")

    for old_key, value in state_dict.items():
        new_key = old_key

        # Handle the main naming changes for 1D models
        if effective_model_type in ['LOCO', 'Hybrid']:
            # Map sno_conv.weight to spectral_weights for 1D LOCO blocks
            if 'sno_conv.weight' in old_key:
                new_key = old_key.replace('sno_conv.weight', 'spectral_weights')
            # Map sno_block to loco_block for Hybrid models
            elif 'sno_block.' in old_key:
                new_key = old_key.replace('sno_block.', 'loco_block.')

        # No changes needed for FNO models - they don't use sno_conv

        new_state_dict[new_key] = value

    return new_state_dict


def convert_model_type_for_compatibility(model_type):
    """
    Convert old model types to new model types for backwards compatibility.

    Args:
        model_type: Original model type ('sno', 'fsno', 'LOCO', 'Hybrid', 'FNO', etc.)

    Returns:
        Converted model type for current codebase
    """
    if model_type == 'sno':
        return 'LOCO'
    elif model_type == 'fsno':
        return 'Hybrid'
    else:
        return model_type


def count_parameters(model, model_name="Model"):
    """
    Count the total number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model to count parameters for
        model_name (str): Name of the model for display purposes

    Returns:
        int: Total number of parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"--- {model_name} Parameter Count ---")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if total_params != trainable_params:
        frozen_params = total_params - trainable_params
        print(f"Frozen parameters: {frozen_params:,}")

    # Convert to millions for easier reading if large
    if total_params >= 1_000_000:
        print(f"Total parameters: {total_params / 1_000_000:.2f}M")
    elif total_params >= 1_000:
        print(f"Total parameters: {total_params / 1_000:.1f}K")

    print("-" * (len(model_name) + 25))

    return total_params

def print_model_summary(models_dict):
    """
    Print parameter counts for multiple models.

    Args:
        models_dict (dict): Dictionary with model names as keys and models as values
    """
    print("\n" + "="*50)
    print("MODEL PARAMETER SUMMARY")
    print("="*50)

    total_all_params = 0
    for model_name, model in models_dict.items():
        params = count_parameters(model, model_name)
        total_all_params += params
        print()

    print(f"TOTAL PARAMETERS ACROSS ALL MODELS: {total_all_params:,}")
    if total_all_params >= 1_000_000:
        print(f"TOTAL PARAMETERS: {total_all_params / 1_000_000:.2f}M")
    elif total_all_params >= 1_000:
        print(f"TOTAL PARAMETERS: {total_all_params / 1_000:.1f}K")
    print("="*50)


def analyze_rollout_loss_1d(device, rollout_steps=50, num_samples=10, data_dir='data', checkpoints_dir='checkpoints', plots_dir='plots', equation='Burgers'):
    """
    Analyzes rollout loss for 1D equations using parallel-trained models.
    Loads best models from parallel training and computes rollout statistics.
    """

    print(f"🔍 Analyzing rollout loss for {equation} equation...")
    print(f"Using {num_samples} samples, {rollout_steps} rollout steps")
    print(f"🔍 Device: {device}")

    # Check if we have parallel training results
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        print("Please run parallel training first to generate model checkpoints.")
        return

    # Import data loading function and models based on equation type
    if equation.lower() == 'burgers':
        from burgers_1d import initialize_models
        from utils1D.data import BurgersDataset
        BurgersDataset('data/Burgers', split='test', prediction_gap=10)
    elif equation.lower() == 'kdv':
        from kdv_1d import initialize_models
        from utils1D.data import KdVDataset
        KdVDataset('data/KdV', split='test', prediction_gap=10)
    else:
        raise ValueError(f"Unknown equation: {equation}")

    # Initialize models
    models_dict = initialize_models(device)
    model_names = list(models_dict.keys())

    # Load best models from checkpoints
    loaded_models = {}
    for model_name in model_names:
        # Try different checkpoint naming patterns
        checkpoint_paths = [
            os.path.join(checkpoints_dir, f'{model_name}_best.pth'),
            os.path.join(checkpoints_dir, f'{model_name.lower()}_model.pt'),
            os.path.join(checkpoints_dir, f'{model_name}_model.pt'),
        ]

        checkpoint_loaded = False
        for checkpoint_path in checkpoint_paths:
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            # Direct state dict (OrderedDict or dict)
                            state_dict = checkpoint
                    elif isinstance(checkpoint, tuple | list):
                        # Handle tuple format - typically (model_state_dict, optimizer_state_dict, ...)
                        state_dict = checkpoint[0]
                    else:
                        state_dict = checkpoint

                    # models_dict[model_name] is a tuple: (model, type, checkpoint_path, optimizer_config)
                    model = models_dict[model_name][0]  # Get the actual model

                    # Apply compatibility mapping for old naming conventions
                    compatible_state_dict = map_old_keys_to_new(state_dict, model_name)
                    model.load_state_dict(compatible_state_dict)
                    model.eval()
                    loaded_models[model_name] = model
                    print(f"Loaded {model_name} model from {checkpoint_path}")
                    checkpoint_loaded = True
                    break
                except Exception as e:
                    import traceback
                    print(f"Failed to load {model_name} from {checkpoint_path}: {e}")
                    print(f"Full traceback: {traceback.format_exc()}")

        if not checkpoint_loaded:
            print(f"No valid checkpoint found for {model_name} in: {checkpoint_paths}")

    if not loaded_models:
        print("No models could be loaded. Check checkpoint paths.")
        return

    # Initialize loss function - use same as training (nn.MSELoss from burgers_1d.py line 214)
    mse_loss = nn.MSELoss()  # Same loss as used in training

    # Get full trajectories for rollout analysis
    # Load the raw data to get full trajectories
    if equation.lower() == 'burgers':
        data_file = 'data/Burgers/burgers_data.pt'
    else:
        data_file = 'data/KdV/kdv_data.pt'

    raw_data = torch.load(data_file, map_location='cpu', weights_only=False)
    all_trajectories = raw_data['snapshots']  # Shape: [num_runs, time_steps, channels, spatial]

    # Use test split trajectories (same logic as dataset)
    num_runs = raw_data['num_runs']
    train_split_ratio = 0.8
    split_idx = int(num_runs * train_split_ratio)
    test_trajectories = all_trajectories[split_idx:]  # Test trajectories

    # Calculate prediction gap from training (should be 10)
    prediction_gap = 10  # Both Burgers and KdV use prediction_gap=10

    # Generate random samples from available test trajectories
    # Each sample is (trajectory_idx, start_time) pair
    available_trajectories = len(test_trajectories)
    samples = []

    for _ in range(num_samples):
        # Random trajectory
        traj_idx = torch.randint(0, available_trajectories, (1,)).item()
        # Random start time (ensure we have enough timesteps for rollout)
        max_start_time = test_trajectories[traj_idx].shape[0] - (rollout_steps * prediction_gap + prediction_gap)
        if max_start_time > 0:
            start_time = torch.randint(0, max_start_time, (1,)).item()
            samples.append((traj_idx, start_time))

    print(f"Using {len(samples)} samples from {available_trajectories} trajectories")

    # Check maximum possible rollout steps based on available timesteps
    max_timesteps = raw_data['time_steps']
    max_possible_rollout = (max_timesteps - prediction_gap) // prediction_gap

    if rollout_steps > max_possible_rollout:
        print(f"Requested {rollout_steps} rollout steps, but only {max_timesteps} available")
        print(f"🔧 Adjusting rollout steps to maximum possible: {max_possible_rollout}")
        rollout_steps = max_possible_rollout

    # Results storage
    rollout_results = {}
    for model_name in loaded_models:
        rollout_results[model_name] = {
            'mse_losses': []
        }

    # Perform rollout analysis with sampled trajectories
    with torch.no_grad():
        for sample_idx in tqdm(range(len(samples)), desc="Processing samples"):
            traj_idx, start_time = samples[sample_idx]
            trajectory = test_trajectories[traj_idx]  # Shape: [time_steps, channels, spatial]

            for model_name, model in loaded_models.items():
                model.eval()

                sample_mse_losses = []

                # Get model type from models_dict tuple
                model_type = models_dict[model_name][1]

                # Start autoregressive rollout from start_time
                current_time = start_time

                # Get initial input at start_time
                current_state = trajectory[current_time]  # Shape: [channels, spatial]
                # Convert to [spatial, channels] format expected by dataset
                if equation.lower() == 'kdv':
                    current_input_data = current_state.transpose(0, 1)  # [spatial, channels]
                else:  # Burgers
                    current_input_data = current_state.unsqueeze(-1)  # [spatial, 1]

                # Rollout for specified number of steps
                for _step in range(rollout_steps):
                    target_time = current_time + prediction_gap

                    if target_time >= trajectory.shape[0]:
                        break  # Not enough timesteps left

                    # Prepare model input
                    model_input = current_input_data.unsqueeze(0).to(device)  # Add batch dim: [1, spatial, channels]

                    # Get ground truth target at target_time
                    target_state = trajectory[target_time]  # Shape: [channels, spatial]
                    if equation.lower() == 'kdv':
                        target_data = target_state.transpose(0, 1)  # [spatial, channels]
                    else:  # Burgers
                        target_data = target_state.unsqueeze(-1)  # [spatial, 1]
                    model_target = target_data.unsqueeze(0).to(device)  # Add batch dim

                    # Apply model-specific data formatting
                    if model_type == 'fno':
                        model_input = model_input.transpose(1, 2)  # [1, channels, spatial]
                        model_target = model_target.transpose(1, 2)

                    # Make prediction
                    pred = model(model_input)

                    # Compute MSE loss against ground truth
                    mse_loss_val = mse_loss(pred, model_target).item()
                    sample_mse_losses.append(mse_loss_val)

                    # Use prediction as input for next step (autoregressive)
                    if model_type == 'fno':
                        pred_for_next = pred.transpose(1, 2)  # Convert back to [1, spatial, channels]
                    else:
                        pred_for_next = pred

                    current_input_data = pred_for_next.squeeze(0).cpu()  # Remove batch dim, move to CPU
                    current_time = target_time

                rollout_results[model_name]['mse_losses'].append(sample_mse_losses)

    # Compute statistics
    rollout_stats = {}
    for model_name in loaded_models:
        mse_array = np.array(rollout_results[model_name]['mse_losses'])

        rollout_stats[model_name] = {
            'mse_mean': np.mean(mse_array, axis=0),
            'mse_p10': np.percentile(mse_array, 10, axis=0),
            'mse_p90': np.percentile(mse_array, 90, axis=0)
        }

    # Plot results - only MSE loss
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (model_name, stats) in enumerate(rollout_stats.items()):
        color = colors[i % len(colors)]
        steps = np.arange(len(stats['mse_mean']))

        # Plot MSE rollout loss with simplified percentiles (mean + 10-90 percentile band only)
        ax.plot(steps, stats['mse_mean'], label=f'{model_name}', color=color, linewidth=2)
        ax.fill_between(steps, stats['mse_p10'], stats['mse_p90'], alpha=0.2, color=color)

    # Format MSE plot
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'{equation} Equation - Rollout MSE Loss (Mean + 10th-90th Percentile)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, f'{equation.lower()}_rollout_analysis.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Rollout analysis plot saved to: {save_path}")

    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"ROLLOUT LOSS SUMMARY - {equation} Equation")
    print(f"{'='*120}")
    print(f"{'Model':<10} | {'Samples':<7} | {'Step 0':<35} | {'Step 5':<35} | {'Step 10':<35}")
    print(f"{'='*120}")

    for model_name, stats in rollout_stats.items():
        mse_mean = stats['mse_mean']
        mse_p10 = stats['mse_p10']
        mse_p90 = stats['mse_p90']

        # Format showing mean and [p10-p90] range
        step0 = f"{mse_mean[0]:.2e}[{mse_p10[0]:.2e}-{mse_p90[0]:.2e}]" if len(mse_mean) > 0 else "N/A"
        step5 = f"{mse_mean[5]:.2e}[{mse_p10[5]:.2e}-{mse_p90[5]:.2e}]" if len(mse_mean) > 5 else "N/A"
        step10 = f"{mse_mean[10]:.2e}[{mse_p10[10]:.2e}-{mse_p90[10]:.2e}]" if len(mse_mean) > 10 else "N/A"

        print(f"{model_name:<10} | {num_samples:<7} | {step0:<35} | {step5:<35} | {step10:<35}")

    print(f"{'='*120}")

    return rollout_stats


def analyze_spacetime_1d(device, rollout_steps=35, num_samples=1, data_dir='data', checkpoints_dir='checkpoints', plots_dir='plots', equation='Burgers', sample_id=100):
    """
    Analyzes spacetime evolution for 1D equations using parallel-trained models.
    Creates a single combined plot showing one sample with all models and their differences to ground truth.
    """
    from .plotting_utils import create_combined_model_comparison_plot

    print(f"🎨 Analyzing spacetime evolution for {equation} equation...")
    print(f"Using {rollout_steps} rollout steps for model comparison")
    print(f"🎨 Device: {device}")

    # Setup paths
    equation_plots_dir = os.path.join(plots_dir, equation)
    os.makedirs(equation_plots_dir, exist_ok=True)

    # Check if we have parallel training results
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        print("Please run parallel training first to generate model checkpoints.")
        return

    # Import data loading function and models based on equation type
    if equation.lower() == 'burgers':
        from burgers_1d import initialize_models
    elif equation.lower() == 'kdv':
        from kdv_1d import initialize_models
    else:
        raise ValueError(f"Unknown equation: {equation}")

    # Initialize models
    models_dict = initialize_models(device)
    model_names = list(models_dict.keys())
    print(f"📂 Found {len(models_dict)} model types: {model_names}")

    # Load best models from checkpoints (same logic as rollout-loss analysis)
    loaded_models_dict = {}
    for model_name in model_names:
        # Try different checkpoint naming patterns
        checkpoint_paths = [
            os.path.join(checkpoints_dir, f'{model_name}_best.pth'),
            os.path.join(checkpoints_dir, f'{model_name.lower()}_model.pt'),
            os.path.join(checkpoints_dir, f'{model_name}_model.pt')
        ]

        model_loaded = False
        for checkpoint_path in checkpoint_paths:
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

                    # Extract model and model type from models_dict
                    model_info = models_dict[model_name]
                    model = model_info[0]  # Extract model from tuple
                    model_type = model_info[1]  # Extract model type

                    # Load state dict with backwards compatibility
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint

                    # Apply backwards compatibility mapping
                    compatible_state_dict = map_old_keys_to_new(state_dict, model_type)
                    model.load_state_dict(compatible_state_dict)

                    model.to(device)
                    model.eval()

                    # Store in format expected by plotting function: {name: (model, model_type)}
                    loaded_models_dict[model_name] = (model, model_type)
                    print(f"Loaded {model_name} from {checkpoint_path}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load {model_name} from {checkpoint_path}: {e}")
                    continue

        if not model_loaded:
            print(f"Could not find valid checkpoint for {model_name} in:")
            for path in checkpoint_paths:
                print(f"  - {path} (exists: {os.path.exists(path)})")

    if len(loaded_models_dict) == 0:
        print("❌ No models loaded successfully. Aborting analysis.")
        return

    # Create test dataset for plotting function
    if equation.lower() == 'burgers':
        from utils1D.data import BurgersDataset
        test_dataset = BurgersDataset(data_dir='data/Burgers', split='test', prediction_gap=10)
    else:
        from utils1D.data import KdVDataset
        test_dataset = KdVDataset(data_dir='data/KdV', split='test', prediction_gap=10)

    print(f"📂 Created test dataset with {len(test_dataset)} samples")

    # Check maximum possible rollout steps based on dataset
    if hasattr(test_dataset, 'time_steps'):
        max_timesteps = test_dataset.time_steps
        max_possible_rollout = (max_timesteps - 10) // 10  # prediction_gap = 10

        if rollout_steps > max_possible_rollout:
            print(f"Requested {rollout_steps} rollout steps, but only {max_timesteps} available")
            print(f"🔧 Adjusting rollout steps to maximum possible: {max_possible_rollout}")
            rollout_steps = max_possible_rollout

    # Generate the combined spacetime plot
    print("\n🎨 Generating combined spacetime visualization...")
    print("Layout: Ground Truth + All Models (top row) | Model Differences (bottom row)")

    create_combined_model_comparison_plot(
        models_dict=loaded_models_dict,
        test_dataset=test_dataset,
        device=device,
        num_rollout_steps=rollout_steps,
        plots_dir=equation_plots_dir,
        title_suffix=f"{equation.lower()}_spacetime_comparison",
        sample_id=sample_id,
        equation=equation
    )

    print("\n🎨 Spacetime analysis complete!")
    print(f"Generated spacetime plot comparing {len(loaded_models_dict)} models")
    print(f"📁 Plot saved in: {equation_plots_dir}")
    print(f"{'='*70}")
