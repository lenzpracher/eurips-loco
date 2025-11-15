"""
1D Training Utilities for Neural Operators

This module provides training utilities for 1D neural operator models.


Functions:
- train_model: Single epoch training
- evaluate_model: Model evaluation
- run_parallel_training: Parallel training with multiple seeds (TODO)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, max_steps_per_epoch=100, model_type='sno'):
    """
    Train model for one epoch



    Args:
        model: Neural operator model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device (CPU/GPU)
        max_steps_per_epoch: Maximum steps per epoch
        model_type: 'sno' or 'fno' for proper tensor handling

    Returns:
        train_loss, val_loss
    """
    model.train()
    total_loss, step_count = 0, 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for data in pbar:
        if step_count >= max_steps_per_epoch:
            break

        x, y = data['x'].to(device), data['y'].to(device)

        # FNO expects [batch, channels, spatial], SNO expects [batch, spatial, channels]
        if model_type == 'fno':
            x, y = x.transpose(1, 2), y.transpose(1, 2)

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step_count += 1
        pbar.set_postfix(loss=f'{total_loss / step_count:.6f}')

    train_loss = total_loss / step_count
    val_loss = evaluate_model(model, val_loader, loss_fn, device, model_type, max_eval_batches=100)

    return train_loss, val_loss


def evaluate_model(model, test_loader, loss_fn, device, model_type='sno', max_eval_batches=100):
    """
    Evaluate model on test set



    Args:
        model: Neural operator model
        test_loader: Test data loader
        loss_fn: Loss function
        device: Device (CPU/GPU)
        model_type: 'sno' or 'fno' for proper tensor handling
        max_eval_batches: Maximum evaluation batches

    Returns:
        Average test loss
    """
    model.eval()
    total_loss = 0
    batch_count = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            if batch_count >= max_eval_batches:
                break
            x, y = data['x'].to(device), data['y'].to(device)

            # FNO expects [batch, channels, spatial], others expect [batch, spatial, channels]
            if model_type == 'fno':
                x, y = x.transpose(1, 2), y.transpose(1, 2)

            out = model(x)
            total_loss += loss_fn(out, y).item()
            batch_count += 1
    return total_loss / batch_count


def initialize_models_1d(device, experiment='burgers'):
    """
    Initialize 1D models with proper configurations

    Args:
        device: Device to place models on
        experiment: 'burgers' or 'kdv' for proper hyperparameters

    Returns:
        Dictionary of initialized models
    """
    from models1D.fno import create_burgers_fno, create_kdv_fno
    from models1D.hybrid import create_burgers_hybrid, create_kdv_hybrid
    from models1D.loco import create_burgers_loco, create_kdv_loco

    if experiment == 'burgers':
        models_dict = {
            "LOCO": create_burgers_loco().to(device),
            "Hybrid": create_burgers_hybrid().to(device),
            "FNO": create_burgers_fno().to(device)
        }
    elif experiment == 'kdv':
        models_dict = {
            "LOCO": create_kdv_loco().to(device),
            "Hybrid": create_kdv_hybrid().to(device),
            "FNO": create_kdv_fno().to(device)
        }
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    return models_dict


def get_model_type_from_name(model_name):
    """
    Get model type string for proper tensor handling

    From old_scripts - SNO and FSNO both use 'sno' type, FNO uses 'fno'

    Args:
        model_name: 'LOCO', 'Hybrid', or 'FNO'

    Returns:
        Model type string for training functions
    """
    if model_name == 'FNO':
        return 'fno'
    elif model_name in ['LOCO', 'Hybrid']:  # SNO and FSNO in old scripts
        return 'sno'  # Both use same tensor format as original SNO
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_single_training(
    experiment='burgers',
    gpu_id=0,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    max_steps_per_epoch=100,
    data_dir=None,
    prediction_gap=10
):
    """
    Run training for all models in an experiment

    Args:
        experiment: 'burgers' or 'kdv'
        gpu_id: GPU ID to use
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_steps_per_epoch: Max steps per epoch
        data_dir: Data directory (default: data/{experiment})
        prediction_gap: Prediction gap

    Returns:
        Dictionary of training results
    """
    from neuralop.training import AdamW

    # Set up device
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    if data_dir is None:
        data_dir = f'data/{experiment.capitalize()}'

    if experiment == 'burgers':
        train_dataset, test_dataset = create_burgers_datasets(
            data_dir=data_dir, prediction_gap=prediction_gap
        )
    elif experiment == 'kdv':
        train_dataset, test_dataset = create_kdv_datasets(
            data_dir=data_dir, prediction_gap=prediction_gap
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize models
    models_dict = initialize_models_1d(device, experiment)

    # Training
    loss_fn = nn.MSELoss()
    all_losses = {}

    for model_name, model in models_dict.items():
        print(f"\nTraining {model_name}...")

        # Set up optimizer
        if model_name == 'FNO':
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        else:
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        model_type = get_model_type_from_name(model_name)

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            train_loss, val_loss = train_model(
                model, train_loader, test_loader, optimizer, loss_fn, device,
                max_steps_per_epoch, model_type
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, {model_name} Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        all_losses[model_name] = {
            'train': train_losses,
            'val': val_losses
        }

    return all_losses


def run_training_with_exact_config(
    experiment='burgers',
    gpu_id=0,
    data_dir=None,
    plots_dir=None,
    checkpoints_dir=None
):
    """
    Run training exactly as in old_scripts/Burgers_KdV/

    This matches the exact training pipeline from the old scripts including:
    - Exact hyperparameters
    - Exact model configurations
    - Exact training loop
    - Exact plotting and saving

    Args:
        experiment: 'burgers' or 'kdv'
        gpu_id: GPU ID to use
        data_dir: Data directory
        plots_dir: Plots directory
        checkpoints_dir: Checkpoints directory
    """
    import os

    from neuralop.training import AdamW
    from torch.utils.data import DataLoader

    from utils1D.data import BurgersDataset, KdVDataset
    from utils1D.models import (
        initialize_burgers_models,
        initialize_kdv_models,
    )
    from utils1D.plotting_utils import evaluate_rollout_loss, plot_losses
    from utils1D.utils import print_model_summary

    # Exact hyperparameters from old scripts
    if experiment == 'burgers':
        BATCH_SIZE = 32
        EPOCHS = 100
        LEARNING_RATE = 1e-3
        MAX_STEPS_PER_EPOCH = 100
        PREDICTION_GAP = 10
        NUM_ROLLOUT_STEPS = 40

        if data_dir is None:
            data_dir = './data/Burgers'
        if plots_dir is None:
            plots_dir = 'plots/Burgers'
        if checkpoints_dir is None:
            checkpoints_dir = 'checkpoints/Burgers'

    elif experiment == 'kdv':
        BATCH_SIZE = 32
        EPOCHS = 100
        LEARNING_RATE = 1e-3
        MAX_STEPS_PER_EPOCH = 100
        PREDICTION_GAP = 10
        NUM_ROLLOUT_STEPS = 40

        if data_dir is None:
            data_dir = './data/KdV'
        if plots_dir is None:
            plots_dir = 'plots/KdV'
        if checkpoints_dir is None:
            checkpoints_dir = 'checkpoints/KdV'
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    # Create directories
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets exactly as in old scripts
    if experiment == 'burgers':
        train_dataset = BurgersDataset(data_dir, split='train', prediction_gap=PREDICTION_GAP)
        test_dataset = BurgersDataset(data_dir, split='test', prediction_gap=PREDICTION_GAP)
        models_dict = initialize_burgers_models(device)
    else:
        train_dataset = KdVDataset(data_dir, split='train', prediction_gap=PREDICTION_GAP)
        test_dataset = KdVDataset(data_dir, split='test', prediction_gap=PREDICTION_GAP)
        models_dict = initialize_kdv_models(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Print parameter counts exactly as in old scripts
    summary_models_dict = {name: model_info[0] for name, model_info in models_dict.items()}
    print_model_summary(summary_models_dict)

    # Training exactly as in old scripts
    mse_loss = nn.MSELoss()
    all_losses = {}

    for name, (model, model_type, path, optim_info) in models_dict.items():
        print(f"\nTraining {name}...")

        # Exact optimizer setup from old scripts
        if optim_info['type'] == 'adam' or optim_info['type'] == 'adamw':
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=optim_info.get('weight_decay', 1e-4))
        else:
            raise ValueError(f"Unsupported optimizer type: {optim_info['type']}")

        train_losses = []
        val_losses = []
        for epoch in range(EPOCHS):
            train_loss, val_loss = train_model(model, train_loader, test_loader, optimizer, mse_loss, device, MAX_STEPS_PER_EPOCH, model_type)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}, {name} Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        all_losses[name] = {'train': train_losses, 'val': val_losses}

        # Update path to be absolute
        models_dict[name] = (model, model_type, os.path.join(checkpoints_dir, path), optim_info)

    # Plot training losses exactly as in old scripts
    plot_losses(all_losses, title_suffix=experiment.capitalize(),
                save_path=os.path.join(plots_dir, 'training_losses.png'))

    # Save models and loss data exactly as in old scripts
    print("\nSaving models and loss data...")
    for _name, (model, _, path, _) in models_dict.items():
        torch.save(model.state_dict(), path)

    torch.save(all_losses, os.path.join(checkpoints_dir, 'loss_data.pt'))
    print(f"Models and loss data saved in '{checkpoints_dir}/' directory.")

    # Evaluate models and generate spacetime diagrams exactly as in old scripts
    print("\nEvaluating models and generating spacetime diagrams...")
    for name, (model, model_type, _, _) in models_dict.items():
        evaluate_rollout_loss(model, test_dataset, device, model_type=model_type,
                            num_rollout_steps=NUM_ROLLOUT_STEPS, generate_plot=True,
                            title_prefix=name, plots_dir=plots_dir)

    return all_losses, models_dict


# TODO: Implement parallel training functions
# These would be complex extractions from the old_scripts
def run_parallel_training():
    """
    Run parallel training with multiple seeds

    TODO: Extract complex parallel training logic from old_scripts
    This involves multiprocessing, seed management, and result aggregation.
    """
    raise NotImplementedError("Parallel training not yet implemented. Use run_training_with_exact_config for now.")
