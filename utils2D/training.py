"""
2D Training Utilities for Neural Operators

This module provides training utilities for 2D neural operator models.

Functions:
- train_model: Single epoch training with optional rollout loss
- evaluate_model: Model evaluation
- custom_collate_fn: Custom collate function for rollout data
- initialize_models_2d: Initialize 2D models with proper configurations
- get_model_type_from_name: Get model type string for proper tensor handling
"""


import torch
import torch.distributed as dist
from neuralop import H1Loss
from neuralop.training import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(model, train_loader, optimizer, loss_fn, device, model_type='sno',
                rank=0, model_name_for_tqdm="Model", tqdm_pos=0, max_steps_per_epoch=None,
                activate_rollout_loss=False, rollout_probs=(0.1, 0.1, 0.1), max_rollout_steps=10):
    """
    Train a model for one epoch with optional multi-step rollout loss.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Training device
        model_type: Type of model ('sno', 'fno', etc.)
        rank: Process rank for distributed training
        model_name_for_tqdm: Name for progress bar
        tqdm_pos: Position for progress bar
        max_steps_per_epoch: Maximum steps per epoch
        activate_rollout_loss: Whether to use rollout loss
        rollout_probs: Probabilities for different rollout steps
        max_rollout_steps: Maximum rollout steps

    Returns:
        float: Average training loss
    """
    model.train()
    total_loss, step_count = 0, 0
    rollout_loss_count = 0
    rollout_counts = {'2step': 0, '3step': 0, '10step': 0}

    pbar = tqdm(train_loader, desc=f"Training {model_name_for_tqdm}", leave=False,
                disable=(rank != 0), position=tqdm_pos)

    for i, data in enumerate(pbar):
        if max_steps_per_epoch and i >= max_steps_per_epoch:
            break

        x, y = data['x'].to(device), data['y'].to(device)

        # Get all future timesteps for rollout loss
        y_future = {}
        for step in range(1, max_rollout_steps):
            key = 'y_next' if step == 1 else f'y_next{step}'
            y_future[key] = data.get(key, None)

        if model_type in ['loco', 'hybrid', 'fno']:
            x = x.permute(0, 3, 1, 2)
            y = y.permute(0, 3, 1, 2)

        optimizer.zero_grad()

        # Decide rollout type using configurable probability distribution
        rollout_type = None
        rollout_steps = 1

        if activate_rollout_loss:
            rand_val = torch.rand(1).item()
            prob_2step, prob_3step, prob_10step = rollout_probs

            if rand_val < prob_2step:  # 2-step rollout
                rollout_type = '2step'
                rollout_steps = 2
            elif rand_val < prob_2step + prob_3step:  # 3-step rollout
                rollout_type = '3step'
                rollout_steps = 3
            elif rand_val < prob_2step + prob_3step + prob_10step:  # max-step rollout
                rollout_type = '10step'
                rollout_steps = min(max_rollout_steps, 10)  # Use configurable max, but cap at 10 for now

        if rollout_type is not None:
            # Check if we have enough future timesteps for this rollout
            required_key = 'y_next' if rollout_steps == 2 else f'y_next{rollout_steps-1}'
            y_target_list = y_future.get(required_key)

            if y_target_list is not None:
                # Filter out samples where target is None
                valid_indices = [idx for idx in range(len(y_target_list)) if y_target_list[idx] is not None]

                if len(valid_indices) > 0:
                    # Keep only valid samples for rollout training
                    x_rollout = x[valid_indices]
                    y_target_rollout = torch.stack([y_target_list[idx] for idx in valid_indices]).to(device)

                    if model_type in ['loco', 'hybrid', 'fno']:
                        y_target_rollout = y_target_rollout.permute(0, 3, 1, 2)

                    # Perform N-step rollout
                    predictions = perform_n_step_rollout(model, x_rollout, rollout_steps, model_type)
                    final_prediction = predictions[-1]  # Use the final prediction

                    # Calculate loss against target
                    if model_type in ['loco', 'hybrid', 'fno']:
                        loss = loss_fn(final_prediction, y_target_rollout)
                    else:  # MLP
                        loss = loss_fn(final_prediction.permute(0, 3, 1, 2), y_target_rollout.permute(0, 3, 1, 2))

                    rollout_loss_count += 1
                    rollout_counts[rollout_type] += 1
                else:
                    rollout_type = None  # Fall back to regular training
            else:
                rollout_type = None  # Fall back to regular training

        if rollout_type is None:
            # Regular single-step training
            out = model(x)

            # Permute output back if necessary for loss calculation
            if model_type in ['loco', 'hybrid', 'fno']:
                loss = loss_fn(out, y)
            else: # MLP
                loss = loss_fn(out.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2))

        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        step_count += 1
        if rank == 0:
            avg_loss = total_loss / step_count
            postfix_dict = {'loss': f'{avg_loss:.6f}'}
            if activate_rollout_loss:
                # Simple rollout display format: [2:count,3:count,10:count]
                rollout_display = f"[2:{rollout_counts['2step']},3:{rollout_counts['3step']},10:{rollout_counts['10step']}]"
                postfix_dict['rollout'] = rollout_display
            pbar.set_postfix(postfix_dict)

    return total_loss / step_count if step_count > 0 else 0.0


def evaluate_model(model, test_loader, loss_fn, device, model_type='sno', max_eval_samples=None, tqdm_pos=0):
    model.eval()
    total_loss = 0
    total_samples = 0
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating", disable=(rank != 0), position=tqdm_pos):
            if max_eval_samples is not None and total_samples >= max_eval_samples:
                break
            x, y = data['x'].to(device), data['y'].to(device)
            batch_size = x.size(0)

            if model_type in ['loco', 'hybrid', 'fno']:
                x = x.permute(0, 3, 1, 2)
                y = y.permute(0, 3, 1, 2)

            out = model(x)

            if model_type in ['loco', 'hybrid', 'fno']:
                loss = loss_fn(out, y)
            else: # MLP
                loss = loss_fn(out.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2))

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    if use_ddp:
        # In DDP, each process has a portion of the total loss and samples.
        # We need to sum them up across all processes.
        loss_tensor = torch.tensor([total_loss, total_samples], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor[0].item() / loss_tensor[1].item() if loss_tensor[1] > 0 else 0

    return avg_loss


def custom_collate_fn(batch):
    from torch.utils.data._utils.collate import default_collate

    # Separate x and y
    x_batch = [item['x'] for item in batch]
    y_batch = [item['y'] for item in batch]

    # Collate x and y normally
    x_collated = default_collate(x_batch)
    y_collated = default_collate(y_batch)

    # Handle all y_next fields - keep as lists since some may be None
    result = {
        'x': x_collated,
        'y': y_collated
    }

    # Add all y_next fields from the batch (up to max available)
    # We check what's actually in the batch rather than assuming a fixed range
    first_item_keys = batch[0].keys()
    for key in first_item_keys:
        if key.startswith('y_next'):
            result[key] = [item[key] for item in batch]

    return result



def perform_n_step_rollout(model, x, n_steps, model_type='loco'):
    """
    Perform n-step rollout prediction

    Args:
        model: Neural operator model
        x: Initial condition [batch, channels, height, width]
        n_steps: Number of rollout steps
        model_type: Type of model for proper handling

    Returns:
        List of predictions for each step
    """
    model.eval()
    predictions = []
    current_x = x.clone()

    with torch.no_grad():
        for step in range(n_steps):
            pred = model(current_x)
            predictions.append(pred)

            # Update input for next step by sliding the time window
            if step < n_steps - 1:  # Don't update on the last step
                # Remove oldest timestep (first channel), add prediction as newest
                current_x = torch.cat([
                    current_x[:, 1:, :, :],  # Remove first channel (oldest)
                    pred.detach()            # Add prediction as newest
                ], dim=1)

    return predictions


def initialize_models_2d(device, training_resolution=64):
    """
    Initialize 2D models with proper configurations

    Args:
        device: Device to place models on
        training_resolution: Spatial resolution for training

    Returns:
        Dictionary of initialized models
    """
    from models2D.fno import create_ns2d_fno
    from models2D.hybrid import create_ns2d_hybrid
    from models2D.loco import create_ns2d_loco

    models_dict = {
        "LOCO": {
            "model": create_ns2d_loco().to(device),
            "type": "loco",
            "optimizer": AdamW(create_ns2d_loco().parameters(), lr=1e-3, weight_decay=1e-4),
            "scheduler": torch.optim.lr_scheduler.StepLR(
                AdamW(create_ns2d_loco().parameters(), lr=1e-3),
                step_size=500, gamma=0.5
            )
        },
        "Hybrid": {
            "model": create_ns2d_hybrid().to(device),
            "type": "hybrid",
            "optimizer": AdamW(create_ns2d_hybrid().parameters(), lr=1e-3, weight_decay=1e-4),
            "scheduler": torch.optim.lr_scheduler.StepLR(
                AdamW(create_ns2d_hybrid().parameters(), lr=1e-3),
                step_size=500, gamma=0.5
            )
        },
        "FNO": {
            "model": create_ns2d_fno().to(device),
            "type": "fno",
            "optimizer": AdamW(create_ns2d_fno().parameters(), lr=1e-3, weight_decay=1e-4),
            "scheduler": torch.optim.lr_scheduler.StepLR(
                AdamW(create_ns2d_fno().parameters(), lr=1e-3),
                step_size=500, gamma=0.5
            )
        }
    }

    return models_dict


def get_model_type_from_name(model_name):
    """
    Get model type string for proper tensor handling

    Args:
        model_name: 'LOCO', 'Hybrid', or 'FNO'

    Returns:
        Model type string for training functions
    """
    if model_name == 'FNO':
        return 'fno'
    elif model_name == 'LOCO':
        return 'loco'
    elif model_name == 'Hybrid':
        return 'hybrid'
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_single_training_2d(
    models_to_train=None,
    gpu_id=0,
    epochs=1500,
    batch_size=8,
    learning_rate=1e-3,
    data_dir='data',
    training_resolution=64,
    activate_rollout_loss=False,
    rollout_probs=(0.1, 0.1, 0.1),
    max_rollout_steps=10,
    enable_augmentation=False,
    augmentation_noise_levels=None,
    augmentation_probability=0.5
):
    """
    Run training for 2D models

    Args:
        models_to_train: List of model names to train
        gpu_id: GPU ID to use
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        data_dir: Data directory
        training_resolution: Training spatial resolution
        activate_rollout_loss: Whether to use rollout loss
        rollout_probs: Probabilities for different rollout steps
        max_rollout_steps: Maximum rollout steps
        enable_augmentation: Whether to enable data augmentation
        augmentation_noise_levels: List of noise levels for augmentation
        augmentation_probability: Probability of applying augmentation

    Returns:
        Dictionary of training results
    """
    # Initialize mutable defaults
    if models_to_train is None:
        models_to_train = ['LOCO', 'Hybrid', 'FNO']
    if augmentation_noise_levels is None:
        augmentation_noise_levels = [0.01, 0.03]

    from utils2D.data import create_ns2d_datasets

    # Set up device
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_dataset, test_dataset = create_ns2d_datasets(
        data_dir=data_dir,
        target_resolution=training_resolution,
        max_rollout_steps=max_rollout_steps,
        subset_mode=True,
        subset_size=1000,  # Standard FNO paper size
        enable_augmentation=enable_augmentation,
        augmentation_noise_levels=augmentation_noise_levels,
        augmentation_probability=augmentation_probability
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # Initialize models
    models_dict = initialize_models_2d(device, training_resolution)

    # Training
    loss_fn = H1Loss(d=2)
    all_losses = {}

    for model_name in models_to_train:
        if model_name not in models_dict:
            print(f"Warning: Model {model_name} not found in models_dict. Skipping.")
            continue

        print(f"\\nTraining {model_name}...")

        model_info = models_dict[model_name]
        model_type = get_model_type_from_name(model_name)

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            train_loss = train_model(
                model_info['model'], train_loader, model_info['optimizer'],
                loss_fn, device, model_type,
                model_name_for_tqdm=model_name,
                activate_rollout_loss=activate_rollout_loss,
                rollout_probs=rollout_probs,
                max_rollout_steps=max_rollout_steps
            )
            val_loss = evaluate_model(
                model_info['model'], test_loader, loss_fn, device, model_type
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Learning rate scheduling
            if epoch < 1500:
                model_info['scheduler'].step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, {model_name} Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        all_losses[model_name] = {
            'train': train_losses,
            'val': val_losses
        }

    return all_losses


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
