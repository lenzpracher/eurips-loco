#!/usr/bin/env python3
"""
Navier-Stokes 2D Experiment

Neural operator training for the 2D Navier-Stokes equation.

Usage:
    python navier_stokes_2d.py --mode parallel-train
    python navier_stokes_2d.py --mode parallel-plot
    python navier_stokes_2d.py --mode rollout-loss
"""

import argparse
import multiprocessing
import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.fft
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Global Data Directory
DATA_DIR = 'data/NS2D'
PLOTS_DIR = 'plots/NS2D'
CHECKPOINTS_DIR = 'checkpoints/NS2D'

# Global Hyperparameters
GLOBAL_BATCH_SIZE = 8
DEFAULT_EPOCHS = 1500
LEARNING_RATE = 1e-3
MODES_X, MODES_Y = 12, 12
N_INPUT_TIMESTEPS = 10  # Use last 10 time steps (1.0 time units) as input
N_OUTPUT_TIMESTEPS = 1  # Predict 1 time step ahead (0.1 time units)
CHANNELS = N_INPUT_TIMESTEPS
HIDDEN_CHANNELS = 32
NUM_BLOCKS = 4
FULL_RESOLUTION_X, FULL_RESOLUTION_Y = 256, 256
TRAINING_RESOLUTION = 64
PREDICTION_GAP = 1
MODELS_TO_TRAIN = ['Hybrid', 'FNO', 'LOCO']
PAPER_TRAINING_SAMPLES = 1000  # As per FNO paper

# Import neuraloperator library components
from neuralop import H1Loss
from neuralop.training import AdamW

from models2D.fno import FNO
from models2D.hybrid import Hybrid

# Placeholder imports for 2D models (these need to be implemented)
from models2D.loco import LOCO
from utils2D.analysis import analyze_rollout_loss_paper_runs
from utils2D.data import NS2DDataset
from utils2D.training import (
    custom_collate_fn,
    evaluate_model,
    print_model_summary,
    train_model,
)


def initialize_models(device, training_resolution=64):
    """
    Initialize 2D models with configurations for the experiments.
    """
    # LOCO model with exact parameters from old scripts
    loco_model = LOCO(
        in_channels=CHANNELS,  # N_INPUT_TIMESTEPS = 10
        out_channels=N_OUTPUT_TIMESTEPS,  # N_OUTPUT_TIMESTEPS = 1
        hidden_channels=HIDDEN_CHANNELS,  # HIDDEN_CHANNELS = 32
        num_blocks=NUM_BLOCKS,  # NUM_BLOCKS = 4
        modes_x=MODES_X,  # MODES_X = 12
        modes_y=MODES_Y,  # MODES_Y = 12
        use_positional_embedding=True
    ).to(device)

    # Hybrid model with exact parameters from old scripts
    hybrid_model = Hybrid(
        in_channels=CHANNELS,  # N_INPUT_TIMESTEPS = 10
        out_channels=N_OUTPUT_TIMESTEPS,  # N_OUTPUT_TIMESTEPS = 1
        hidden_channels=HIDDEN_CHANNELS,  # HIDDEN_CHANNELS = 32
        n_layers=NUM_BLOCKS,  # NUM_BLOCKS = 4
        modes_x=MODES_X,  # MODES_X = 12
        modes_y=MODES_Y,  # MODES_Y = 12
    ).to(device)

    # FNO model from neuraloperator library with exact parameters
    fno_model = FNO(
        in_channels=CHANNELS,  # N_INPUT_TIMESTEPS = 10
        out_channels=N_OUTPUT_TIMESTEPS,  # N_OUTPUT_TIMESTEPS = 1
        hidden_channels=HIDDEN_CHANNELS,  # HIDDEN_CHANNELS = 32
        n_layers=NUM_BLOCKS,  # NUM_BLOCKS = 4
        modes_x=MODES_X,  # MODES_X = 12
        modes_y=MODES_Y,  # MODES_Y = 12
    ).to(device)

    # Create optimizers first
    loco_optimizer = AdamW(loco_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    hybrid_optimizer = AdamW(hybrid_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    fno_optimizer = AdamW(fno_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    models_dict = {
        'LOCO': {
            'model': loco_model,
            'type': 'loco',
            'optimizer': loco_optimizer,
            'scheduler': torch.optim.lr_scheduler.StepLR(
                loco_optimizer, step_size=200, gamma=0.5
            )
        },
        'Hybrid': {
            'model': hybrid_model,
            'type': 'hybrid',
            'optimizer': hybrid_optimizer,
            'scheduler': torch.optim.lr_scheduler.StepLR(
                hybrid_optimizer, step_size=200, gamma=0.5
            )
        },
        'FNO': {
            'model': fno_model,
            'type': 'fno',
            'optimizer': fno_optimizer,
            'scheduler': torch.optim.lr_scheduler.StepLR(
                fno_optimizer, step_size=200, gamma=0.5
            )
        }
    }

    return models_dict






def plot_losses_2d():
    """
    Plot training losses for 2D models
    """
    plt.figure(figsize=(12, 5))

    for _i, model_name in enumerate(MODELS_TO_TRAIN):
        results_path = os.path.join(CHECKPOINTS_DIR, f'{model_name}_losses.pt')
        if os.path.exists(results_path):
            results = torch.load(results_path)

            plt.subplot(1, 2, 1)
            plt.plot(results['train_losses'], label=f'{model_name} Train')
            plt.subplot(1, 2, 2)
            plt.plot(results['val_losses'], label=f'{model_name} Val')

    plt.subplot(1, 2, 1)
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title('Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'training_losses_2d.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")


def train_process_wrapper(log_file, **kwargs):
    """Redirects stdout/stderr of a process to a log file."""
    sys.stdout.flush()
    sys.stderr.flush()
    with open(log_file, 'w') as f:
        sys.stdout = f
        sys.stderr = f
        safe_run_training(**kwargs)


def safe_run_training(**kwargs):
    """Wrapper for run_training that catches and reports exceptions."""
    try:
        run_training_with_config(**kwargs)
    except Exception as e:
        gpu_id = kwargs.get('gpu_id', 'unknown')
        model_name = kwargs.get('model_to_train', 'unknown')
        print(f"[GPU {gpu_id}] ERROR: {model_name} training failed")
        print(f"[GPU {gpu_id}] {type(e).__name__}: {str(e)}")
        print(f"[GPU {gpu_id}] Traceback:")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise  # Re-raise to ensure proper exit code


def run_training_with_config(use_ddp=False, rank=0, world_size=1, model_to_train='all', gpu_id=0, max_epochs=1500, train_data=None, test_data=None, tqdm_pos=0, activate_rollout_loss=False, rollout_probs=(0.1, 0.1, 0.1), max_rollout_steps=10, enable_augmentation=False, augmentation_noise_levels=None, augmentation_probability=0.5, augmentation_subset_size=1000, run_id=None, run_dir=None, seed=None):
    # Handle run-specific directories for paper experiments
    if augmentation_noise_levels is None:
        augmentation_noise_levels = [0.01, 0.03]
    if run_dir is not None:
        # Use provided run directory for multiple runs mode
        checkpoints_dir = run_dir
        plots_dir = run_dir
    else:
        # Use separate directories for rollout loss and augmentation experiments
        dir_suffix = ''
        if activate_rollout_loss and enable_augmentation:
            dir_suffix = '_rollout_noise'
        elif activate_rollout_loss:
            dir_suffix = '_rollout'
        elif enable_augmentation:
            dir_suffix = '_noise'

        plots_dir = 'plots/moNS2D' + dir_suffix
        checkpoints_dir = 'checkpoints/moNS2D' + dir_suffix

    if rank == 0:
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        if activate_rollout_loss:
            print(f"[GPU {gpu_id}] Rollout mode - separate directories:")
            print(f"[GPU {gpu_id}] Checkpoints: {checkpoints_dir}")
            print(f"[GPU {gpu_id}] Plots: {plots_dir}")

    device = torch.device(f'cuda:{gpu_id if not use_ddp else rank}')

    # Set random seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"[GPU {gpu_id}] Seed: {seed}")

    run_info = f" (run {run_id})" if run_id is not None else ""
    print(f"[GPU {gpu_id}] Training {model_to_train}{run_info}")
    print(f"[GPU {gpu_id}] Device: {device}")

    # Add delay to stagger dataset loading and reduce I/O contention
    import time
    time.sleep(gpu_id * 2)  # Stagger by 2 seconds per GPU

    print(f"[GPU {gpu_id}] Loading training data")
    import sys
    sys.stdout.flush()
    train_dataset = NS2DDataset(
        'data',
        split='train',
        data_tensor=train_data,
        target_resolution=64,
        max_rollout_steps=max_rollout_steps,
        subset_mode=True,
        subset_size=augmentation_subset_size if enable_augmentation else 1000,  # Use configurable size for augmentation
        enable_augmentation=enable_augmentation,
        augmentation_noise_levels=augmentation_noise_levels,
        augmentation_probability=augmentation_probability
    )
    print(f"[GPU {gpu_id}] Loading test data")
    sys.stdout.flush()
    test_dataset = NS2DDataset(
        'data',
        split='test',
        data_tensor=test_data,
        target_resolution=64,
        max_rollout_steps=max_rollout_steps,
        subset_mode=True,
        subset_size=200  # Use 200 testing samples as per FNO paper (no augmentation for test)
    )
    print(f"[GPU {gpu_id}] Datasets loaded")

    # Check memory usage
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        print(f"[GPU {gpu_id}] Memory: {memory_info.percent:.1f}% ({memory_info.used / 1024**3:.1f}GB/{memory_info.total / 1024**3:.1f}GB)")
    except ImportError:
        print(f"[GPU {gpu_id}] psutil unavailable for memory monitoring")

    sys.stdout.flush()

    # Use a small number of workers for parallel training to avoid overloading the CPU.
    # This enables asynchronous data loading.
    # Set to 0 when augmentation is enabled to avoid multiprocessing issues
    num_dataloader_workers = 0 if enable_augmentation else 1

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if use_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=8 // (world_size if use_ddp else 1),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_dataloader_workers,
        pin_memory=True,
        persistent_workers=num_dataloader_workers > 0,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8 // (world_size if use_ddp else 1),
        shuffle=True,
        num_workers=num_dataloader_workers,
        pin_memory=True,
        persistent_workers=num_dataloader_workers > 0,
        collate_fn=custom_collate_fn
    )

    models_dict = initialize_models(device, training_resolution=64)

    model_names_to_train = [model_to_train] if model_to_train != 'all' else ['Hybrid', 'FNO', 'LOCO']
    print(f"[GPU {gpu_id}] Training: {model_names_to_train}")
    print(f"[GPU {gpu_id}] Available: {list(models_dict.keys())}")

    # Filter models to only those that exist in models_dict
    filtered_models = [m for m in model_names_to_train if m in models_dict]
    print(f"[GPU {gpu_id}] Filtered: {filtered_models}")
    model_names_to_train = filtered_models

    if use_ddp:
        for name in model_names_to_train:
            models_dict[name]['model'] = DDP(models_dict[name]['model'], device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        summary_models = {name: models_dict[name]['model'] for name in model_names_to_train}
        print_model_summary(summary_models)

    # Initialize H1Loss for a 2D problem on a unit domain
    loss_fn = H1Loss(d=2)

    # Dictionary to store loss histories for all models
    all_losses = {}

    # Calculate steps per epoch to match paper
    effective_batch_size = 8 // (world_size if use_ddp else 1)
    max_steps = 1000 // effective_batch_size
    print(f"[GPU {gpu_id}] Max steps/epoch: {max_steps}")

    for name in model_names_to_train:
        model_info = models_dict[name]
        print(f"\n[GPU {gpu_id}] Training {name}")
        print(f"[GPU {gpu_id}] Model: {model_info['type']}")

        # Initialize loss tracking for this model
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        for epoch in range(max_epochs):
            if use_ddp:
                train_sampler.set_epoch(epoch)

            train_loss = train_model(model_info['model'], train_loader, model_info['optimizer'], loss_fn, device, model_info['type'], rank, model_name_for_tqdm=name, tqdm_pos=tqdm_pos, max_steps_per_epoch=None, activate_rollout_loss=activate_rollout_loss, rollout_probs=rollout_probs, max_rollout_steps=max_rollout_steps)
            val_loss = evaluate_model(model_info['model'], test_loader, loss_fn, device, model_info['type'], tqdm_pos=tqdm_pos, max_eval_samples=None)

            if use_ddp:
                # Average losses across processes
                train_loss_tensor = torch.tensor(train_loss).to(device)
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                train_loss = train_loss_tensor.item() / world_size

            # Keep learning rate constant after epoch 2000
            if epoch < 1500:
                model_info['scheduler'].step()

            if rank == 0:
                # Track losses
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"{name} | Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                # Save best model only
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_path = os.path.join(checkpoints_dir, f'{name}_best.pth')
                    model_to_save = model_info['model'].module if use_ddp else model_info['model']
                    torch.save({
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': model_info['optimizer'].state_dict(),
                        'epoch': epoch + 1,
                        'best_val_loss': best_val_loss,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    }, best_checkpoint_path)
                    print(f"New best model saved: {best_checkpoint_path} (Val Loss: {val_loss:.4f})")

        if rank == 0:
            # Store loss history for this model
            all_losses[name] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }

            # Save loss data with run-specific filename
            if run_id is not None:
                loss_filename = f'{name}_run{run_id}_losses.pt'
            else:
                loss_filename = f'{name}_losses.pt'
            loss_data_path = os.path.join(checkpoints_dir, loss_filename)
            torch.save(all_losses[name], loss_data_path)
            print(f"Loss data saved: {loss_data_path}")

    print(f"🎉 [GPU {gpu_id}] Training completed successfully for {model_to_train}!")


def run_parallel_training_runs(models_to_train, num_runs, runs_per_gpu, gpu_ids, epochs, train_data, test_data, **kwargs):
    """
    Run multiple independent training runs for each model across multiple GPUs for paper results.

    Args:
        models_to_train: List of model names to train
        num_runs: Number of independent runs per model
        runs_per_gpu: Number of runs to execute in parallel per GPU
        gpu_ids: List of GPU IDs to use
        epochs: Number of epochs per run
        train_data: Training data tensor
        test_data: Test data tensor
        **kwargs: Additional arguments (rollout loss, augmentation, etc.)
    """
    import time

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Start method already set

    # Create base directory for paper runs
    base_runs_dir = CHECKPOINTS_DIR + '_paper_runs'
    os.makedirs(base_runs_dir, exist_ok=True)

    # Calculate total runs and distribute across GPUs (limited to 5 hardcoded seeds)
    max_runs_per_model = min(num_runs, 5)
    total_runs = len(models_to_train) * max_runs_per_model
    total_gpu_slots = len(gpu_ids) * runs_per_gpu

    print("Starting paper training runs:")
    print(f"   Models: {models_to_train}")
    print(f"   Runs per model: {max_runs_per_model} (limited by hardcoded seeds)")
    print(f"   Total runs: {total_runs}")
    print(f"   Available GPU slots: {total_gpu_slots} ({len(gpu_ids)} GPUs × {runs_per_gpu} runs/GPU)")
    print(f"   Epochs per run: {epochs}")
    print("   Hardcoded seeds: [42, 123, 456, 789, 1024]")

    if total_runs > total_gpu_slots:
        print(f"Warning: {total_runs} runs > {total_gpu_slots} GPU slots, queueing")

    # Hardcoded seeds for reproducibility
    HARDCODED_SEEDS = [42, 123, 456, 789, 1024]

    # Create run assignments
    run_queue = []
    for model_name in models_to_train:
        for run_id in range(1, max_runs_per_model + 1):
            run_queue.append({
                'model_name': model_name,
                'run_id': run_id,
                'seed': HARDCODED_SEEDS[run_id - 1]  # Use hardcoded seed
            })

    # Distribute runs across GPU slots
    active_processes = []
    completed_runs = 0

    def start_run_process(run_info, gpu_id, process_slot):
        """Start a single training run process."""
        model_name = run_info['model_name']
        run_id = run_info['run_id']
        seed = run_info['seed']

        # Create unique output directory
        run_dir = os.path.join(base_runs_dir, model_name, f'run_{run_id}')
        os.makedirs(run_dir, exist_ok=True)

        # Create log file
        log_file = os.path.join(run_dir, f'training_{model_name}_run{run_id}_gpu{gpu_id}.log')

        print(f"Starting {model_name} run {run_id} (GPU {gpu_id})")

        p = multiprocessing.Process(
            target=train_process_wrapper,
            args=(log_file,),
            kwargs={
                'use_ddp': False,
                'rank': 0,
                'world_size': 1,
                'model_to_train': model_name,
                'gpu_id': gpu_id,
                'max_epochs': epochs,
                'train_data': train_data,
                'test_data': test_data,
                'tqdm_pos': process_slot,
                'run_id': run_id,
                'run_dir': run_dir,
                'seed': seed,
                **kwargs
            }
        )
        p.start()
        return p, run_info, gpu_id, process_slot

    # Main execution loop
    run_index = 0

    while run_index < len(run_queue) or active_processes:
        # Start new processes if slots available and runs remaining
        while len(active_processes) < total_gpu_slots and run_index < len(run_queue):
            # Find next available GPU slot
            gpu_slot = len(active_processes)
            gpu_id = gpu_ids[gpu_slot // runs_per_gpu]
            process_slot = gpu_slot % runs_per_gpu

            # Start the run
            process_info = start_run_process(run_queue[run_index], gpu_id, process_slot)
            active_processes.append(process_info)
            run_index += 1

        # Check for completed processes
        if active_processes:
            time.sleep(5)  # Check every 5 seconds

            for i in range(len(active_processes) - 1, -1, -1):
                p, run_info, gpu_id, process_slot = active_processes[i]

                if not p.is_alive():
                    p.join()
                    completed_runs += 1

                    if p.exitcode == 0:
                        print(f"{run_info['model_name']} run {run_info['run_id']} complete ({completed_runs}/{total_runs})")
                    else:
                        print(f"{run_info['model_name']} run {run_info['run_id']} failed (code {p.exitcode})")

                    # Remove completed process
                    active_processes.pop(i)

    print(f"🎉 All paper training runs completed! ({completed_runs}/{total_runs} successful)")
    print(f"📁 Results saved in: {base_runs_dir}")
    print("Use --mode plot-paper-losses for averaged plots")

def load_ns2d_data():
    """Load NS2D data - simplified version from old scripts"""
    train_data_path = os.path.join(DATA_DIR, f"nsforcing_train_T100_{FULL_RESOLUTION_X}.pt")
    if not os.path.exists(train_data_path):
        train_data_path = os.path.join(DATA_DIR, f"nsforcing_train_{FULL_RESOLUTION_X}.pt")

    test_data_path = os.path.join(DATA_DIR, f"nsforcing_test_T100_{FULL_RESOLUTION_X}.pt")
    if not os.path.exists(test_data_path):
        test_data_path = os.path.join(DATA_DIR, f"nsforcing_test_{FULL_RESOLUTION_X}.pt")

    train_data_tensor = torch.load(train_data_path, map_location='cpu')['u']
    test_data_tensor = torch.load(test_data_path, map_location='cpu')['u']
    # Downsample for training
    if TRAINING_RESOLUTION and train_data_tensor.shape[-1] != TRAINING_RESOLUTION:
        print(f"Downsampling training data to {TRAINING_RESOLUTION}x{TRAINING_RESOLUTION}")
        train_data_tensor = torch.nn.functional.interpolate(
            train_data_tensor, size=(TRAINING_RESOLUTION, TRAINING_RESOLUTION), mode='bicubic', align_corners=False
        )
        print(f"Downsampling test data to {TRAINING_RESOLUTION}x{TRAINING_RESOLUTION}")
        test_data_tensor = torch.nn.functional.interpolate(
            test_data_tensor, size=(TRAINING_RESOLUTION, TRAINING_RESOLUTION), mode='bicubic', align_corners=False
        )
    train_data_tensor.share_memory_()
    test_data_tensor.share_memory_()

    return train_data_tensor, test_data_tensor


def aggregate_multiple_runs(model_dir, model_name):
    """Aggregate loss data from multiple runs - simplified version from old scripts"""
    print(f"Looking for runs in: {model_dir}")

    if not os.path.exists(model_dir):
        print(f"Model directory does not exist: {model_dir}")
        return None, None

    # Find run directories
    run_dirs = [d for d in os.listdir(model_dir)
                if d.startswith('run_') and os.path.isdir(os.path.join(model_dir, d))]

    if not run_dirs:
        print(f"No run directories found in {model_dir}")
        return None, None

    print(f"Found {len(run_dirs)} runs for {model_name}")

    all_train_losses = []
    all_val_losses = []

    for run_dir in sorted(run_dirs):
        run_path = os.path.join(model_dir, run_dir)

        # Look for loss files (adapt this based on your actual file structure)
        loss_files = [f for f in os.listdir(run_path) if f.endswith('.pt') and 'loss' in f.lower()]

        if loss_files:
            try:
                # Load the first loss file found
                loss_file = os.path.join(run_path, loss_files[0])
                data = torch.load(loss_file, weights_only=False, map_location='cpu')

                # Extract train and val losses (adapt based on your data structure)
                if isinstance(data, dict):
                    train_losses = data.get('train_losses', data.get('train', []))
                    val_losses = data.get('val_losses', data.get('val', []))
                else:
                    # Fallback - assume data is a list of losses
                    train_losses = val_losses = data

                if train_losses and val_losses:
                    all_train_losses.append(train_losses)
                    all_val_losses.append(val_losses)
                    print(f"Loaded losses from {run_dir}")
                else:
                    print(f"No loss data in {loss_files[0]}")

            except Exception as e:
                print(f"Failed to load {loss_files[0]}: {e}")
        else:
            print(f"No loss files in {run_dir}")

    if not all_train_losses:
        print(f"No valid loss data found for {model_name}")
        return None, None

    # Convert to numpy arrays and compute statistics

    # Ensure all runs have the same length by taking minimum
    min_len = min(len(losses) for losses in all_train_losses)
    train_array = np.array([losses[:min_len] for losses in all_train_losses])
    val_array = np.array([losses[:min_len] for losses in all_val_losses])

    # Compute mean, std, and percentiles
    train_stats = {
        'mean': np.mean(train_array, axis=0),
        'std': np.std(train_array, axis=0),
        'p10': np.percentile(train_array, 10, axis=0),
        'p90': np.percentile(train_array, 90, axis=0),
        'runs': len(all_train_losses)
    }

    val_stats = {
        'mean': np.mean(val_array, axis=0),
        'std': np.std(val_array, axis=0),
        'p10': np.percentile(val_array, 10, axis=0),
        'p90': np.percentile(val_array, 90, axis=0),
        'runs': len(all_val_losses)
    }

    print(f"Aggregated {len(all_train_losses)} runs for {model_name}")
    return train_stats, val_stats


def plot_paper_losses(aggregated_losses, plots_dir):
    """Plot aggregated losses from paper runs - simplified version from old scripts"""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (model, data) in enumerate(aggregated_losses.items()):
        color = colors[i % len(colors)]

        train_mean = data['train']['mean']
        train_p10 = data['train']['p10']
        train_p90 = data['train']['p90']
        val_mean = data['val']['mean']
        val_p10 = data['val']['p10']
        val_p90 = data['val']['p90']

        epochs = np.arange(1, len(train_mean) + 1)

        # Plot training losses
        ax1.plot(epochs, train_mean, label=f'{model}', color=color, linewidth=2)
        ax1.fill_between(epochs, train_p10, train_p90, alpha=0.2, color=color)

        # Plot validation losses
        ax2.plot(epochs, val_mean, label=f'{model}', color=color, linewidth=2)
        ax2.fill_between(epochs, val_p10, val_p90, alpha=0.2, color=color)

    # Format training plot
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss (Mean + 10th-90th Percentile)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Format validation plot
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss (Mean + 10th-90th Percentile)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    save_path = os.path.join(plots_dir, 'paper_losses.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Paper losses plot saved to: {save_path}")


def run_plot_paper_losses():
    """Load and plot paper training results"""
    print("🔍 Aggregating and plotting paper results...")

    # Aggregate results for LOCO, Hybrid, FNO models
    paper_models = ['LOCO', 'Hybrid', 'FNO']

    # Check if we have paper runs directory
    paper_runs_dir = CHECKPOINTS_DIR + '_paper_runs'
    if not os.path.exists(paper_runs_dir):
        print(f"Paper runs directory not found: {paper_runs_dir}")
        print("Please run --mode parallel-train first to generate paper training data.")
        return

    print(f"Looking for paper runs in: {paper_runs_dir}")

    aggregated_losses = {}
    for model in paper_models:
        model_dir = os.path.join(paper_runs_dir, model)
        if os.path.exists(model_dir):
            print(f"Aggregating {model} results...")
            train_stats, val_stats = aggregate_multiple_runs(model_dir, model)
            if train_stats is not None and val_stats is not None:
                aggregated_losses[model] = {
                    'train': train_stats,
                    'val': val_stats
                }
                print(f"Aggregated {model} results")
            else:
                print(f"No valid loss data for {model}")
        else:
            print(f"Model directory not found: {model_dir}")

    if aggregated_losses:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        print("Creating paper plots...")
        plot_paper_losses(aggregated_losses, PLOTS_DIR)
        print(f"Plots saved to {PLOTS_DIR}")
    else:
        print("No aggregated loss data for plotting")




def main():
    """
    Main function with simplified argument parsing
    """
    parser = argparse.ArgumentParser(description="2D Navier-Stokes Neural Operator Training")
    parser.add_argument('--mode', type=str, default='parallel-train',
                       choices=['parallel-train', 'parallel-plot', 'rollout-loss'],
                       help="Mode to run")
    parser.add_argument('--model', type=str, default='all',
                       choices=['LOCO', 'Hybrid', 'FNO', 'all'],
                       help="Model to train")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID")
    parser.add_argument('--gpu-ids', type=str, default='4,5,6,7', help="Comma-separated GPU IDs for parallel training")
    parser.add_argument('--parallel-runs', type=int, default=5, help="Number of parallel training runs")
    parser.add_argument('--runs-per-gpu', type=int, default=5, help="Number of runs per GPU")
    parser.add_argument('--epochs', type=int, default=1500, help="Number of epochs")
    parser.add_argument('--rollout-steps', type=int, default=20, help="Number of rollout steps for analysis")
    parser.add_argument('--num-samples', type=int, default=20, help="Number of samples for analysis")
    args = parser.parse_args()

    if args.mode == 'parallel-train':
        # NS2D parallel training for paper results
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')] if args.gpu_ids else [args.gpu]

        # Load data first
        print("Loading NS2D data...")
        train_data, test_data = load_ns2d_data()

        models_to_train = ['LOCO', 'Hybrid', 'FNO'] if args.model == 'all' else [args.model]

        run_parallel_training_runs(
            models_to_train=models_to_train,
            num_runs=args.parallel_runs,
            runs_per_gpu=args.runs_per_gpu or 1,
            gpu_ids=gpu_ids,
            epochs=args.epochs or DEFAULT_EPOCHS,
            train_data=train_data,
            test_data=test_data
        )

        # Automatically create loss plots after training
        print("\nCreating loss plots...")
        run_plot_paper_losses()

    elif args.mode == 'parallel-plot':
        run_plot_paper_losses()

    elif args.mode == 'rollout-loss':
        # Rollout loss analysis mode (always uses paper runs)
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print("🔍 Running rollout loss analysis using paper runs...")
        print(f"🔍 Device: {device}")
        print(f"🔍 Rollout steps: {args.rollout_steps}")
        print(f"🔍 Number of samples: {args.num_samples}")

        # Always use paper runs for simplified script
        analyze_rollout_loss_paper_runs(
            device,
            rollout_steps=args.rollout_steps,
            num_samples=args.num_samples
        )


if __name__ == "__main__":
    main()
