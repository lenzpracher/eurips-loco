"""
Simplified training utilities for neural operator experiments.

This module contains functions for training and evaluating neural operator models
with a focus on simplicity and reproducibility.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass
import time
import threading
import queue
import multiprocessing as mp

from .data import PDEDataset


def train_model(
    model: nn.Module,
    train_dataset: PDEDataset,
    val_dataset: PDEDataset,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    save_path: Optional[str] = None,
    verbose: bool = True,
    progress_prefix: str = "",
    seed: int = None
) -> Dict[str, List[float]]:
    """
    Train a neural operator model.
    
    Args:
        model: Neural operator model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on
        save_path: Path to save the best model (optional)
        verbose: Whether to print training progress
        
    Returns:
        Dictionary containing training and validation loss histories
    """
    model = model.to(device)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs//3), gamma=0.5)
    
    # History tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    if verbose:
        print(f"Starting training for {epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(epochs):
        # Update epoch for resampling if using UnifiedBurgersDataset
        if hasattr(train_dataset, 'set_epoch'):
            train_dataset.set_epoch(epoch)
        if hasattr(val_dataset, 'set_epoch'):
            val_dataset.set_epoch(epoch)
            
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        if verbose:
            desc = f"{progress_prefix} " if progress_prefix else ""
            if seed is not None:
                desc += f"[Seed {seed}] "
            desc += f"Epoch {epoch+1}/{epochs}"
            # Add sample count info
            desc += f" ({len(train_dataset)} samples)"
            pbar = tqdm(train_loader, desc=desc)
        else:
            pbar = train_loader
            
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if verbose and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
                val_batches += 1
        
        # Average losses
        train_loss /= train_batches
        val_loss /= val_batches
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if save_path and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, save_path)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    if verbose:
        print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def train_model_with_progress_tracking(
    model: nn.Module,
    train_dataset: PDEDataset,
    val_dataset: PDEDataset,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    save_path: Optional[str] = None,
    seed: int = None,
    progress_tracker: Optional['SharedProgressTracker'] = None
) -> Dict[str, List[float]]:
    """
    Train a neural operator model with shared progress tracking.
    
    This is similar to train_model but updates a shared progress tracker
    for multiseed training coordination.
    """
    model = model.to(device)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs//3), gamma=0.5)
    
    # History tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Mark seed as started for immediate progress feedback
    if progress_tracker and seed is not None:
        progress_tracker.mark_seed_started(seed)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        total_batches = len(train_loader)
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress within epoch every 20% for cleaner display
            if progress_tracker and seed is not None and batch_idx % max(20, total_batches // 5) == 0:
                batch_fraction = round((batch_idx + 1) / total_batches, 1)  # Round to 0.2, 0.4, 0.6, 0.8
                current_train_loss = train_loss / train_batches
                progress_tracker.update_progress(seed, epoch, current_train_loss, 0.0, batch_fraction)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
                val_batches += 1
        
        # Average losses
        train_loss /= train_batches
        val_loss /= val_batches
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Update progress tracker (epoch completed)
        if progress_tracker and seed is not None:
            progress_tracker.update_progress(seed, epoch, train_loss, val_loss, 1.0)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if save_path and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, save_path)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def evaluate_model(
    model: nn.Module,
    test_dataset: PDEDataset,
    batch_size: int = 32,
    device: str = "cpu",
    compute_rollout: bool = False,
    rollout_steps: int = 10
) -> Dict[str, float]:
    """
    Evaluate a trained neural operator model.
    
    Args:
        model: Trained neural operator model
        test_dataset: Test dataset
        batch_size: Batch size for evaluation
        device: Device to evaluate on
        compute_rollout: Whether to compute autoregressive rollout error
        rollout_steps: Number of steps for rollout evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    
    # Single-step evaluation
    total_loss = 0.0
    total_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Evaluating"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            
            total_loss += loss.item()
            total_batches += 1
            
            all_predictions.append(pred.cpu())
            all_targets.append(batch_y.cpu())
    
    avg_loss = total_loss / total_batches
    
    # Combine all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute additional metrics
    mse = nn.MSELoss()(all_predictions, all_targets).item()
    mae = nn.L1Loss()(all_predictions, all_targets).item()
    
    # Relative L2 error
    relative_l2 = torch.norm(all_predictions - all_targets, dim=(-2, -1) if len(all_targets.shape) > 3 else (-1,)) / torch.norm(all_targets, dim=(-2, -1) if len(all_targets.shape) > 3 else (-1,))
    relative_l2 = relative_l2.mean().item()
    
    results = {
        'test_loss': avg_loss,
        'mse': mse,
        'mae': mae,
        'relative_l2': relative_l2
    }
    
    # Autoregressive rollout evaluation (if requested)
    if compute_rollout:
        rollout_loss = evaluate_rollout(model, test_dataset, rollout_steps, device)
        results['rollout_loss'] = rollout_loss
    
    return results


def evaluate_rollout(
    model: nn.Module,
    dataset: PDEDataset,
    rollout_steps: int,
    device: str = "cpu",
    n_samples: int = 100
) -> float:
    """
    Evaluate autoregressive rollout performance.
    
    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        rollout_steps: Number of autoregressive steps
        device: Device to evaluate on
        n_samples: Number of samples to evaluate on
        
    Returns:
        Average rollout loss
    """
    model.eval()
    criterion = nn.MSELoss()
    
    # Get a subset of data for rollout evaluation
    indices = torch.randperm(len(dataset))[:n_samples]
    rollout_losses = []
    
    with torch.no_grad():
        for i in tqdm(indices, desc=f"Rollout evaluation ({rollout_steps} steps)"):
            x, y_true = dataset[i]
            x = x.unsqueeze(0).to(device)  # Add batch dimension
            
            # Perform rollout
            x_rollout = x.clone()
            rollout_loss = 0.0
            
            for step in range(rollout_steps):
                pred = model(x_rollout)
                
                # Update input for next step (assuming time series structure)
                if len(x_rollout.shape) == 4:  # 2D case: [batch, channels, H, W]
                    if x_rollout.shape[1] > 1:  # Multi-timestep input
                        x_rollout = torch.cat([x_rollout[:, 1:], pred], dim=1)
                    else:
                        x_rollout = pred
                else:  # 1D case: [batch, channels, N]
                    if x_rollout.shape[1] > 1:  # Multi-timestep input
                        x_rollout = torch.cat([x_rollout[:, 1:], pred], dim=1)
                    else:
                        x_rollout = pred
                
                # Accumulate error (compared to ground truth if available)
                if step == 0:  # For first step, compare to immediate target
                    y_step = y_true.unsqueeze(0).to(device)
                    step_loss = criterion(pred, y_step)
                    rollout_loss += step_loss.item()
            
            rollout_losses.append(rollout_loss / rollout_steps)
    
    return np.mean(rollout_losses)


def load_model(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cpu"
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model: Model architecture (uninitialized)
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SharedProgressTracker:
    """Shared progress tracker for multiseed training using simple threading."""
    def __init__(self, n_seeds: int, epochs: int):
        self.n_seeds = n_seeds
        self.epochs = epochs
        self.seed_progress = {}  # seed -> current_epoch (fractional for within-epoch progress)
        self.seed_losses = {}    # seed -> latest_loss
        self.completed_seeds = set()
        self.lock = threading.Lock()
        self._running = True
        self.seed_started = set()  # Track which seeds have started training
        
    def update_progress(self, seed: int, epoch: int, train_loss: float, val_loss: float, batch_fraction: float = 1.0):
        """Update progress for a specific seed."""
        with self.lock:
            self.seed_started.add(seed)
            # Use fractional progress: completed epochs + fraction of current epoch
            self.seed_progress[seed] = epoch + batch_fraction
            self.seed_losses[seed] = {'train': train_loss, 'val': val_loss}
    
    def mark_seed_started(self, seed: int):
        """Mark that a seed has started training (for immediate feedback)."""
        with self.lock:
            self.seed_started.add(seed)
            if seed not in self.seed_progress:
                self.seed_progress[seed] = 0.0  # Just started
            
    def mark_completed(self, seed: int):
        """Mark a seed as completed."""
        with self.lock:
            self.completed_seeds.add(seed)
            
    def get_average_progress(self):
        """Get average progress across all seeds."""
        with self.lock:
            if not self.seed_progress:
                return 0, 0, 0  # avg_epoch, avg_train_loss, avg_val_loss
                
            total_progress = sum(self.seed_progress.values())
            avg_epoch = total_progress / len(self.seed_progress)
            
            # Calculate average losses
            train_losses = [losses['train'] for losses in self.seed_losses.values() if isinstance(losses, dict)]
            val_losses = [losses['val'] for losses in self.seed_losses.values() if isinstance(losses, dict)]
            avg_train_loss = np.mean(train_losses) if train_losses else 0
            avg_val_loss = np.mean(val_losses) if val_losses else 0
            
            return avg_epoch, avg_train_loss, avg_val_loss
            
    def get_completion_status(self):
        """Get completion status."""
        with self.lock:
            return len(self.completed_seeds), self.n_seeds
            
    def stop(self):
        """Stop the progress tracker."""
        with self.lock:
            self._running = False
            
    def is_running(self):
        """Check if the progress tracker is still running."""
        with self.lock:
            return self._running


def train_single_seed_wrapper(args):
    """Wrapper function for parallel training with multiprocessing."""
    (seed, model_class, model_kwargs, train_dataset, val_dataset, 
     epochs, batch_size, learning_rate, device, save_dir, progress_tracker) = args
    
    # Set seed for this process
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    # Create fresh model instance
    model = model_class(**model_kwargs)
    
    # Create unique save path for this seed
    seed_save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        seed_save_path = os.path.join(save_dir, f"seed_{seed}_best.pt")
    
    # Train model with progress tracking
    history = train_model_with_progress_tracking(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        save_path=seed_save_path,
        seed=seed,
        progress_tracker=progress_tracker
    )
    
    # Load best model for evaluation
    if seed_save_path and os.path.exists(seed_save_path):
        checkpoint = torch.load(seed_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    test_results = evaluate_model(
        model=model,
        test_dataset=val_dataset,
        batch_size=batch_size,
        device=device,
        compute_rollout=True,
        rollout_steps=10
    )
    
    # Mark seed as completed
    if progress_tracker:
        progress_tracker.mark_completed(seed)
    
    return {
        'seed': seed,
        'train_losses': history['train_losses'],
        'val_losses': history['val_losses'],
        'final_train_loss': history['train_losses'][-1],
        'final_val_loss': history['val_losses'][-1],
        'test_results': test_results
    }


def train_model_multiseed(
    model_class,
    model_kwargs: Dict,
    train_dataset: PDEDataset,
    val_dataset: PDEDataset,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    save_dir: Optional[str] = None,
    n_seeds: int = 5,
    base_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, List]:
    """
    Train a model multiple times with different seeds for statistical robustness.
    
    Args:
        model_class: Model class to instantiate
        model_kwargs: Arguments for model instantiation
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of epochs to train
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints (optional)
        n_seeds: Number of different seeds to run
        base_seed: Base seed (will use base_seed + i for each run)
        verbose: Whether to print progress
        
    Returns:
        Dictionary with lists of results for each seed
    """
    import concurrent.futures
    
    results = {
        'train_losses': [],
        'val_losses': [],
        'final_train_loss': [],
        'final_val_loss': [],
        'test_results': [],
        'seeds': []
    }
    
    # Create shared progress tracker
    progress_tracker = SharedProgressTracker(n_seeds, epochs)
    
    # Prepare arguments for parallel execution
    seeds = [base_seed + i for i in range(n_seeds)]
    args_list = [(seed, model_class, model_kwargs, train_dataset, val_dataset, 
                  epochs, batch_size, learning_rate, device, save_dir, progress_tracker) for seed in seeds]
    
    if verbose:
        print(f"Training {n_seeds} models with seeds {seeds}")
        
        # Create a temporary model instance for summary
        temp_model = model_class(**model_kwargs)
        
        # Get input shape from dataset
        sample_input, _ = train_dataset[0]
        input_shape = sample_input.shape
        
        print(f"\n{'='*50}")
        print("MODEL SUMMARY")
        print(f"{'='*50}")
        print_model_summary(temp_model, input_shape)
        print(f"{'='*50}")
        
        del temp_model  # Free memory
        
        print(f"Training dataset size: {len(train_dataset)} samples")
        print(f"Validation dataset size: {len(val_dataset)} samples")
        print("Using threading for concurrent training")
    
    # For GPU training, use threading with limited workers to avoid memory issues
    max_workers = min(n_seeds, 9) if device.startswith('cuda') else n_seeds
    
    if verbose and max_workers < n_seeds:
        print(f"Limiting concurrent GPU training to {max_workers} workers to manage memory")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        def train_single_seed_inline(args):
            return train_single_seed_wrapper(args)
        futures = [executor.submit(train_single_seed_inline, args) for args in args_list]
        
        # Start progress monitoring in a separate thread
        total_progress = n_seeds * epochs  # Total epochs across all seeds
        
        def progress_monitor():
            """Monitor progress from all training processes."""
            pbar = tqdm(total=total_progress, desc=f"Multi-seed training ({max_workers} parallel)", 
                       unit="epoch", dynamic_ncols=True)
            last_total_progress = 0
            startup_displayed = False
            
            try:
                while True:
                    time.sleep(0.5)  # Update every 0.5 seconds
                    
                    try:
                        avg_epoch, avg_train_loss, avg_val_loss = progress_tracker.get_average_progress()
                        completed_seeds, total_seeds = progress_tracker.get_completion_status()
                        started_seeds = len(progress_tracker.seed_started)
                        
                        # Calculate total progress (sum of epochs completed across all seeds)
                        # Round progress to 1 decimal place for cleaner display
                        total_epochs_completed = sum(round(p, 1) for p in progress_tracker.seed_progress.values()) if progress_tracker.seed_progress else 0
                        
                        progress_increase = total_epochs_completed - last_total_progress
                        
                        if progress_increase >= 0.1:  # Only update for meaningful progress (0.1 epoch increments)
                            pbar.update(progress_increase)
                            last_total_progress = total_epochs_completed
                        
                        # Update progress bar with current averages
                        postfix = {
                            'Complete': f"{completed_seeds}/{total_seeds}",
                            'Started': f"{started_seeds}/{total_seeds}"
                        }
                        
                        if avg_epoch > 0:
                            # Round to 1 decimal place for cleaner display
                            postfix['Avg Epoch'] = f"{avg_epoch:.1f}/{epochs}"
                            if avg_train_loss > 0:
                                postfix['Train Loss'] = f"{avg_train_loss:.4f}"
                            if avg_val_loss > 0:
                                postfix['Val Loss'] = f"{avg_val_loss:.4f}"
                        
                        # Show startup message for first few updates
                        if not startup_displayed and started_seeds > 0:
                            if verbose:
                                tqdm.write(f"Training started - {started_seeds}/{total_seeds} seeds initialized")
                            startup_displayed = True
                        
                        # Add debug info occasionally
                        update_counter = getattr(progress_monitor, 'counter', 0) + 1
                        progress_monitor.counter = update_counter
                        
                        if update_counter % 120 == 0:  # Every 60 seconds
                            postfix['Debug'] = f"Updates:{update_counter}"
                            if verbose:
                                tqdm.write(f"[DEBUG] Progress - Started: {started_seeds}, Progress: {dict(progress_tracker.seed_progress)}")
                        
                        pbar.set_postfix(postfix)
                        
                        # Check if all training is complete
                        if completed_seeds >= total_seeds:
                            break
                            
                    except Exception as e:
                        # If there's an error in progress tracking, just continue
                        # This prevents the progress monitor from crashing
                        if verbose:
                            tqdm.write(f"Progress tracking error: {e}")
                        continue
                        
            finally:
                pbar.close()
        
        # Start progress monitoring thread
        progress_thread = threading.Thread(target=progress_monitor, daemon=True)
        progress_thread.start()
        
        # Wait for all futures to complete and collect results
        try:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                seed = result['seed']
                
                if verbose:
                    tqdm.write(f"\nCompleted seed {seed}")
                    tqdm.write(f"  Final train loss: {result['final_train_loss']:.6f}")
                    tqdm.write(f"  Final val loss: {result['final_val_loss']:.6f}")
                    tqdm.write(f"  Test loss: {result['test_results']['test_loss']:.6f}")
                
                results['seeds'].append(seed)
                results['train_losses'].append(result['train_losses'])
                results['val_losses'].append(result['val_losses'])
                results['final_train_loss'].append(result['final_train_loss'])
                results['final_val_loss'].append(result['final_val_loss'])
                results['test_results'].append(result['test_results'])
        
        finally:
            # Stop progress tracking and wait for thread to finish
            progress_tracker.stop()
            progress_thread.join(timeout=3)
    
    # Compute statistics
    results['mean_final_train_loss'] = np.mean(results['final_train_loss'])
    results['std_final_train_loss'] = np.std(results['final_train_loss'])
    results['mean_final_val_loss'] = np.mean(results['final_val_loss'])
    results['std_final_val_loss'] = np.std(results['final_val_loss'])
    
    # Compute test statistics
    test_metrics = ['test_loss', 'mse', 'mae', 'relative_l2', 'rollout_loss']
    for metric in test_metrics:
        if metric in results['test_results'][0]:
            values = [result[metric] for result in results['test_results']]
            results[f'mean_{metric}'] = np.mean(values)
            results[f'std_{metric}'] = np.std(values)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"MULTI-SEED RESULTS SUMMARY ({n_seeds} seeds)")
        print(f"{'='*60}")
        print(f"Seeds used: {results['seeds']}")
        print(f"")
        print(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print(f"{'-'*68}")
        
        metrics = {
            'Train Loss': ('final_train_loss', 'mean_final_train_loss', 'std_final_train_loss'),
            'Val Loss': ('final_val_loss', 'mean_final_val_loss', 'std_final_val_loss'),
            'Test Loss': ('test_loss', 'mean_test_loss', 'std_test_loss'),
            'MAE': ('mae', 'mean_mae', 'std_mae'),
            'Relative L2': ('relative_l2', 'mean_relative_l2', 'std_relative_l2'),
            'Rollout Loss': ('rollout_loss', 'mean_rollout_loss', 'std_rollout_loss')
        }
        
        for metric_name, (raw_key, mean_key, std_key) in metrics.items():
            if mean_key in results:
                if raw_key == 'test_loss' or raw_key == 'mae' or raw_key == 'relative_l2' or raw_key == 'rollout_loss':
                    # These are in test_results
                    values = [res[raw_key] for res in results['test_results'] if raw_key in res]
                else:
                    # These are direct keys
                    values = results[raw_key]
                
                if values:
                    min_val = min(values)
                    max_val = max(values)
                    mean_val = results[mean_key]
                    std_val = results[std_key]
                    
                    print(f"{metric_name:<20} {mean_val:<12.6f} {std_val:<12.6f} {min_val:<12.6f} {max_val:<12.6f}")
        
        print(f"{'='*60}")
    
    return results


def print_model_summary(model: nn.Module, input_shape: Tuple[int, ...]):
    """Print a summary of the model architecture."""
    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Create a dummy input to get output shape
    if len(input_shape) == 3:  # 1D case
        dummy_input = torch.randn(1, *input_shape)
    else:  # 2D case
        dummy_input = torch.randn(1, *input_shape)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # For full reproducibility, also set these
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False