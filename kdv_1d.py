#!/usr/bin/env python3
"""
KdV 1D Experiment

Neural operator training for the Korteweg-de Vries equation.

Usage:
    python kdv_1d.py --mode parallel-train
    python kdv_1d.py --mode parallel-plot
    python kdv_1d.py --mode rollout-loss
    python kdv_1d.py --mode space-time
"""

import argparse
import atexit
import contextlib
import datetime
import multiprocessing
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from neuralop.models import FNO
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from models1D.hybrid import Hybrid
from models1D.loco import LOCO
from utils1D.data import KdVDataset
from utils1D.plotting_utils import plot_losses
from utils1D.utils import print_model_summary

# Global Directories
PLOTS_DIR = 'plots/KdV'
CHECKPOINTS_DIR = 'checkpoints/KdV'
DATA_DIR = 'data/KdV'  # Updated to use local data

# Global Model Configuration
# This parameter controls whether the ChannelMLP is used within the spectral convolution blocks
# of the custom LOCO models.
USE_SPECTRAL_CHANNEL_MLP = True

# Fixed seeds for reproducible parallel training
PARALLEL_TRAINING_SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768, 12312, 2542, 23452, 1231, 12342315, 5234, 523452, 23452, 65475, 879768]

# =============================================================================
# GLOBAL MODEL PARAMETERS
# =============================================================================

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
MAX_STEPS_PER_EPOCH = 100

# Model Architecture Parameters
MODES = 16
CHANNELS = 1  # We have 1D data, so 1 channel
HIDDEN_CHANNELS = 26  # Balanced to match FNO parameter count (~49k params)
NUM_BLOCKS = 4
SPATIAL_POINTS = 128  # Number of spatial points in the simulation

# Data Parameters
PREDICTION_GAP = 10
TRAIN_SPLIT_RATIO = 0.8

# Model-specific Parameters
FNO_HIDDEN_CHANNELS = 32
FNO_LIFTING_CHANNEL_RATIO = 2
FNO_PROJECTION_CHANNEL_RATIO = 1

# Evaluation Parameters
NUM_ROLLOUT_STEPS = 40
NUM_TRAJECTORIES_FOR_AVG = 100
NUM_COMPARISON_TRAJECTORIES = 5


def initialize_models(device):
    """Initializes all models and returns them in a dictionary."""
    loco_model = LOCO(
        channels=CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        num_blocks=NUM_BLOCKS,
        modes=MODES,
        use_mlp=USE_SPECTRAL_CHANNEL_MLP
    ).to(device)

    hybrid_model = Hybrid(
        channels=CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        num_blocks=NUM_BLOCKS,
        modes=MODES,
        use_mlp=USE_SPECTRAL_CHANNEL_MLP
    ).to(device)

    fno_model = FNO(
        n_modes=(MODES,),
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        hidden_channels=FNO_HIDDEN_CHANNELS,
        lifting_channel_ratio=FNO_LIFTING_CHANNEL_RATIO,
        projection_channel_ratio=FNO_PROJECTION_CHANNEL_RATIO,
        n_layers=NUM_BLOCKS,
        positional_embedding='grid',
        use_mlp=USE_SPECTRAL_CHANNEL_MLP,
        mlp_dropout=0,
        mlp_expansion=1,
        non_linearity=nn.GELU(),
        stabilizer=None
    ).to(device)

    # Clean model names
    models_dict = {
        "LOCO": (loco_model, 'sno', os.path.join(CHECKPOINTS_DIR, 'loco_model.pt'), {'type': 'adamw', 'weight_decay': 1e-4}),
        "Hybrid": (hybrid_model, 'sno', os.path.join(CHECKPOINTS_DIR, 'hybrid_model.pt'), {'type': 'adamw', 'weight_decay': 1e-4}),
        "FNO": (fno_model, 'fno', os.path.join(CHECKPOINTS_DIR, 'fno_model.pt'), {'type': 'adamw', 'weight_decay': 1e-4})
    }
    return models_dict


def _pool_init():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, max_steps_per_epoch=100, model_type='sno'):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    step_count = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for _i, data in enumerate(pbar):
        if step_count >= max_steps_per_epoch:
            break

        x, y = data['x'].to(device), data['y'].to(device)

        if model_type == 'fno':
            x, y = x.transpose(1, 2), y.transpose(1, 2)  # Right format for neuralop FNO model

        optimizer.zero_grad()
        out = model(x)

        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step_count += 1

        # Display running average L2 loss in the progress bar
        pbar.set_postfix(loss=f'{total_loss / step_count:.6f}', step=f'{step_count}/{max_steps_per_epoch}')

    train_loss = total_loss / step_count
    val_loss = evaluate_model(model, val_loader, loss_fn, device, model_type, max_eval_batches=100)

    return train_loss, val_loss


def evaluate_model(model, test_loader, loss_fn, device, model_type='sno', max_eval_batches=100):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    batch_count = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            if batch_count >= max_eval_batches:
                break
            x, y = data['x'].to(device), data['y'].to(device)

            if model_type == 'fno':
                x, y = x.transpose(1, 2), y.transpose(1, 2)

            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count




def train_single_run_with_seed(model_name, model_info, train_dataset, test_dataset,
                              seed, run_idx, gpu_id, epochs, batch_size, max_steps_per_epoch,
                              show_progress=False, progress_queue=None, parallel_runs_dir=None):
    """Train a single model with a specific seed"""
    import os

    # Set up signal handler for graceful termination
    def signal_handler(signum, frame):
        import os
        print(f"Process {os.getpid()} received signal {signum}, terminating gracefully...")
        os._exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Set all random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Create data loaders with seeded generator
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model (will use the set torch.manual_seed)
    model, model_type, _, optim_info = model_info
    model = model.to(device)

    # Initialize optimizer
    if optim_info['type'] == 'adamw':
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=optim_info.get('weight_decay', 1e-4))
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_info['type']}")

    loss_fn = nn.MSELoss()
    train_losses = []
    val_losses = []

    # Progress reporting - only show if requested
    if show_progress:
        pbar = tqdm(range(epochs), desc=f"GPU{gpu_id} {model_name} (seed={seed})", leave=True)
    else:
        pbar = None

    for epoch in range(epochs):
        train_loss, val_loss = train_model(model, train_loader, test_loader, optimizer,
                                         loss_fn, device, max_steps_per_epoch, model_type)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Update progress
        if pbar:
            pbar.set_postfix({'train': f'{train_loss:.6f}', 'val': f'{val_loss:.6f}'})
            pbar.update(1)

        # Send progress to parent process for aggregated display
        if progress_queue:
            progress_queue.put({
                'model_name': model_name,
                'run_idx': run_idx,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'total_epochs': epochs
            })

    if pbar:
        pbar.close()

    # Save individual model checkpoint if parallel_runs_dir is provided
    model_state_dict = None
    if parallel_runs_dir:
        run_dir = os.path.join(parallel_runs_dir, f'run_{run_idx}')
        os.makedirs(run_dir, exist_ok=True)
        checkpoint_path = os.path.join(run_dir, f'{model_name}_model.pt')
        model_state_dict = model.state_dict().copy()  # Create a copy for safe return
        torch.save(model_state_dict, checkpoint_path)

    # Return results
    return {
        'model_name': model_name,
        'seed': seed,
        'run_idx': run_idx,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'model_state_dict': model_state_dict,
        'checkpoint_path': checkpoint_path if parallel_runs_dir else None
    }


def run_parallel_training_process(args):
    """Wrapper function for multiprocessing"""
    return train_single_run_with_seed(*args)


def aggregate_parallel_results(all_results, model_names):
    """Aggregate results from parallel training runs"""
    aggregated = {}

    for model_name in model_names:
        model_results = [r for r in all_results if r['model_name'] == model_name]

        if not model_results:
            continue

        # Extract data
        train_losses_all = [r['train_losses'] for r in model_results]
        val_losses_all = [r['val_losses'] for r in model_results]
        seeds = [r['seed'] for r in model_results]

        # Convert to numpy arrays for easier computation
        train_losses_array = np.array(train_losses_all)  # Shape: (num_runs, num_epochs)
        val_losses_array = np.array(val_losses_all)

        # Compute statistics
        mean_train = np.mean(train_losses_array, axis=0)
        std_train = np.std(train_losses_array, axis=0)
        mean_val = np.mean(val_losses_array, axis=0)
        std_val = np.std(val_losses_array, axis=0)

        aggregated[model_name] = {
            'runs': [{'train': r['train_losses'], 'val': r['val_losses'], 'seed': r['seed']}
                    for r in model_results],
            'mean_train': mean_train.tolist(),
            'std_train': std_train.tolist(),
            'mean_val': mean_val.tolist(),
            'std_val': std_val.tolist(),
            'seeds': seeds,
            'num_runs': len(model_results)
        }

    return aggregated


def select_and_save_best_models(all_results, models_dict, checkpoints_dir, parallel_runs_dir):
    """Select and save the best models based on lowest validation loss."""
    print("Selecting and saving best models...")

    # Group results by model name
    model_results = {}
    for result in all_results:
        model_name = result['model_name']
        if model_name not in model_results:
            model_results[model_name] = []
        model_results[model_name].append(result)

    # For each model type, find the best run (lowest final validation loss)
    for model_name in models_dict:
        if model_name not in model_results:
            print(f"Warning: No results found for {model_name}")
            continue

        results = model_results[model_name]
        best_result = min(results, key=lambda x: x['final_val_loss'])

        # Load the best model from the parallel runs directory
        best_run_idx = best_result['run_idx']
        source_path = os.path.join(parallel_runs_dir, f'run_{best_run_idx}', f'{model_name}_model.pt')

        if os.path.exists(source_path):
            # Copy to main checkpoints directory
            _, _, main_checkpoint_path, _ = models_dict[model_name]
            os.makedirs(os.path.dirname(main_checkpoint_path), exist_ok=True)

            # Load and save to ensure consistency
            state_dict = torch.load(source_path, map_location='cpu', weights_only=False)
            torch.save(state_dict, main_checkpoint_path)

            print(f"Best {model_name} model: Run {best_run_idx} (seed={best_result['seed']}) "
                  f"with final val loss: {best_result['final_val_loss']:.6f}")
            print(f"  Saved to: {main_checkpoint_path}")
        else:
            print(f"Warning: Best model checkpoint not found at {source_path}")


def save_seeds_summary(aggregated_losses, checkpoints_dir, num_runs):
    """Save a summary of seeds and results"""
    summary_path = os.path.join(checkpoints_dir, 'seeds_summary.txt')

    with open(summary_path, 'w') as f:
        f.write("KdV Parallel Training Summary\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Runs: {num_runs}\n")
        f.write(f"Models: {', '.join(aggregated_losses.keys())}\n")
        f.write(f"Hardcoded Seeds: {PARALLEL_TRAINING_SEEDS[:num_runs]}\n\n")

        f.write("Run Assignments:\n")
        for i in range(num_runs):
            f.write(f"Run {i}: seed={PARALLEL_TRAINING_SEEDS[i]}\n")
        f.write("\n")

        f.write("Model Training Results:\n")
        for model_name, data in aggregated_losses.items():
            final_val_mean = data['mean_val'][-1]
            final_val_std = data['std_val'][-1]
            f.write(f"{model_name}: {data['num_runs']} runs completed, "
                   f"mean final val loss: {final_val_mean:.6f} ± {final_val_std:.6f}\n")

    print(f"Seeds summary saved to {summary_path}")


def monitor_progress(progress_queue, total_jobs, models_dict, num_runs, epochs):
    """Monitor and display aggregated progress from all training processes"""
    # Initialize progress tracking
    progress_data = {}
    for model_name in models_dict:
        progress_data[model_name] = {
            'completed_epochs': [0] * num_runs,
            'latest_train_loss': [float('inf')] * num_runs,
            'latest_val_loss': [float('inf')] * num_runs
        }

    # Create master progress bar
    total_epochs = total_jobs * epochs
    master_pbar = tqdm(total=total_epochs, desc="Overall Progress", position=0, leave=True)

    completed_epochs = 0

    try:
        while completed_epochs < total_epochs:
            try:
                # Get progress update with timeout
                update = progress_queue.get(timeout=1.0)

                model_name = update['model_name']
                run_idx = update['run_idx']
                epoch = update['epoch']

                # Update progress tracking
                old_epoch = progress_data[model_name]['completed_epochs'][run_idx]
                progress_data[model_name]['completed_epochs'][run_idx] = epoch + 1
                progress_data[model_name]['latest_train_loss'][run_idx] = update['train_loss']
                progress_data[model_name]['latest_val_loss'][run_idx] = update['val_loss']

                # Update master progress bar
                epochs_increment = (epoch + 1) - old_epoch
                master_pbar.update(epochs_increment)
                completed_epochs += epochs_increment

                # Calculate average losses across all runs for display
                avg_losses = {}
                for mn in models_dict:
                    valid_train = [loss for loss in progress_data[mn]['latest_train_loss'] if loss != float('inf')]
                    valid_val = [loss for loss in progress_data[mn]['latest_val_loss'] if loss != float('inf')]
                    if valid_train and valid_val:
                        avg_losses[mn] = f"T:{np.mean(valid_train):.4f} V:{np.mean(valid_val):.4f}"

                # Update display
                if avg_losses:
                    loss_str = " | ".join([f"{mn}: {loss}" for mn, loss in avg_losses.items()])
                    master_pbar.set_postfix_str(loss_str)

            except Exception:
                # Timeout or other error - continue monitoring
                continue

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        master_pbar.close()


def run_parallel_training(gpu_ids=None, num_runs=10, runs_per_gpu=None, show_individual_progress=False):
    """Run parallel training with multiple seeds."""
    if gpu_ids is None:
        gpu_ids = [0]
    if num_runs > len(PARALLEL_TRAINING_SEEDS):
        raise ValueError(f"Cannot run more than {len(PARALLEL_TRAINING_SEEDS)} parallel runs")

    # Auto-calculate runs per GPU if not specified
    if runs_per_gpu is None:
        runs_per_gpu = max(1, num_runs // len(gpu_ids))

    total_processes = runs_per_gpu * len(gpu_ids)

    print(f"=== Running Parallel Training with {num_runs} runs ===")
    print(f"Using GPUs: {gpu_ids}")
    print(f"Runs per GPU: {runs_per_gpu}")
    print(f"Total parallel processes: {total_processes}")
    print(f"Seeds: {PARALLEL_TRAINING_SEEDS[:num_runs]}")
    print(f"Individual progress bars: {'Enabled' if show_individual_progress else 'Disabled (aggregated only)'}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # Create parallel runs directory
    parallel_runs_dir = os.path.join(CHECKPOINTS_DIR, 'parallel_runs')
    os.makedirs(parallel_runs_dir, exist_ok=True)

    # Load datasets once (shared across all processes)
    print("Loading datasets...")
    train_dataset = KdVDataset(DATA_DIR, split='train', prediction_gap=PREDICTION_GAP)
    test_dataset = KdVDataset(DATA_DIR, split='test', prediction_gap=PREDICTION_GAP)

    # Initialize models dictionary (template for each run)
    device = torch.device('cpu')  # Initialize on CPU, move to GPU in each process
    models_dict = initialize_models(device)

    # Print model summary once
    summary_models_dict = {name: model_info[0] for name, model_info in models_dict.items()}
    print_model_summary(summary_models_dict)

    # Set up progress monitoring with manager for proper sharing
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue() if not show_individual_progress else None

    # Prepare arguments for parallel execution
    all_args = []
    for run_idx in range(num_runs):
        seed = PARALLEL_TRAINING_SEEDS[run_idx]
        gpu_id = gpu_ids[run_idx % len(gpu_ids)]

        for model_name, model_info in models_dict.items():
            args = (model_name, model_info, train_dataset, test_dataset,
                   seed, run_idx, gpu_id, EPOCHS, BATCH_SIZE, MAX_STEPS_PER_EPOCH,
                   show_individual_progress, progress_queue, parallel_runs_dir)
            all_args.append(args)

    # Set up process group for cleanup
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    print(f"\nStarting parallel training with {len(all_args)} total jobs...")
    start_time = time.time()

    try:
        # Start worker processes
        with multiprocessing.Pool(processes=total_processes, initializer=_pool_init) as pool:
            # Register cleanup function
            def cleanup_processes():
                print("\nCleaning up worker processes...")
                pool.terminate()
                pool.join()

            atexit.register(cleanup_processes)

            # Start progress monitoring if not showing individual progress
            monitor_process = None
            if not show_individual_progress:
                monitor_process = multiprocessing.Process(
                    target=monitor_progress,
                    args=(progress_queue, len(all_args), models_dict, num_runs, EPOCHS)
                )
                monitor_process.start()

            # Restore signal handler for main process
            signal.signal(signal.SIGINT, original_sigint_handler)

            # Run training
            all_results = pool.map(run_parallel_training_process, all_args)

            # Stop progress monitoring
            if monitor_process:
                monitor_process.terminate()
                monitor_process.join()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        return

    end_time = time.time()
    print(f"\nParallel training completed in {end_time - start_time:.2f} seconds")

    # Aggregate results
    print("Aggregating results...")
    model_names = list(models_dict.keys())
    aggregated_losses = aggregate_parallel_results(all_results, model_names)

    # Select and save best models to main checkpoints directory
    select_and_save_best_models(all_results, models_dict, CHECKPOINTS_DIR, parallel_runs_dir)

    # Save aggregated results
    torch.save(aggregated_losses, os.path.join(CHECKPOINTS_DIR, 'aggregated_losses.pt'))

    # Save seeds summary
    save_seeds_summary(aggregated_losses, CHECKPOINTS_DIR, num_runs)

    # Plot aggregated results using the specialized parallel plotting function
    run_plot_parallel()

    print(f"Parallel training results saved in '{CHECKPOINTS_DIR}/' directory.")
    print("Results include:")
    print("- Individual model checkpoints in parallel_runs/ subdirectories")
    print("- Best models saved to main checkpoints directory")
    print("- aggregated_losses.pt: All training data")
    print("- seeds_summary.txt: Seed assignments and final results")
    print("- parallel_training_losses.png: Training curves with error bars")


def run_plot_parallel():
    """Load and plot existing parallel training results"""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Check if parallel results exist
    aggregated_path = os.path.join(CHECKPOINTS_DIR, 'aggregated_losses.pt')
    if not os.path.exists(aggregated_path):
        print(f"Error: No parallel training results found at {aggregated_path}")
        print("Please run with --mode parallel-train first to generate the data.")
        return

    print(f"Loading parallel training results from {aggregated_path}...")
    aggregated_losses = torch.load(aggregated_path, weights_only=False)

    # Use the aggregated losses directly (already clean names)
    renamed_losses = aggregated_losses

    # Plot with updated utilities (clean legends and proper error bands)
    plot_losses(renamed_losses, title_suffix="KdV",
                save_path=os.path.join(PLOTS_DIR, 'parallel_training_losses_updated.pdf'))

    print(f"Updated parallel training plot saved to {os.path.join(PLOTS_DIR, 'parallel_training_losses_updated.pdf')}")
    print("Plot features:")
    print("- Clean model names (LOCO, Hybrid, FNO)")
    print("- Error bands showing std across multiple runs")
    print("- Publication-ready styling")

    print("\nParallel plot generation complete!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train and evaluate models on KdV equation data.")
    parser.add_argument('--mode', type=str, default='parallel-train',
                       choices=['parallel-train', 'parallel-plot', 'rollout-loss', 'space-time'],
                       help="Mode to run the script in")
    parser.add_argument('--gpu-ids', type=str, default='0,1,2,3,4,5,6,7', help="Comma-separated GPU IDs for parallel training")
    parser.add_argument('--parallel-runs', type=int, default=20, help="Number of parallel training runs")
    parser.add_argument('--runs-per-gpu', type=int, default=5, help="Number of runs per GPU")
    parser.add_argument('--show-individual-progress', action='store_true', help="Show individual progress bars")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID for rollout analysis")
    parser.add_argument('--rollout-steps', type=int, default=35, help="Number of rollout steps for analysis")
    parser.add_argument('--num-samples', type=int, default=100, help="Number of samples for rollout analysis")
    parser.add_argument('--sample-id', type=int, default=5947, help="Sample ID for space-time visualization (default: 100)")
    args = parser.parse_args()

    if args.mode == 'parallel-train':
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        num_runs = min(args.parallel_runs, len(PARALLEL_TRAINING_SEEDS))
        run_parallel_training(gpu_ids=gpu_ids, num_runs=num_runs, runs_per_gpu=args.runs_per_gpu,
                             show_individual_progress=args.show_individual_progress)
    elif args.mode == 'parallel-plot':
        run_plot_parallel()
    elif args.mode == 'rollout-loss':
        # Rollout loss analysis mode
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print("🔍 Running rollout loss analysis for KdV equation...")
        print(f"🔍 Device: {device}")
        print(f"🔍 Rollout steps: {args.rollout_steps}")
        print(f"🔍 Number of samples: {args.num_samples}")

        # Import and run rollout analysis
        from utils1D.utils import analyze_rollout_loss_1d
        analyze_rollout_loss_1d(
            device=device,
            rollout_steps=args.rollout_steps,
            num_samples=args.num_samples,
            checkpoints_dir=CHECKPOINTS_DIR,
            plots_dir=PLOTS_DIR,
            equation='KdV'
        )
    elif args.mode == 'space-time':
        # Space-time visualization mode
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print("🎨 Running space-time analysis for KdV equation...")
        print(f"🎨 Device: {device}")
        print(f"🎨 Rollout steps: {args.rollout_steps}")
        print(f"🎨 Sample ID: {args.sample_id}")

        # Import and run space-time analysis
        from utils1D.utils import analyze_spacetime_1d
        analyze_spacetime_1d(
            device=device,
            rollout_steps=args.rollout_steps,
            num_samples=1,  # Space-time mode uses single sample with all models
            checkpoints_dir=CHECKPOINTS_DIR,
            plots_dir=PLOTS_DIR,
            equation='KdV',
            sample_id=args.sample_id
        )


if __name__ == "__main__":
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn")

    # Ensure subprocesses use the same Python executable
    import sys
    multiprocessing.set_executable(sys.executable)

    main()
