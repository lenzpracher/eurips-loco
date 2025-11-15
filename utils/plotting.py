"""
Plotting utilities for neural operator experiments.

This module contains functions for visualizing training progress,
model predictions, and comparative analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import os


def plot_training_losses(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
    title: str = "Training Progress"
) -> None:
    """
    Plot training and validation losses.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_predictions_1d(
    x: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    x_coords: Optional[torch.Tensor] = None,
    n_samples: int = 5,
    save_path: Optional[str] = None,
    title: str = "Model Predictions"
) -> None:
    """
    Plot 1D predictions vs ground truth.
    
    Args:
        x: Input data
        y_true: Ground truth
        y_pred: Model predictions
        x_coords: Spatial coordinates (optional)
        n_samples: Number of samples to plot
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    n_samples = min(n_samples, len(y_true))
    
    if x_coords is None:
        x_coords = torch.arange(y_true.shape[-1])
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        ax = axes[i]
        
        # Handle different input shapes
        if len(x.shape) == 3 and x.shape[1] > 1:  # Multi-timestep input
            input_data = x[i, -1]  # Show last timestep
        else:
            input_data = x[i].squeeze()
        
        # Plot input, ground truth, and prediction
        ax.plot(x_coords, input_data, 'k--', label='Input', alpha=0.7)
        ax.plot(x_coords, y_true[i].squeeze(), 'b-', label='Ground Truth', linewidth=2)
        ax.plot(x_coords, y_pred[i].squeeze(), 'r--', label='Prediction', linewidth=2)
        
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f'Sample {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_predictions_2d(
    x: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    n_samples: int = 3,
    save_path: Optional[str] = None,
    title: str = "Model Predictions"
) -> None:
    """
    Plot 2D predictions vs ground truth.
    
    Args:
        x: Input data
        y_true: Ground truth
        y_pred: Model predictions
        n_samples: Number of samples to plot
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    n_samples = min(n_samples, len(y_true))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Handle different input shapes
        if len(x.shape) == 4 and x.shape[1] > 1:  # Multi-timestep input
            input_data = x[i, -1]  # Show last timestep
        else:
            input_data = x[i].squeeze()
        
        # Input
        im1 = axes[i, 0].imshow(input_data, cmap='viridis')
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # Ground truth
        im2 = axes[i, 1].imshow(y_true[i].squeeze(), cmap='viridis')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # Prediction
        im3 = axes[i, 2].imshow(y_pred[i].squeeze(), cmap='viridis')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_predictions(
    x: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    x_coords: Optional[torch.Tensor] = None,
    n_samples: int = 5,
    save_path: Optional[str] = None,
    title: str = "Model Predictions"
) -> None:
    """
    Plot predictions vs ground truth (automatically detects 1D vs 2D).
    
    Args:
        x: Input data
        y_true: Ground truth
        y_pred: Model predictions
        x_coords: Spatial coordinates for 1D (optional)
        n_samples: Number of samples to plot
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    # Determine if data is 1D or 2D
    if len(y_true.shape) == 2:  # 1D case: [batch, spatial]
        plot_predictions_1d(x, y_true, y_pred, x_coords, n_samples, save_path, title)
    elif len(y_true.shape) == 3:  # 2D case: [batch, height, width]
        plot_predictions_2d(x, y_true, y_pred, n_samples, save_path, title)
    else:
        raise ValueError(f"Unsupported data shape: {y_true.shape}")


def plot_rollout_comparison(
    model_results: Dict[str, Dict],
    rollout_steps: List[int],
    save_path: Optional[str] = None,
    title: str = "Rollout Comparison"
) -> None:
    """
    Plot rollout performance comparison across models.
    
    Args:
        model_results: Dictionary of {model_name: {rollout_step: error}}
        rollout_steps: List of rollout steps
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, results in model_results.items():
        errors = [results.get(step, np.nan) for step in rollout_steps]
        plt.plot(rollout_steps, errors, 'o-', label=model_name, linewidth=2, markersize=6)
    
    plt.xlabel('Rollout Steps')
    plt.ylabel('L2 Error')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rollout comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['test_loss', 'mae', 'relative_l2'],
    save_path: Optional[str] = None,
    title: str = "Model Comparison"
) -> None:
    """
    Plot comparison of different models across multiple metrics.
    
    Args:
        results: Dictionary of {model_name: {metric: value}}
        metrics: List of metrics to plot
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    model_names = list(results.keys())
    
    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, np.nan) for model in model_names]
        
        bars = axes[i].bar(model_names, values)
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.4f}', ha='center', va='bottom')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_rollout_gif_1d(
    data: torch.Tensor,
    x_coords: Optional[torch.Tensor] = None,
    save_path: str = "rollout.gif",
    fps: int = 10,
    title: str = "Rollout Evolution"
) -> None:
    """
    Create a GIF showing the evolution of a 1D rollout.
    
    Args:
        data: Time series data of shape (n_time, spatial)
        x_coords: Spatial coordinates (optional)
        save_path: Path to save the GIF
        fps: Frames per second
        title: Plot title
    """
    if x_coords is None:
        x_coords = torch.arange(data.shape[-1])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up the plot
    y_min, y_max = data.min().item(), data.max().item()
    margin = (y_max - y_min) * 0.1
    
    line, = ax.plot([], [], 'b-', linewidth=2)
    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.grid(True, alpha=0.3)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    def animate(frame):
        line.set_data(x_coords, data[frame])
        time_text.set_text(f'Time step: {frame}')
        return line, time_text
    
    anim = animation.FuncAnimation(fig, animate, frames=len(data), 
                                 interval=1000//fps, blit=True, repeat=True)
    
    plt.title(title)
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    print(f"Rollout GIF saved to {save_path}")


def create_rollout_gif_2d(
    data: torch.Tensor,
    save_path: str = "rollout.gif",
    fps: int = 10,
    title: str = "Rollout Evolution"
) -> None:
    """
    Create a GIF showing the evolution of a 2D rollout.
    
    Args:
        data: Time series data of shape (n_time, height, width)
        save_path: Path to save the GIF
        fps: Frames per second
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set up the plot
    vmin, vmax = data.min().item(), data.max().item()
    im = ax.imshow(data[0], cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                       fontsize=12, color='white', weight='bold')
    
    def animate(frame):
        im.set_array(data[frame])
        time_text.set_text(f'Time step: {frame}')
        return [im, time_text]
    
    anim = animation.FuncAnimation(fig, animate, frames=len(data), 
                                 interval=1000//fps, blit=True, repeat=True)
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    print(f"Rollout GIF saved to {save_path}")


def plot_multiseed_training(
    multiseed_results: Dict,
    save_path: Optional[str] = None,
    title: str = "Multi-seed Training Progress"
) -> None:
    """
    Plot training curves with mean and variance from multi-seed results.
    
    Args:
        multiseed_results: Results from train_model_multiseed
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    set_plot_style()
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Extract training curves
    train_losses = multiseed_results['train_losses']
    val_losses = multiseed_results['val_losses']
    
    # Convert to numpy arrays for easier computation
    max_epochs = min(len(losses) for losses in train_losses)
    train_array = np.array([losses[:max_epochs] for losses in train_losses])
    val_array = np.array([losses[:max_epochs] for losses in val_losses])
    
    epochs = np.arange(1, max_epochs + 1)
    
    # Compute mean and std
    train_mean = np.mean(train_array, axis=0)
    train_std = np.std(train_array, axis=0)
    val_mean = np.mean(val_array, axis=0)
    val_std = np.std(val_array, axis=0)
    
    # Plot training losses
    ax1.plot(epochs, train_mean, 'b-', label='Training Loss', linewidth=2)
    ax1.fill_between(epochs, train_mean - train_std, train_mean + train_std, 
                     alpha=0.3, color='blue')
    ax1.plot(epochs, val_mean, 'r-', label='Validation Loss', linewidth=2)
    ax1.fill_between(epochs, val_mean - val_std, val_mean + val_std, 
                     alpha=0.3, color='red')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-seed training plot saved to {save_path}")
    
    plt.close()


def plot_multiseed_comparison(
    model_results: Dict[str, Dict],
    metrics: List[str] = ['mean_test_loss', 'mean_mae', 'mean_relative_l2', 'mean_rollout_loss'],
    save_path: Optional[str] = None,
    title: str = "Multi-seed Model Comparison"
) -> None:
    """
    Plot comparison of multiple models with error bars from multi-seed results.
    
    Args:
        model_results: Dict mapping model names to multi-seed results
        metrics: List of metrics to plot
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    set_plot_style()
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    model_names = list(model_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        x_pos = np.arange(len(model_names))
        means = []
        stds = []
        
        for model_name in model_names:
            results = model_results[model_name]
            if metric in results:
                means.append(results[metric])
                std_key = metric.replace('mean_', 'std_')
                stds.append(results.get(std_key, 0))
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                     color=colors[:len(model_names)], alpha=0.7)
        
        # Add individual data points if available
        for j, model_name in enumerate(model_names):
            results = model_results[model_name]
            if 'test_results' in results:
                raw_metric = metric.replace('mean_', '')
                values = [res[raw_metric] for res in results['test_results'] 
                         if raw_metric in res]
                if values:
                    ax.scatter([j] * len(values), values, color='black', 
                             alpha=0.6, s=20, zorder=10)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('mean_', '').replace('_', ' ').title())
        ax.set_title(metric.replace('mean_', '').replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-seed comparison plot saved to {save_path}")
    
    plt.close()


def plot_all_models_training(
    model_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Training Progress Comparison"
) -> None:
    """
    Plot training and validation curves for all models in one figure.
    Left panel shows training losses, right panel shows validation losses.
    
    Args:
        model_results: Dict mapping model names to multi-seed results containing 'train_losses' and 'val_losses'
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define colors for consistency
    colors = {'LOCO': '#1f77b4', 'FNO': '#ff7f0e', 'Hybrid': '#2ca02c'}
    
    for model_name, results in model_results.items():
        # Extract training curves
        train_losses = results['train_losses']
        val_losses = results['val_losses']
        
        # Convert to numpy arrays for easier computation
        max_epochs = min(len(losses) for losses in train_losses)
        train_array = np.array([losses[:max_epochs] for losses in train_losses])
        val_array = np.array([losses[:max_epochs] for losses in val_losses])
        
        epochs = np.arange(1, max_epochs + 1)
        
        # Compute mean and std
        train_mean = np.mean(train_array, axis=0)
        train_std = np.std(train_array, axis=0)
        val_mean = np.mean(val_array, axis=0)
        val_std = np.std(val_array, axis=0)
        
        color = colors.get(model_name, plt.cm.tab10(list(model_results.keys()).index(model_name)))
        
        # Plot training losses (left panel)
        ax1.plot(epochs, train_mean, '-', color=color, label=model_name, linewidth=2)
        ax1.fill_between(epochs, train_mean - train_std, train_mean + train_std, 
                         alpha=0.2, color=color)
        
        # Plot validation losses (right panel)
        ax2.plot(epochs, val_mean, '-', color=color, label=model_name, linewidth=2)
        ax2.fill_between(epochs, val_mean - val_std, val_mean + val_std, 
                         alpha=0.2, color=color)
    
    # Configure left panel (Training)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Configure right panel (Validation)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training comparison plot saved to {save_path}")
    
    plt.close()


def plot_spacetime(
    data: torch.Tensor,
    x_coords: Optional[torch.Tensor] = None,
    t_coords: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    title: str = "Space-Time Evolution",
    equation: str = "PDE"
) -> None:
    """
    Create a space-time plot showing the evolution of a 1D solution.
    
    Args:
        data: Time series data of shape (n_time, n_spatial)
        x_coords: Spatial coordinates (optional)
        t_coords: Time coordinates (optional)
        save_path: Path to save the plot (optional)
        title: Plot title
        equation: Equation name for axis labels
    """
    set_plot_style()
    
    n_time, n_spatial = data.shape
    
    if x_coords is None:
        x_coords = torch.linspace(0, 2*np.pi, n_spatial)
    if t_coords is None:
        t_coords = torch.linspace(0, 1, n_time)
    
    # Create meshgrid for plotting (transposed: time on x-axis, space on y-axis)
    T, X = np.meshgrid(t_coords.numpy(), x_coords.numpy())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create contour plot (transpose data to match new axis orientation)
    data_transposed = data.numpy().T
    
    vmin, vmax = data_transposed.min(), data_transposed.max()
    cmap = 'viridis'
    
    levels = 25
    contour = ax.contourf(T, X, data_transposed, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    contour_lines = ax.contour(T, X, data_transposed, levels=levels, colors='black', alpha=0.2, linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(f'u(x,t)', rotation=270, labelpad=15)
    
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Space (x)')
    ax.set_title(title)
    
    # Add some contour labels for clarity
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Space-time plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_rollout_frames_2d(
    data: torch.Tensor,
    save_dir: str,
    title_prefix: str = "Rollout",
    fps: int = 10
) -> None:
    """
    Create individual frame images from 2D rollout data.
    
    Args:
        data: Time series data of shape (n_time, height, width)
        save_dir: Directory to save frame images
        title_prefix: Prefix for frame titles
        fps: Frames per second (used for filename ordering)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_time = data.shape[0]
    vmin, vmax = data.min().item(), data.max().item()
    
    print(f"Creating {n_time} frames in {save_dir}")
    
    for frame in range(n_time):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        im = ax.imshow(data[frame].numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
        
        
        # Save frame
        frame_path = os.path.join(save_dir, f'frame_{frame:03d}.png')
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Frames saved to {save_dir}")
    print(f"To create GIF: convert -delay {100//fps} {save_dir}/frame_*.png {save_dir}/rollout.gif")


def _select_interesting_burgers_sample(data: torch.Tensor) -> int:
    """
    Auto-select most interesting Burgers sample based on variance and peak count.
    
    Args:
        data: Burgers data tensor of shape (n_samples, n_time, n_spatial)
        
    Returns:
        Index of most interesting sample
    """
    try:
        n_samples = data.shape[0]
        best_score = -1
        best_idx = 0
        
        # Look at later half of samples (typically more interesting)
        start_idx = max(0, n_samples // 2)
        
        for i in range(start_idx, n_samples):
            u_initial = data[i, 0, :]  # Initial condition
            
            # Calculate variance (measure of complexity)
            variance = u_initial.var().item()
            
            # Count peaks (measure of structure)
            n_peaks = ((u_initial[1:-1] > u_initial[:-2]) & 
                      (u_initial[1:-1] > u_initial[2:])).sum().item()
            
            # Combined score: higher variance + more peaks = more interesting
            score = variance + 0.1 * n_peaks
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        print(f"Auto-selected sample {best_idx} (variance={data[best_idx, 0, :].var():.4f}, "
              f"peaks={((data[best_idx, 0, 1:-1] > data[best_idx, 0, :-2]) & (data[best_idx, 0, 1:-1] > data[best_idx, 0, 2:])).sum()}, "
              f"score={best_score:.4f})")
        
        return best_idx
        
    except Exception as e:
        print(f"Warning: Failed to auto-select sample ({e}), using fallback sample 100")
        return min(100, data.shape[0] - 1)  # Fallback to sample 100 or last available


def demonstrate_rollout(
    equation: str,
    model_path: Optional[str] = None,
    data_path: str = "data",
    save_dir: str = "rollouts",
    n_steps: int = 400,
    sample_idx: Optional[int] = None
) -> None:
    """
    Create rollout demonstration for a specific equation.
    
    Args:
        equation: Type of equation ('burgers', 'kdv', 'ns2d')
        model_path: Path to trained model (if None, uses ground truth)
        data_path: Path to data
        save_dir: Directory to save visualizations
        n_steps: Number of rollout steps
        sample_idx: Index of sample to use (if None, auto-selects most interesting for Burgers)
    """
    from .data import generate_data
    
    equation = equation.lower()
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Creating rollout demonstration for {equation.upper()}")
    
    # Load raw data to get time coordinates
    raw_data = None
    if equation in ['burgers', 'kdv']:
        raw_data_path = os.path.join(data_path, f'{equation}_test.pt')
        if os.path.exists(raw_data_path):
            raw_data = torch.load(raw_data_path)
    
    # Generate or load data using updated configurations
    if equation == 'burgers':
        _, test_dataset = generate_data(
            equation='burgers',
            n_train=1, n_test=20,
            N=512, T=2.5, dt=1e-4, time_gap=10,  # Updated to new config
            nu=0.01,  # Add missing Burgers parameter
            data_path=data_path
        )
    elif equation == 'kdv':
        _, test_dataset = generate_data(
            equation='kdv', 
            n_train=1, n_test=5,
            N=128, T=4.0, dt=0.001, time_gap=10,  # Smaller dt for stability
            a=6.0, b=1.0, use_solitons=True,  # Add missing KdV parameters
            data_path=data_path
        )
    elif equation == 'ns2d':
        _, test_dataset = generate_data(
            equation='ns2d',
            n_train=1, n_test=5,
            N=64, T=110.0, dt=1e-4, time_gap=1000,  # Longer T to support time_gap=1000
            visc=1e-3,  # Add missing NS2D parameters
            data_path=data_path
        )
    else:
        raise ValueError(f"Unknown equation: {equation}")
    
    # Get a sample for rollout - use provided index or auto-select for Burgers
    auto_select_burgers = (sample_idx is None and equation == 'burgers')
    if sample_idx is None:
        sample_idx = 0  # Default, will be updated for Burgers if auto_select_burgers is True
    
    if hasattr(test_dataset.data, 'shape') and len(test_dataset.data.shape) >= 3:
        # Auto-select interesting sample for Burgers if needed
        if auto_select_burgers:
            sample_idx = _select_interesting_burgers_sample(test_dataset.data)
        
        # Get full time series for rollout
        if equation in ['burgers', 'kdv']:
            # 1D equations: shape (n_samples, n_time, n_spatial)
            rollout_data = test_dataset.data[sample_idx, :n_steps]  # (n_time, n_spatial)
            
            # Get real time and space coordinates
            t_coords = None
            x_coords = None
            if equation in ['burgers', 'kdv'] and raw_data is not None:
                t_coords = raw_data['t'][:n_steps]
                x_coords = raw_data['x']
            elif hasattr(test_dataset, 't'):
                t_coords = test_dataset.t[:n_steps]
            elif hasattr(test_dataset, 'data_dict') and 't' in test_dataset.data_dict:
                t_coords = test_dataset.data_dict['t'][:n_steps]
            
            if equation in ['burgers', 'kdv'] and raw_data is not None and x_coords is None:
                x_coords = raw_data['x']
            elif hasattr(test_dataset, 'x'):
                x_coords = test_dataset.x
            elif hasattr(test_dataset, 'data_dict') and 'x' in test_dataset.data_dict:
                x_coords = test_dataset.data_dict['x']
            
            # Create space-time plot
            save_path = os.path.join(save_dir, f"{equation}_spacetime_rollout.png")
            plot_spacetime(
                rollout_data,
                x_coords=x_coords,
                t_coords=t_coords,
                save_path=save_path,
                title=f"{equation.upper()} Space-Time Evolution",
                equation=equation
            )
            
        elif equation == 'ns2d':
            # 2D equation: shape (n_samples, n_time, height, width)
            rollout_data = test_dataset.data[sample_idx, :n_steps]  # (n_time, H, W)
            
            # Create frame sequence
            frames_dir = os.path.join(save_dir, f"{equation}_frames")
            create_rollout_frames_2d(
                rollout_data,
                frames_dir,
                title_prefix=f"{equation.upper()} Rollout"
            )
    
    print(f"Rollout visualization saved to {save_dir}")


def set_plot_style():
    """Set a nice default plot style."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['grid.alpha'] = 0.3