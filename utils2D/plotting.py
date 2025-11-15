"""
2D Plotting Utilities for Neural Operators

This module provides plotting utilities for 2D PDE experiments.

Functions:
- plot_training_losses: Plot training/validation losses with smoothing
- load_and_plot_losses: Load saved loss data and create plots
- aggregate_and_plot_losses: Aggregate individual model losses and plot
- plot_2d_field: Plot 2D scalar field with colorbar
- create_ns2d_comparison: Create NS2D model comparison plots
- plot_rollout_comparison: Plot rollout performance comparison
"""

import glob
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch


def _rolling_mean_std(values, window=20):
    """
    Centered rolling mean/std (unbiased=False). For edges, uses the available
    neighborhood. Returns two Python lists.


    """
    if not isinstance(values, list | tuple):
        values = list(values)
    n = len(values)
    if n == 0:
        return [], []
    t = torch.tensor(values, dtype=torch.float32)
    means = torch.empty(n, dtype=torch.float32)
    stds = torch.empty(n, dtype=torch.float32)
    half = window // 2
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        seg = t[s:e]
        means[i] = seg.mean()
        # use population std to keep shading tighter; set to small epsilon if 0
        std = seg.std(unbiased=False)
        stds[i] = std if std > 0 else 1e-12
    return means.tolist(), stds.tolist()


def plot_training_losses(all_losses, save_dir='plots'):
    """
    Plot training and validation losses for all models with transparency and rolling stats.



    Args:
        all_losses: Dictionary of model losses
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Appearance controls
    raw_alpha = 0.05          # very faint raw curve
    mean_alpha = 0.9          # popping smoothed curve
    shade_alpha = 0.07        # faint variance shading
    linewidth_raw = 0.6
    linewidth_mean = 1.6
    window = 20
    eps = 1e-12
    y_min, y_max = 1e-3, 1e-1

    plt.figure(figsize=(16, 10))

    # Deterministic bright colors per model (use tab10 cycle for "pop")
    model_names_order = list(all_losses.keys())
    tab10_colors = list(plt.get_cmap('tab10').colors)
    color_map = {name: tab10_colors[i % len(tab10_colors)] for i, name in enumerate(model_names_order)}

    # Plot 1: Training losses
    plt.subplot(2, 2, 1)
    for model_name, losses in all_losses.items():
        tr = losses.get('train_losses', losses.get('train', []))
        if len(tr) == 0:
            continue
        m, s = _rolling_mean_std(tr, window=window)
        x = range(len(tr))
        color = color_map[model_name]
        plt.plot(x, tr, alpha=raw_alpha, linewidth=linewidth_raw, color=color)
        # Convert to percentile-based approach (approximating p10-p90 from std)
        lower = np.maximum((torch.tensor(m) - 1.28 * torch.tensor(s)), eps).tolist()
        upper = (torch.tensor(m) + 1.28 * torch.tensor(s)).tolist()

        plt.plot(x, m, label=f'{model_name}', linewidth=linewidth_mean, alpha=mean_alpha, color=color, zorder=3)
        plt.fill_between(x, lower, upper, alpha=shade_alpha, linewidth=0, facecolor=color, zorder=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss (20-epoch mean + 10th-90th percentile)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.ylim(y_min, y_max)
    plt.legend(framealpha=0.2, fontsize='small')

    # Plot 2: Validation losses
    plt.subplot(2, 2, 2)
    for model_name, losses in all_losses.items():
        vl = losses.get('val_losses', losses.get('val', []))
        if len(vl) == 0:
            continue
        m, s = _rolling_mean_std(vl, window=window)
        x = range(len(vl))
        color = color_map[model_name]
        plt.plot(x, vl, alpha=raw_alpha, linewidth=linewidth_raw, color=color)
        # Convert to percentile-based approach
        lower = np.maximum((torch.tensor(m) - 1.28 * torch.tensor(s)), eps).tolist()
        upper = (torch.tensor(m) + 1.28 * torch.tensor(s)).tolist()

        plt.plot(x, m, label=f'{model_name}', linewidth=linewidth_mean, alpha=mean_alpha, color=color, zorder=3)
        plt.fill_between(x, lower, upper, alpha=shade_alpha, linewidth=0, facecolor=color, zorder=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss (20-epoch mean + 10th-90th percentile)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.ylim(y_min, y_max)
    plt.legend(framealpha=0.2, fontsize='small')

    # Plot 3: Combined
    plt.subplot(2, 2, 3)
    for model_name, losses in all_losses.items():
        tr = losses.get('train_losses', losses.get('train', []))
        vl = losses.get('val_losses', losses.get('val', []))
        if len(tr) > 0:
            color = color_map[model_name]
            plt.plot(range(len(tr)), tr, '--', alpha=raw_alpha, linewidth=linewidth_raw, label=f'{model_name} (train raw)', color=color)
        if len(vl) > 0:
            m, s = _rolling_mean_std(vl, window=window)
            x = range(len(vl))
            color = color_map[model_name]
            plt.plot(x, m, linewidth=linewidth_mean, alpha=mean_alpha, label=f'{model_name} (val mean)', color=color, zorder=3)
            lower = (torch.tensor(m) - torch.tensor(s)).clamp(min=eps).tolist()
            upper = (torch.tensor(m) + torch.tensor(s)).tolist()
            plt.fill_between(x, lower, upper, alpha=shade_alpha, linewidth=0, facecolor=color, zorder=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training (raw) vs Validation (mean ± std)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.ylim(y_min, y_max)
    plt.legend(ncol=2, framealpha=0.2, fontsize='small')

    # Plot 4: Best validation losses
    plt.subplot(2, 2, 4)
    model_names = list(all_losses.keys())
    best_losses = [all_losses[name].get('best_val_loss', min(all_losses[name].get('val_losses', all_losses[name].get('val', [1.0])))) for name in model_names]
    bars = plt.bar(model_names, best_losses, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'][:len(model_names)])
    plt.xlabel('Model')
    plt.ylabel('Best Validation Loss')
    plt.title('Best Validation Loss by Model')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    for bar, loss in zip(bars, best_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(best_losses)*0.01, f'{loss:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_losses.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Loss plots saved: {plot_path}")
    plt.close()

    # Individual plots per model
    for model_name, losses in all_losses.items():
        tr = losses.get('train_losses', losses.get('train', []))
        vl = losses.get('val_losses', losses.get('val', []))
        plt.figure(figsize=(10, 6))
        if len(tr) > 0:
            mt, st = _rolling_mean_std(tr, window=window)
            x = range(len(tr))
            color = color_map[model_name]
            plt.plot(x, tr, label='Training (raw)', alpha=raw_alpha, linewidth=linewidth_raw, color=color)
            plt.plot(x, mt, label='Training (mean)', linewidth=linewidth_mean, alpha=mean_alpha, color=color, zorder=3)
            lower = (torch.tensor(mt) - torch.tensor(st)).clamp(min=eps).tolist()
            upper = (torch.tensor(mt) + torch.tensor(st)).tolist()
            plt.fill_between(x, lower, upper, alpha=shade_alpha, linewidth=0, facecolor=color, zorder=2)
        if len(vl) > 0:
            mv, sv = _rolling_mean_std(vl, window=window)
            x = range(len(vl))
            color = color_map[model_name]
            plt.plot(x, vl, label='Validation (raw)', alpha=raw_alpha, linewidth=linewidth_raw, color=color)
            plt.plot(x, mv, label='Validation (mean)', linewidth=linewidth_mean, alpha=mean_alpha, color=color, zorder=3)
            lower = (torch.tensor(mv) - torch.tensor(sv)).clamp(min=eps).tolist()
            upper = (torch.tensor(mv) + torch.tensor(sv)).tolist()
            plt.fill_between(x, lower, upper, alpha=shade_alpha, linewidth=0, facecolor=color, zorder=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} - Training Progress (20-epoch mean ± std)')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.ylim(y_min, y_max)
        plt.legend(framealpha=0.2, fontsize='small')
        individual_plot_path = os.path.join(save_dir, f'{model_name}_losses.png')
        plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Individual loss plots saved in: {save_dir}")


def load_and_plot_losses(checkpoint_dir, plots_dir='plots'):
    """
    Load saved loss data and create plots.



    Args:
        checkpoint_dir: Directory containing loss files
        plots_dir: Directory to save plots

    Returns:
        Dictionary of loaded losses or None
    """
    combined_loss_path = os.path.join(checkpoint_dir, 'all_losses.pt')
    if os.path.exists(combined_loss_path):
        all_losses = torch.load(combined_loss_path, map_location='cpu')
        plot_training_losses(all_losses, save_dir=plots_dir)
        return all_losses
    else:
        print(f"No combined loss data found at {combined_loss_path}")
        return None


def aggregate_and_plot_losses(checkpoint_dir, plots_dir='plots'):
    """
    Finds all individual model loss files, combines them, plots them,
    and saves the combined data.



    Args:
        checkpoint_dir: Directory containing individual loss files
        plots_dir: Directory to save plots
    """
    all_losses = {}
    loss_files = glob.glob(os.path.join(checkpoint_dir, '*_losses.pt'))

    if not loss_files:
        print("No individual loss files found to aggregate.")
        return

    print(f"Found {len(loss_files)} loss files. Aggregating...")
    for f_path in loss_files:
        # Extract model name from filename, e.g., 'LOCO_losses.pt' -> 'LOCO'
        model_name = os.path.basename(f_path).replace('_losses.pt', '')
        if model_name == 'all':
            continue # Skip the old combined file

        try:
            loss_data = torch.load(f_path, map_location='cpu')
            all_losses[model_name] = loss_data
            print(f"  - Loaded data for {model_name}")
        except Exception as e:
            print(f"  - Failed to load {f_path}: {e}")

    if not all_losses:
        print("No valid model loss data found to plot.")
        return

    # Plot all losses using the existing function
    plot_training_losses(all_losses, save_dir=plots_dir)

    # Save the newly combined loss data
    combined_loss_path = os.path.join(checkpoint_dir, 'all_losses.pt')
    torch.save(all_losses, combined_loss_path)
    print(f"Combined loss data for {len(all_losses)} models saved: {combined_loss_path}")


def plot_2d_field(field, title="2D Field", figsize=(8, 6), cmap='viridis', save_path=None):
    """
    Plot a 2D scalar field with colorbar

    Args:
        field: 2D numpy array or torch tensor
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure
    """
    if torch.is_tensor(field):
        field = field.detach().cpu().numpy()

    plt.figure(figsize=figsize)
    im = plt.imshow(field, cmap=cmap, origin='lower', aspect='equal')
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def create_ns2d_comparison(models_dict, test_dataset, device, num_samples=3, save_dir='plots'):
    """
    Create NS2D model comparison plots showing ground truth vs predictions

    Args:
        models_dict: Dictionary of trained models
        test_dataset: Test dataset
        device: Computing device
        num_samples: Number of samples to compare
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_samples):
        # Get random test sample
        idx = np.random.randint(0, len(test_dataset))
        sample = test_dataset[idx]
        x = sample['x'].unsqueeze(0).to(device)  # Add batch dimension
        y_true = sample['y'].squeeze().cpu().numpy()  # Remove channel dim

        # Get predictions from all models
        predictions = {}
        for model_name, model_info in models_dict.items():
            model = model_info['model']
            model_type = model_info['type']
            model.eval()

            with torch.no_grad():
                # Handle tensor format for different models
                if model_type in ['loco', 'hybrid', 'fno']:
                    x_model = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                else:
                    x_model = x

                pred = model(x_model)

                # Convert back to [H, W] format
                if model_type in ['loco', 'hybrid', 'fno']:
                    pred = pred.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

                predictions[model_name] = pred.squeeze().cpu().numpy()

        # Create comparison plot
        num_models = len(models_dict) + 1  # +1 for ground truth
        cols = min(4, num_models)
        rows = int(np.ceil(num_models / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Plot ground truth
        vmin, vmax = y_true.min(), y_true.max()
        im = axes[0].imshow(y_true, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        # Plot predictions
        for idx, (model_name, pred) in enumerate(predictions.items()):
            ax_idx = idx + 1
            axes[ax_idx].imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            axes[ax_idx].set_title(f'{model_name} Prediction')
            axes[ax_idx].axis('off')

        # Hide unused subplots
        for idx in range(num_models, len(axes)):
            axes[idx].axis('off')

        # Add colorbar
        fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        plt.suptitle(f'NS2D Model Comparison - Sample {i+1}', fontsize=14)

        save_path = os.path.join(save_dir, f'ns2d_comparison_sample_{i+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"NS2D comparison plots saved in: {save_dir}")


def plot_rollout_comparison(models_dict, test_dataset, device, num_steps=10, save_dir='plots'):
    """
    Plot rollout performance comparison for 2D models

    Args:
        models_dict: Dictionary of trained models
        test_dataset: Test dataset
        device: Computing device
        num_steps: Number of rollout steps
        save_dir: Directory to save plots
    """
    from utils2D.training import perform_n_step_rollout

    os.makedirs(save_dir, exist_ok=True)

    # Get random test sample
    idx = np.random.randint(0, len(test_dataset))
    sample = test_dataset[idx]
    x0 = sample['x'].unsqueeze(0).to(device)  # Add batch dimension

    # Generate rollouts for each model
    rollouts = {}
    for model_name, model_info in models_dict.items():
        model = model_info['model']
        model_type = model_info['type']
        model.eval()

        # Handle tensor format
        if model_type in ['loco', 'hybrid', 'fno']:
            x_model = x0.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        else:
            x_model = x0

        predictions = perform_n_step_rollout(model, x_model, num_steps, model_type)

        # Convert predictions back to [H, W] format
        rollout_frames = []
        for pred in predictions:
            if model_type in ['loco', 'hybrid', 'fno']:
                pred = pred.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
            rollout_frames.append(pred.squeeze().cpu().numpy())

        rollouts[model_name] = rollout_frames

    # Create rollout comparison plot
    fig, axes = plt.subplots(len(models_dict), num_steps, figsize=(2*num_steps, 2*len(models_dict)))
    if len(models_dict) == 1:
        axes = axes.reshape(1, -1)

    for model_idx, (model_name, frames) in enumerate(rollouts.items()):
        for step_idx, frame in enumerate(frames):
            ax = axes[model_idx, step_idx]
            ax.imshow(frame, cmap='viridis', origin='lower')
            ax.set_title(f'{model_name} Step {step_idx+1}' if model_idx == 0 else f'Step {step_idx+1}')
            ax.axis('off')

            if step_idx == 0:
                ax.text(-0.1, 0.5, model_name, rotation=90, va='center', ha='right',
                       transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'rollout_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Rollout comparison plot saved: {save_path}")


def create_animated_comparison(models_dict, test_dataset, device, num_steps=20, fps=2, save_dir='plots'):
    """
    Create animated GIF comparing model rollouts

    Args:
        models_dict: Dictionary of trained models
        test_dataset: Test dataset
        device: Computing device
        num_steps: Number of rollout steps
        fps: Frames per second for animation
        save_dir: Directory to save animation
    """
    from utils2D.training import perform_n_step_rollout

    os.makedirs(save_dir, exist_ok=True)

    # Get random test sample
    idx = np.random.randint(0, len(test_dataset))
    sample = test_dataset[idx]
    x0 = sample['x'].unsqueeze(0).to(device)

    # Generate rollouts for each model
    rollouts = {}
    for model_name, model_info in models_dict.items():
        model = model_info['model']
        model_type = model_info['type']
        model.eval()

        if model_type in ['loco', 'hybrid', 'fno']:
            x_model = x0.permute(0, 3, 1, 2)
        else:
            x_model = x0

        predictions = perform_n_step_rollout(model, x_model, num_steps, model_type)

        rollout_frames = []
        for pred in predictions:
            if model_type in ['loco', 'hybrid', 'fno']:
                pred = pred.permute(0, 2, 3, 1)
            rollout_frames.append(pred.squeeze().cpu().numpy())

        rollouts[model_name] = rollout_frames

    # Create animation
    fig, axes = plt.subplots(1, len(models_dict), figsize=(4*len(models_dict), 4))
    if len(models_dict) == 1:
        axes = [axes]

    def animate(frame):
        for ax in axes:
            ax.clear()

        for model_idx, (model_name, frames) in enumerate(rollouts.items()):
            ax = axes[model_idx]
            ax.imshow(frames[frame], cmap='viridis', origin='lower')
            ax.set_title(f'{model_name} - Step {frame+1}')
            ax.axis('off')

        return axes

    ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=1000//fps, blit=False)

    save_path = os.path.join(save_dir, 'rollout_animation.gif')
    ani.save(save_path, writer='pillow', fps=fps)
    plt.close()

    print(f"Rollout animation saved: {save_path}")
