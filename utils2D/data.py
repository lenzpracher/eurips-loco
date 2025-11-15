"""
2D Dataset Classes for Neural Operators

This module provides dataset classes for 2D PDE problems.


Classes:
- NS2DDataset: Dataset for 2D Navier-Stokes equation
"""

import os

import torch
from torch.utils.data import Dataset


class NS2DDataset(Dataset):
    """
    Dataset for 2D Navier-Stokes time-series data.



    Data format:
    - Time steps: 0.1, 0.2, 0.3, ..., 100.0 (1000 total steps)
    - Input: Last 10 time steps (1.0 time units)
    - Output: Next 1 time step (0.1 time units ahead)
    - Spatial resolution: 256x256

    Args:
        data_dir: Directory containing NS2D data
        split: 'train' or 'test' split
        n_input_timesteps: Number of input timesteps (default: 10)
        n_output_timesteps: Number of output timesteps (default: 1)
        train_split_ratio: Train/test split ratio (unused if data_tensor provided)
        data_tensor: Pre-loaded data tensor
        target_resolution: Target spatial resolution for downsampling
        max_rollout_steps: Maximum rollout steps for rollout loss
        subset_mode: Whether to use subset of data
        subset_size: Size of subset if subset_mode=True
        enable_augmentation: Whether to enable data augmentation
        augmentation_noise_levels: List of noise levels for augmentation
        augmentation_probability: Probability of applying augmentation
    """
    def __init__(self, data_dir, split='train',
                 n_input_timesteps=10,
                 n_output_timesteps=1,
                 train_split_ratio=0.8, data_tensor=None,
                 target_resolution=None, max_rollout_steps=10,
                 subset_mode=False, subset_size=None,
                 enable_augmentation=False, augmentation_noise_levels=None,
                 augmentation_probability=0.5):
        if augmentation_noise_levels is None:
            augmentation_noise_levels = [0.01, 0.03]
        self.data_dir = data_dir
        self.n_input_timesteps = n_input_timesteps
        self.n_output_timesteps = n_output_timesteps
        self.total_timesteps = n_input_timesteps + n_output_timesteps
        self.max_rollout_steps = max_rollout_steps

        # Augmentation parameters
        self.enable_augmentation = enable_augmentation and split == 'train'  # Only augment training data
        self.augmentation_noise_levels = augmentation_noise_levels
        self.augmentation_probability = augmentation_probability

        if self.enable_augmentation:
            self.augmentation_generator = torch.Generator()
            self.augmentation_generator.manual_seed(42)  # Fixed seed for reproducibility

        data_dict = {}
        if data_tensor is not None:
            print(f"[{os.getpid()}] Loading data from pre-loaded tensor for split: {split}")
            self.data = data_tensor
        else:
            # Try new filename format first (with T=100), then fall back to old format
            if split == 'train':
                file_path = os.path.join(data_dir, "nsforcing_train_T100_256.pt")
                if not os.path.exists(file_path):
                    file_path = os.path.join(data_dir, "nsforcing_train_256.pt")
            else:
                file_path = os.path.join(data_dir, "nsforcing_test_T100_256.pt")
                if not os.path.exists(file_path):
                    file_path = os.path.join(data_dir, "nsforcing_test_256.pt")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")

            print(f"Loading data from: {file_path}")
            data_dict = torch.load(file_path, map_location='cpu')

            # Use 'u' tensor which has shape [N, T, H, W]
            if 'u' in data_dict:
                self.data = data_dict['u']
            else:
                raise ValueError("Could not find 'u' tensor in the dataset file.")

        # Downsample if needed to match FNO paper setup
        if target_resolution and self.data.shape[-1] != target_resolution:
            print(f"Downsampling data from {self.data.shape[-1]}x{self.data.shape[-2]} to {target_resolution}x{target_resolution}")
            # data is [N, T, H, W]
            self.data = torch.nn.functional.interpolate(
                self.data,
                size=(target_resolution, target_resolution),
                mode='bicubic',
                align_corners=False
            )
            print(f"New data shape: {self.data.shape}")

        self.num_sims, self.T, self.H, self.W = self.data.shape
        self.num_samples_per_sim = self.T - self.total_timesteps + 1

        self.subset_mode = subset_mode
        if self.subset_mode:
            if subset_size is None:
                raise ValueError("subset_size must be provided when subset_mode is True")

            # Create a fixed list of all possible (sim_idx, start_t) pairs
            all_possible_indices = []
            for i in range(self.num_sims):
                for j in range(self.num_samples_per_sim):
                    all_possible_indices.append((i, j))

            # Create a reproducible random subset of these pairs
            import numpy as np
            rng = np.random.default_rng(seed=42) # Use a fixed seed

            total_available_samples = len(all_possible_indices)
            if subset_size > total_available_samples:
                print(f"Warning: subset_size {subset_size} is larger than total available samples {total_available_samples}. Using all samples.")
                subset_size = total_available_samples

            chosen_indices = rng.choice(np.arange(total_available_samples), size=subset_size, replace=False)
            self.subset_indices = [all_possible_indices[i] for i in chosen_indices]

        # Print dataset information
        print(f"Dataset info - Split: {split}")
        print(f"  Data shape: {self.data.shape}")
        print(f"  Number of simulations: {self.num_sims}")
        print(f"  Time steps per simulation: {self.T}")
        print(f"  Spatial resolution: {self.H}x{self.W}")
        if self.subset_mode:
            print(f"  Mode: SUBSET ({len(self.subset_indices)} samples)")
        else:
            print(f"  Mode: FULL ({self.num_sims * self.num_samples_per_sim} total samples)")
        if self.enable_augmentation:
            print(f"  Augmentation: ENABLED (noise_levels={self.augmentation_noise_levels}, prob={self.augmentation_probability})")

        # Check time information if available
        if 't' in data_dict:
            time_info = data_dict['t']
            print(f"  Time range: {time_info[0]:.1f} to {time_info[-1]:.1f}")
            print(f"  Time step size: {time_info[1] - time_info[0]:.1f}")
        else:
            print("  No time information found in dataset")

    def __len__(self):
        if self.subset_mode:
            return len(self.subset_indices)
        return self.num_sims * self.num_samples_per_sim

    def __getitem__(self, idx):
        if self.subset_mode:
            sim_idx, start_t = self.subset_indices[idx]
        else:
            sim_idx = idx // self.num_samples_per_sim
            start_t = idx % self.num_samples_per_sim

        time_slice = self.data[sim_idx, start_t : start_t + self.total_timesteps]

        x = time_slice[:self.n_input_timesteps]
        y = time_slice[self.n_input_timesteps : self.total_timesteps]

        # Reshape for model: [H, W, C]
        x = x.permute(1, 2, 0)
        y = y.permute(1, 2, 0)

        # For rollout loss, get additional future timesteps if available
        # We need up to (max_rollout_steps-1) additional timesteps
        y_next_dict = {}

        for step in range(1, self.max_rollout_steps):
            key = 'y_next' if step == 1 else f'y_next{step}'
            y_next_dict[key] = None

            if start_t + self.total_timesteps + step <= self.T:
                y_next_slice = self.data[sim_idx, start_t + self.total_timesteps + step - 1 : start_t + self.total_timesteps + step]
                y_next_dict[key] = y_next_slice.permute(1, 2, 0)

        # Apply augmentation if enabled
        if self.enable_augmentation and torch.rand(1, generator=self.augmentation_generator).item() < self.augmentation_probability:
                # Randomly select noise level
                noise_idx = torch.randint(0, len(self.augmentation_noise_levels), (1,), generator=self.augmentation_generator).item()
                noise_std = self.augmentation_noise_levels[noise_idx]

                # Add noise to input
                x = x + torch.randn(x.shape, generator=self.augmentation_generator, dtype=x.dtype, device=x.device) * noise_std

                # Add noise to output
                y = y + torch.randn(y.shape, generator=self.augmentation_generator, dtype=y.dtype, device=y.device) * noise_std

                # Add noise to future timesteps
                for key in y_next_dict:
                    if y_next_dict[key] is not None:
                        y_future = y_next_dict[key]
                        y_next_dict[key] = y_future + torch.randn(y_future.shape, generator=self.augmentation_generator, dtype=y_future.dtype, device=y_future.device) * noise_std

        result = {'x': x, 'y': y}
        result.update(y_next_dict)
        return result


# Utility functions for dataset creation
def create_ns2d_datasets(
    data_dir='data',
    target_resolution=64,
    max_rollout_steps=10,
    subset_mode=False,
    subset_size=None,
    enable_augmentation=False,
    augmentation_noise_levels=None,
    augmentation_probability=0.5
):
    """
    Create train and test datasets for 2D Navier-Stokes equation

    Args:
        data_dir: Directory containing NS2D data
        target_resolution: Target spatial resolution for downsampling
        max_rollout_steps: Maximum rollout steps for rollout loss
        subset_mode: Whether to use subset of data
        subset_size: Size of subset if subset_mode=True
        enable_augmentation: Whether to enable data augmentation
        augmentation_noise_levels: List of noise levels for augmentation
        augmentation_probability: Probability of applying augmentation

    Returns:
        train_dataset, test_dataset
    """
    if augmentation_noise_levels is None:
        augmentation_noise_levels = [0.01, 0.03]
    train_dataset = NS2DDataset(
        data_dir=data_dir,
        split='train',
        target_resolution=target_resolution,
        max_rollout_steps=max_rollout_steps,
        subset_mode=subset_mode,
        subset_size=subset_size,
        enable_augmentation=enable_augmentation,
        augmentation_noise_levels=augmentation_noise_levels,
        augmentation_probability=augmentation_probability
    )

    test_dataset = NS2DDataset(
        data_dir=data_dir,
        split='test',
        target_resolution=target_resolution,
        max_rollout_steps=max_rollout_steps,
        subset_mode=False,  # No subset for test data
        enable_augmentation=False  # No augmentation for test data
    )

    return train_dataset, test_dataset
