"""
1D Dataset Classes for Neural Operators

This module provides dataset classes for 1D PDE problems.


Classes:
- BurgersDataset: Dataset for 1D Burgers equation
- KdVDataset: Dataset for 1D Korteweg-de Vries equation
"""

import os

import torch
from torch.utils.data import Dataset


class BurgersDataset(Dataset):
    def __init__(self, data_dir, prediction_gap=1, split='train', train_split_ratio=0.8):
        self.data_dir = data_dir
        self.prediction_gap = prediction_gap

        # Load aggregated data file
        aggregated_file = os.path.join(data_dir, "burgers_data.pt")
        if not os.path.exists(aggregated_file):
            raise FileNotFoundError(f"Aggregated data file not found: {aggregated_file}. Run 'python aggregate_data.py --experiment burgers' to create it.")

        print(f"Loading aggregated data from {aggregated_file}")
        self.data = torch.load(aggregated_file, map_location='cpu')

        num_runs = self.data['num_runs']
        split_idx = int(num_runs * train_split_ratio)

        if split == 'train':
            self.run_indices = list(range(split_idx))
        else:
            self.run_indices = list(range(split_idx, num_runs))

        # Create data map
        self.data_map = []
        num_timesteps = self.data['time_steps']
        num_samples_per_run = num_timesteps - self.prediction_gap

        for run_idx in self.run_indices:
            for time_idx in range(num_samples_per_run):
                self.data_map.append((run_idx, time_idx))

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        run_idx, start_time_idx = self.data_map[idx]
        snapshots = self.data['snapshots'][run_idx]  # Shape: [time, spatial]

        in_slice = snapshots[start_time_idx]
        out_slice = snapshots[start_time_idx + self.prediction_gap]

        # Reshape to [spatial, channels=1]
        in_data = in_slice.unsqueeze(-1)
        out_data = out_slice.unsqueeze(-1)

        return {'x': in_data, 'y': out_data}


class KdVDataset(Dataset):
    """
    Dataset class for 1D Korteweg-de Vries equation



    Args:
        data_dir: Directory containing simulation_run_*.pt files
        n_input_timesteps: Number of input timesteps (currently supports 1)
        prediction_gap: Gap between input and prediction timesteps
        split: 'train' or 'test' split
        train_split_ratio: Ratio of files used for training
    """

    def __init__(self, data_dir, n_input_timesteps=1, prediction_gap=1, split='train', train_split_ratio=0.8):
        self.data_dir = data_dir
        self.n_input_timesteps = n_input_timesteps
        self.prediction_gap = prediction_gap
        self.split = split

        # Load aggregated data file
        aggregated_file = os.path.join(data_dir, "kdv_data.pt")
        if not os.path.exists(aggregated_file):
            raise FileNotFoundError(f"Aggregated data file not found: {aggregated_file}. Run 'python aggregate_data.py --experiment kdv' to create it.")

        print(f"Loading aggregated KdV data from {aggregated_file}")
        self.data = torch.load(aggregated_file, map_location='cpu', weights_only=False)

        num_runs = self.data['num_runs']
        split_idx = int(num_runs * train_split_ratio)

        if split == 'train':
            self.run_indices = list(range(split_idx))
        else:
            self.run_indices = list(range(split_idx, num_runs))

        # Create data map
        self.data_map = []
        num_timesteps = self.data['time_steps']
        num_samples_per_run = num_timesteps - self.n_input_timesteps - self.prediction_gap + 1

        for run_idx in self.run_indices:
            for time_idx in range(num_samples_per_run):
                self.data_map.append((run_idx, time_idx))

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        run_idx, start_time_idx = self.data_map[idx]
        snapshots = self.data['snapshots'][run_idx]  # Shape: [time, channels, spatial]

        # Predict a single timestep with specified gap
        start_idx = start_time_idx
        in_slice = snapshots[start_idx]      # Shape: (channels, spatial_points)
        out_slice = snapshots[start_idx + self.prediction_gap]  # Target is 'gap' steps away

        # Ensure correct format: [spatial_points, channels]
        # Snapshots are in format [time, channels, spatial] where channels=1
        # So in_slice and out_slice are [1, 128], we want [128, 1]
        in_data = in_slice.transpose(0, 1)   # From [1, 128] to [128, 1]
        out_data = out_slice.transpose(0, 1) # From [1, 128] to [128, 1]

        return {'x': in_data, 'y': out_data}


# Utility functions for dataset creation
def create_burgers_datasets(data_dir='data/Burgers', prediction_gap=10, train_split_ratio=0.8):
    """
    Create train and test datasets for Burgers equation

    Args:
        data_dir: Directory containing Burgers data
        prediction_gap: Prediction gap (default from original: 10)
        train_split_ratio: Train/test split ratio

    Returns:
        train_dataset, test_dataset
    """
    train_dataset = BurgersDataset(
        data_dir=data_dir,
        prediction_gap=prediction_gap,
        split='train',
        train_split_ratio=train_split_ratio
    )
    test_dataset = BurgersDataset(
        data_dir=data_dir,
        prediction_gap=prediction_gap,
        split='test',
        train_split_ratio=train_split_ratio
    )
    return train_dataset, test_dataset


def create_kdv_datasets(data_dir='data/KdV', prediction_gap=10, train_split_ratio=0.8):
    """
    Create train and test datasets for KdV equation

    Args:
        data_dir: Directory containing KdV data
        prediction_gap: Prediction gap (default from original: 10)
        train_split_ratio: Train/test split ratio

    Returns:
        train_dataset, test_dataset
    """
    train_dataset = KdVDataset(
        data_dir=data_dir,
        prediction_gap=prediction_gap,
        split='train',
        train_split_ratio=train_split_ratio
    )
    test_dataset = KdVDataset(
        data_dir=data_dir,
        prediction_gap=prediction_gap,
        split='test',
        train_split_ratio=train_split_ratio
    )
    return train_dataset, test_dataset
