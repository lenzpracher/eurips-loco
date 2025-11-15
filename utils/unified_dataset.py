"""
Unified dataset approach that replicates the old burgers.py file-based loading
but stores everything in memory and samples exactly 1000/200 per epoch.
"""

import torch
from torch.utils.data import Dataset
import glob
import os
import numpy as np
from typing import List, Tuple, Optional

class UnifiedBurgersDataset(Dataset):
    """
    Unified Burgers dataset that:
    1. Loads all files into memory at initialization (like neurips approach)
    2. Maintains strict train/test split based on file indices 
    3. Samples exactly 1000 train / 200 test samples per epoch
    4. Resamples after each epoch for diversity (like original file-based approach)
    """
    
    def __init__(
        self, 
        data_dir: str,
        prediction_gap: int = 10,
        split: str = 'train',
        train_split_ratio: float = 0.8,
        samples_per_epoch: int = None,  # 1000 for train, 200 for test
        seed: int = 42,
        epoch: int = 0,
        normalize: bool = True
    ):
        self.data_dir = data_dir
        self.prediction_gap = prediction_gap
        self.split = split
        self.train_split_ratio = train_split_ratio
        self.seed = seed
        self.epoch = epoch
        self.normalize = normalize
        
        # Set samples per epoch based on split if not specified
        if samples_per_epoch is None:
            self.samples_per_epoch = 1000 if split == 'train' else 200
        else:
            self.samples_per_epoch = samples_per_epoch
            
        # Load all data files
        self.files = sorted(glob.glob(os.path.join(data_dir, "run_*.pt")))
        if not self.files:
            raise FileNotFoundError(f"No data files found in {data_dir}")
            
        # Split files into train/test based on indices (not per-file sampling)
        split_idx = int(len(self.files) * train_split_ratio)
        if split == 'train':
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]
            
        print(f"Loading {len(self.files)} files for {split} split...")
        
        # Load all data into memory upfront
        self.all_data = []  # List of tensors
        self.valid_samples = []  # List of (file_idx, time_idx) pairs
        
        for file_idx, file_path in enumerate(self.files):
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            snapshots = data['snapshots']  # Shape: [time, spatial]
            
            # Store the tensor
            self.all_data.append(snapshots)
            
            # Create all valid (file_idx, time_idx) pairs for this file
            num_samples_in_file = snapshots.shape[0] - self.prediction_gap
            for time_idx in range(num_samples_in_file):
                self.valid_samples.append((file_idx, time_idx))
        
        print(f"Loaded {len(self.all_data)} files with {len(self.valid_samples)} total valid samples")
        
        # Compute normalization statistics if needed
        if self.normalize:
            print("Computing normalization statistics...")
            all_tensors = torch.cat(self.all_data, dim=0)  # Concatenate all time steps
            self.mean = all_tensors.mean()
            self.std = all_tensors.std()
            
            # Normalize all data in place
            for i in range(len(self.all_data)):
                self.all_data[i] = (self.all_data[i] - self.mean) / self.std
        else:
            self.mean = 0.0
            self.std = 1.0
        
        # Generate sample indices for this epoch
        self._resample_epoch()
    
    def _resample_epoch(self):
        """Resample indices for the current epoch."""
        # Create epoch-specific random generator for reproducibility
        rng = np.random.default_rng(seed=self.seed + self.epoch)
        
        # Sample exactly samples_per_epoch indices
        if len(self.valid_samples) >= self.samples_per_epoch:
            chosen_indices = rng.choice(len(self.valid_samples), size=self.samples_per_epoch, replace=False)
            self.epoch_samples = [self.valid_samples[i] for i in chosen_indices]
        else:
            # If not enough samples, use all and pad with repeats
            self.epoch_samples = self.valid_samples.copy()
            remaining = self.samples_per_epoch - len(self.valid_samples)
            extra_indices = rng.choice(len(self.valid_samples), size=remaining, replace=True)
            self.epoch_samples.extend([self.valid_samples[i] for i in extra_indices])
            
        print(f"Epoch {self.epoch}: Sampled {len(self.epoch_samples)} samples for {self.split}")
    
    def set_epoch(self, epoch: int):
        """Set the epoch and resample data."""
        self.epoch = epoch
        self._resample_epoch()
    
    def __len__(self):
        return len(self.epoch_samples)
    
    def __getitem__(self, idx):
        file_idx, time_idx = self.epoch_samples[idx]
        snapshots = self.all_data[file_idx]
        
        in_slice = snapshots[time_idx]
        out_slice = snapshots[time_idx + self.prediction_gap]
        
        # Reshape to [spatial, channels=1] to match original format
        in_data = in_slice.unsqueeze(-1)
        out_data = out_slice.unsqueeze(-1)
        
        return in_data, out_data
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data back to original scale."""
        if self.normalize:
            return data * self.std + self.mean
        return data


def create_unified_burgers_datasets(
    data_dir: str,
    prediction_gap: int = 10,
    train_split_ratio: float = 0.8,
    seed: int = 42,
    normalize: bool = True
) -> Tuple[UnifiedBurgersDataset, UnifiedBurgersDataset]:
    """
    Create unified train and test datasets for Burgers equation.
    
    Args:
        data_dir: Directory containing run_*.pt files
        prediction_gap: Time gap between input and output
        train_split_ratio: Ratio for train/test file split
        seed: Random seed for reproducibility
        normalize: Whether to normalize the data
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_dataset = UnifiedBurgersDataset(
        data_dir=data_dir,
        prediction_gap=prediction_gap,
        split='train',
        train_split_ratio=train_split_ratio,
        samples_per_epoch=1000,
        seed=seed,
        epoch=0,
        normalize=normalize
    )
    
    test_dataset = UnifiedBurgersDataset(
        data_dir=data_dir,
        prediction_gap=prediction_gap,
        split='test',
        train_split_ratio=train_split_ratio,
        samples_per_epoch=200,
        seed=seed,
        epoch=0,
        normalize=False  # Use train stats for normalization
    )
    
    # Share normalization statistics
    test_dataset.mean = train_dataset.mean
    test_dataset.std = train_dataset.std
    if normalize:
        # Apply train normalization to test data
        for i in range(len(test_dataset.all_data)):
            test_dataset.all_data[i] = (test_dataset.all_data[i] - test_dataset.mean) / test_dataset.std
    
    return train_dataset, test_dataset