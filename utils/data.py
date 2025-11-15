"""
Data handling utilities for neural operator experiments.

This module contains dataset classes and data generation utilities
for different types of PDE problems.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Tuple, Optional, Union
import os

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers import BurgersSolver, KdVSolver, NavierStokesSolver
from .unified_dataset import create_unified_burgers_datasets

# Paper-compliant sample counts
PAPER_TRAIN_SAMPLES = 1000
PAPER_TEST_SAMPLES = 200


class PDEDataset(Dataset):
    """
    Generic dataset class for PDE data.
    
    Handles time series data where we predict the next timestep(s) from
    previous timestep(s).
    """
    
    def __init__(
        self,
        data: torch.Tensor,
        input_length: int = 1,
        output_length: int = 1,
        time_gap: int = 1,
        normalize: bool = True
    ):
        """
        Initialize PDE dataset.
        
        Args:
            data: Time series data of shape (n_samples, n_time, *spatial_dims)
            input_length: Number of input timesteps
            output_length: Number of output timesteps to predict
            time_gap: Gap between input and output timesteps
            normalize: Whether to normalize the data
        """
        self.data = data
        self.input_length = input_length
        self.output_length = output_length
        self.time_gap = time_gap
        self.normalize = normalize
        
        # Compute normalization statistics
        if normalize:
            self.mean = data.mean()
            self.std = data.std()
            self.data = (data - self.mean) / self.std
        else:
            self.mean = 0.0
            self.std = 1.0
        
        # Compute valid time indices
        self.n_samples, self.n_time = data.shape[:2]
        self.max_input_time = self.n_time - self.input_length - self.time_gap - self.output_length + 1
        
        if self.max_input_time <= 0:
            raise ValueError(f"Not enough timesteps in data. Need at least {self.input_length + self.time_gap + self.output_length}")
    
    def __len__(self):
        return self.n_samples * self.max_input_time
    
    def __getitem__(self, idx):
        # Convert flat index to sample and time indices
        sample_idx = idx // self.max_input_time
        time_idx = idx % self.max_input_time
        
        # Extract input sequence
        input_start = time_idx
        input_end = time_idx + self.input_length
        x = self.data[sample_idx, input_start:input_end]
        
        # Extract output sequence
        output_start = input_end + self.time_gap
        output_end = output_start + self.output_length
        y = self.data[sample_idx, output_start:output_end]
        
        # Reshape for neural operator input/output format
        if len(x.shape) == 2:  # 1D spatial data
            # For 1D: [time, spatial] -> [time, spatial] (keep as is)
            pass
        elif len(x.shape) == 3:  # 2D spatial data
            # For 2D: [time, height, width] -> [time, height, width] (keep as is)
            pass
        
        # If single timestep output, remove time dimension
        if self.output_length == 1:
            y = y.squeeze(0)
        
        # If single timestep input, remove time dimension
        if self.input_length == 1:
            x = x.squeeze(0)
            
        # Add channel dimension for 1D data if needed
        if len(x.shape) == 1:  # 1D spatial data needs channel dim: (N,) -> (1, N)
            x = x.unsqueeze(0) 
        if len(y.shape) == 1:  # 1D spatial data needs channel dim: (N,) -> (1, N)
            y = y.unsqueeze(0)
        
        return x, y
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data back to original scale."""
        if self.normalize:
            return data * self.std + self.mean
        return data


class SimplePDEDataset(Dataset):
    """
    Simplified dataset class that uses one sample per simulation for faster training.
    
    Instead of using all possible time windows, this samples one random time window
    per simulation, resulting in exactly n_samples training examples per epoch.
    """
    
    def __init__(
        self,
        data: torch.Tensor,
        input_length: int = 1,
        output_length: int = 1,
        time_gap: int = 1,
        normalize: bool = True,
        samples_per_sim: int = 1,
        limit_samples: bool = False,
        max_samples: int = 1000,
        split: str = 'train'
    ):
        """
        Initialize simplified PDE dataset.
        
        Args:
            data: Time series data of shape (n_samples, n_time, *spatial_dims)
            input_length: Number of input timesteps
            output_length: Number of output timesteps to predict
            time_gap: Gap between input and output timesteps
            normalize: Whether to normalize the data
            samples_per_sim: Number of random samples per simulation (default: 1)
            limit_samples: Whether to limit total samples to max_samples (default: False)
            max_samples: Maximum number of samples when limit_samples=True (default: 1000)
            split: Dataset split ('train' or 'test') for sample limiting
        """
        self.data = data
        self.input_length = input_length
        self.output_length = output_length
        self.time_gap = time_gap
        self.normalize = normalize
        self.samples_per_sim = samples_per_sim
        self.limit_samples = limit_samples
        self.max_samples = max_samples
        self.split = split
        
        # Compute normalization statistics
        if normalize:
            self.mean = data.mean()
            self.std = data.std()
            self.data = (data - self.mean) / self.std
        else:
            self.mean = 0.0
            self.std = 1.0
        
        # Compute valid time indices
        self.n_samples, self.n_time = data.shape[:2]
        self.max_input_time = self.n_time - self.input_length - self.time_gap - self.output_length + 1
        
        if self.max_input_time <= 0:
            raise ValueError(f"Not enough timesteps in data. Need at least {self.input_length + self.time_gap + self.output_length}")
        
        # Generate sample pairs
        if self.limit_samples:
            # Generate all possible (sim_idx, time_idx) pairs
            all_samples = []
            for sim_idx in range(self.n_samples):
                for time_idx in range(self.max_input_time):
                    all_samples.append((sim_idx, time_idx))
            
            # Sample exactly the requested number with fixed seed for reproducibility
            import numpy as np
            rng = np.random.default_rng(seed=42)
            if len(all_samples) > self.max_samples:
                chosen_indices = rng.choice(len(all_samples), size=self.max_samples, replace=False)
                self.sample_pairs = [all_samples[i] for i in chosen_indices]
            else:
                self.sample_pairs = all_samples
            
            print(f"Sample limiting enabled: Using {len(self.sample_pairs)} samples from {len(all_samples)} possible samples (split: {self.split})")
        else:
            # Original behavior: one random time index per simulation
            self.time_indices = []
            for _ in range(self.n_samples):
                for _ in range(self.samples_per_sim):
                    time_idx = torch.randint(0, self.max_input_time, (1,)).item()
                    self.time_indices.append(time_idx)
    
    def __len__(self):
        if self.limit_samples:
            return len(self.sample_pairs)
        else:
            return self.n_samples * self.samples_per_sim
    
    def __getitem__(self, idx):
        if self.limit_samples:
            sample_idx, time_idx = self.sample_pairs[idx]
        else:
            sample_idx = idx // self.samples_per_sim
            time_idx = self.time_indices[idx]
        
        # Extract input sequence
        input_start = time_idx
        input_end = time_idx + self.input_length
        x = self.data[sample_idx, input_start:input_end]
        
        # Extract output sequence
        output_start = input_end + self.time_gap
        output_end = output_start + self.output_length
        y = self.data[sample_idx, output_start:output_end]
        
        # Reshape for neural operator input/output format
        if len(x.shape) == 2:  # 1D spatial data
            pass
        elif len(x.shape) == 3:  # 2D spatial data
            pass
        
        # If single timestep output, remove time dimension
        if self.output_length == 1:
            y = y.squeeze(0)
        
        # If single timestep input, remove time dimension
        if self.input_length == 1:
            x = x.squeeze(0)
            
        # Add channel dimension for 1D data if needed
        if len(x.shape) == 1:  # 1D spatial data needs channel dim: (N,) -> (1, N)
            x = x.unsqueeze(0) 
        if len(y.shape) == 1:  # 1D spatial data needs channel dim: (N,) -> (1, N)
            y = y.unsqueeze(0)
        
        return x, y
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data back to original scale."""
        if self.normalize:
            return data * self.std + self.mean
        return data


def create_burgers_dataset(
    n_train: int = 1000,
    n_test: int = 200,
    input_length: int = 1,
    output_length: int = 1,
    data_path: Optional[str] = None,
    device: str = "cpu",
    **solver_kwargs
) -> Tuple[PDEDataset, PDEDataset]:
    """
    Create Burgers equation datasets.
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        input_length: Number of input timesteps
        output_length: Number of output timesteps
        data_path: Path to existing data (if None, generates new data)
        device: Device for data generation
        **solver_kwargs: Additional arguments for solver
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Check if we have file-based data (run_*.pt files) - use unified dataset approach
    if data_path and os.path.exists(data_path):
        run_files = [f for f in os.listdir(data_path) if f.startswith('run_') and f.endswith('.pt')]
        if run_files:
            print(f"Found {len(run_files)} run files in {data_path}, using UnifiedBurgersDataset")
            time_gap = solver_kwargs.get('time_gap', 10)
            return create_unified_burgers_datasets(
                data_dir=data_path,
                prediction_gap=time_gap,
                train_split_ratio=0.8,
                seed=42,
                normalize=True
            )
    
    # Fallback to original approach for generated data
    if data_path and os.path.exists(os.path.join(data_path, 'burgers_train.pt')):
        print(f"Loading Burgers data from {data_path}")
        from solvers.burgers import load_burgers_data
        train_data, test_data = load_burgers_data(data_path)
    else:
        print("Generating new Burgers data...")
        from solvers.burgers import generate_burgers_data
        # Filter out dataset-specific parameters
        solver_params = {k: v for k, v in solver_kwargs.items() 
                        if k not in ['time_gap']}
        train_data, test_data = generate_burgers_data(
            n_train=n_train,
            n_test=n_test,
            device=device,
            save_path=data_path,
            **solver_params
        )
    
    # Extract time_gap if provided
    time_gap = solver_kwargs.get('time_gap', 1)
    print(f"Dataset time_gap: {time_gap}")
    
    # Create datasets - use SimplePDEDataset for faster training
    train_dataset = SimplePDEDataset(
        train_data['u'],
        input_length=input_length,
        output_length=output_length,
        time_gap=time_gap,
        limit_samples=True,
        max_samples=PAPER_TRAIN_SAMPLES,
        split='train'
    )
    
    test_dataset = SimplePDEDataset(
        test_data['u'],
        input_length=input_length,
        output_length=output_length,
        time_gap=time_gap,
        normalize=False,  # Use training stats for normalization
        limit_samples=True,
        max_samples=PAPER_TEST_SAMPLES,
        split='test'
    )
    test_dataset.mean = train_dataset.mean
    test_dataset.std = train_dataset.std
    if test_dataset.normalize:
        test_dataset.data = (test_data['u'] - test_dataset.mean) / test_dataset.std
    
    return train_dataset, test_dataset


def create_kdv_dataset(
    n_train: int = 1000,
    n_test: int = 200,
    input_length: int = 1,
    output_length: int = 1,
    data_path: Optional[str] = None,
    device: str = "cpu",
    **solver_kwargs
) -> Tuple[PDEDataset, PDEDataset]:
    """
    Create KdV equation datasets.
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        input_length: Number of input timesteps
        output_length: Number of output timesteps
        data_path: Path to existing data (if None, generates new data)
        device: Device for data generation
        **solver_kwargs: Additional arguments for solver
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if data_path and os.path.exists(os.path.join(data_path, 'kdv_train.pt')):
        print(f"Loading KdV data from {data_path}")
        from solvers.kdv import load_kdv_data
        train_data, test_data = load_kdv_data(data_path)
    else:
        print("Generating new KdV data...")
        from solvers.kdv import generate_kdv_data
        # Filter out dataset-specific parameters
        solver_params = {k: v for k, v in solver_kwargs.items() 
                        if k not in ['time_gap']}
        train_data, test_data = generate_kdv_data(
            n_train=n_train,
            n_test=n_test,
            device=device,
            save_path=data_path,
            **solver_params
        )
    
    # Extract time_gap if provided
    time_gap = solver_kwargs.get('time_gap', 1)
    print(f"Dataset time_gap: {time_gap}")
    
    # Create datasets - use SimplePDEDataset for faster training
    train_dataset = SimplePDEDataset(
        train_data['u'],
        input_length=input_length,
        output_length=output_length,
        time_gap=time_gap,
        limit_samples=True,
        max_samples=PAPER_TRAIN_SAMPLES,
        split='train'
    )
    
    test_dataset = SimplePDEDataset(
        test_data['u'],
        input_length=input_length,
        output_length=output_length,
        time_gap=time_gap,
        normalize=False,  # Use training stats for normalization
        limit_samples=True,
        max_samples=PAPER_TEST_SAMPLES,
        split='test'
    )
    test_dataset.mean = train_dataset.mean
    test_dataset.std = train_dataset.std
    if test_dataset.normalize:
        test_dataset.data = (test_data['u'] - test_dataset.mean) / test_dataset.std
    
    return train_dataset, test_dataset


def create_ns2d_dataset(
    n_train: int = 1000,
    n_test: int = 200,
    input_length: int = 10,  # Common for 2D NS
    output_length: int = 1,
    data_path: Optional[str] = None,
    device: str = "cpu",
    **solver_kwargs
) -> Tuple[PDEDataset, PDEDataset]:
    """
    Create 2D Navier-Stokes datasets.
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        input_length: Number of input timesteps
        output_length: Number of output timesteps
        data_path: Path to existing data (if None, generates new data)
        device: Device for data generation
        **solver_kwargs: Additional arguments for solver
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if data_path and os.path.exists(os.path.join(data_path, 'ns2d_train.pt')):
        print(f"Loading 2D Navier-Stokes data from {data_path}")
        from solvers.navier_stokes import load_ns2d_data
        train_data, test_data = load_ns2d_data(data_path)
    else:
        print("Generating new 2D Navier-Stokes data...")
        from solvers.navier_stokes import generate_ns2d_data
        # Filter out dataset-specific parameters
        solver_params = {k: v for k, v in solver_kwargs.items() 
                        if k not in ['time_gap', 'record_steps']}
        train_data, test_data = generate_ns2d_data(
            n_train=n_train,
            n_test=n_test,
            device=device,
            save_path=data_path,
            **solver_params
        )
    
    # Extract time_gap if provided
    time_gap = solver_kwargs.get('time_gap', 1)
    print(f"Dataset time_gap: {time_gap}")
    
    # Create datasets - use SimplePDEDataset for faster training
    train_dataset = SimplePDEDataset(
        train_data['u'],
        input_length=input_length,
        output_length=output_length,
        time_gap=time_gap,
        limit_samples=True,
        max_samples=PAPER_TRAIN_SAMPLES,
        split='train'
    )
    
    test_dataset = SimplePDEDataset(
        test_data['u'],
        input_length=input_length,
        output_length=output_length,
        time_gap=time_gap,
        normalize=False,  # Use training stats for normalization
        limit_samples=True,
        max_samples=PAPER_TEST_SAMPLES,
        split='test'
    )
    test_dataset.mean = train_dataset.mean
    test_dataset.std = train_dataset.std
    if test_dataset.normalize:
        test_dataset.data = (test_data['u'] - test_dataset.mean) / test_dataset.std
    
    return train_dataset, test_dataset


def generate_data(
    equation: str,
    n_train: int = 1000,
    n_test: int = 200,
    input_length: int = 1,
    output_length: int = 1,
    data_path: Optional[str] = None,
    device: str = "cpu",
    **kwargs
) -> Tuple[PDEDataset, PDEDataset]:
    """
    Unified interface for generating datasets for different equations.
    
    Args:
        equation: Type of equation ('burgers', 'kdv', 'ns2d')
        n_train: Number of training samples
        n_test: Number of test samples
        input_length: Number of input timesteps
        output_length: Number of output timesteps
        data_path: Path to save/load data
        device: Device for computation
        **kwargs: Additional arguments for specific equations
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    equation = equation.lower()
    
    if equation == 'burgers':
        return create_burgers_dataset(
            n_train=n_train,
            n_test=n_test,
            input_length=input_length,
            output_length=output_length,
            data_path=data_path,
            device=device,
            **kwargs
        )
    elif equation == 'kdv':
        return create_kdv_dataset(
            n_train=n_train,
            n_test=n_test,
            input_length=input_length,
            output_length=output_length,
            data_path=data_path,
            device=device,
            **kwargs
        )
    elif equation == 'ns2d':
        # Default to multi-timestep input for 2D NS
        if input_length == 1:
            input_length = 10
        return create_ns2d_dataset(
            n_train=n_train,
            n_test=n_test,
            input_length=input_length,
            output_length=output_length,
            data_path=data_path,
            device=device,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown equation type: {equation}. Supported: 'burgers', 'kdv', 'ns2d'")


def get_data_info(equation: str) -> Dict:
    """
    Get standard parameters for different equation types.
    
    Args:
        equation: Type of equation
        
    Returns:
        Dictionary with standard parameters
    """
    info = {
        'burgers': {
            'spatial_dim': 1,
            'default_resolution': 256,
            'default_input_length': 1,
            'default_timesteps': 200,
            'typical_domain': [0, 2*np.pi],
            'description': '1D Burgers equation: u_t + u*u_x = nu*u_xx'
        },
        'kdv': {
            'spatial_dim': 1,
            'default_resolution': 128,
            'default_input_length': 1,
            'default_timesteps': 400,
            'typical_domain': [-np.pi, np.pi],
            'description': '1D KdV equation: u_t + a*u*u_x + b*u_xxx = 0'
        },
        'ns2d': {
            'spatial_dim': 2,
            'default_resolution': 64,
            'default_input_length': 10,
            'default_timesteps': 100,
            'typical_domain': [0, 1],
            'description': '2D Navier-Stokes: vorticity form with forcing'
        }
    }
    
    equation = equation.lower()
    if equation not in info:
        raise ValueError(f"Unknown equation: {equation}")
    
    return info[equation]