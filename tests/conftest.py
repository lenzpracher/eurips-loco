"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def device():
    """Return available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def small_1d_data():
    """Generate small 1D test data."""
    torch.manual_seed(42)
    batch_size, n_time, n_spatial = 10, 20, 32
    data = torch.randn(batch_size, n_time, n_spatial)
    return data


@pytest.fixture
def small_2d_data():
    """Generate small 2D test data."""
    torch.manual_seed(42)
    batch_size, n_time, height, width = 5, 10, 16, 16
    data = torch.randn(batch_size, n_time, height, width)
    return data


@pytest.fixture
def model_config_1d():
    """Standard configuration for 1D models."""
    return {
        'in_channels': 1,
        'out_channels': 1,
        'hidden_channels': 16,
        'num_blocks': 2,
        'modes_x': 8
    }


@pytest.fixture
def model_config_2d():
    """Standard configuration for 2D models."""
    return {
        'in_channels': 1,
        'out_channels': 1,
        'hidden_channels': 16,
        'num_blocks': 2,
        'modes_x': 8,
        'modes_y': 8
    }


@pytest.fixture
def solver_config_burgers():
    """Configuration for Burgers solver."""
    return {
        'N': 32,
        'T': 1.0,
        'dt': 0.1,
        'nu': 0.01
    }


@pytest.fixture
def solver_config_kdv():
    """Configuration for KdV solver.""" 
    return {
        'N': 64,  # Increased resolution for stability
        'T': 0.5,  # Reduced time for testing
        'dt': 0.01,  # Much smaller timestep for stability
        'a': 6.0,
        'b': 1.0
    }


@pytest.fixture
def solver_config_ns2d():
    """Configuration for 2D Navier-Stokes solver."""
    return {
        'N': 16,
        'T': 2.0,
        'record_steps': 20,
        'visc': 1e-3
    }