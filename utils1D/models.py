"""
Model initialization utilities for 1D experiments

This module provides model initialization functions that match the exact
configurations used in old_scripts/Burgers_KdV/ experiments.
"""

import os

import torch
from neuralop.training import AdamW

from models1D.fno import create_burgers_fno, create_kdv_fno
from models1D.hybrid import create_burgers_hybrid, create_kdv_hybrid
from models1D.loco import create_burgers_loco, create_kdv_loco


def initialize_burgers_models(device):
    """
    Initialize all models for Burgers equation with exact old_scripts parameters

    From old_scripts/Burgers_KdV/burgers.py:
    - MODES = 16
    - CHANNELS = 1
    - HIDDEN_CHANNELS = 24
    - NUM_BLOCKS = 4
    - FNO_HIDDEN_CHANNELS = 32
    - USE_SPECTRAL_CHANNEL_MLP = True
    """

    # Create models with exact parameters from old scripts
    loco_model = create_burgers_loco(
        hidden_channels=24,  # HIDDEN_CHANNELS = 24
        num_blocks=4,        # NUM_BLOCKS = 4
        modes=16,           # MODES = 16
        use_mlp=True        # USE_SPECTRAL_CHANNEL_MLP = True
    ).to(device)

    hybrid_model = create_burgers_hybrid(
        hidden_channels=24,  # HIDDEN_CHANNELS = 24
        num_blocks=4,        # NUM_BLOCKS = 4
        modes=16,           # MODES = 16
        use_mlp=True        # USE_SPECTRAL_CHANNEL_MLP = True
    ).to(device)

    fno_model = create_burgers_fno(
        modes=16,           # MODES = 16
        hidden_channels=32, # FNO_HIDDEN_CHANNELS = 32
        n_layers=4,         # NUM_BLOCKS = 4
        use_mlp=True        # USE_SPECTRAL_CHANNEL_MLP = True
    ).to(device)

    # Model dictionary with exact structure from old scripts
    models_dict = {
        "Hybrid": (hybrid_model, 'sno', 'hybrid_model.pt', {'type': 'adam'}),
        "LOCO": (loco_model, 'sno', 'loco_model.pt', {'type': 'adam'}),
        "FNO": (fno_model, 'fno', 'fno_model.pt', {'type': 'adamw', 'weight_decay': 1e-4})
    }

    return models_dict


def initialize_kdv_models(device):
    """
    Initialize all models for KdV equation with exact old_scripts parameters

    From old_scripts/Burgers_KdV/KdV.py:
    - MODES = 16
    - CHANNELS = 1
    - HIDDEN_CHANNELS = 26  # Different from Burgers!
    - NUM_BLOCKS = 4
    - FNO_HIDDEN_CHANNELS = 32
    - FNO_LIFTING_CHANNEL_RATIO = 2
    - FNO_PROJECTION_CHANNEL_RATIO = 1
    - USE_SPECTRAL_CHANNEL_MLP = False  # Different from Burgers!
    """

    # Create models with exact parameters from old scripts
    loco_model = create_kdv_loco(
        hidden_channels=26,  # HIDDEN_CHANNELS = 26
        num_blocks=4,        # NUM_BLOCKS = 4
        modes=16,           # MODES = 16
        use_mlp=False       # USE_SPECTRAL_CHANNEL_MLP = False
    ).to(device)

    hybrid_model = create_kdv_hybrid(
        hidden_channels=26,  # HIDDEN_CHANNELS = 26
        num_blocks=4,        # NUM_BLOCKS = 4
        modes=16,           # MODES = 16
        use_mlp=False       # USE_SPECTRAL_CHANNEL_MLP = False
    ).to(device)

    fno_model = create_kdv_fno(
        modes=16,                        # MODES = 16
        hidden_channels=32,              # FNO_HIDDEN_CHANNELS = 32
        lifting_channel_ratio=2,         # FNO_LIFTING_CHANNEL_RATIO = 2
        projection_channel_ratio=1,      # FNO_PROJECTION_CHANNEL_RATIO = 1
        n_layers=4,                     # NUM_BLOCKS = 4
        use_mlp=False                   # USE_SPECTRAL_CHANNEL_MLP = False
    ).to(device)

    # Model dictionary with exact structure from old scripts
    models_dict = {
        "LOCO": (loco_model, 'sno', 'loco_model.pt', {'type': 'adam'}),
        "Hybrid": (hybrid_model, 'sno', 'hybrid_model.pt', {'type': 'adam'}),
        "FNO": (fno_model, 'fno', 'fno_model.pt', {'type': 'adamw', 'weight_decay': 1e-4})
    }

    return models_dict


def create_optimizers(models_dict, learning_rate=1e-3):
    """
    Create optimizers for models with exact configurations from old scripts

    Args:
        models_dict: Dictionary of models with optimizer configurations
        learning_rate: Learning rate (default 1e-3 from old scripts)

    Returns:
        Dictionary mapping model names to optimizers
    """
    optimizers = {}

    for name, (model, _model_type, _path, optim_info) in models_dict.items():
        if optim_info['type'] == 'adam':
            # Use AdamW but with Adam-like parameters
            optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=optim_info.get('weight_decay', 1e-4)
            )
        elif optim_info['type'] == 'adamw':
            optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=optim_info.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optim_info['type']}")

        optimizers[name] = optimizer

    return optimizers


def save_models(models_dict, checkpoints_dir):
    """
    Save model checkpoints

    Args:
        models_dict: Dictionary of models
        checkpoints_dir: Directory to save checkpoints
    """
    os.makedirs(checkpoints_dir, exist_ok=True)

    for name, (model, _, filename, _) in models_dict.items():
        path = os.path.join(checkpoints_dir, filename)
        torch.save(model.state_dict(), path)
        print(f"Saved {name} model to {path}")


def load_models(models_dict, checkpoints_dir, device):
    """
    Load model checkpoints

    Args:
        models_dict: Dictionary of models
        checkpoints_dir: Directory containing checkpoints
        device: Device to load models on
    """
    for name, (model, _, filename, _) in models_dict.items():
        path = os.path.join(checkpoints_dir, filename)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
            model.eval()
            print(f"Loaded {name} model from {path}")
        else:
            print(f"Warning: Model checkpoint not found for {name} at {path}")

    return models_dict
