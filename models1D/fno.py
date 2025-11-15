"""
1D Fourier Neural Operator Implementation

This module provides a clean 1D FNO implementation using the neuraloperator library,
adapted from the usage patterns in old_scripts/Burgers_KdV/.
"""

import torch.nn as nn
from neuralop.models import FNO as NeuralOpFNO


class FNO(nn.Module):
    """
    1D Fourier Neural Operator

    Wrapper around the neuraloperator library's FNO implementation,
    configured for 1D PDE problems (Burgers, KdV equations).

    Args:
        modes: Number of Fourier modes to keep
        in_channels: Number of input channels
        out_channels: Number of output channels
        hidden_channels: Hidden channel width
        n_layers: Number of FNO layers
        use_mlp: Whether to use MLP in spectral convolutions
    """

    def __init__(
        self,
        modes=16,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
        use_mlp=True
    ):
        super().__init__()

        # Use neuraloperator FNO with 1D configuration
        self.fno = NeuralOpFNO(
            n_modes=(modes,),  # 1D modes tuple
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            use_mlp=use_mlp
        )

        # Store configuration for compatibility
        self.modes = modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

    def forward(self, x):
        """
        Forward pass through 1D FNO

        Args:
            x: Input tensor of shape [batch, spatial, channels] or [batch, channels, spatial]

        Returns:
            Output tensor of same shape as input
        """
        return self.fno(x)

    def count_params(self):
        """Count the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_burgers_fno(
    modes=16,
    hidden_channels=32,  # FNO_HIDDEN_CHANNELS = 32 from old_scripts
    n_layers=4,
    use_mlp=True  # USE_SPECTRAL_CHANNEL_MLP = True for Burgers
):
    """
    Create FNO model with Burgers equation configuration

    Exact parameters from old_scripts/Burgers_KdV/burgers.py:
    - MODES = 16
    - CHANNELS = 1
    - FNO_HIDDEN_CHANNELS = 32
    - NUM_BLOCKS = 4
    - USE_SPECTRAL_CHANNEL_MLP = True
    """
    return FNO(
        modes=modes,
        in_channels=1,  # CHANNELS = 1 for Burgers
        out_channels=1,
        hidden_channels=hidden_channels,  # FNO_HIDDEN_CHANNELS = 32
        n_layers=n_layers,  # NUM_BLOCKS = 4
        use_mlp=use_mlp  # USE_SPECTRAL_CHANNEL_MLP = True
    )


def create_kdv_fno(
    modes=16,
    hidden_channels=32,  # FNO_HIDDEN_CHANNELS = 32 from old_scripts
    lifting_channel_ratio=2,  # FNO_LIFTING_CHANNEL_RATIO = 2
    projection_channel_ratio=1,  # FNO_PROJECTION_CHANNEL_RATIO = 1
    n_layers=4,
    use_mlp=True  # USE_SPECTRAL_CHANNEL_MLP = False for KdV
):
    """
    Create FNO model with KdV equation configuration

    Exact parameters from old_scripts/Burgers_KdV/KdV.py:
    - MODES = 16
    - CHANNELS = 1
    - FNO_HIDDEN_CHANNELS = 32
    - FNO_LIFTING_CHANNEL_RATIO = 2
    - FNO_PROJECTION_CHANNEL_RATIO = 1
    - NUM_BLOCKS = 4
    - USE_SPECTRAL_CHANNEL_MLP = True
    """
    return NeuralOpFNO(
        n_modes=(modes,),
        in_channels=1,  # CHANNELS = 1 for KdV
        out_channels=1,
        hidden_channels=hidden_channels,  # FNO_HIDDEN_CHANNELS = 32
        lifting_channel_ratio=lifting_channel_ratio,
        projection_channel_ratio=projection_channel_ratio,
        n_layers=n_layers,  # NUM_BLOCKS = 4
        positional_embedding='grid',
        use_mlp=use_mlp,  # USE_SPECTRAL_CHANNEL_MLP = False
        mlp_dropout=0,
        mlp_expansion=1,
        non_linearity=nn.GELU(),
        stabilizer=None
    )
