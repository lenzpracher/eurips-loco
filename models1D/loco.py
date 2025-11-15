"""
1D LOCO Implementation

This module provides the LOCO (Local Convolution Operator) model implementation for 1D PDEs.
"""

import numpy as np
import torch
import torch.nn as nn
from neuralop.layers.channel_mlp import ChannelMLP


class SpectralConvolutionBlock(nn.Module):
    """
    Spectral convolution block with learnable weights in frequency domain (FNO-comparable):

    1. Input is already in hidden channel space
    2. Skip connection (save input)
    3. FFT(u) -> û
    4. Apply learnable weights W in frequency domain: W*û
    5. Apply nonlinearity: σ(W*û)
    6. Spectral convolution: û ⊛ σ(W*û) (pointwise multiplication in frequency domain)
    7. IFFT back to spatial domain
    8. Skip connection [hidden_channels, hidden_channels]
    9. Optional Channel MLP
    10. Add skip connection + nonlinearity
    """

    def __init__(self, hidden_channels, modes=None, activation='gelu', use_mlp=True, full_block=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.modes = modes if modes is not None else 64
        self.use_mlp = use_mlp
        self.full_block = full_block

        self.spectral_weights = nn.Parameter(
            torch.randn(self.modes, hidden_channels, hidden_channels, dtype=torch.cfloat) * 0.01
        )

        if self.full_block:
            self.skip_connection = nn.Conv1d(hidden_channels, hidden_channels, 1)
            if use_mlp:
                self.channel_mlp = ChannelMLP(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    hidden_channels=hidden_channels // 2,
                    n_layers=2,
                    n_dim=1
                )

        if activation == 'gelu':
            self.sigma = nn.GELU()
        else:
            self.sigma = activation

    def spectral_conv(self, u):
        batch_size, spatial_points, _ = u.shape
        u_hat = torch.fft.fft(u.to(torch.cfloat), dim=1)

        modes_to_use = min(self.modes, spatial_points)
        u_hat_truncated = u_hat[:, :modes_to_use, :]

        Wu_hat = torch.einsum("bmh,mhH->bmH", u_hat_truncated, self.spectral_weights[:modes_to_use])
        sigma_Wu_hat = self.sigma(Wu_hat.real) + 1j * self.sigma(Wu_hat.imag)

        u_hat_fft = torch.fft.fft(u_hat_truncated, dim=1)
        sigma_Wu_hat_fft = torch.fft.fft(sigma_Wu_hat, dim=1)
        conv_result_hat = torch.fft.ifft(u_hat_fft * sigma_Wu_hat_fft / np.sqrt(modes_to_use), dim=1)

        padded_result = torch.zeros(batch_size, spatial_points, self.hidden_channels,
                                  dtype=conv_result_hat.dtype, device=conv_result_hat.device)
        padded_result[:, :modes_to_use, :] = conv_result_hat

        return torch.fft.ifft(padded_result, dim=1).real

    def forward(self, u):
        if not self.full_block:
            return self.spectral_conv(u)

        u_skip = u
        conv_result = self.spectral_conv(u)

        skip_result = self.skip_connection(u_skip.transpose(1, 2)).transpose(1, 2)

        activated = self.sigma(conv_result + skip_result)

        if self.use_mlp:
            output = self.channel_mlp(activated.transpose(1, 2)).transpose(1, 2)
        else:
            output = activated

        return output


class LOCO(nn.Module):
    """
    LOCO Neural Operator Architecture
    """

    def __init__(self, channels, hidden_channels, num_blocks, modes=None, activation='gelu', use_mlp=True):
        super().__init__()

        # Lifting layer (like FNO, using ChannelMLP)
        self.lifting = ChannelMLP(
            in_channels=channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels * 2, # Standard FNO lifting ratio
            n_layers=2,
            n_dim=1
        )

        # Spectral blocks
        self.blocks = nn.ModuleList([
            SpectralConvolutionBlock(hidden_channels, modes, activation, use_mlp)
            for _ in range(num_blocks)
        ])

        # Projection layer (like FNO, using ChannelMLP)
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=channels,
            hidden_channels=hidden_channels * 2, # Standard FNO projection ratio
            n_layers=2,
            n_dim=1
        )

    def forward(self, u):
        """
        Apply lifting -> spectral blocks -> projection
        """
        # ChannelMLP expects [batch, channels, spatial], LOCO expects [batch, spatial, channels]
        u = u.transpose(1, 2)

        # Lift to hidden channels
        x = self.lifting(u)

        # Transpose back to LOCO's native [batch, spatial, channels] format for the spectral blocks
        x = x.transpose(1, 2)

        # Apply spectral blocks
        for block in self.blocks:
            x = block(x)

        # Transpose for the projection layer
        x = x.transpose(1, 2)

        # Project back to output channels
        x = self.projection(x)

        # Transpose back to the original [batch, spatial, channels] format
        x = x.transpose(1, 2)

        return x

    def count_params(self):
        """Count the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_burgers_loco(
    hidden_channels=24,
    num_blocks=4,
    modes=16,
    use_mlp=True
):
    """
    Create LOCO model with Burgers equation configuration
    """
    return LOCO(
        channels=1,  # CHANNELS = 1 for Burgers
        hidden_channels=hidden_channels,  # HIDDEN_CHANNELS = 24
        num_blocks=num_blocks,  # NUM_BLOCKS = 4
        modes=modes,  # MODES = 16
        use_mlp=use_mlp  # USE_SPECTRAL_CHANNEL_MLP
    )


def create_kdv_loco(
    hidden_channels=26,
    num_blocks=4,
    modes=16,
    use_mlp=True
):
    """
    Create LOCO model with KdV equation configuration
    """
    return LOCO(
        channels=1,  # CHANNELS = 1 for KdV
        hidden_channels=hidden_channels,  # HIDDEN_CHANNELS = 26
        num_blocks=num_blocks,  # NUM_BLOCKS = 4
        modes=modes,  # MODES = 16
        use_mlp=use_mlp  # USE_SPECTRAL_CHANNEL_MLP = False for KdV
    )
