"""
1D Hybrid Implementation

This module provides the Hybrid model implementation for 1D PDEs.
"""

import numpy as np
import torch
import torch.nn as nn
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.embeddings import GridEmbeddingND


class Hybrid(nn.Module):
    """Hybrid Operator between FNO and LOCO.

    Now supports an optional **grid positional embedding** (same idea as in
    `neuralop.models.FNO`). When enabled, two extra channels containing the
    1-D grid coordinates are concatenated to the input before the lifting
    layer.  This provides the model with explicit spatial information instead
    of forcing it to infer position implicitly.
    """

    def __init__(self, channels, hidden_channels, num_blocks, modes=None,
                 activation='gelu', use_mlp=True, use_positional_embedding=True):
        super().__init__()

        self.use_positional_embedding = use_positional_embedding

        if self.use_positional_embedding:
            # 1-D positional embedding adds *one* extra channel (the coordinate)
            self.positional_embedding = GridEmbeddingND(in_channels=channels, dim=1,
                                                       grid_boundaries=[[0.0, 1.0]])
            lifting_in_channels = channels + 1  # original channels + x-coord channel
        else:
            self.positional_embedding = None
            lifting_in_channels = channels

        # Lifting
        self.lifting = ChannelMLP(in_channels=lifting_in_channels,
                                  out_channels=hidden_channels,
                                  hidden_channels=hidden_channels * 2,
                                  n_layers=2, n_dim=1)

        # Spectral blocks
        self.blocks = nn.ModuleList([
            FactorizedSpectralBlock(hidden_channels, modes, activation, use_mlp)
            for _ in range(num_blocks)
        ])

        # Projection
        self.projection = ChannelMLP(in_channels=hidden_channels,
                                     out_channels=channels,
                                     hidden_channels=hidden_channels * 2,
                                     n_layers=2, n_dim=1)

    def forward(self, u):
        # u shape: [batch, spatial, channels]

        # Move channels first to satisfy ChannelMLP expectation
        u = u.transpose(1, 2)  # [batch, channels, spatial]

        # Append grid coordinates if requested
        if self.positional_embedding is not None:
            u = self.positional_embedding(u)  # now channels = orig + 1

        # Lifting to hidden dimension
        x = self.lifting(u)

        # Bring spatial dim to middle for spectral blocks: [batch, channels, spatial] -> [batch, spatial, channels]
        x = x.transpose(1, 2)

        for block in self.blocks:
            x = block(x)

        # Back to ChannelMLP format and project to output channels
        x = x.transpose(1, 2)
        x = self.projection(x)

        # Return to original ordering [batch, spatial, channels]
        x = x.transpose(1, 2)
        return x

    def count_params(self):
        """Count the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FactorizedSpectralBlock(nn.Module):
    """
    A spectral block that factorizes the channel dimension, sending one half
    through a full SpectralConvolutionBlock and the other half through a
    standard FNOSpectralConvBlock.
    """
    def __init__(self, hidden_channels, modes=None, activation='gelu', use_mlp=True):
        super().__init__()
        if hidden_channels % 2 != 0:
            raise ValueError("hidden_channels must be divisible by 2 for FactorizedSpectralBlock")

        self.hidden_channels = hidden_channels
        self.half_channels = hidden_channels // 2
        modes = modes if modes is not None else 64

        self.loco_block = SpectralConvolutionBlock(
            self.half_channels, modes, activation, use_mlp=False, full_block=False
        )

        self.fno_block = FNOSpectralConvBlock(
            self.half_channels, modes, activation
        )

        self.skip_connection = nn.Conv1d(hidden_channels, hidden_channels, 1)

        if activation == 'gelu':
            self.sigma = nn.GELU()
        else:
            self.sigma = activation

        self.use_mlp = use_mlp
        if use_mlp:
            self.channel_mlp = ChannelMLP(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                hidden_channels=hidden_channels, # No expansion
                n_layers=1,
                n_dim=1
            )

    def forward(self, u):
        u_loco_in = u[..., :self.half_channels]
        u_fno_in = u[..., self.half_channels:]

        out_loco = self.loco_block(u_loco_in)
        out_fno = self.fno_block(u_fno_in)

        spectral_out = torch.cat([out_loco, out_fno], dim=-1)
        skip_out = self.skip_connection(u.transpose(1, 2)).transpose(1, 2)

        activated = self.sigma(spectral_out + skip_out)

        if self.use_mlp:
            output = self.channel_mlp(activated.transpose(1, 2)).transpose(1, 2)
        else:
            output = activated

        return output


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


class FNOSpectralConvBlock(nn.Module):
    """
    A simplified spectral convolution block that performs a standard FNO operation.
    It applies learnable weights in the frequency domain, without the
    more complex non-linear convolution of the main SpectralConvolutionBlock.
    """
    def __init__(self, hidden_channels, modes, activation='gelu'):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.modes = modes

        self.spectral_weights = nn.Parameter(
            torch.randn(self.modes, hidden_channels, hidden_channels, dtype=torch.cfloat) * 0.01
        )

    def forward(self, u):
        u_hat = torch.fft.fft(u.to(torch.cfloat), dim=1)

        modes_to_use = min(self.modes, u.shape[1])
        u_hat_truncated = u_hat[:, :modes_to_use, :]

        f_u_hat = torch.einsum("bmh,mhH->bmH", u_hat_truncated, self.spectral_weights[:modes_to_use])

        padded_result = torch.zeros(u.shape[0], u.shape[1], self.hidden_channels, dtype=torch.cfloat, device=u.device)
        padded_result[:, :modes_to_use, :] = f_u_hat

        f_u = torch.fft.ifft(padded_result, dim=1).real

        return f_u

def create_burgers_hybrid(
    hidden_channels=24,
    num_blocks=4,
    modes=16,
    use_mlp=True
):
    """
    Create Hybrid model with Burgers equation configuration
    """
    return Hybrid(
        channels=1,  # CHANNELS = 1 for Burgers
        hidden_channels=hidden_channels,  # HIDDEN_CHANNELS = 24
        num_blocks=num_blocks,  # NUM_BLOCKS = 4
        modes=modes,  # MODES = 16
        use_mlp=use_mlp  # USE_SPECTRAL_CHANNEL_MLP
    )


def create_kdv_hybrid(
    hidden_channels=26,
    num_blocks=4,
    modes=16,
    use_mlp=True
):
    """
    Create Hybrid model with KdV equation configuration
    """
    return Hybrid(
        channels=1,  # CHANNELS = 1 for KdV
        hidden_channels=hidden_channels,  # HIDDEN_CHANNELS = 26
        num_blocks=num_blocks,  # NUM_BLOCKS = 4
        modes=modes,  # MODES = 16
        use_mlp=use_mlp  # USE_SPECTRAL_CHANNEL_MLP = False for KdV
    )
