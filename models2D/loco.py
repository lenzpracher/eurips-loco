"""
2D LOCO Implementation

This module provides the 2D LOCO (Local Convolution Operator) implementation
for Navier-Stokes equation experiments.
"""

import math

import torch
import torch.nn as nn
from neuralop.layers.embeddings import GridEmbeddingND


class LOCOSpectralConv2D(nn.Module):
    """
    2D LOCO convolution

    Steps:
    1. FFT to spectral domain and truncate to low modes
    2. Linear mixing in spectral space (learnable complex weights)
    3. Apply non-linearity to real & imag parts in spectral domain
    4. Perform FFT convolution using convolution theorem
    5. Zero-pad to full mode range and inverse FFT to spatial domain
    """

    def __init__(self, in_channels, out_channels, modes_x=16, modes_y=16, activation='gelu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        # Smaller init to compensate for quadratic energy build-up
        std = 1.0 / (math.sqrt(in_channels) * math.sqrt(modes_x * modes_y))
        self.weight = nn.Parameter(
            torch.randn(modes_x, modes_y, in_channels, out_channels, dtype=torch.cfloat) * std
        )

        self.act = nn.GELU() if activation == 'gelu' else activation

    def forward(self, u):
        B, C, H, W = u.shape

        # 1) FFT to spectral domain
        u_hat = torch.fft.fft2(u.to(torch.cfloat), dim=[2, 3], norm='ortho')  # B,C,H,Wc

        # 2) keep low modes and apply linear mixing
        u_low = u_hat[:, :, : self.modes_x, : self.modes_y]  # B,C,mx,my
        Wu_hat = torch.einsum("bcxy,xyco->boxy", u_low, self.weight)  # B,O,mx,my

        # 3) non-linearity in spectral domain (separate real/imag)
        sigma_Wu_hat = self.act(Wu_hat.real) + 1j * self.act(Wu_hat.imag)

        # 4) 2-D FFT over mode grid again and convolution via element-wise mult
        u_low_fft = torch.fft.fft2(u_low, dim=[2, 3], norm='ortho')
        sigma_fft = torch.fft.fft2(sigma_Wu_hat, dim=[2, 3], norm='ortho')
        conv_hat = torch.fft.ifft2(u_low_fft * sigma_fft / math.sqrt(self.modes_x * self.modes_y), dim=[2, 3], norm='ortho')

        # 5) embed back into full spectrum (zero-pad) and iFFT to spatial domain
        out_full_hat = torch.zeros(B, self.out_channels, H, W, dtype=torch.cfloat, device=u.device)
        out_full_hat[:, :, : self.modes_x, : self.modes_y] = conv_hat

        out = torch.fft.ifft2(out_full_hat, dim=[2, 3], norm='ortho').real  # back to spatial domain
        return out


class LOCOBlock2D(nn.Module):
    """2D LOCO Block with spectral convolution and MLP"""

    def __init__(self, hidden_channels, modes_x=16, modes_y=16, expansion_ratio=2):
        super().__init__()

        self.loco_conv = LOCOSpectralConv2D(hidden_channels, hidden_channels, modes_x, modes_y)

        self.skip_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        # Start gate bias at -2 → initial sigmoid ≈ 0.12 (mostly skip connection)
        self.gate = nn.Parameter(torch.full((1, hidden_channels, 1, 1), -2.0))
        self.norm1 = nn.InstanceNorm2d(hidden_channels, affine=True)

        expanded = int(hidden_channels * expansion_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(hidden_channels, expanded, 1),
            nn.GELU(),
            nn.Conv2d(expanded, hidden_channels, 1)
        )
        self.mlp_gate = nn.Parameter(torch.full((1, hidden_channels, 1, 1), -2.0))
        self.norm2 = nn.InstanceNorm2d(hidden_channels, affine=True)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        x_spec = self.loco_conv(x)
        x_proc = self.act(self.norm1(x_spec + torch.sigmoid(self.gate) * self.skip_conv(x)))
        x_mlp = self.mlp(x_proc)
        x_proc = self.act(self.norm2(x_mlp + torch.sigmoid(self.mlp_gate) * x_proc))
        return identity + x_proc


class LOCO(nn.Module):
    """
    2D LOCO (Local Convolution Operator)

    Full LOCO model built from LOCOBlock2D with positional embedding.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        hidden_channels: Hidden channel width
        num_blocks: Number of LOCO blocks
        modes_x: Number of Fourier modes in x direction
        modes_y: Number of Fourier modes in y direction
        use_positional_embedding: Whether to use positional embedding
    """

    def __init__(self, in_channels, out_channels, hidden_channels, num_blocks, modes_x=16, modes_y=16, use_positional_embedding=True):
        super().__init__()

        if use_positional_embedding:
            self.pos_embed = GridEmbeddingND(in_channels=in_channels, dim=2, grid_boundaries=[[0,1],[0,1]])
            in_channels += 2
        else:
            self.pos_embed = None

        self.lift = nn.Conv2d(in_channels, hidden_channels, 1)

        self.blocks = nn.ModuleList([
            LOCOBlock2D(hidden_channels, modes_x, modes_y) for _ in range(num_blocks)
        ])

        self.proj = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, u):
        if self.pos_embed is not None:
            x = self.pos_embed(u)
        else:
            x = u
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.proj(x)
        return x

    def count_params(self):
        """Count the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_ns2d_loco(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    num_blocks=4,
    modes_x=12,
    modes_y=12,
    use_positional_embedding=True
):
    """
    Create LOCO model with NS2D configuration
    """
    return LOCO(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,  # HIDDEN_CHANNELS = 32
        num_blocks=num_blocks,  # NUM_BLOCKS = 4
        modes_x=modes_x,  # MODES_X = 12
        modes_y=modes_y,  # MODES_Y = 12
        use_positional_embedding=use_positional_embedding
    )
