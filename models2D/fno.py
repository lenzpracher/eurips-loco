"""
2D Fourier Neural Operator Implementation

This module provides the 2D FNO implementation extracted from
old_scripts/NS2D/models2D.py for Navier-Stokes equation experiments.
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


class NFNOSpectralConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x=16, modes_y=16):
        super().__init__()
        # Smaller init to compensate for quadratic energy build-up
        std = 1.0 / (math.sqrt(in_channels) * math.sqrt(modes_x * modes_y))
        self.weight = nn.Parameter(
            torch.randn(modes_x, modes_y, in_channels, out_channels, dtype=torch.cfloat) * std
        )
        self.modes_x = modes_x
        self.modes_y = modes_y

    def _c_mul(self, x, w):
        # x: [B, in_c, Mx, My]  w: [Mx, My, in_c, out_c] -> [B, out_c, Mx, My]
        return torch.einsum("bcxy,xyco->boxy", x, w)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')
        out_ft = torch.zeros(B, self.weight.shape[-1], H, W//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, : self.modes_x, : self.modes_y] = self._c_mul(x_ft[:, :, : self.modes_x, : self.modes_y], self.weight)
        return torch.fft.irfft2(out_ft, s=(H,W), norm='ortho')


class FNOBlock2D(nn.Module):
    def __init__(self, hidden_channels, modes_x=16, modes_y=16, expansion_ratio=2, use_loco=False):
        super().__init__()
        self.use_loco = use_loco

        if self.use_loco and hidden_channels % 2 != 0:
            raise ValueError("hidden_channels must be divisible by 2 when use_sno=True")

        if self.use_loco:
            half = hidden_channels // 2
            self.sno_conv = LOCOSpectralConv2D(half, half, modes_x, modes_y)
            self.fourier = NFNOSpectralConv2D(half, half, modes_x, modes_y)
        else:
            self.fourier = NFNOSpectralConv2D(hidden_channels, hidden_channels, modes_x, modes_y)

        self.skip_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        # Start gate bias at -2 → initial sigmoid ≈ 0.12 (mostly skip connection)
        self.gate = nn.Parameter(torch.full((1, hidden_channels, 1, 1), -2.0))
        self.norm1 = nn.InstanceNorm2d(hidden_channels, affine=True)

        expanded = int(hidden_channels*expansion_ratio)
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
        if self.use_loco:
            x_sno, x_fno = torch.split(x, x.shape[1] // 2, dim=1)
            x_spec = torch.cat([
                self.sno_conv(x_sno),
                self.fourier(x_fno)
            ], dim=1)
        else:
            x_spec = self.fourier(x)

        x_proc = self.act(self.norm1(x_spec + torch.sigmoid(self.gate)*self.skip_conv(x)))
        x_mlp = self.mlp(x_proc)
        x_proc = self.act(self.norm2(x_mlp + torch.sigmoid(self.mlp_gate)*x_proc))
        return identity + x_proc

class FNO(nn.Module):
    """
    FNO: NFNO architecture with SNO turned off (use_sno=False).

    This is identical to Hybrid but uses only FNO blocks without any SNO branch.
    Uses linear Conv2d lifting and projection layers like the original NFNO2D.
    Perfect for comparing pure FNO vs hybrid FNO-SNO approaches.
    """
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=32, modes_x=16, modes_y=16, n_layers=4):
        super().__init__()
        self.use_pos = True
        if self.use_pos:
            self.pos_embed = GridEmbeddingND(in_channels=in_channels, dim=2, grid_boundaries=[[0,1],[0,1]])
            in_channels += 2

        # Linear lifting layer (same as NFNO2D)
        self.lift = nn.Conv2d(in_channels, hidden_channels, 1)

        # FNO blocks with SNO disabled (use_sno=False)
        self.blocks = nn.ModuleList([FNOBlock2D(hidden_channels, modes_x, modes_y, use_loco=False) for _ in range(n_layers)])

        # Linear projection layer (same as NFNO2D)
        self.proj = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x):
        if self.use_pos:
            x = self.pos_embed(x)
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)

    def count_params(self):
        """Count the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_ns2d_fno(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    modes_x=12,
    modes_y=12,
    n_layers=4
):
    """
    Create FNO model with NS2D configuration

    Based on parameters from old_scripts/NS2D/moNS2D.py
    """
    return FNO(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,  # HIDDEN_CHANNELS = 32
        modes_x=modes_x,  # MODES_X = 12
        modes_y=modes_y,  # MODES_Y = 12
        n_layers=n_layers  # NUM_BLOCKS = 4
    )
