"""
Fourier Neural Operator (FNO) implementation for 1D and 2D PDEs.

This module contains a unified FNO implementation that can handle both 
1D and 2D problems through automatic dimension detection. It uses standard
Fourier transforms without the spectral domain nonlinearity of LOCO.
"""

import torch
import torch.nn as nn
import torch.fft
import math
from typing import Optional


class FourierConv1D(nn.Module):
    """1D Fourier convolution layer for FNO - matches FNOSpectralConvBlock from models.py."""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Learnable weights in spectral domain - same initialization as FNOSpectralConvBlock
        self.spectral_weights = nn.Parameter(
            torch.randn(self.modes, in_channels, out_channels, dtype=torch.cfloat) * 0.01
        )
        
    def forward(self, x):
        # x: [batch, channels, spatial] -> convert to [batch, spatial, channels] like FNOSpectralConvBlock
        u = x.transpose(1, 2)  # [batch, spatial, channels]
        
        # Exact implementation from FNOSpectralConvBlock
        u_hat = torch.fft.fft(u.to(torch.cfloat), dim=1)
        
        modes_to_use = min(self.modes, u.shape[1])
        u_hat_truncated = u_hat[:, :modes_to_use, :]
        
        f_u_hat = torch.einsum("bmh,mhH->bmH", u_hat_truncated, self.spectral_weights[:modes_to_use])

        padded_result = torch.zeros(u.shape[0], u.shape[1], self.out_channels, dtype=torch.cfloat, device=u.device)
        padded_result[:, :modes_to_use, :] = f_u_hat

        f_u = torch.fft.ifft(padded_result, dim=1).real
        
        # Convert back to [batch, channels, spatial]
        return f_u.transpose(1, 2)


class FourierConv2D(nn.Module):
    """2D Fourier convolution layer for FNO."""
    
    def __init__(self, in_channels: int, out_channels: int, modes_x: int = 16, modes_y: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        
        # Learnable weights in spectral domain
        std = 1.0 / (math.sqrt(in_channels) * math.sqrt(modes_x * modes_y))
        self.weight = nn.Parameter(
            torch.randn(modes_x, modes_y, in_channels, out_channels, dtype=torch.cfloat) * std
        )
        
    def forward(self, x):
        # x: [batch, channels, height, width]
        B, C, H, W = x.shape
        
        # FFT to spectral domain
        x_ft = torch.fft.rfft2(x, norm='ortho')
        
        # Truncate to low modes and apply weights
        out_ft = torch.zeros(B, self.out_channels, H, W//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes_x, :self.modes_y] = torch.einsum(
            'bcxy,xyco->boxy', 
            x_ft[:, :, :self.modes_x, :self.modes_y], 
            self.weight
        )
        
        # Inverse FFT back to spatial domain
        return torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')


class FNOBlock(nn.Module):
    """FNO block with Fourier convolution and MLP."""
    
    def __init__(self, hidden_channels: int, modes: int = 16, modes_y: Optional[int] = None, expansion_ratio: float = 2.0):
        super().__init__()
        self.is_2d = modes_y is not None
        
        # Fourier convolution
        if self.is_2d:
            self.fourier_conv = FourierConv2D(hidden_channels, hidden_channels, modes, modes_y)
            conv_layer = nn.Conv2d
            norm_layer = nn.InstanceNorm2d
        else:
            self.fourier_conv = FourierConv1D(hidden_channels, hidden_channels, modes)
            conv_layer = nn.Conv1d
            norm_layer = nn.InstanceNorm1d
            
        # Skip connection
        self.skip_conv = conv_layer(hidden_channels, hidden_channels, 1)
        self.gate = nn.Parameter(torch.full((1, hidden_channels) + (1,) * (2 if self.is_2d else 1), -2.0))
        self.norm1 = norm_layer(hidden_channels, affine=True)
        
        # MLP
        expanded = int(hidden_channels * expansion_ratio)
        self.mlp = nn.Sequential(
            conv_layer(hidden_channels, expanded, 1),
            nn.GELU(),
            conv_layer(expanded, hidden_channels, 1)
        )
        self.mlp_gate = nn.Parameter(torch.full((1, hidden_channels) + (1,) * (2 if self.is_2d else 1), -2.0))
        self.norm2 = norm_layer(hidden_channels, affine=True)
        self.act = nn.GELU()
        
    def forward(self, x):
        identity = x
        
        # Fourier convolution with gated skip connection
        x_fourier = self.fourier_conv(x)
        x_proc = self.act(self.norm1(x_fourier + torch.sigmoid(self.gate) * self.skip_conv(x)))
        
        # MLP with gated skip connection  
        x_mlp = self.mlp(x_proc)
        x_proc = self.act(self.norm2(x_mlp + torch.sigmoid(self.mlp_gate) * x_proc))
        
        # Residual connection
        return identity + x_proc


class FourierNeuralOperator(nn.Module):
    """
    Unified Fourier Neural Operator for 1D and 2D PDEs.
    
    Automatically detects input dimensionality and applies appropriate Fourier convolutions.
    This is the standard FNO without spectral domain nonlinearity.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        hidden_channels: Number of hidden channels
        num_blocks: Number of FNO blocks
        modes_x: Number of Fourier modes in x-direction (or only direction for 1D)
        modes_y: Number of Fourier modes in y-direction (None for 1D)
        use_positional_embedding: Whether to add positional encoding
    """
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        hidden_channels: int,
        num_blocks: int,
        modes_x: int = 16,
        modes_y: Optional[int] = None,
        use_positional_embedding: bool = True
    ):
        super().__init__()
        self.is_2d = modes_y is not None
        self.use_pos_embed = use_positional_embedding
        
        # Positional embedding (optional)
        if use_positional_embedding:
            if self.is_2d:
                self.pos_embed = GridEmbedding2D()
                in_channels += 2
            else:
                self.pos_embed = GridEmbedding1D()
                in_channels += 1
        else:
            self.pos_embed = None
            
        # Lifting layer
        conv_layer = nn.Conv2d if self.is_2d else nn.Conv1d
        self.lift = conv_layer(in_channels, hidden_channels, 1)
        
        # FNO blocks
        self.blocks = nn.ModuleList([
            FNOBlock(hidden_channels, modes_x, modes_y) for _ in range(num_blocks)
        ])
        
        # Projection layer
        self.proj = conv_layer(hidden_channels, out_channels, 1)
        
    def forward(self, x):
        # Add positional embedding if enabled
        if self.pos_embed is not None:
            x = self.pos_embed(x)
            
        # Lift to hidden dimension
        x = self.lift(x)
        
        # Apply FNO blocks
        for block in self.blocks:
            x = block(x)
            
        # Project to output dimension
        x = self.proj(x)
        return x


class GridEmbedding1D(nn.Module):
    """1D positional encoding."""
    
    def forward(self, x):
        # x: [batch, channels, spatial]
        B, C, N = x.shape
        
        # Create spatial grid
        grid = torch.linspace(0, 1, N, device=x.device, dtype=x.dtype)
        grid = grid.expand(B, 1, N)
        
        return torch.cat([x, grid], dim=1)


class GridEmbedding2D(nn.Module):
    """2D positional encoding."""
    
    def forward(self, x):
        # x: [batch, channels, height, width]
        B, C, H, W = x.shape
        
        # Create spatial grids
        grid_y = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype)
        grid_x = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        
        # Expand to batch dimension
        grid_y = grid_y.expand(B, 1, H, W)
        grid_x = grid_x.expand(B, 1, H, W)
        
        return torch.cat([x, grid_y, grid_x], dim=1)


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Convenience functions for creating models
def create_fno_1d(in_channels=1, out_channels=1, hidden_channels=32, num_blocks=4, modes=16):
    """Create a 1D FNO model with standard parameters."""
    return FourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels, 
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        modes_x=modes,
        modes_y=None
    )


def create_fno_2d(in_channels=1, out_channels=1, hidden_channels=32, num_blocks=4, modes_x=16, modes_y=16):
    """Create a 2D FNO model with standard parameters.""" 
    return FourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels, 
        num_blocks=num_blocks,
        modes_x=modes_x,
        modes_y=modes_y
    )