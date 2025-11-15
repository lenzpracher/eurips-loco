"""
Hybrid NFNO-SNO model implementation for 1D and 2D PDEs.

This module contains a hybrid neural operator that combines Fourier Neural Operator (FNO)
and Spectral Neural Operator (SNO) approaches by splitting the hidden channels between
FNO and SNO branches within each block.
"""

import torch
import torch.nn as nn
import torch.fft
import math
from typing import Optional

# Import the individual components from our FNO and SNO implementations
from .fno import FourierConv1D, FourierConv2D, GridEmbedding1D, GridEmbedding2D
from .loco import SpectralConv1D, SpectralConv2D


class HybridBlock1D(nn.Module):
    """Hybrid FNO-SNO block for 1D problems."""
    
    def __init__(self, hidden_channels: int, modes: int = 16, expansion_ratio: float = 2.0):
        super().__init__()
        
        if hidden_channels % 2 != 0:
            raise ValueError("hidden_channels must be divisible by 2 for hybrid model")
            
        half_channels = hidden_channels // 2
        
        # Split channels between FNO and LOCO
        self.fno_conv = FourierConv1D(half_channels, half_channels, modes)
        self.loco_conv = SpectralConv1D(half_channels, half_channels, modes)
        
        # Skip connection and normalization
        self.skip_conv = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.gate = nn.Parameter(torch.full((1, hidden_channels, 1), -2.0))
        self.norm1 = nn.InstanceNorm1d(hidden_channels, affine=True)
        
        # MLP
        expanded = int(hidden_channels * expansion_ratio)
        self.mlp = nn.Sequential(
            nn.Conv1d(hidden_channels, expanded, 1),
            nn.GELU(),
            nn.Conv1d(expanded, hidden_channels, 1)
        )
        self.mlp_gate = nn.Parameter(torch.full((1, hidden_channels, 1), -2.0))
        self.norm2 = nn.InstanceNorm1d(hidden_channels, affine=True)
        self.act = nn.GELU()
        
    def forward(self, x):
        identity = x
        
        # Split channels between FNO and LOCO branches
        x_fno, x_loco = torch.split(x, x.shape[1] // 2, dim=1)
        
        # Apply FNO and LOCO convolutions separately
        x_fno_out = self.fno_conv(x_fno)
        x_loco_out = self.loco_conv(x_loco)
        
        # Concatenate results
        x_spec = torch.cat([x_fno_out, x_loco_out], dim=1)
        
        # Apply normalization and skip connection
        x_proc = self.act(self.norm1(x_spec + torch.sigmoid(self.gate) * self.skip_conv(x)))
        
        # MLP with gated skip connection  
        x_mlp = self.mlp(x_proc)
        x_proc = self.act(self.norm2(x_mlp + torch.sigmoid(self.mlp_gate) * x_proc))
        
        # Residual connection
        return identity + x_proc


class HybridBlock2D(nn.Module):
    """Hybrid FNO-LOCO block for 2D problems."""
    
    def __init__(self, hidden_channels: int, modes_x: int = 16, modes_y: int = 16, expansion_ratio: float = 2.0):
        super().__init__()
        
        if hidden_channels % 2 != 0:
            raise ValueError("hidden_channels must be divisible by 2 for hybrid model")
            
        half_channels = hidden_channels // 2
        
        # Split channels between FNO and LOCO
        self.fno_conv = FourierConv2D(half_channels, half_channels, modes_x, modes_y)
        self.loco_conv = SpectralConv2D(half_channels, half_channels, modes_x, modes_y)
        
        # Skip connection and normalization
        self.skip_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.gate = nn.Parameter(torch.full((1, hidden_channels, 1, 1), -2.0))
        self.norm1 = nn.InstanceNorm2d(hidden_channels, affine=True)
        
        # MLP
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
        
        # Split channels between FNO and LOCO branches
        x_fno, x_loco = torch.split(x, x.shape[1] // 2, dim=1)
        
        # Apply FNO and LOCO convolutions separately
        x_fno_out = self.fno_conv(x_fno)
        x_loco_out = self.loco_conv(x_loco)
        
        # Concatenate results
        x_spec = torch.cat([x_fno_out, x_loco_out], dim=1)
        
        # Apply normalization and skip connection
        x_proc = self.act(self.norm1(x_spec + torch.sigmoid(self.gate) * self.skip_conv(x)))
        
        # MLP with gated skip connection  
        x_mlp = self.mlp(x_proc)
        x_proc = self.act(self.norm2(x_mlp + torch.sigmoid(self.mlp_gate) * x_proc))
        
        # Residual connection
        return identity + x_proc


class HybridOperator(nn.Module):
    """
    Hybrid FNO-LOCO Neural Operator for 1D and 2D PDEs.
    
    This model combines the strengths of both FNO and LOCO by splitting the hidden channels
    between FNO and LOCO branches within each block. The FNO branch handles linear spectral
    mixing while the LOCO branch provides spectral domain nonlinearity.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        hidden_channels: Number of hidden channels (must be even for channel splitting)
        num_blocks: Number of hybrid blocks
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
        
        if hidden_channels % 2 != 0:
            raise ValueError("hidden_channels must be even for hybrid FNO-LOCO model")
        
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
        
        # Hybrid blocks
        if self.is_2d:
            self.blocks = nn.ModuleList([
                HybridBlock2D(hidden_channels, modes_x, modes_y) for _ in range(num_blocks)
            ])
        else:
            self.blocks = nn.ModuleList([
                HybridBlock1D(hidden_channels, modes_x) for _ in range(num_blocks)
            ])
        
        # Projection layer
        self.proj = conv_layer(hidden_channels, out_channels, 1)
        
    def forward(self, x):
        # Add positional embedding if enabled
        if self.pos_embed is not None:
            x = self.pos_embed(x)
            
        # Lift to hidden dimension
        x = self.lift(x)
        
        # Apply hybrid blocks
        for block in self.blocks:
            x = block(x)
            
        # Project to output dimension
        x = self.proj(x)
        return x


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Convenience functions for creating models
def create_hybrid_1d(in_channels=1, out_channels=1, hidden_channels=32, num_blocks=4, modes=16):
    """
    Create a 1D Hybrid FNO-LOCO model with standard parameters.
    
    Note: hidden_channels will be rounded to nearest even number if odd.
    """
    if hidden_channels % 2 != 0:
        hidden_channels = hidden_channels + 1
        print(f"Warning: Adjusted hidden_channels to {hidden_channels} (must be even for hybrid model)")
        
    return HybridOperator(
        in_channels=in_channels,
        out_channels=out_channels, 
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        modes_x=modes,
        modes_y=None
    )


def create_hybrid_2d(in_channels=1, out_channels=1, hidden_channels=32, num_blocks=4, modes_x=16, modes_y=16):
    """
    Create a 2D Hybrid FNO-LOCO model with standard parameters.
    
    Note: hidden_channels will be rounded to nearest even number if odd.
    """
    if hidden_channels % 2 != 0:
        hidden_channels = hidden_channels + 1
        print(f"Warning: Adjusted hidden_channels to {hidden_channels} (must be even for hybrid model)")
        
    return HybridOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels, 
        num_blocks=num_blocks,
        modes_x=modes_x,
        modes_y=modes_y
    )