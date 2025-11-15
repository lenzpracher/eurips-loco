"""
Local Operator (LOCO) implementation for 1D and 2D PDEs.

This module contains a unified LOCO implementation that can handle both 
1D and 2D problems through automatic dimension detection.

For 1D: LOCO is identical to SNO (Spectral Neural Operator)
For 2D: LOCO maintains its own implementation
"""

import torch
import torch.nn as nn
import torch.fft
import math
import numpy as np
from typing import Optional
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.embeddings import GridEmbeddingND


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


class SpectralConv1D(nn.Module):
    """1D Spectral convolution layer - wrapper around SpectralConvolutionBlock for consistency."""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int = 16):
        super().__init__()
        assert in_channels == out_channels, "SpectralConvolutionBlock requires in_channels == out_channels"
        self.spectral_block = SpectralConvolutionBlock(in_channels, modes, 'gelu', use_mlp=False, full_block=False)
        
    def forward(self, x):
        # Convert from [batch, channels, spatial] to [batch, spatial, channels] for SpectralConvolutionBlock
        x = x.transpose(1, 2)
        x = self.spectral_block(x)
        # Convert back to [batch, channels, spatial]
        return x.transpose(1, 2)


class SpectralConv2D(nn.Module):
    """2D Spectral convolution layer for LOCO."""
    
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
        # x: [batch, channels, height, width] -> convert to SNO format [batch, height, width, channels]
        B, C, H, W = x.shape
        u = x.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        
        # SNO-style 2D spectral convolution
        modes_x_to_use = min(self.modes_x, H)
        modes_y_to_use = min(self.modes_y, W)
        
        u_hat = torch.fft.fft2(u.to(torch.cfloat), dim=(1, 2))
        u_hat_truncated = u_hat[:, :modes_x_to_use, :modes_y_to_use, :]
        
        # Apply learnable weights: W*û
        Wu_hat = torch.einsum("bxyh,xyhH->bxyH", u_hat_truncated, self.weight[:modes_x_to_use, :modes_y_to_use])
        
        # Complex nonlinearity: σ(W*û)
        sigma_Wu_hat = torch.relu(Wu_hat.real) + 1j * torch.relu(Wu_hat.imag)
        
        # Double FFT for convolution in frequency domain
        u_hat_fft = torch.fft.fft2(u_hat_truncated, dim=(1, 2))
        sigma_Wu_hat_fft = torch.fft.fft2(sigma_Wu_hat, dim=(1, 2))
        
        # Spectral convolution: û ⊛ σ(W*û) with normalization
        conv_result_hat = torch.fft.ifft2(u_hat_fft * sigma_Wu_hat_fft / math.sqrt(modes_x_to_use * modes_y_to_use), dim=(1, 2))
        
        # Pad back to full size
        padded_result = torch.zeros(B, H, W, self.out_channels, dtype=conv_result_hat.dtype, device=conv_result_hat.device)
        padded_result[:, :modes_x_to_use, :modes_y_to_use, :] = conv_result_hat
        
        # IFFT back to spatial domain and convert to original format [batch, channels, height, width]
        result = torch.fft.ifft2(padded_result, dim=(1, 2)).real
        return result.permute(0, 3, 1, 2)


class LOCOBlock(nn.Module):
    """LOCO block with spectral convolution and MLP."""
    
    def __init__(self, hidden_channels: int, modes: int = 16, modes_y: Optional[int] = None, expansion_ratio: float = 2.0):
        super().__init__()
        self.is_2d = modes_y is not None
        
        # For 1D: use SpectralConvolutionBlock directly, for 2D: use SpectralConv2D
        if self.is_2d:
            self.spectral_conv = SpectralConv2D(hidden_channels, hidden_channels, modes, modes_y)
            conv_layer = nn.Conv2d
            norm_layer = nn.InstanceNorm2d
        else:
            # For 1D, use the exact same block structure as SNO
            self.spectral_conv = SpectralConvolutionBlock(hidden_channels, modes, 'gelu', use_mlp=True, full_block=True)
            # No additional layers needed - SpectralConvolutionBlock handles everything
            return
            
        # Skip connection (only for 2D case)
        self.skip_conv = conv_layer(hidden_channels, hidden_channels, 1)
        self.gate = nn.Parameter(torch.full((1, hidden_channels) + (1,) * (2 if self.is_2d else 1), -2.0))
        self.norm1 = norm_layer(hidden_channels, affine=True)
        
        # MLP (only for 2D case)
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
        if not self.is_2d:
            # For 1D: convert to SNO format and use SpectralConvolutionBlock directly
            x = x.transpose(1, 2)  # [batch, channels, spatial] -> [batch, spatial, channels]
            x = self.spectral_conv(x)
            return x.transpose(1, 2)  # [batch, spatial, channels] -> [batch, channels, spatial]
        
        # 2D case
        identity = x
        
        # Spectral convolution with gated skip connection
        x_spec = self.spectral_conv(x)
        x_proc = self.act(self.norm1(x_spec + torch.sigmoid(self.gate) * self.skip_conv(x)))
        
        # MLP with gated skip connection  
        x_mlp = self.mlp(x_proc)
        x_proc = self.act(self.norm2(x_mlp + torch.sigmoid(self.mlp_gate) * x_proc))
        
        # Residual connection
        return identity + x_proc


class LocalOperator(nn.Module):
    """
    Unified Local Operator (LOCO) for 1D and 2D PDEs.
    
    For 1D: Identical to SNO (SpectralNeuralOperator)
    For 2D: Custom LOCO implementation
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        hidden_channels: Number of hidden channels
        num_blocks: Number of LOCO blocks
        modes_x: Number of Fourier modes in x-direction (or only direction for 1D)
        modes_y: Number of Fourier modes in y-direction (None for 1D)
        use_positional_embedding: Whether to add positional encoding (ignored for 1D SNO)
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
        
        if not self.is_2d:
            # For 1D: Use exact SNO implementation
            self.lifting = ChannelMLP(
                in_channels=in_channels,
                out_channels=hidden_channels,
                hidden_channels=hidden_channels * 2,
                n_layers=2,
                n_dim=1
            )
            
            self.blocks = nn.ModuleList([
                SpectralConvolutionBlock(hidden_channels, modes_x, 'gelu', use_mlp=True) 
                for _ in range(num_blocks)
            ])
            
            self.projection = ChannelMLP(
                in_channels=hidden_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels * 2,
                n_layers=2,
                n_dim=1
            )
        else:
            # For 2D: Use LOCO implementation
            # Positional embedding (optional)
            if use_positional_embedding:
                self.pos_embed = GridEmbedding2D()
                in_channels += 2
            else:
                self.pos_embed = None
                
            # Lifting layer
            self.lift = nn.Conv2d(in_channels, hidden_channels, 1)
            
            # LOCO blocks
            self.blocks = nn.ModuleList([
                LOCOBlock(hidden_channels, modes_x, modes_y) for _ in range(num_blocks)
            ])
            
            # Projection layer
            self.proj = nn.Conv2d(hidden_channels, out_channels, 1)
        
    def forward(self, x):
        if not self.is_2d:
            # 1D case: Exact SNO forward pass
            # Input: [batch, channels, spatial]
            # ChannelMLP expects [batch, channels, spatial], SNO blocks expect [batch, spatial, channels]
            
            # Lift to hidden channels (ChannelMLP expects [batch, channels, spatial])
            x = self.lifting(x)
            
            # Transpose to SNO's native [batch, spatial, channels] format for the spectral blocks
            x = x.transpose(1, 2)

            # Apply spectral blocks
            for block in self.blocks:
                x = block(x)
                
            # Transpose back for the projection layer (ChannelMLP expects [batch, channels, spatial])
            x = x.transpose(1, 2)
            
            # Project back to output channels
            x = self.projection(x)
            
            # Output: [batch, channels, spatial] - matches input format
            return x
        else:
            # 2D case: LOCO implementation
            # Add positional embedding if enabled
            if self.pos_embed is not None:
                x = self.pos_embed(x)
                
            # Lift to hidden dimension
            x = self.lift(x)
            
            # Apply LOCO blocks
            for block in self.blocks:
                x = block(x)
                
            # Project to output dimension
            x = self.proj(x)
            return x


class GridEmbedding1D(nn.Module):
    """1D positional encoding."""
    
    def forward(self, x):
        # x: [batch, channels, spatial]
        B, _, N = x.shape
        
        # Create spatial grid
        grid = torch.linspace(0, 1, N, device=x.device, dtype=x.dtype)
        grid = grid.expand(B, 1, N)
        
        return torch.cat([x, grid], dim=1)


class GridEmbedding2D(nn.Module):
    """2D positional encoding."""
    
    def forward(self, x):
        # x: [batch, channels, height, width]
        B, _, H, W = x.shape
        
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
def create_loco_1d(in_channels=1, out_channels=1, hidden_channels=32, num_blocks=4, modes=16):
    """Create a 1D LOCO model (identical to SNO) with standard parameters."""
    return LocalOperator(
        in_channels=in_channels,
        out_channels=out_channels, 
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        modes_x=modes,
        modes_y=None
    )


def create_loco_2d(in_channels=1, out_channels=1, hidden_channels=32, num_blocks=4, modes_x=16, modes_y=16):
    """Create a 2D LOCO model with standard parameters.""" 
    return LocalOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels, 
        num_blocks=num_blocks,
        modes_x=modes_x,
        modes_y=modes_y
    )