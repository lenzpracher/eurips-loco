"""
Tests for neural operator models.
"""

import pytest
import torch
import torch.nn as nn

from models import LocalOperator, FourierNeuralOperator, HybridOperator


class TestLocalOperator:
    """Tests for LOCO model."""
    
    def test_loco_1d_creation(self, model_config_1d):
        """Test 1D LOCO model creation."""
        model = LocalOperator(**model_config_1d)
        assert isinstance(model, nn.Module)
        
        # Check parameter count is reasonable
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 1000  # Should have substantial parameters
        assert param_count < 1e6   # But not too many for test model
    
    def test_loco_2d_creation(self, model_config_2d):
        """Test 2D LOCO model creation."""
        model = LocalOperator(**model_config_2d)
        assert isinstance(model, nn.Module)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 1000
        assert param_count < 1e6
    
    def test_loco_1d_forward(self, model_config_1d, device):
        """Test 1D LOCO forward pass."""
        model = LocalOperator(**model_config_1d).to(device)
        
        # Test with single timestep input
        batch_size, spatial_dim = 4, 32
        x = torch.randn(batch_size, 1, spatial_dim).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, spatial_dim)
        assert torch.isfinite(output).all()
    
    def test_loco_2d_forward(self, model_config_2d, device):
        """Test 2D LOCO forward pass."""
        model = LocalOperator(**model_config_2d).to(device)
        
        # Test with single timestep input
        batch_size, height, width = 2, 16, 16
        x = torch.randn(batch_size, 1, height, width).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, height, width)
        assert torch.isfinite(output).all()
    
    def test_loco_multi_timestep_input(self, device):
        """Test LOCO with multiple timestep input."""
        model = LocalOperator(
            in_channels=5, out_channels=1, hidden_channels=16,
            num_blocks=2, modes_x=8
        ).to(device)
        
        batch_size, spatial_dim = 3, 32
        x = torch.randn(batch_size, 5, spatial_dim).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, spatial_dim)
    
    def test_loco_gradient_flow(self, model_config_1d):
        """Test that gradients flow through LOCO."""
        model = LocalOperator(**model_config_1d)
        
        x = torch.randn(2, 1, 32, requires_grad=True)
        target = torch.randn(2, 1, 32)
        
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()


class TestFourierNeuralOperator:
    """Tests for FNO model."""
    
    def test_fno_1d_creation(self, model_config_1d):
        """Test 1D FNO model creation."""
        model = FourierNeuralOperator(**model_config_1d)
        assert isinstance(model, nn.Module)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 1000
        assert param_count < 1e6
    
    def test_fno_2d_creation(self, model_config_2d):
        """Test 2D FNO model creation."""
        model = FourierNeuralOperator(**model_config_2d)
        assert isinstance(model, nn.Module)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 1000
        assert param_count < 1e6
    
    def test_fno_1d_forward(self, model_config_1d, device):
        """Test 1D FNO forward pass."""
        model = FourierNeuralOperator(**model_config_1d).to(device)
        
        batch_size, spatial_dim = 4, 32
        x = torch.randn(batch_size, 1, spatial_dim).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, spatial_dim)
        assert torch.isfinite(output).all()
    
    def test_fno_2d_forward(self, model_config_2d, device):
        """Test 2D FNO forward pass."""
        model = FourierNeuralOperator(**model_config_2d).to(device)
        
        batch_size, height, width = 2, 16, 16
        x = torch.randn(batch_size, 1, height, width).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, height, width)
        assert torch.isfinite(output).all()
    
    def test_fno_gradient_flow(self, model_config_1d):
        """Test that gradients flow through FNO."""
        model = FourierNeuralOperator(**model_config_1d)
        
        x = torch.randn(2, 1, 32, requires_grad=True)
        target = torch.randn(2, 1, 32)
        
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()


class TestHybridOperator:
    """Tests for Hybrid model."""
    
    def test_hybrid_1d_creation(self, model_config_1d):
        """Test 1D Hybrid model creation."""
        model = HybridOperator(**model_config_1d)
        assert isinstance(model, nn.Module)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 1000
        assert param_count < 1e6
    
    def test_hybrid_2d_creation(self, model_config_2d):
        """Test 2D Hybrid model creation."""
        model = HybridOperator(**model_config_2d)
        assert isinstance(model, nn.Module)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 1000
        assert param_count < 1e6
    
    def test_hybrid_even_channels_requirement(self):
        """Test that hybrid model requires even hidden channels."""
        with pytest.raises(ValueError, match="hidden_channels must be even"):
            HybridOperator(
                in_channels=1, out_channels=1, hidden_channels=17,  # Odd number
                num_blocks=2, modes_x=8
            )
    
    def test_hybrid_1d_forward(self, model_config_1d, device):
        """Test 1D Hybrid forward pass."""
        model = HybridOperator(**model_config_1d).to(device)
        
        batch_size, spatial_dim = 4, 32
        x = torch.randn(batch_size, 1, spatial_dim).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, spatial_dim)
        assert torch.isfinite(output).all()
    
    def test_hybrid_2d_forward(self, model_config_2d, device):
        """Test 2D Hybrid forward pass."""
        model = HybridOperator(**model_config_2d).to(device)
        
        batch_size, height, width = 2, 16, 16
        x = torch.randn(batch_size, 1, height, width).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, height, width)
        assert torch.isfinite(output).all()
    
    def test_hybrid_gradient_flow(self, model_config_1d):
        """Test that gradients flow through Hybrid model."""
        model = HybridOperator(**model_config_1d)
        
        x = torch.randn(2, 1, 32, requires_grad=True)
        target = torch.randn(2, 1, 32)
        
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()


class TestModelComparison:
    """Comparative tests across models."""
    
    def test_model_output_shapes_consistent(self, device):
        """Test that all models produce consistent output shapes."""
        config = {
            'in_channels': 1, 'out_channels': 1, 'hidden_channels': 16,
            'num_blocks': 2, 'modes_x': 8
        }
        
        models = [
            LocalOperator(**config),
            FourierNeuralOperator(**config),
            HybridOperator(**config)
        ]
        
        x = torch.randn(3, 1, 32).to(device)
        
        outputs = []
        for model in models:
            model = model.to(device)
            with torch.no_grad():
                output = model(x)
            outputs.append(output)
        
        # All outputs should have same shape
        reference_shape = outputs[0].shape
        for output in outputs[1:]:
            assert output.shape == reference_shape
    
    def test_model_parameter_counts_reasonable(self):
        """Test that model parameter counts are in reasonable ranges."""
        config = {
            'in_channels': 1, 'out_channels': 1, 'hidden_channels': 32,
            'num_blocks': 4, 'modes_x': 16, 'modes_y': 16
        }
        
        models = [
            LocalOperator(**config),
            FourierNeuralOperator(**config),
            HybridOperator(**config)
        ]
        
        param_counts = []
        for model in models:
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_counts.append(count)
            
            # Each model should have reasonable parameter count
            assert 10000 < count < 5e6  # Between 10K and 5M parameters
        
        # Parameter counts should be in similar ranges (within factor of 5)
        max_count = max(param_counts)
        min_count = min(param_counts)
        assert max_count / min_count < 5.0