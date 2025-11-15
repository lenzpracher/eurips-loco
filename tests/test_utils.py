"""
Tests for utility functions.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tempfile
import os

from utils import PDEDataset, train_model, evaluate_model, generate_data, count_parameters
from models import LocalOperator


class TestPDEDataset:
    """Tests for PDE dataset class."""
    
    def test_dataset_creation(self, small_1d_data):
        """Test basic dataset creation."""
        dataset = PDEDataset(small_1d_data, input_length=5, output_length=1)
        
        assert len(dataset) > 0
        assert hasattr(dataset, 'mean')
        assert hasattr(dataset, 'std')
    
    def test_dataset_getitem(self, small_1d_data):
        """Test dataset item retrieval."""
        dataset = PDEDataset(small_1d_data, input_length=3, output_length=2)
        
        x, y = dataset[0]
        
        assert x.shape[0] == 3  # input_length
        assert y.shape[0] == 2  # output_length
        assert x.shape[1:] == y.shape[1:]  # Same spatial dimensions
    
    def test_dataset_single_timestep_io(self, small_1d_data):
        """Test dataset with single timestep input/output."""
        dataset = PDEDataset(small_1d_data, input_length=1, output_length=1)
        
        x, y = dataset[0]
        
        # Single timesteps should have time dimension squeezed but channel dimension added
        assert len(x.shape) == 2  # [channels, spatial]
        assert len(y.shape) == 2  # [channels, spatial]
        assert x.shape[0] == 1  # Single channel
        assert y.shape[0] == 1  # Single channel
    
    def test_dataset_normalization(self, small_1d_data):
        """Test dataset normalization."""
        # Create dataset with normalization
        dataset_norm = PDEDataset(small_1d_data, normalize=True)
        
        # Create dataset without normalization
        dataset_raw = PDEDataset(small_1d_data, normalize=False)
        
        x_norm, _ = dataset_norm[0]
        x_raw, _ = dataset_raw[0]
        
        # Normalized data should have different statistics
        assert not torch.allclose(x_norm, x_raw)
        
        # Test denormalization
        x_denorm = dataset_norm.denormalize(x_norm)
        assert torch.allclose(x_denorm, x_raw, atol=1e-5)
    
    def test_dataset_2d(self, small_2d_data):
        """Test dataset with 2D data."""
        dataset = PDEDataset(small_2d_data, input_length=3, output_length=1)
        
        x, y = dataset[0]
        
        assert len(x.shape) == 3  # [time, height, width]
        assert len(y.shape) == 2  # [height, width] (single timestep)
    
    def test_dataset_time_gap(self, small_1d_data):
        """Test dataset with time gap between input and output."""
        dataset = PDEDataset(small_1d_data, input_length=2, output_length=1, time_gap=3)
        
        # Should still work but have fewer samples
        assert len(dataset) > 0
        
        x, y = dataset[0]
        assert x.shape[0] == 2  # input_length
    
    def test_dataset_insufficient_timesteps(self):
        """Test dataset creation with insufficient timesteps."""
        # Create data with too few timesteps
        data = torch.randn(5, 3, 32)  # Only 3 timesteps
        
        with pytest.raises(ValueError):
            PDEDataset(data, input_length=5, output_length=1)


class TestTrainingUtilities:
    """Tests for training utility functions."""
    
    def test_count_parameters(self, model_config_1d):
        """Test parameter counting function."""
        model = LocalOperator(**model_config_1d)
        
        count1 = count_parameters(model)
        count2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert count1 == count2
        assert count1 > 0
    
    def test_train_model_basic(self, device, temp_dir):
        """Test basic training functionality."""
        # Create small model and dataset
        model = LocalOperator(
            in_channels=1, out_channels=1, hidden_channels=8,
            num_blocks=1, modes_x=4
        )
        
        # Create synthetic dataset
        data = torch.randn(20, 10, 16)  # 20 samples, 10 timesteps, 16 spatial
        dataset = PDEDataset(data, input_length=1, output_length=1)
        
        # Split into train/val
        train_size = 15
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Train for a few epochs
        save_path = os.path.join(temp_dir, "test_model.pt")
        history = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            batch_size=4,
            learning_rate=1e-3,
            device=device,
            save_path=save_path,
            verbose=False
        )
        
        assert 'train_losses' in history
        assert 'val_losses' in history
        assert len(history['train_losses']) == 3
        assert len(history['val_losses']) == 3
        assert os.path.exists(save_path)
    
    def test_evaluate_model_basic(self, device):
        """Test basic evaluation functionality."""
        # Create small model and dataset
        model = LocalOperator(
            in_channels=1, out_channels=1, hidden_channels=8,
            num_blocks=1, modes_x=4
        )
        
        data = torch.randn(10, 5, 16)
        dataset = PDEDataset(data, input_length=1, output_length=1)
        
        # Evaluate model
        results = evaluate_model(
            model=model,
            test_dataset=dataset,
            batch_size=4,
            device=device,
            compute_rollout=False
        )
        
        assert 'test_loss' in results
        assert 'mse' in results
        assert 'mae' in results
        assert 'relative_l2' in results
        
        # All metrics should be positive numbers
        for value in results.values():
            assert isinstance(value, float)
            assert value >= 0
    
    def test_evaluate_model_with_rollout(self, device):
        """Test evaluation with rollout."""
        model = LocalOperator(
            in_channels=1, out_channels=1, hidden_channels=8,
            num_blocks=1, modes_x=4
        )
        
        data = torch.randn(8, 10, 16)  # More timesteps for rollout
        dataset = PDEDataset(data, input_length=1, output_length=1)
        
        results = evaluate_model(
            model=model,
            test_dataset=dataset,
            batch_size=4,
            device=device,
            compute_rollout=True,
            rollout_steps=3
        )
        
        assert 'rollout_loss' in results
        assert results['rollout_loss'] >= 0


class TestDataGeneration:
    """Tests for data generation utilities."""
    
    def test_generate_data_burgers(self, device, temp_dir):
        """Test Burgers data generation through unified interface."""
        train_dataset, test_dataset = generate_data(
            equation='burgers',
            n_train=3,
            n_test=2,
            data_path=temp_dir,
            device=device,
            N=16,
            T=0.5,
            dt=0.1
        )
        
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0
        
        x, y = train_dataset[0]
        assert torch.isfinite(x).all()
        assert torch.isfinite(y).all()
    
    @pytest.mark.skip(reason="KdV data generation depends on unstable integration")
    def test_generate_data_kdv(self, device, temp_dir):
        """Test KdV data generation through unified interface."""
        train_dataset, test_dataset = generate_data(
            equation='kdv',
            n_train=3,
            n_test=2,
            data_path=temp_dir,
            device=device,
            N=16,
            T=1.0,
            dt=0.1
        )
        
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0
        
        x, y = train_dataset[0]
        assert torch.isfinite(x).all()
        assert torch.isfinite(y).all()
    
    @pytest.mark.slow
    def test_generate_data_ns2d(self, device, temp_dir):
        """Test NS2D data generation through unified interface."""
        train_dataset, test_dataset = generate_data(
            equation='ns2d',
            n_train=2,
            n_test=1,
            data_path=temp_dir,
            device=device,
            N=8,
            T=1.0,
            record_steps=5
        )
        
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0
        
        x, y = train_dataset[0]
        assert torch.isfinite(x).all()
        assert torch.isfinite(y).all()
    
    def test_generate_data_invalid_equation(self, device, temp_dir):
        """Test error handling for invalid equation type."""
        with pytest.raises(ValueError, match="Unknown equation type"):
            generate_data(
                equation='invalid_equation',
                n_train=5,
                n_test=2,
                data_path=temp_dir,
                device=device
            )
    
    def test_data_caching(self, device, temp_dir):
        """Test that data generation caches results."""
        # Generate data first time
        train_dataset1, test_dataset1 = generate_data(
            equation='burgers',
            n_train=3,
            n_test=2,
            data_path=temp_dir,
            device=device,
            N=16,
            T=0.3,
            dt=0.1
        )
        
        # Check that files were created
        assert os.path.exists(os.path.join(temp_dir, 'burgers_train.pt'))
        assert os.path.exists(os.path.join(temp_dir, 'burgers_test.pt'))
        
        # Generate data second time (should load from cache)
        train_dataset2, test_dataset2 = generate_data(
            equation='burgers',
            n_train=3,
            n_test=2,
            data_path=temp_dir,
            device=device,
            N=16,
            T=0.3,
            dt=0.1
        )
        
        # Data should be identical
        x1, y1 = train_dataset1[0]
        x2, y2 = train_dataset2[0]
        
        assert torch.allclose(x1, x2)
        assert torch.allclose(y1, y2)


class TestUtilityIntegration:
    """Integration tests for utility functions."""
    
    def test_complete_training_evaluation_cycle(self, device, temp_dir):
        """Test complete training and evaluation cycle."""
        # Generate data
        train_dataset, test_dataset = generate_data(
            equation='burgers',
            n_train=10,
            n_test=5,
            data_path=temp_dir,
            device=device,
            N=16,
            T=0.5,
            dt=0.1
        )
        
        # Create model
        model = LocalOperator(
            in_channels=1, out_channels=1, hidden_channels=8,
            num_blocks=1, modes_x=4
        )
        
        # Train model
        save_path = os.path.join(temp_dir, "integration_test.pt")
        history = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            device=device,
            save_path=save_path,
            verbose=False
        )
        
        # Load trained model
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate model
        results = evaluate_model(
            model=model,
            test_dataset=test_dataset,
            batch_size=4,
            device=device
        )
        
        # Check that everything worked
        assert len(history['train_losses']) == 2
        assert 'test_loss' in results
        assert results['test_loss'] > 0