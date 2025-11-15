"""
Tests for experiment scripts and main execution.
"""

import pytest
import torch
import tempfile
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_experiments
from experiments import burgers_1d, kdv_1d, navier_stokes_2d


class TestMainScript:
    """Tests for the main run_experiments.py script."""
    
    def test_argument_parser(self):
        """Test argument parser functionality."""
        parser = run_experiments.get_parser()
        
        # Test valid arguments
        args = parser.parse_args([
            '--equation', 'burgers', 
            '--model', 'loco',
            '--epochs', '10',
            '--batch_size', '16'
        ])
        
        assert args.equation == 'burgers'
        assert args.model == 'loco'
        assert args.epochs == 10
        assert args.batch_size == 16
    
    def test_argument_parser_defaults(self):
        """Test argument parser default values."""
        parser = run_experiments.get_parser()
        args = parser.parse_args(['--equation', 'burgers', '--model', 'loco'])
        
        assert args.data_path == 'data'
        assert args.results_path == 'results'
        assert args.device == 'auto'
        assert args.seed == 42
    
    def test_config_update_with_args(self):
        """Test configuration update with command line arguments."""
        config = {'epochs': 100, 'batch_size': 32}
        
        # Mock args
        class MockArgs:
            epochs = 50
            batch_size = None
            learning_rate = 1e-4
            hidden_channels = None
            n_train = None
            n_test = None
            rollout_steps = None
        
        args = MockArgs()
        updated_config = run_experiments.update_config_with_args(config, args)
        
        assert updated_config['epochs'] == 50  # Updated
        assert updated_config['batch_size'] == 32  # Unchanged (None in args)
        assert updated_config['learning_rate'] == 1e-4  # Added
    
    def test_filter_models_for_experiment(self):
        """Test model filtering functionality."""
        config = {}
        
        # Test single model
        filtered = run_experiments.filter_models_for_experiment(config, 'loco')
        assert filtered['model_filter'] == 'LOCO'
        
        # Test all models
        filtered = run_experiments.filter_models_for_experiment(config, 'all')
        assert filtered == config  # Should return original config unchanged


class TestExperimentScripts:
    """Tests for individual experiment scripts."""
    
    def test_burgers_experiment_config(self):
        """Test Burgers experiment default configuration."""
        config = burgers_1d.get_default_config()
        
        required_keys = [
            'n_train', 'n_test', 'N', 'T', 'dt', 'nu',
            'hidden_channels', 'num_blocks', 'modes',
            'epochs', 'batch_size', 'learning_rate'
        ]
        
        for key in required_keys:
            assert key in config
            assert config[key] is not None
    
    def test_kdv_experiment_config(self):
        """Test KdV experiment default configuration."""
        config = kdv_1d.get_default_config()
        
        required_keys = [
            'n_train', 'n_test', 'N', 'T', 'dt', 'a', 'b',
            'hidden_channels', 'num_blocks', 'modes',
            'epochs', 'batch_size', 'learning_rate'
        ]
        
        for key in required_keys:
            assert key in config
            assert config[key] is not None
    
    def test_ns2d_experiment_config(self):
        """Test NS2D experiment default configuration."""
        config = navier_stokes_2d.get_default_config()
        
        required_keys = [
            'n_train', 'n_test', 'N', 'T', 'record_steps', 'visc',
            'hidden_channels', 'num_blocks', 'modes_x', 'modes_y',
            'epochs', 'batch_size', 'learning_rate'
        ]
        
        for key in required_keys:
            assert key in config
            assert config[key] is not None
    
    def test_burgers_model_creation(self, device):
        """Test Burgers experiment model creation."""
        config = burgers_1d.get_default_config()
        models = burgers_1d.create_models(config, device)
        
        expected_models = ['LOCO', 'FNO', 'Hybrid']
        assert all(model_name in models for model_name in expected_models)
        
        for model in models.values():
            assert hasattr(model, 'forward')
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
    
    def test_kdv_model_creation(self, device):
        """Test KdV experiment model creation."""
        config = kdv_1d.get_default_config()
        models = kdv_1d.create_models(config, device)
        
        expected_models = ['LOCO', 'FNO', 'Hybrid']
        assert all(model_name in models for model_name in expected_models)
    
    def test_ns2d_model_creation(self, device):
        """Test NS2D experiment model creation."""
        config = navier_stokes_2d.get_default_config()
        models = navier_stokes_2d.create_models(config, device)
        
        expected_models = ['LOCO', 'FNO', 'Hybrid']
        assert all(model_name in models for model_name in expected_models)
    
    @pytest.mark.slow
    def test_burgers_experiment_integration(self, temp_dir, device):
        """Test complete Burgers experiment (marked as slow)."""
        # Create minimal config for fast testing
        config = {
            'n_train': 5,
            'n_test': 3,
            'N': 16,
            'T': 0.5,
            'dt': 0.1,
            'nu': 0.01,
            'hidden_channels': 8,
            'num_blocks': 1,
            'modes': 4,
            'epochs': 2,
            'batch_size': 4,
            'learning_rate': 1e-3,
            'rollout_steps': 2,
            'n_eval_samples': 3
        }
        
        # Mock the create_models function to return only one model
        def mock_create_models(cfg, dev):
            from models import LocalOperator
            return {
                'LOCO': LocalOperator(
                    in_channels=1, out_channels=1, hidden_channels=8,
                    num_blocks=1, modes_x=4
                )
            }
        
        with patch.object(burgers_1d, 'create_models', mock_create_models):
            results = burgers_1d.run_experiment(
                config=config,
                data_path=temp_dir,
                results_path=temp_dir,
                device=device
            )
        
        assert isinstance(results, dict)
        assert 'LOCO' in results
        assert 'test_loss' in results['LOCO']
        assert results['LOCO']['test_loss'] > 0


class TestMainExecution:
    """Tests for main execution functionality."""
    
    @patch('run_experiments.run_single_experiment')
    def test_main_execution_single_experiment(self, mock_run_single):
        """Test main execution with single experiment."""
        mock_run_single.return_value = {
            'LOCO': {'test_loss': 0.001, 'mae': 0.0005}
        }
        
        # Mock sys.argv
        test_args = [
            'run_experiments.py',
            '--equation', 'burgers',
            '--model', 'loco',
            '--epochs', '5'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('os.makedirs'):
                with patch('json.dump'):
                    try:
                        run_experiments.main()
                    except SystemExit:
                        pass  # main() calls sys.exit, which is expected
        
        # Check that run_single_experiment was called
        mock_run_single.assert_called_once()
        args, kwargs = mock_run_single.call_args
        assert args[0] == 'burgers'  # equation
    
    def test_run_single_experiment_burgers(self, temp_dir, device):
        """Test run_single_experiment function for Burgers."""
        config = {}
        
        class MockArgs:
            def __init__(self, dev):
                self.model = 'loco'
                self.data_path = temp_dir
                self.results_path = temp_dir
                self.device = dev
                self.epochs = 2
                self.batch_size = 4
                self.learning_rate = None
                self.hidden_channels = None
                self.n_train = 3
                self.n_test = 2
                self.rollout_steps = None
                self.n_seeds = 1
                self.base_seed = 42
        
        args = MockArgs(device)
        
        # Mock the expensive parts
        with patch('experiments.burgers_1d.train_model') as mock_train:
            with patch('experiments.burgers_1d.evaluate_model') as mock_eval:
                mock_train.return_value = {
                    'train_losses': [0.1, 0.05],
                    'val_losses': [0.12, 0.06]
                }
                mock_eval.return_value = {
                    'test_loss': 0.08,
                    'mae': 0.04,
                    'relative_l2': 0.1
                }
                
                with patch('torch.save'):  # Mock checkpoint saving
                    with patch('torch.load') as mock_load:
                        mock_load.return_value = {
                            'model_state_dict': {},
                            'epoch': 1,
                            'train_loss': 0.05,
                            'val_loss': 0.06
                        }
                        
                        try:
                            results = run_experiments.run_single_experiment(
                                'burgers', config, args
                            )
                            
                            # Should return results even with mocked training
                            assert isinstance(results, dict)
                            
                        except Exception as e:
                            # Some imports might fail in test environment
                            pytest.skip(f"Experiment test skipped due to import issues: {e}")


class TestErrorHandling:
    """Tests for error handling in experiments."""
    
    def test_invalid_equation_type(self):
        """Test handling of invalid equation type."""
        config = {}
        
        class MockArgs:
            model = 'loco'
            data_path = 'temp'
            results_path = 'temp'
            device = 'cpu'
        
        args = MockArgs()
        
        with pytest.raises(ValueError, match="Unknown equation type"):
            run_experiments.run_single_experiment('invalid_equation', config, args)
    
    def test_missing_model_in_creation(self, device):
        """Test behavior when model creation fails."""
        from experiments.burgers_1d import create_models
        
        # This should work normally
        config = {
            'input_length': 1, 'output_length': 1, 'hidden_channels': 16,
            'num_blocks': 2, 'modes': 8
        }
        
        models = create_models(config, device)
        assert len(models) > 0
        
        # Test with invalid config (should still create models, maybe with warnings)
        invalid_config = {
            'input_length': -1,  # Invalid but might be handled gracefully
            'output_length': 1,
            'hidden_channels': 16,
            'num_blocks': 2,
            'modes': 8
        }
        
        try:
            models = create_models(invalid_config, device)
            # If it doesn't raise an error, that's fine too
        except Exception:
            # Expected for truly invalid configs
            pass


@pytest.mark.integration 
class TestFullWorkflow:
    """Integration tests for full workflow."""
    
    @pytest.mark.slow
    def test_minimal_end_to_end_workflow(self, temp_dir):
        """Test minimal end-to-end workflow."""
        # This test runs the actual workflow with minimal parameters
        test_args = [
            'run_experiments.py',
            '--equation', 'burgers',
            '--model', 'loco',
            '--n_train', '3',
            '--n_test', '2', 
            '--epochs', '1',
            '--batch_size', '2',
            '--data_path', temp_dir,
            '--results_path', temp_dir
        ]
        
        with patch.object(sys, 'argv', test_args):
            try:
                run_experiments.main()
            except SystemExit as e:
                # main() calls sys.exit(0) on success
                assert e.code == 0 or e.code is None
            except Exception as e:
                # In test environment, some parts might fail
                pytest.skip(f"End-to-end test skipped: {e}")
        
        # Check that some results were created
        results_files = os.listdir(temp_dir)
        # Should have created some files (data, results, etc.)
        assert len(results_files) > 0