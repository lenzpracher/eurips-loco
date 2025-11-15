"""
1D Burgers equation experiment.

This script provides a complete experiment setup for training and evaluating
neural operators on the 1D Burgers equation.
"""

import torch
import os
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LocalOperator, FourierNeuralOperator, HybridOperator
from utils import (train_model, evaluate_model, generate_data, plot_training_losses, 
                   plot_predictions, plot_rollout_comparison, train_model_multiseed,
                   plot_multiseed_training, plot_multiseed_comparison, plot_all_models_training)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for Burgers experiment."""
    return {
        # Data parameters
        'n_train': 1000,
        'n_test': 200,
        'N': 512,  # Spatial resolution
        'T': 2.5,  # Final time - much shorter for testing
        'dt': 1e-4,  # Time step - matches original solver
        'nu': 0.01,  # Viscosity - matches original solver
        'input_length': 1,
        'output_length': 1,
        
        # Model parameters
        'hidden_channels': 24,  # SNO/LOCO channels - match old burgers.py
        'fno_hidden_channels': 32,  # FNO channels - match old burgers.py  
        'num_blocks': 4,
        'modes': 16,
        
        # Training parameters
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-3,
        
        # Evaluation parameters
        'rollout_steps': 10,
        'n_eval_samples': 50,
        
        # Time gap parameters
        'time_gap': 10  # Gap between input and output timesteps
    }


def get_model_classes() -> Dict[str, tuple]:
    """Get model classes and their parameters."""
    return {
        'LOCO': (LocalOperator, {}),
        'FNO': (FourierNeuralOperator, {}),
        'Hybrid': (HybridOperator, {})
    }


def create_models(config: Dict[str, Any], device: str) -> Dict[str, torch.nn.Module]:
    """Create all models for comparison."""
    models = {}
    
    # LOCO
    models['LOCO'] = LocalOperator(
        in_channels=config['input_length'],
        out_channels=config['output_length'],
        hidden_channels=config['hidden_channels'],
        num_blocks=config['num_blocks'],
        modes_x=config['modes']
    )
    
    # FNO
    models['FNO'] = FourierNeuralOperator(
        in_channels=config['input_length'],
        out_channels=config['output_length'],
        hidden_channels=config['fno_hidden_channels'],  # Use FNO-specific channels
        num_blocks=config['num_blocks'],
        modes_x=config['modes']
    )
    
    # Hybrid FNO-LOCO
    models['Hybrid'] = HybridOperator(
        in_channels=config['input_length'],
        out_channels=config['output_length'],
        hidden_channels=config['hidden_channels'],
        num_blocks=config['num_blocks'],
        modes_x=config['modes']
    )
    
    return models


def run_multiseed_experiment(
    config: Dict[str, Any] = None,
    data_path: str = "data",
    results_path: str = "results",
    device: str = None,
    n_seeds: int = 5,
    base_seed: int = 42,
    model_filter: str = None
) -> Dict[str, Dict]:
    """
    Run multi-seed Burgers experiment for statistical robustness.
    
    Args:
        config: Experiment configuration
        data_path: Path to save/load data
        results_path: Path to save results
        device: Device to run on
        n_seeds: Number of seeds to run for each model
        model_filter: Optional model name filter (e.g., 'LOCO', 'FNO', 'Hybrid', or 'all')
        
    Returns:
        Dictionary containing multi-seed results for all models
    """
    if config is None:
        config = get_default_config()
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("1D Burgers Equation Multi-seed Experiment")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Number of seeds: {n_seeds}")
    print(f"Configuration: {config}")
    
    # Create directories
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    burgers_results_path = os.path.join(results_path, "burgers")
    os.makedirs(burgers_results_path, exist_ok=True)
    
    # Generate/load data
    print("\nGenerating data...")
    train_dataset, test_dataset = generate_data(
        equation='burgers',
        n_train=config['n_train'],
        n_test=config['n_test'],
        input_length=config['input_length'],
        output_length=config['output_length'],
        data_path=data_path,
        device=device,
        N=config['N'],
        T=config['T'],
        dt=config['dt'],
        nu=config['nu'],
        time_gap=config['time_gap']
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Get model classes
    model_classes = get_model_classes()
    
    # Filter models if requested
    if model_filter and model_filter.lower() != 'all':
        # Handle case insensitivity
        model_filter_upper = model_filter.upper()
        if model_filter_upper in model_classes:
            model_classes = {model_filter_upper: model_classes[model_filter_upper]}
        else:
            raise ValueError(f"Unknown model filter: {model_filter}. Available: {list(model_classes.keys())}")
    
    # Train all models with multiple seeds
    all_results = {}
    
    for model_name, (model_class, model_extra_kwargs) in model_classes.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name} with {n_seeds} seeds")
        print(f"{'='*50}")
        
        # Prepare model kwargs with model-specific hidden channels
        if model_name == 'FNO':
            hidden_channels = config['fno_hidden_channels']
        else:
            hidden_channels = config['hidden_channels']
            
        model_kwargs = {
            'in_channels': config['input_length'],
            'out_channels': config['output_length'],
            'hidden_channels': hidden_channels,
            'num_blocks': config['num_blocks'],
            'modes_x': config['modes']
        }
        model_kwargs.update(model_extra_kwargs)
        
        # Create save directory for this model
        model_save_dir = os.path.join(burgers_results_path, model_name.lower())
        
        # Train with multiple seeds
        multiseed_results = train_model_multiseed(
            model_class=model_class,
            model_kwargs=model_kwargs,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            device=device,
            save_dir=model_save_dir,
            n_seeds=n_seeds,
            base_seed=base_seed,
            verbose=True
        )
        
        all_results[model_name] = multiseed_results
        
        # Create individual model plots
        training_plot_path = os.path.join(burgers_results_path, f"{model_name.lower()}_training.png")
        plot_multiseed_training(
            multiseed_results,
            save_path=training_plot_path,
            title=f"{model_name} Training - Burgers Equation"
        )
    
    # Create comparison plots
    print(f"\n{'='*50}")
    print("Creating comparison plots...")
    
    # Combined training curves plot
    training_comparison_path = os.path.join(burgers_results_path, "training_comparison.png")
    plot_all_models_training(
        all_results,
        save_path=training_comparison_path,
        title="Training Progress Comparison - Burgers Equation"
    )
    
    comparison_plot_path = os.path.join(burgers_results_path, "model_comparison.png")
    plot_multiseed_comparison(
        all_results,
        save_path=comparison_plot_path,
        title="Model Comparison - Burgers Equation"
    )
    
    # Save detailed results
    results_file = os.path.join(burgers_results_path, "results.txt")
    with open(results_file, 'w') as f:
        f.write("1D Burgers Equation Experiment Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Configuration: {config}\n")
        f.write(f"Number of seeds: {n_seeds}\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"{model_name} Results (Mean ± Std):\n")
            f.write(f"  Train Loss: {results['mean_final_train_loss']:.6f} ± {results['std_final_train_loss']:.6f}\n")
            f.write(f"  Val Loss: {results['mean_final_val_loss']:.6f} ± {results['std_final_val_loss']:.6f}\n")
            f.write(f"  Test Loss: {results['mean_test_loss']:.6f} ± {results['std_test_loss']:.6f}\n")
            f.write(f"  MAE: {results['mean_mae']:.6f} ± {results['std_mae']:.6f}\n")
            f.write(f"  Relative L2: {results['mean_relative_l2']:.6f} ± {results['std_relative_l2']:.6f}\n")
            f.write(f"  Rollout Loss: {results['mean_rollout_loss']:.6f} ± {results['std_rollout_loss']:.6f}\n")
            f.write(f"  Seeds: {results['seeds']}\n\n")
    
    print(f"Results saved to {burgers_results_path}")
    print("Experiment completed!")
    
    return all_results


def run_experiment(
    config: Dict[str, Any] = None,
    data_path: str = "data",
    results_path: str = "results",
    device: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Run complete Burgers experiment.
    
    Args:
        config: Experiment configuration
        data_path: Path to save/load data
        results_path: Path to save results
        device: Device to run on
        
    Returns:
        Dictionary containing results for all models
    """
    if config is None:
        config = get_default_config()
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("1D Burgers Equation Experiment")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Configuration: {config}")
    
    # Create directories
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    burgers_results_path = os.path.join(results_path, "burgers")
    os.makedirs(burgers_results_path, exist_ok=True)
    
    # Generate/load data
    print("\nGenerating data...")
    train_dataset, test_dataset = generate_data(
        equation='burgers',
        n_train=config['n_train'],
        n_test=config['n_test'],
        input_length=config['input_length'],
        output_length=config['output_length'],
        data_path=data_path,
        device=device,
        N=config['N'],
        T=config['T'],
        dt=config['dt'],
        nu=config['nu'],
        time_gap=config['time_gap']
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create models
    models = create_models(config, device)
    
    # Train and evaluate each model
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\n{'-'*40}")
        print(f"Training {model_name}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"{'-'*40}")
        
        # Train model
        model_save_path = os.path.join(burgers_results_path, f"{model_name.lower()}_best.pt")
        history = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=test_dataset,  # Using test as validation for simplicity
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            device=device,
            save_path=model_save_path,
            verbose=True
        )
        
        # Plot training curves
        plot_path = os.path.join(burgers_results_path, f"{model_name.lower()}_training.png")
        plot_training_losses(
            history['train_losses'],
            history['val_losses'],
            save_path=plot_path,
            title=f"{model_name} Training Progress - Burgers"
        )
        
        # Load best model for evaluation
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate model
        print(f"Evaluating {model_name}...")
        results = evaluate_model(
            model=model,
            test_dataset=test_dataset,
            batch_size=config['batch_size'],
            device=device,
            compute_rollout=True,
            rollout_steps=config['rollout_steps']
        )
        
        all_results[model_name] = results
        
        print(f"{model_name} Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.6f}")
        
        # Generate sample predictions
        model.eval()
        with torch.no_grad():
            # Get a few test samples
            test_indices = torch.randperm(len(test_dataset))[:5]
            test_x = torch.stack([test_dataset[i][0] for i in test_indices])
            test_y = torch.stack([test_dataset[i][1] for i in test_indices])
            
            test_x = test_x.to(device)
            test_pred = model(test_x).cpu()
            
            # Create spatial coordinates
            x_coords = torch.linspace(0, 2*3.14159, config['N'])
            
            # Plot predictions
            pred_plot_path = os.path.join(burgers_results_path, f"{model_name.lower()}_predictions.png")
            plot_predictions(
                test_x.cpu(),
                test_y,
                test_pred,
                x_coords=x_coords,
                save_path=pred_plot_path,
                title=f"{model_name} Predictions - Burgers"
            )
    
    # Create comparison plots
    print(f"\n{'-'*40}")
    print("Creating comparison plots...")
    
    # Model comparison
    comparison_plot_path = os.path.join(burgers_results_path, "model_comparison.png")
    plot_rollout_comparison(
        all_results,
        metrics=['test_loss', 'mae', 'relative_l2', 'rollout_loss'],
        save_path=comparison_plot_path,
        title="Model Comparison - Burgers Equation"
    )
    
    # Save results
    results_file = os.path.join(burgers_results_path, "results.txt")
    with open(results_file, 'w') as f:
        f.write("1D Burgers Equation Experiment Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Configuration:\n{config}\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"{model_name} Results:\n")
            for metric, value in results.items():
                f.write(f"  {metric}: {value:.6f}\n")
            f.write("\n")
    
    print(f"Results saved to {burgers_results_path}")
    print("Experiment completed!")
    
    return all_results


if __name__ == "__main__":
    # Run with default configuration
    results = run_experiment()
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    for model_name, model_results in results.items():
        print(f"{model_name:>10}: Test Loss = {model_results['test_loss']:.6f}, "
              f"Rollout Loss = {model_results.get('rollout_loss', 'N/A')}")