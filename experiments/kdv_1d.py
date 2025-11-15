"""
1D KdV equation experiment.

This script provides a complete experiment setup for training and evaluating
neural operators on the 1D Korteweg-de Vries equation.
"""

import torch
import os
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LocalOperator, FourierNeuralOperator, HybridOperator
from utils import train_model, evaluate_model, generate_data, plot_training_losses, plot_predictions, plot_rollout_comparison


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for KdV experiment."""
    return {
        # Data parameters
        'n_train': 1000,
        'n_test': 200,
        'N': 128,  # Spatial resolution
        'T': 4.0,  # Final time
        'dt': 0.01,  # Time step
        'a': 6.0,  # Nonlinear coefficient
        'b': 1.0,  # Dispersion coefficient
        'use_solitons': True,  # Use soliton-based initial conditions
        'input_length': 1,
        'output_length': 1,
        
        # Model parameters
        'hidden_channels': 32,
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
        hidden_channels=config['hidden_channels'],
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


def run_experiment(
    config: Dict[str, Any] = None,
    data_path: str = "data",
    results_path: str = "results",
    device: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Run complete KdV experiment.
    
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
    print("1D KdV Equation Experiment")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Configuration: {config}")
    
    # Create directories
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    kdv_results_path = os.path.join(results_path, "kdv")
    os.makedirs(kdv_results_path, exist_ok=True)
    
    # Generate/load data
    print("\nGenerating data...")
    train_dataset, test_dataset = generate_data(
        equation='kdv',
        n_train=config['n_train'],
        n_test=config['n_test'],
        input_length=config['input_length'],
        output_length=config['output_length'],
        data_path=data_path,
        device=device,
        N=config['N'],
        T=config['T'],
        dt=config['dt'],
        a=config['a'],
        b=config['b'],
        use_solitons=config['use_solitons'],
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
        model_save_path = os.path.join(kdv_results_path, f"{model_name.lower()}_best.pt")
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
        plot_path = os.path.join(kdv_results_path, f"{model_name.lower()}_training.png")
        plot_training_losses(
            history['train_losses'],
            history['val_losses'],
            save_path=plot_path,
            title=f"{model_name} Training Progress - KdV"
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
            x_coords = torch.linspace(-3.14159, 3.14159, config['N'])
            
            # Plot predictions
            pred_plot_path = os.path.join(kdv_results_path, f"{model_name.lower()}_predictions.png")
            plot_predictions(
                test_x.cpu(),
                test_y,
                test_pred,
                x_coords=x_coords,
                save_path=pred_plot_path,
                title=f"{model_name} Predictions - KdV"
            )
    
    # Create comparison plots
    print(f"\n{'-'*40}")
    print("Creating comparison plots...")
    
    # Model comparison
    comparison_plot_path = os.path.join(kdv_results_path, "model_comparison.png")
    plot_rollout_comparison(
        all_results,
        metrics=['test_loss', 'mae', 'relative_l2', 'rollout_loss'],
        save_path=comparison_plot_path,
        title="Model Comparison - KdV Equation"
    )
    
    # Save results
    results_file = os.path.join(kdv_results_path, "results.txt")
    with open(results_file, 'w') as f:
        f.write("1D KdV Equation Experiment Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Configuration:\n{config}\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"{model_name} Results:\n")
            for metric, value in results.items():
                f.write(f"  {metric}: {value:.6f}\n")
            f.write("\n")
    
    print(f"Results saved to {kdv_results_path}")
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