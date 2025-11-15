"""
Main execution script for neural operator experiments.

This script provides a unified command-line interface for running
experiments on different PDE types with various neural operator models.

Usage:
    python run_experiments.py --equation burgers --model loco
    python run_experiments.py --equation kdv --model fno 
    python run_experiments.py --equation ns2d --model hybrid
    python run_experiments.py --equation all --model all  # Run all combinations
"""

import argparse
import torch
import os
import sys
from typing import Dict, Any, List
import json
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.burgers_1d import run_experiment as run_burgers_experiment
from experiments.burgers_1d import run_multiseed_experiment as run_burgers_multiseed
from experiments.kdv_1d import run_experiment as run_kdv_experiment  
from experiments.navier_stokes_2d import run_experiment as run_ns2d_experiment
from utils.plotting import set_plot_style, demonstrate_rollout
from utils.training import set_seed
from utils.data import generate_data


def get_parser():
    """Create argument parser for the experiment script."""
    parser = argparse.ArgumentParser(
        description="Neural Operator Experiments for PDEs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run single experiment (5 seeds by default for statistical robustness)
    python run_experiments.py --equation burgers --model loco
    
    # Run with custom parameters and fewer seeds for quick testing
    python run_experiments.py --equation kdv --model fno --epochs 200 --batch_size 64 --n_seeds 3
    
    # Run all model comparisons for one equation
    python run_experiments.py --equation ns2d --model all
    
    # Run with single seed for debugging (not recommended for results)
    python run_experiments.py --equation burgers --model loco --n_seeds 1
    
    # Run all combinations (WARNING: takes a long time!)
    python run_experiments.py --equation all --model all --n_seeds 5
    
    # Use different data and results directories
    python run_experiments.py --equation burgers --model hybrid --data_path my_data --results_path my_results
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'generate-data', 'rollout-examples'],
        default='train',
        help='Operation mode: train models, generate data, or create rollout examples (default: train)'
    )
    
    # Main experiment selection (only required for train mode)
    parser.add_argument(
        '--equation', 
        type=str, 
        choices=['burgers', 'kdv', 'ns2d', 'all'],
        help='Type of PDE equation to solve (required for train and rollout-examples modes)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['loco', 'fno', 'hybrid', 'all'],
        help='Neural operator model to use (required for train mode)'
    )
    
    # Paths
    parser.add_argument(
        '--data_path',
        type=str,
        default='data',
        help='Directory to save/load data (default: data)'
    )
    
    parser.add_argument(
        '--results_path',
        type=str, 
        default='results',
        help='Directory to save results (default: results)'
    )
    
    # Hardware
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to run on (default: auto)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Training parameters (optional overrides)
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides default)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size for training (overrides default)'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='Learning rate (overrides default)'
    )
    
    parser.add_argument(
        '--hidden_channels',
        type=int,
        help='Number of hidden channels (overrides default)'
    )
    
    # Data parameters
    parser.add_argument(
        '--n_train',
        type=int,
        help='Number of training samples (overrides default)'
    )
    
    parser.add_argument(
        '--n_test',
        type=int,
        help='Number of test samples (overrides default)'
    )
    
    # Evaluation
    parser.add_argument(
        '--rollout_steps',
        type=int,
        help='Number of rollout steps for evaluation (overrides default)'
    )
    
    parser.add_argument(
        '--sample_idx',
        type=int,
        help='Sample index for rollout examples (if not specified, auto-selects interesting sample for Burgers)'
    )
    
    # Miscellaneous
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--save_config',
        action='store_true',
        help='Save configuration to results directory'
    )
    
    # Multi-seed options (always enabled for statistical robustness)
    parser.add_argument(
        '--n_seeds',
        type=int,
        default=5,
        help='Number of seeds for statistical robustness (default: 5)'
    )
    
    parser.add_argument(
        '--base_seed',
        type=int,
        default=42,
        help='Base seed for multi-seed experiments (uses base_seed + i) (default: 42)'
    )
    
    return parser


def update_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update configuration with command line arguments."""
    # Training parameters
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.hidden_channels is not None:
        config['hidden_channels'] = args.hidden_channels
    
    # Data parameters
    if args.n_train is not None:
        config['n_train'] = args.n_train
    if args.n_test is not None:
        config['n_test'] = args.n_test
    
    # Evaluation parameters
    if args.rollout_steps is not None:
        config['rollout_steps'] = args.rollout_steps
    
    return config


def filter_models_for_experiment(config: Dict[str, Any], model_filter: str) -> Dict[str, Any]:
    """Filter config to run only specified model(s)."""
    if model_filter == 'all':
        return config  # Run all models
    
    # Map command line model names to experiment function expectations
    model_mapping = {
        'loco': 'LOCO',
        'fno': 'FNO', 
        'hybrid': 'Hybrid'
    }
    
    # This is a placeholder - the actual filtering happens in each experiment
    # by modifying the create_models function to return only the requested model
    config['model_filter'] = model_mapping.get(model_filter, model_filter.upper())
    return config


def run_single_experiment(equation: str, config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    """Run a single experiment for the specified equation."""
    print(f"\n{'='*20} {equation.upper()} EXPERIMENT {'='*20}")
    
    if equation == 'burgers':
        from experiments.burgers_1d import get_default_config, create_models
        default_config = get_default_config()
        default_config.update(config)
        config = update_config_with_args(default_config, args)
        
        # Multi-seed training for statistical robustness
        print(f"Running multi-seed experiment with {args.n_seeds} seeds (base_seed={args.base_seed})")
        model_filter = args.model if args.model != 'all' else None
        return run_burgers_multiseed(
            config=config,
            data_path=args.data_path,
            results_path=args.results_path,
            device=args.device,
            n_seeds=args.n_seeds,
            base_seed=args.base_seed,
            model_filter=model_filter
        )
        
    elif equation == 'kdv':
        from experiments.kdv_1d import get_default_config, create_models
        default_config = get_default_config()
        default_config.update(config)
        config = update_config_with_args(default_config, args)
        
        # Filter models if requested
        if hasattr(args, 'model') and args.model != 'all':
            original_create_models = create_models
            def filtered_create_models(cfg, device):
                all_models = original_create_models(cfg, device)
                model_name = config.get('model_filter', args.model.upper())
                if model_name in all_models:
                    return {model_name: all_models[model_name]}
                return all_models
            # Patch the function temporarily
            import experiments.kdv_1d
            experiments.kdv_1d.create_models = filtered_create_models
        
        return run_kdv_experiment(config, args.data_path, args.results_path, args.device)
        
    elif equation == 'ns2d':
        from experiments.navier_stokes_2d import get_default_config, create_models
        default_config = get_default_config()
        default_config.update(config)
        config = update_config_with_args(default_config, args)
        
        # Filter models if requested
        if hasattr(args, 'model') and args.model != 'all':
            original_create_models = create_models
            def filtered_create_models(cfg, device):
                all_models = original_create_models(cfg, device)
                model_name = config.get('model_filter', args.model.upper())
                if model_name in all_models:
                    return {model_name: all_models[model_name]}
                return all_models
            # Patch the function temporarily
            import experiments.navier_stokes_2d
            experiments.navier_stokes_2d.create_models = filtered_create_models
        
        return run_ns2d_experiment(config, args.data_path, args.results_path, args.device)
    
    else:
        raise ValueError(f"Unknown equation type: {equation}")


def main():
    """Main execution function."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Validate required arguments based on mode
    if args.mode == 'train':
        if not args.equation or not args.model:
            parser.error("--equation and --model are required for train mode")
    elif args.mode == 'rollout-examples':
        if not args.equation:
            parser.error("--equation is required for rollout-examples mode")
    
    # Set up environment
    set_seed(args.seed)
    set_plot_style()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    args.device = device
    
    print("Neural Operator Framework")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    if args.equation:
        print(f"Equation(s): {args.equation}")
    if args.model:
        print(f"Model(s): {args.model}")
    print(f"Random seed: {args.seed}")
    print(f"Data path: {args.data_path}")
    print(f"Results path: {args.results_path}")
    
    # Create directories
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.results_path, exist_ok=True)
    
    # Execute based on mode
    if args.mode == 'train':
        run_training_mode(args)
    elif args.mode == 'generate-data':
        run_data_generation_mode(args)
    elif args.mode == 'rollout-examples':
        run_rollout_examples_mode(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def run_data_generation_mode(args):
    """Generate all datasets."""
    from generate_all_data import main as generate_main
    import sys
    
    # Override sys.argv to pass arguments to generate_all_data.py
    original_argv = sys.argv[:]
    sys.argv = [
        'generate_all_data.py',
        '--data_path', args.data_path,
        '--device', args.device,
        '--equations', 'all'
    ]
    
    try:
        generate_main()
    finally:
        sys.argv = original_argv


def run_rollout_examples_mode(args):
    """Create rollout visualization examples."""
    save_dir = os.path.join(args.results_path, "rollout_examples")
    
    # Determine which equations to visualize
    if args.equation == 'all':
        equations = ['burgers', 'kdv', 'ns2d']
    else:
        equations = [args.equation]
    
    print(f"Creating rollout examples for: {equations}")
    
    for equation in equations:
        print(f"\n{'='*50}")
        print(f"Creating {equation.upper()} rollout example")
        print(f"{'='*50}")
        
        # Set appropriate number of steps based on equation
        if equation == 'burgers':
            n_steps = 25001  # Full Burgers evolution to T=2.5
        elif equation == 'kdv':
            n_steps = 4002   # Full KdV evolution to T=4.0
        else:
            n_steps = 400    # Default for other equations
            
        demonstrate_rollout(
            equation=equation,
            data_path=args.data_path,
            save_dir=save_dir,
            n_steps=n_steps,
            sample_idx=getattr(args, 'sample_idx', None)
        )
    
    print(f"\n{'='*50}")
    print("ROLLOUT EXAMPLES COMPLETE")
    print(f"{'='*50}")
    print(f"All visualizations saved to: {os.path.abspath(save_dir)}")


def run_training_mode(args):
    """Run the original training mode."""
    
    # Determine which experiments to run
    if args.equation == 'all':
        equations = ['burgers', 'kdv', 'ns2d']
    else:
        equations = [args.equation]
    
    # Run experiments
    all_results = {}
    total_start_time = time.time()
    
    for equation in equations:
        start_time = time.time()
        
        try:
            config = filter_models_for_experiment({}, args.model)
            results = run_single_experiment(equation, config, args)
            all_results[equation] = results
            
            end_time = time.time()
            print(f"\n{equation.upper()} experiment completed in {end_time - start_time:.1f} seconds")
            
        except Exception as e:
            print(f"ERROR in {equation} experiment: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue
    
    total_end_time = time.time()
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Total runtime: {total_end_time - total_start_time:.1f} seconds")
    print()
    
    for equation, equation_results in all_results.items():
        print(f"{equation.upper()} Results:")
        for model_name, model_results in equation_results.items():
            test_loss = model_results.get('test_loss', 'N/A')
            rollout_loss = model_results.get('rollout_loss', 'N/A')
            print(f"  {model_name:>10}: Test Loss = {test_loss}, Rollout Loss = {rollout_loss}")
        print()
    
    # Save summary results
    summary_path = os.path.join(args.results_path, "experiment_summary.json")
    summary_data = {
        'args': vars(args),
        'results': all_results,
        'total_runtime': total_end_time - total_start_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"Experiment summary saved to {summary_path}")
    print("All experiments completed!")


if __name__ == "__main__":
    main()