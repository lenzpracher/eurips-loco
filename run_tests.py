#!/usr/bin/env python3
"""
Test runner script for neurips-neural-operators.

This script provides different test running options:
- Quick tests: Fast unit tests only
- Full tests: All tests including slow ones
- Specific test categories
"""

import subprocess
import sys
import argparse
import os


def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ pytest not found. Please install it: pip install pytest")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test runner for neurips-neural-operators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tests.py                    # Run quick tests
    python run_tests.py --full            # Run all tests
    python run_tests.py --models          # Test only models
    python run_tests.py --solvers         # Test only solvers
    python run_tests.py --slow            # Run only slow tests
    python run_tests.py --integration     # Run integration tests
    python run_tests.py --coverage        # Run with coverage report
        """
    )
    
    parser.add_argument('--full', action='store_true',
                       help='Run all tests including slow ones')
    parser.add_argument('--quick', action='store_true', default=True,
                       help='Run only quick tests (default)')
    parser.add_argument('--models', action='store_true',
                       help='Run only model tests')
    parser.add_argument('--solvers', action='store_true',
                       help='Run only solver tests') 
    parser.add_argument('--utils', action='store_true',
                       help='Run only utility tests')
    parser.add_argument('--experiments', action='store_true',
                       help='Run only experiment tests')
    parser.add_argument('--slow', action='store_true',
                       help='Run only slow tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--coverage', action='store_true',
                       help='Run tests with coverage report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--parallel', '-n', type=int,
                       help='Run tests in parallel (requires pytest-xdist)')
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ['python', '-m', 'pytest']
    
    # Add verbosity
    if args.verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(['-n', str(args.parallel)])
    
    # Add coverage
    if args.coverage:
        cmd.extend(['--cov=models', '--cov=solvers', '--cov=utils', '--cov=experiments'])
        cmd.extend(['--cov-report=html', '--cov-report=term'])
    
    # Test selection
    test_files = []
    markers = []
    
    if args.models:
        test_files.append('tests/test_models.py')
    elif args.solvers:
        test_files.append('tests/test_solvers.py')
    elif args.utils:
        test_files.append('tests/test_utils.py')
    elif args.experiments:
        test_files.append('tests/test_experiments.py')
    elif args.slow:
        markers.append('slow')
    elif args.integration:
        markers.append('integration')
    elif args.full:
        # Run all tests
        pass
    else:
        # Default: quick tests (exclude slow and integration)
        markers.extend(['not slow', 'not integration'])
    
    # Add test files
    if test_files:
        cmd.extend(test_files)
    
    # Add markers
    if markers:
        cmd.extend(['-m', ' and '.join(markers)])
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    print("Neural Operators Test Suite")
    print("=" * 60)
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Test command: {' '.join(cmd)}")
    
    # Check if pytest is available
    try:
        subprocess.run(['python', '-m', 'pytest', '--version'], 
                      check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ pytest not found. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytest'], check=True)
    
    # Run tests
    try:
        result = subprocess.run(cmd, env=env)
        
        if result.returncode == 0:
            print("\n🎉 All tests passed!")
            
            if args.coverage:
                print("\n📊 Coverage report generated in htmlcov/")
                
        else:
            print(f"\n❌ Tests failed with exit code {result.returncode}")
            
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)