# Neural Operators for PDEs: A NeurIPS Workshop Submission

This repository provides a simplified and reproducible implementation of neural operator methods for solving partial differential equations (PDEs). It focuses on comparing three key architectures: **Local Operators (LOCO)**, **Fourier Neural Operators (FNO)**, and **Hybrid FNO-LOCO** models.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd neurips-neural-operators

# Install dependencies using pixi (recommended)
pixi install

# Or install using pip
pip install -r requirements.txt
```

### Run Your First Experiment

```bash
# Generate all datasets first (recommended)
pixi run python run_experiments.py --mode generate-data

# Run Burgers equation with LOCO model (default: 5 seeds for statistical robustness)
pixi run python run_experiments.py --equation burgers --model loco

# Run KdV equation with FNO model  
pixi run python run_experiments.py --equation kdv --model fno

# Run 2D Navier-Stokes with hybrid model
pixi run python run_experiments.py --equation ns2d --model hybrid

# Compare all models on Burgers equation
pixi run python run_experiments.py --equation burgers --model all

# Create rollout visualization examples
pixi run python run_experiments.py --mode rollout-examples --equation all
```

## 📊 Supported Equations

| Equation | Dimension | Description | Domain |
|----------|-----------|-------------|---------|
| **Burgers** | 1D | Viscous Burgers: `u_t + u*u_x = ν*u_xx` | `[0, 2π]` |
| **KdV** | 1D | Korteweg-de Vries: `u_t + a*u*u_x + b*u_xxx = 0` | `[-π, π]` |
| **Navier-Stokes** | 2D | Vorticity form with forcing | `[0, 1]²` |

## 🧠 Neural Operator Models

### 1. Local Operator (LOCO)
- **Key Feature**: Nonlinearity applied in spectral domain
- **Architecture**: `FFT → Spectral weights → ReLU(iFFT) → FFT → Convolution → iFFT`
- **Best For**: PDEs with complex spectral behavior

### 2. Fourier Neural Operator (FNO)
- **Key Feature**: Linear mixing in spectral domain
- **Architecture**: `FFT → Spectral weights → iFFT`
- **Best For**: Dispersive PDEs (e.g., KdV)

### 3. Hybrid FNO-LOCO
- **Key Feature**: Combines FNO and LOCO branches
- **Architecture**: Channel splitting between FNO and LOCO paths
- **Best For**: Mixed PDE characteristics

## 📁 Repository Structure

```
neurips-neural-operators/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── pixi.toml                   # Pixi configuration
├── run_experiments.py          # Main execution script
├── models/                     # Neural operator implementations
│   ├── loco.py                 # Local Operator
│   ├── fno.py                  # Fourier Neural Operator
│   └── hybrid.py               # Hybrid model
├── solvers/                    # PDE solvers for data generation
│   ├── burgers.py              # 1D Burgers solver
│   ├── kdv.py                  # 1D KdV solver
│   └── navier_stokes.py        # 2D Navier-Stokes solver
├── experiments/                # Complete experiment scripts
│   ├── burgers_1d.py           # Burgers experiment
│   ├── kdv_1d.py              # KdV experiment
│   └── navier_stokes_2d.py     # Navier-Stokes experiment
├── utils/                      # Utilities for training and visualization
│   ├── training.py             # Training pipeline
│   ├── data.py                 # Dataset handling
│   └── plotting.py             # Visualization functions
├── data/                       # Generated datasets (created automatically)
└── results/                    # Experiment results (created automatically)
```

## 🔧 Usage Examples

### Data Generation and Visualization

```bash
# Generate all datasets (Burgers, KdV, Navier-Stokes)
pixi run python run_experiments.py --mode generate-data

# Create rollout visualization examples
pixi run python run_experiments.py --mode rollout-examples --equation all

# Create specific visualizations
pixi run python run_experiments.py --mode rollout-examples --equation burgers
```

### Basic Training Experiments

```bash
# Single equation, single model (5 seeds by default)
pixi run python run_experiments.py --equation burgers --model loco

# Custom training parameters
pixi run python run_experiments.py --equation kdv --model fno \
    --epochs 200 --batch_size 64 --learning_rate 5e-4 --n_seeds 3

# Use different data/results paths
pixi run python run_experiments.py --equation ns2d --model hybrid \
    --data_path my_data --results_path my_results
```

### Comprehensive Comparisons

```bash
# Compare all models on one equation
pixi run python run_experiments.py --equation burgers --model all

# Run all combinations (WARNING: takes several hours!)
pixi run python run_experiments.py --equation all --model all
```

### Advanced Configuration

```bash
# Override default data generation parameters
pixi run python run_experiments.py --equation burgers --model loco \
    --n_train 2000 --n_test 400 --rollout_steps 20

# Use CPU (default is auto GPU/CPU detection)
pixi run python run_experiments.py --equation kdv --model fno --device cpu

# Set random seed for reproducibility
pixi run python run_experiments.py --equation ns2d --model hybrid --seed 123
```

## 📈 Results and Visualization

### Training Experiments Generate:

1. **Combined Training Curves**: All models on one plot (train/validation split)
2. **Individual Model Plots**: Training progress for each model
3. **Model Comparison**: Statistical comparison with error bars
4. **Model Predictions**: Sample input/output comparisons  
5. **Numerical Results**: Detailed metrics with mean ± std across seeds

### Rollout Visualizations Generate:

1. **Burgers/KdV**: Space-time contour plots showing wave evolution
2. **Navier-Stokes 2D**: Frame sequences for creating animations
3. **Ground Truth Data**: Reference solutions for comparison

Results are organized by equation type:
```
results/
├── burgers/
│   ├── training_comparison.png     # NEW: All models combined
│   ├── loco_training.png
│   ├── model_comparison.png
│   └── results.txt
├── kdv/
├── ns2d/
└── rollout_examples/
    ├── burgers_spacetime_rollout.png
    ├── kdv_spacetime_rollout.png
    └── ns2d_frames/
        ├── frame_001.png
        ├── frame_002.png
        └── ...
```

## ⚙️ Configuration

### Default Parameters

Each equation has optimized default parameters:

- **Burgers**: 256 spatial points, ν=0.01, T=2.0
- **KdV**: 128 spatial points, a=6.0, b=1.0, T=4.0  
- **Navier-Stokes**: 64×64 grid, ν=1e-3, T=10.0

### Model Parameters

- **Hidden channels**: 32
- **Number of blocks**: 4
- **Fourier modes**: 16 (1D) or 12×12 (2D)
- **Training epochs**: 100
- **Batch size**: 32 (1D) or 16 (2D)
- **Statistical robustness**: 5 seeds by default

## 🧪 Reproducing Paper Results

To reproduce the main comparison results from our paper:

```bash
# Step 1: Generate all datasets first
pixi run python run_experiments.py --mode generate-data

# Step 2: Generate all comparison results
pixi run python run_experiments.py --equation all --model all

# Step 3: Create rollout visualizations
pixi run python run_experiments.py --mode rollout-examples --equation all

# Quick validation (reduced parameters)
pixi run python run_experiments.py --equation burgers --model all --n_seeds 3 --epochs 50
```

## 📋 Dependencies

### Core Dependencies
- **PyTorch** (≥1.12): Deep learning framework
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization
- **Tqdm**: Progress bars
- **Pillow**: Image processing for GIFs

### Development Environment
- **Pixi**: Package and environment management (recommended)
- **Python** (≥3.8)

### Installation Options

**Option 1: Using Pixi (Recommended)**
```bash
# Install pixi first: https://github.com/prefix-dev/pixi
pixi install
pixi run python run_experiments.py --help
```

**Option 2: Using pip**
```bash
pip install -r requirements.txt
python run_experiments.py --help
```

**Option 3: Using conda**
```bash
conda create -n neural-ops python=3.9
conda activate neural-ops
pip install -r requirements.txt
```

## 🎯 Operation Modes

The framework supports three main operation modes:

### 1. Training Mode (default)
```bash
pixi run python run_experiments.py --equation burgers --model loco
```
- Trains neural operator models with statistical robustness (5 seeds)
- Generates combined training plots showing all models
- Creates detailed comparison metrics with error bars
- Saves individual model checkpoints and predictions

### 2. Data Generation Mode
```bash
pixi run python run_experiments.py --mode generate-data
```
- Generates all PDE datasets (Burgers, KdV, Navier-Stokes)
- Uses optimized parameters for each equation type
- Automatically caches data for reuse in experiments
- Progress bars show generation status

### 3. Rollout Examples Mode
```bash
pixi run python run_experiments.py --mode rollout-examples --equation all
```
- Creates space-time plots for 1D equations (Burgers, KdV)
- Generates frame sequences for 2D equations (Navier-Stokes)
- Shows ground truth rollout evolution for visualization
- Useful for understanding PDE dynamics

## 💡 Key Features

### ✅ Simple and Reproducible
- Three operation modes for different use cases
- Multi-seed training by default (5 seeds for statistical robustness)
- Automatic data generation and caching
- Cleaner file organization (removed "multiseed" naming)
- Deterministic random seeds

### ✅ Publication Ready
- Combined training plots showing all models together
- Statistical analysis with mean ± std across seeds
- Space-time visualizations for understanding dynamics
- Modular architecture for easy extension
- Standard neural operator implementations

### ✅ Educational
- Clear separation of different neural operator types
- Rollout visualizations show PDE evolution
- Comparative analysis across multiple PDEs
- Well-commented implementations with progress tracking

## 🔬 Research Insights

Our experiments reveal important insights about neural operator architectures:

1. **PDE-Specific Performance**: Different architectures excel on different PDE types
   - FNO works best for dispersive equations (KdV)  
   - LOCO shows advantages for diffusive equations (Burgers)
   - Hybrid models provide robustness across equation types

2. **Spectral Domain Nonlinearity**: LOCO's spectral domain nonlinearity provides benefits for certain PDE characteristics

3. **Parameter Efficiency**: All models achieve similar parameter counts (~600K-2M) for fair comparison

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{neurips2024neural,
  title={Neural Operators for PDEs: Comparative Analysis of LOCO, FNO, and Hybrid Approaches},
  author={[Authors]},
  journal={NeurIPS Workshop on [Workshop Name]},
  year={2024}
}
```

## 🤝 Contributing

This repository is designed for reproducible research. For questions or issues:

1. Check the experiment outputs in `results/` directory
2. Review the configuration in generated JSON files
3. Examine the detailed logs and plots
4. Open an issue with specific error messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🧪 Testing

The repository includes comprehensive tests to ensure correctness:

### Quick Testing
```bash
# Run quick unit tests (recommended first step)
pixi run test

# Run tests with coverage report
pixi run test-coverage

# Test specific components
pixi run test-models    # Neural operator models
pixi run test-solvers   # PDE solvers  
pixi run test-utils     # Training utilities
```

### Full Test Suite
```bash
# Run all tests including slow integration tests
pixi run test-full

# Run specific test categories
python run_tests.py --integration  # Integration tests
python run_tests.py --slow         # Slow tests only
```

### Test Categories
- **Unit tests**: Fast tests for individual components
- **Integration tests**: End-to-end workflow tests
- **Slow tests**: Tests involving data generation and training
- **GPU tests**: Tests requiring CUDA (automatically detected)

---

**Quick Test**: Run `pixi run test` to validate the installation, then try `pixi run test-burgers` for a fast experiment!