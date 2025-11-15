# Implementation for "LOCO: Abstracting Spectral Numerical Integrators for PDEs into Neural Operators"

Simplified, streamlined implementation of neural operator experiments for partial differential equations (PDEs). This repository contains implementations for 1D (Burgers, Korteweg-de Vries) and 2D (Navier-Stokes) equation experiments using neural operators.

## Overview

This codebase focuses on the three essential neural operator models with streamlined training workflows and automatic visualization generation.

### Supported Models

- **LOCO** (Local Convolution Operator)
- **Hybrid** (FNO + LOCO Neural Operator)  
- **FNO** (Fourier Neural Operator)

### Supported Experiments

- **Burgers 1D**: 1D viscous Burgers equation
- **KdV 1D**: 1D Korteweg-de Vries equation
- **Navier-Stokes 2D**: 2D incompressible Navier-Stokes equation

## Quick Start

### Prerequisites

This project uses [pixi](https://pixi.sh) for dependency management:

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies
pixi install
```

### Running Experiments

Each training command automatically generates plots and analysis:

```bash
# 1D Experiments (20 parallel runs each)
pixi run burgers      # Train all models + generate plots
pixi run kdv          # Train all models + generate plots

# 2D Experiments (5 parallel runs)
pixi run ns2d         # Train all models + generate plots

# Additional plot generation
pixi run burgers-plot  # Just generate plots from existing results
pixi run kdv-plot      # Just generate plots from existing results  
pixi run ns2d-plot     # Just generate plots from existing results

# 2D Rollout analysis
pixi run ns2d-rollout  # Long-term prediction analysis
```

### Data

Download pre-generated datasets from Hugging Face:

```bash
# Download all datasets (recommended - these are already aggregated)
hf download neurips-loco/Burgers --repo-type dataset --local-dir data/Burgers
hf download neurips-loco/KdV --repo-type dataset --local-dir data/KdV  
hf download neurips-loco/NS2D --repo-type dataset --local-dir data/NS2D
```

## Model Configurations

### 1D Experiments (20 parallel runs each)

| Parameter | Burgers | KdV | 
|-----------|---------|-----|
| `MODES` | 16 | 16 |
| `HIDDEN_CHANNELS` | 24 | 26 |
| `EPOCHS` | 100 | 100 |
| `PARALLEL_RUNS` | 20 | 20 |

### 2D Experiments (5 parallel runs)

| Parameter | Value |
|-----------|-------|
| `MODES_X, MODES_Y` | 12, 12 |
| `HIDDEN_CHANNELS` | 32 |
| `EPOCHS` | 1500 |
| `PARALLEL_RUNS` | 5 |