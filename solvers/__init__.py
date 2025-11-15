"""
PDE Solvers Package

This package contains PDE solvers for generating training data.
Each solver can be run independently to generate datasets.

Modules:
- burgers: 1D Burgers equation solver
- kdv: 1D Korteweg-de Vries equation solver
- navier_stokes: 2D Navier-Stokes equation solver
"""

from .burgers import (
    BurgersSolver,
    generate_burgers_dataset,
    generate_burgers_initial_conditions,
)
from .kdv import KdVSolver, generate_kdv_dataset, generate_kdv_initial_conditions
from .navier_stokes import NavierStokes2DSolver, generate_ns2d_dataset

__all__ = [
    'BurgersSolver',
    'generate_burgers_initial_conditions',
    'generate_burgers_dataset',
    'KdVSolver',
    'generate_kdv_initial_conditions',
    'generate_kdv_dataset',
    'NavierStokes2DSolver',
    'generate_ns2d_dataset'
]
