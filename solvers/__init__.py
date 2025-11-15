"""PDE solvers for data generation."""

from .burgers import BurgersSolver
from .kdv import KdVSolver  
from .navier_stokes import NavierStokesSolver

__all__ = [
    'BurgersSolver',
    'KdVSolver',
    'NavierStokesSolver'
]