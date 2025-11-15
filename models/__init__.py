"""Neural operator models for PDE solving."""

from .loco import LocalOperator
from .fno import FourierNeuralOperator
from .hybrid import HybridOperator

__all__ = [
    'LocalOperator',
    'FourierNeuralOperator', 
    'HybridOperator'
]