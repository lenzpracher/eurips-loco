"""
2D Neural Operator Models Package

This package contains 2D implementations of neural operator models
for use with 2D PDE experiments (Navier-Stokes).

Models:
- fno: 2D Fourier Neural Operator
- loco: 2D LOCO - Spectral Neural Operator
- hybrid: 2D Hybrid (previously NFNO_SNO) - Combined FNO/Spectral approach
"""

from .hybrid import Hybrid
from .loco import LOCO
