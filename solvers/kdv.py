"""
Korteweg-de Vries (KdV) equation solver for data generation.

Implements a GPU-accelerated 1D KdV solver using a conservative Fourier split-step method.
Solves u_t + a*u*u_x + b*u_xxx = 0 with periodic boundary conditions.
Standard KdV corresponds to a=6, b=1.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from typing import Optional, Tuple


class KdVSolver:
    """
    GPU-accelerated, vectorized KdV solver using a conservative Fourier split-step method.
    Solves u_t + a*u*u_x + b*u_xxx = 0.
    Standard KdV corresponds to a=6, b=1.
    """

    def __init__(self, N: int = 128, width: float = 2 * np.pi, a: float = 6.0, b: float = 1.0, device: str = "cpu"):
        """
        Initialize the KdV solver.
        
        Args:
            N: Number of spatial points
            width: Spatial domain width
            a: Nonlinear coefficient (standard KdV: a=6)
            b: Dispersion coefficient (standard KdV: b=1)
            device: Device to run computations on
        """
        self.N = N
        self.width = width
        self.dx = width / N
        self.a = a
        self.b = b
        self.device = device

        # Wavenumbers for spectral differentiation
        k = 2 * np.pi * torch.fft.fftfreq(N, d=self.dx).to(device)
        self.k = k.view(1, 1, -1)  # Shape for broadcasting over batches
        self.ik = 1j * self.k

        # Precompute the linear operator for the split-step method
        # Linear part: u_t = -b * u_xxx → u_hat_t = i * b * k^3 * u_hat
        self.linear_operator = 1j * self.b * self.k**3

    def _nonlinear_dudt(self, u):
        """Computes the time derivative of the nonlinear part: u_t = -a*u*u_x"""
        u_hat = torch.fft.fft(u, dim=-1)
        u_x = torch.fft.ifft(self.ik * u_hat, dim=-1).real
        return -self.a * u * u_x

    def _rk4_step_nonlinear(self, u, dt):
        """A single RK4 step on the nonlinear part in real space."""
        k1 = dt * self._nonlinear_dudt(u)
        k2 = dt * self._nonlinear_dudt(u + 0.5 * k1)
        k3 = dt * self._nonlinear_dudt(u + 0.5 * k2)
        k4 = dt * self._nonlinear_dudt(u + k3)
        return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    def _strang_splitting_step(self, u, dt):
        """A single conservative step using Strang splitting with dealiasing."""
        # Half step on nonlinear part in real space
        u_half_1 = self._rk4_step_nonlinear(u, dt / 2.0)

        # Full step on linear part in Fourier space
        u_half_1_hat = torch.fft.fft(u_half_1, dim=-1)

        # Dealiasing: Zero out the highest 1/3 of frequencies
        k_max_dealias = int(self.N * (1 / 3))
        u_half_1_hat[..., k_max_dealias:-k_max_dealias] = 0

        u_hat_linear = u_half_1_hat * torch.exp(self.linear_operator * dt)
        u_full_linear = torch.fft.ifft(u_hat_linear, dim=-1)

        # Another half step on nonlinear part in real space
        u_next = self._rk4_step_nonlinear(u_full_linear, dt / 2.0)

        return u_next.real

    def integrate(self, u0: torch.Tensor, t_eval: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Integrates a batch of initial conditions.
        
        Args:
            u0: Initial conditions, shape (batch_size, N) or (batch_size, 1, N)
            t_eval: Time points to evaluate at, shape (n_time,)
            
        Returns:
            Solution tensor of shape (batch_size, n_time, N) or None if divergence occurs
        """
        if len(u0.shape) == 2:
            u0 = u0.unsqueeze(1)  # Add channel dim: (batch, 1, N)

        u = u0
        sol = [u.squeeze(1)]  # Remove channel dim for storage

        for i in range(len(t_eval) - 1):
            dt = t_eval[i + 1] - t_eval[i]
            u = self._strang_splitting_step(u, dt)

            # Check for divergence
            if not torch.isfinite(u).all():
                print(f"WARNING: Divergence detected at time step {i}")
                return None

            sol.append(u.squeeze(1))

        return torch.stack(sol, dim=1)  # (batch, time, N)

    def get_mass(self, u):
        """Mass (first conserved quantity): ∫u dx"""
        return u.sum(dim=-1) * self.dx

    def get_momentum(self, u):
        """Momentum (second conserved quantity): ∫u^2 dx"""
        return (u**2).sum(dim=-1) * self.dx

    def get_energy(self, u):
        """Energy (third conserved quantity): ∫(u^3 - (u_x)^2) dx"""
        u_hat = torch.fft.fft(u, dim=-1)
        u_x = torch.fft.ifft(self.ik.squeeze() * u_hat, dim=-1).real
        integrand = u**3 - u_x**2
        return integrand.sum(dim=-1) * self.dx


def generate_soliton_initial_conditions(n_samples: int, N: int = 128, device: str = "cpu") -> torch.Tensor:
    """
    Generate soliton-based initial conditions for KdV equation.
    
    Uses combinations of solitons with random parameters.
    
    Args:
        n_samples: Number of initial conditions to generate
        N: Number of spatial points
        device: Device to generate on
        
    Returns:
        Initial conditions tensor of shape (n_samples, N)
    """
    x = torch.linspace(-np.pi, np.pi, N, device=device)
    u0_list = []
    
    for _ in range(n_samples):
        # Random number of solitons (1-3)
        n_solitons = np.random.randint(1, 4)
        u0 = torch.zeros(N, device=device)
        
        for _ in range(n_solitons):
            # Random soliton parameters
            amplitude = np.random.uniform(0.5, 3.0)  # Height
            center = np.random.uniform(-np.pi/2, np.pi/2)  # Position
            width = np.random.uniform(0.3, 1.0)  # Width parameter
            
            # Soliton formula: amplitude * sech^2(sqrt(amplitude/12) * (x - center) / width)
            arg = np.sqrt(amplitude / 12) * (x - center) / width
            soliton = amplitude * (1 / torch.cosh(arg))**2
            u0 += soliton
            
        u0_list.append(u0)
    
    return torch.stack(u0_list)


def generate_random_initial_conditions(n_samples: int, N: int = 128, device: str = "cpu") -> torch.Tensor:
    """
    Generate random smooth initial conditions for KdV equation.
    
    Uses Fourier series with random coefficients.
    
    Args:
        n_samples: Number of initial conditions to generate
        N: Number of spatial points
        device: Device to generate on
        
    Returns:
        Initial conditions tensor of shape (n_samples, N)
    """
    x = torch.linspace(-np.pi, np.pi, N, device=device)
    u0_list = []
    
    for _ in range(n_samples):
        # Random Fourier series
        n_modes = np.random.randint(3, 8)  # 3-7 modes
        u0 = torch.zeros(N, device=device)
        
        for k in range(1, n_modes + 1):
            amplitude = np.random.uniform(0.1, 1.0) / k  # Decay with frequency
            phase = np.random.uniform(0, 2*np.pi)
            u0 += amplitude * torch.sin(k * x + phase)
            
        u0_list.append(u0)
    
    return torch.stack(u0_list)


def generate_kdv_data(
    n_train: int = 1000,
    n_test: int = 200, 
    N: int = 128,
    T: float = 4.0,
    dt: float = 0.01,
    a: float = 6.0,
    b: float = 1.0,
    device: str = "cpu",
    use_solitons: bool = True,
    save_path: Optional[str] = None
) -> Tuple[dict, dict]:
    """
    Generate training and test datasets for KdV equation.
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        N: Number of spatial points
        T: Final time
        dt: Time step size
        a: Nonlinear coefficient
        b: Dispersion coefficient
        device: Device to run on
        use_solitons: Whether to use soliton-based initial conditions
        save_path: Path to save data (optional)
        
    Returns:
        Tuple of (train_data, test_data) dictionaries
    """
    print(f"Generating KdV equation data...")
    print(f"Parameters: N={N}, T={T}, dt={dt}, a={a}, b={b}")
    print(f"Train samples: {n_train}, Test samples: {n_test}")
    print(f"Initial conditions: {'soliton-based' if use_solitons else 'random Fourier'}")
    
    # Time evaluation points
    t_eval = torch.arange(0, T + dt, dt, device=device)
    n_time = len(t_eval)
    
    # Initialize solver
    solver = KdVSolver(N=N, a=a, b=b, device=device)
    
    def generate_dataset(n_samples, desc):
        # Generate initial conditions
        if use_solitons:
            u0 = generate_soliton_initial_conditions(n_samples, N, device)
        else:
            u0 = generate_random_initial_conditions(n_samples, N, device)
        
        # Solve in batches to manage memory
        batch_size = min(50, n_samples)
        all_solutions = []
        
        for i in tqdm(range(0, n_samples, batch_size), desc=desc):
            end_idx = min(i + batch_size, n_samples)
            batch_u0 = u0[i:end_idx]
            
            # Integrate
            sol = solver.integrate(batch_u0, t_eval)
            if sol is None:
                raise RuntimeError(f"Integration failed for batch {i//batch_size}")
                
            all_solutions.append(sol.cpu())
        
        return {
            'u': torch.cat(all_solutions, dim=0),  # (n_samples, n_time, N)
            't': t_eval.cpu(),                     # (n_time,)
            'x': torch.linspace(-np.pi, np.pi, N), # (N,)
            'a': a,
            'b': b,
            'dt': dt
        }
    
    # Generate datasets
    train_data = generate_dataset(n_train, "Generating train data")
    test_data = generate_dataset(n_test, "Generating test data")
    
    # Save if requested
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        torch.save(train_data, os.path.join(save_path, 'kdv_train.pt'))
        torch.save(test_data, os.path.join(save_path, 'kdv_test.pt'))
        print(f"Data saved to {save_path}")
    
    print("Data generation complete!")
    return train_data, test_data


def load_kdv_data(data_path: str) -> Tuple[dict, dict]:
    """Load KdV equation data from saved files."""
    train_data = torch.load(os.path.join(data_path, 'kdv_train.pt'))
    test_data = torch.load(os.path.join(data_path, 'kdv_test.pt'))
    return train_data, test_data


if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, test_data = generate_kdv_data(
        n_train=100,  # Small example
        n_test=20,
        device=device,
        save_path="data"
    )
    
    print(f"Train data shape: {train_data['u'].shape}")
    print(f"Test data shape: {test_data['u'].shape}")
    print(f"Time points: {len(train_data['t'])}")