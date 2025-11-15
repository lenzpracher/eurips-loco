"""
Burgers equation solver for data generation.

Implements a GPU-accelerated 1D Burgers' equation solver using a semi-implicit Fourier method.
Solves u_t + u * u_x = nu * u_xx with periodic boundary conditions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from typing import Optional, Tuple


class BurgersSolver:
    """
    GPU-accelerated 1D Burgers' equation solver using a semi-implicit Fourier method.
    Solves u_t + u * u_x = nu * u_xx
    
    The linear diffusion term (-nu * k^2 * û) is treated implicitly.
    The nonlinear advection term is treated explicitly using dealiasing.
    """
    
    def __init__(self, N: int = 512, width: float = 2*np.pi, nu: float = 0.01, device: str = "cpu"):
        """
        Initialize the Burgers solver.
        
        Args:
            N: Number of spatial points
            width: Spatial domain width
            nu: Viscosity coefficient  
            device: Device to run computations on
        """
        self.N = N
        self.width = width
        self.dx = width / N
        self.nu = nu
        self.device = device

        # Wavenumbers for spectral differentiation
        k = 2 * np.pi * torch.fft.fftfreq(N, d=self.dx).to(device)
        self.k = k.view(1, 1, -1)  # Shape for broadcasting over batches
        self.ik = 1j * self.k

        # Precompute the linear operator for the semi-implicit scheme
        self.linear_operator = -self.nu * self.k**2

    def _nonlinear_term_hat(self, u):
        """Computes the Fourier transform of the nonlinear advection term: -0.5 * ik * F(u^2)"""
        u2_hat = torch.fft.fft(u**2, dim=-1)
        return -0.5 * self.ik * u2_hat

    def _step(self, u, dt):
        """A single step using semi-implicit Euler."""
        u_hat = torch.fft.fft(u, dim=-1)
        nonlinear_hat = self._nonlinear_term_hat(u)
        
        # Denominator for the implicit update of the linear term
        denominator = 1.0 - dt * self.linear_operator
        
        # Update in Fourier space
        u_hat_next = (u_hat + dt * nonlinear_hat) / denominator
        
        # Dealiasing: Zero out the highest 1/3 of frequencies
        k_max_dealias = int(self.N * (1 / 3))
        u_hat_next[..., k_max_dealias:-k_max_dealias] = 0

        # Return to real space
        return torch.fft.ifft(u_hat_next, dim=-1).real

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
            u = self._step(u, dt)

            # Check for divergence
            if not torch.isfinite(u).all():
                print(f"WARNING: Divergence detected at time step {i}")
                return None

            sol.append(u.squeeze(1))

        return torch.stack(sol, dim=1)  # (batch, time, N)

    def get_kinetic_energy(self, u):
        """Total kinetic energy E = ∫(1/2 * u^2) dx."""
        integrand = 0.5 * u**2
        return integrand.sum(dim=-1) * self.dx


def generate_initial_conditions(n_samples: int, N: int = 256, device: str = "cpu", ic_type: str = 'random', scale: float = 1.0, seed: int = None) -> torch.Tensor:
    """
    Generate random initial conditions for Burgers equation.
    
    Args:
        n_samples: Number of initial conditions to generate
        N: Number of spatial points
        device: Device to generate on
        ic_type: Type of initial condition ('random' or 'sine')
        scale: Scale of the initial condition
        seed: Random seed for reproducibility
        
    Returns:
        Initial conditions tensor of shape (n_samples, N)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if ic_type == 'random':
        # Create a uniform distribution in the range [-scale, scale)
        ics = 2 * scale * torch.rand(n_samples, N, device=device) - scale
    elif ic_type == 'sine':
        x = torch.linspace(0, 2*np.pi, N + 1, device=device)[:-1]
        ics = torch.zeros(n_samples, N, device=device)
        for i in range(n_samples):
            phase = 2 * np.pi * torch.rand(1).item()
            ics[i, :] = scale * torch.sin(x + phase)
    else:
        raise ValueError(f"Unknown initial condition type: {ic_type}")
        
    return ics


def generate_burgers_data(
    n_train: int = 1000,
    n_test: int = 200, 
    N: int = 512,
    T: float = 2.5,
    dt: float = 1e-4,
    nu: float = 0.01,
    device: str = "cpu",
    save_path: Optional[str] = None,
    ic_type: str = 'random',
    ic_scale: float = 1.0,
    seed: int = None
) -> Tuple[dict, dict]:
    """
    Generate training and test datasets for Burgers equation.
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        N: Number of spatial points
        T: Final time
        dt: Time step size
        nu: Viscosity coefficient
        device: Device to run on
        save_path: Path to save data (optional)
        ic_type: Type of initial condition ('random' or 'sine')
        ic_scale: Scale of the initial condition
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, test_data) dictionaries
    """
    print(f"Generating Burgers equation data...")
    print(f"Parameters: N={N}, T={T}, dt={dt}, nu={nu}")
    print(f"Train samples: {n_train}, Test samples: {n_test}")
    
    # Time evaluation points
    t_eval = torch.arange(0, T + dt, dt, device=device)
    n_time = len(t_eval)
    
    # Initialize solver
    solver = BurgersSolver(N=N, nu=nu, device=device)
    
    def generate_dataset(n_samples, desc):
        # Generate initial conditions
        u0 = generate_initial_conditions(n_samples, N, device, ic_type, ic_scale, seed)
        
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
            'x': torch.linspace(0, 2*np.pi, N),   # (N,)
            'nu': nu,
            'dt': dt
        }
    
    # Generate datasets
    train_data = generate_dataset(n_train, "Generating train data")
    test_data = generate_dataset(n_test, "Generating test data")
    
    # Save if requested
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        torch.save(train_data, os.path.join(save_path, 'burgers_train.pt'))
        torch.save(test_data, os.path.join(save_path, 'burgers_test.pt'))
        print(f"Data saved to {save_path}")
    
    print("Data generation complete!")
    return train_data, test_data


def load_burgers_data(data_path: str) -> Tuple[dict, dict]:
    """Load Burgers equation data from saved files."""
    train_data = torch.load(os.path.join(data_path, 'burgers_train.pt'))
    test_data = torch.load(os.path.join(data_path, 'burgers_test.pt'))
    return train_data, test_data


if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, test_data = generate_burgers_data(
        n_train=100,  # Small example
        n_test=20,
        device=device,
        save_path="data",
        ic_type='random',  # Use uniform random noise like your script
        ic_scale=1.0,
        seed=42
    )
    
    print(f"Train data shape: {train_data['u'].shape}")
    print(f"Test data shape: {test_data['u'].shape}")
    print(f"Time points: {len(train_data['t'])}")