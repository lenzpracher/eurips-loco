"""
1D Korteweg-de Vries (KdV) Equation Solver

This module provides a GPU-accelerated solver for the 1D KdV equation.
Extracted from old_scripts/Burgers_KdV/kdv_solver.py for clean organization.

The solver uses a conservative Fourier split-step method:
- Linear dispersive term: spectral treatment (exact)
- Nonlinear advection term: real space RK4 treatment
"""

import numpy as np
import torch
from tqdm import tqdm


class KdVSolver:
    """
    GPU-accelerated, vectorized KdV solver using a conservative Fourier split-step method.

    Solves: u_t + a*u*u_x + b*u_xxx = 0
    Standard KdV corresponds to a=6, b=1.

    Args:
        N: Number of spatial grid points
        width: Spatial domain width
        a: Nonlinear coefficient (default: 6.0)
        b: Dispersion coefficient (default: 1.0)
        device: Computing device ('cpu' or 'cuda:X')
    """

    def __init__(self, N=128, width=2*np.pi, a=6.0, b=1.0, device="cpu"):
        self.N = N
        self.width = width
        self.dx = width / N
        self.a = a
        self.b = b
        self.device = device

        # Wavenumbers for spectral differentiation
        k = 2 * np.pi * torch.fft.fftfreq(N, d=self.dx).to(device)
        # Reshape for broadcasting over batches and channels
        self.k = k.view(1, 1, -1)
        self.ik = 1j * self.k

        # Precompute the linear operator for the split-step method
        # Equation for the linear part: u_t = -b * u_xxx
        # In Fourier space: u_hat_t = -b * (ik)^3 * u_hat = i * b * k^3 * u_hat
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

        # Dealiasing: Zero out the highest 1/3 of frequencies to prevent aliasing instability
        N = self.N
        k_max_dealias = int(N * (1 / 3))
        u_half_1_hat[..., k_max_dealias:-k_max_dealias] = 0

        u_hat_linear = u_half_1_hat * torch.exp(self.linear_operator * dt)
        u_full_linear = torch.fft.ifft(u_hat_linear, dim=-1)

        # Another half step on nonlinear part in real space
        u_next = self._rk4_step_nonlinear(u_full_linear, dt / 2.0)

        return u_next.real

    def integrate(self, u0, t_eval):
        """
        Integrates a batch of initial conditions.

        Args:
            u0: Initial conditions, shape (batch_size, N) or (batch_size, 1, N)
            t_eval: 1D tensor of time points

        Returns:
            Solution tensor (batch_size, time, N) or None if divergence occurs
        """
        if len(u0.shape) == 2:
            u0 = u0.unsqueeze(1)  # Add channel dim: (batch, 1, N)

        u = u0

        # Store initial state
        sol = [u.squeeze(1)]  # Squeeze out channel dim for storage

        pbar = tqdm(range(len(t_eval) - 1), desc=f"Integrating on {u.device}", leave=False)
        for i in pbar:
            dt = t_eval[i + 1] - t_eval[i]
            u = self._strang_splitting_step(u, dt)

            # Check for divergence
            if not torch.isfinite(u).all():
                print(f"\\nWARNING: Divergence detected on device {u.device} at time step {i}. Aborting batch.")
                return None  # Indicate failure

            sol.append(u.squeeze(1))

        return torch.stack(sol, dim=1)  # (batch, time, N)

    def get_mass(self, u):
        """Mass (M = ∫u dx)"""
        return u.sum(dim=-1) * self.dx

    def get_momentum(self, u):
        """Momentum (P = ∫u^2 dx) - standard definition"""
        return (u**2).sum(dim=-1) * self.dx

    def get_energy(self, u):
        """Hamiltonian/Energy (H = ∫(b/2 * u_x^2 - a/6 * u^3) dx)"""
        u_hat = torch.fft.fft(u, dim=-1)
        u_x = torch.fft.ifft(self.ik * u_hat, dim=-1).real

        integrand = (self.b / 2.0) * u_x**2 - (self.a / 6.0) * u**3
        return integrand.sum(dim=-1) * self.dx


def generate_kdv_initial_conditions(num_runs, N, width=2*np.pi, seed=None, device="cpu"):
    """
    Generate initial conditions for KdV equation (two-soliton superposition)

    Args:
        num_runs: Number of initial conditions to generate
        N: Number of spatial grid points
        width: Spatial domain width
        seed: Random seed for reproducibility
        device: Computing device

    Returns:
        Tensor of initial conditions (num_runs, N)
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.arange(N, device=device) * (width / N)
    ics = []

    for _i in range(num_runs):
        c1 = torch.rand(1, device=device) * 1.5 + 0.5  # Speed/amplitude
        c2 = torch.rand(1, device=device) * 1.5 + 0.5
        x1 = torch.rand(1, device=device) * width
        x2 = torch.rand(1, device=device) * width

        # Ensure solitons are somewhat separated
        if torch.abs(x1 - x2) < width / 4:
            x2 = (x1 + width / 2) % width

        ic = create_two_soliton_ic(x, c1, x1, c2, x2)
        ics.append(ic)

    return torch.stack(ics, dim=0)  # (num_runs, N)


def create_two_soliton_ic(x, c1, x1, c2, x2):
    """Generates a two-soliton solution as a torch tensor."""
    def inverse_cosh(z):
        return 1 / torch.cosh(z)

    soliton1 = 2 * c1**2 * inverse_cosh(c1 * (x - x1)) ** 2
    soliton2 = 2 * c2**2 * inverse_cosh(c2 * (x - x2)) ** 2
    return soliton1 + soliton2


def run_kdv_simulation_batch(solver, ics, T, dt, save_interval):
    """
    Run a batch of KdV simulations and compute conservation laws

    Args:
        solver: KdVSolver instance
        ics: Initial conditions (batch_size, N)
        T: Final time
        dt: Time step
        save_interval: Interval for saving snapshots

    Returns:
        List of results dictionaries for each simulation
    """
    # Time points for integration and saving
    t_eval = torch.arange(0, T + dt, dt, device=solver.device)
    save_times = torch.arange(0, T + save_interval, save_interval, device=solver.device)

    # Run integration
    snapshots = solver.integrate(ics, t_eval)

    if snapshots is None:
        return [{"status": "failed", "error_message": "Divergence during integration"} for _ in range(ics.shape[0])]

    # Downsample to save_times
    save_indices = torch.searchsorted(t_eval, save_times)
    save_indices.clamp_max_(snapshots.shape[1] - 1)
    saved_snapshots = snapshots[:, save_indices, :]

    # Compute conservation laws for each saved snapshot
    mass_t = solver.get_mass(saved_snapshots)
    momentum_t = solver.get_momentum(saved_snapshots)
    energy_t = solver.get_energy(saved_snapshots)

    results = []
    for i in range(ics.shape[0]):
        results.append({
            "status": "success",
            "snapshots": saved_snapshots[i].cpu(),
            "times": save_times.cpu(),
            "mass_t": mass_t[i].cpu(),
            "momentum_t": momentum_t[i].cpu(),
            "energy_t": energy_t[i].cpu(),
        })
    return results


def generate_kdv_dataset(
    num_runs=1000,
    N=128,
    width=2*np.pi,
    a=6.0,
    b=1.0,
    T=2.0,
    dt=0.001,
    save_interval=0.001,
    batch_size=32,
    device='cpu',
    seed=None,
    save_dir='data/KdV'
):
    """
    Generate a complete KdV dataset

    Args:
        num_runs: Number of simulations to run
        N: Spatial grid points
        width: Spatial domain width
        a: Nonlinear coefficient
        b: Dispersion coefficient
        T: Final time
        dt: Time step for integration
        save_interval: Time interval for saving snapshots
        batch_size: Batch size for processing
        device: Computing device
        seed: Random seed
        save_dir: Directory to save results

    Returns:
        Dictionary with dataset statistics
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Create solver
    solver = KdVSolver(N=N, width=width, a=a, b=b, device=device)

    # Generate initial conditions
    print(f"Generating {num_runs} KdV initial conditions...")
    all_ics = generate_kdv_initial_conditions(
        num_runs, N, width=width, seed=seed, device='cpu'  # Generate on CPU first
    )

    # Process in batches
    successful_runs = 0
    failed_runs = 0

    print(f"Running {num_runs} KdV simulations...")
    for i in tqdm(range(0, num_runs, batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, num_runs)
        batch_ics = all_ics[i:batch_end].to(device)

        # Run batch simulation
        batch_results = run_kdv_simulation_batch(
            solver, batch_ics, T, dt, save_interval
        )

        # Save individual results
        for j, result in enumerate(batch_results):
            run_id = i + j
            result['run_id'] = run_id

            if result['status'] == 'success':
                save_path = os.path.join(save_dir, f"simulation_run_{run_id}.pt")
                torch.save(result, save_path)
                successful_runs += 1
            else:
                failed_runs += 1

    stats = {
        'total_runs': num_runs,
        'successful_runs': successful_runs,
        'failed_runs': failed_runs,
        'success_rate': successful_runs / num_runs,
        'parameters': {
            'N': N, 'width': width, 'a': a, 'b': b, 'T': T,
            'dt': dt, 'save_interval': save_interval
        }
    }

    print(f"Dataset generation complete: {successful_runs}/{num_runs} successful")
    return stats
