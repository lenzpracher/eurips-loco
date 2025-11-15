"""
1D Burgers Equation Solver

This module provides a GPU-accelerated solver for the 1D Burgers equation.
Extracted from old_scripts/Burgers_KdV/burgers_solver.py for clean organization.

The solver uses a semi-implicit Fourier method:
- Linear diffusion term: implicit treatment (stable)
- Nonlinear advection term: explicit treatment
"""

import numpy as np
import torch
from tqdm import tqdm


class BurgersSolver:
    """
    GPU-accelerated 1D Burgers' equation solver using a semi-implicit Fourier method.

    Solves: u_t + u * u_x = nu * u_xx

    In Fourier space:
    û_t + F(u * u_x) = -nu * k^2 * û
    û_t = -nu * k^2 * û - 0.5 * ik * F(u^2)

    The linear diffusion term is treated implicitly (stable).
    The nonlinear advection term is treated explicitly.

    Args:
        N: Number of spatial grid points
        width: Spatial domain width
        nu: Viscosity parameter
        device: Computing device ('cpu' or 'cuda:X')
    """

    def __init__(self, N=256, width=2*np.pi, nu=0.01, device="cpu"):
        self.N = N
        self.width = width
        self.dx = width / N
        self.nu = nu
        self.device = device

        # Wavenumbers for spectral differentiation
        k = 2 * np.pi * torch.fft.fftfreq(N, d=self.dx).to(device)
        # Reshape for broadcasting over batches and channels
        self.k = k.view(1, 1, -1)
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

        # Dealiasing: Zero out the highest 1/3 of frequencies to prevent aliasing instability
        k_max_dealias = int(self.N * (1 / 3))
        u_hat_next[..., k_max_dealias:-k_max_dealias] = 0

        # Return to real space
        return torch.fft.ifft(u_hat_next, dim=-1).real

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
        sol = [u.squeeze(1)]  # Squeeze out channel dim for storage

        pbar = tqdm(range(len(t_eval) - 1), desc=f"Integrating on {u.device}", leave=False)
        for i in pbar:
            dt = t_eval[i + 1] - t_eval[i]
            u = self._step(u, dt)

            # Check for divergence
            if not torch.isfinite(u).all():
                print(f"\\nWARNING: Divergence detected on device {u.device} at time step {i}. Aborting batch.")
                return None  # Indicate failure

            sol.append(u.squeeze(1))

        return torch.stack(sol, dim=1)  # (batch, time, N)

    def get_kinetic_energy(self, u):
        """
        Total kinetic energy E = ∫(1/2 * u^2) dx

        Args:
            u: Solution field(s)

        Returns:
            Kinetic energy tensor
        """
        integrand = 0.5 * u**2
        return integrand.sum(dim=-1) * self.dx


def generate_burgers_initial_conditions(num_runs, N, ic_type='random', width=2*np.pi, scale=1.0, seed=None, device="cpu"):
    """
    Generate initial conditions for Burgers equation

    Args:
        num_runs: Number of initial conditions to generate
        N: Number of spatial grid points
        ic_type: Type of initial conditions ('random' or 'sine')
        width: Spatial domain width
        scale: Scaling factor for initial conditions
        seed: Random seed for reproducibility
        device: Computing device

    Returns:
        Tensor of initial conditions (num_runs, N)
    """
    if seed is not None:
        torch.manual_seed(seed)

    if ic_type == 'random':
        # Create a uniform distribution in the range [-scale, scale)
        ics = 2 * scale * torch.rand(num_runs, N, device=device) - scale
    elif ic_type == 'sine':
        x = torch.linspace(0, width, N + 1, device=device)[:-1]
        ics = torch.zeros(num_runs, N, device=device)
        for i in range(num_runs):
            phase = 2 * np.pi * torch.rand(1).item()
            ics[i, :] = scale * torch.sin(x + phase)
    else:
        raise ValueError(f"Unknown initial condition type: {ic_type}")

    return ics


def run_burgers_simulation_batch(solver, ics, T, dt, save_interval):
    """
    Run a batch of Burgers simulations and compute statistics

    Args:
        solver: BurgersSolver instance
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

    # Compute physical quantities for each saved snapshot
    energy_t = solver.get_kinetic_energy(saved_snapshots)

    results = []
    for i in range(ics.shape[0]):
        results.append({
            "status": "success",
            "snapshots": saved_snapshots[i].cpu(),
            "times": save_times.cpu(),
            "energy_t": energy_t[i].cpu(),
        })
    return results


def generate_burgers_dataset(
    num_runs=100,
    N=512,
    width=2*np.pi,
    nu=0.01,
    T=20.0,
    dt=1e-4,
    save_interval=0.05,
    ic_type='random',
    ic_scale=1.0,
    batch_size=32,
    device='cpu',
    seed=None,
    save_dir='data/Burgers'
):
    """
    Generate a complete Burgers dataset

    Args:
        num_runs: Number of simulations to run
        N: Spatial grid points
        width: Spatial domain width
        nu: Viscosity
        T: Final time
        dt: Time step for integration
        save_interval: Time interval for saving snapshots
        ic_type: Type of initial conditions
        ic_scale: Scale of initial conditions
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
    solver = BurgersSolver(N=N, width=width, nu=nu, device=device)

    # Generate initial conditions
    print(f"Generating {num_runs} Burgers initial conditions...")
    all_ics = generate_burgers_initial_conditions(
        num_runs, N, ic_type=ic_type, width=width,
        scale=ic_scale, seed=seed, device='cpu'  # Generate on CPU first
    )

    # Process in batches
    successful_runs = 0
    failed_runs = 0

    print(f"Running {num_runs} Burgers simulations...")
    for i in tqdm(range(0, num_runs, batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, num_runs)
        batch_ics = all_ics[i:batch_end].to(device)

        # Run batch simulation
        batch_results = run_burgers_simulation_batch(
            solver, batch_ics, T, dt, save_interval
        )

        # Save individual results
        for j, result in enumerate(batch_results):
            run_id = i + j
            result['run_id'] = run_id

            if result['status'] == 'success':
                save_path = os.path.join(save_dir, f"run_{run_id:05d}.pt")
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
            'N': N, 'width': width, 'nu': nu, 'T': T,
            'dt': dt, 'save_interval': save_interval,
            'ic_type': ic_type, 'ic_scale': ic_scale
        }
    }

    print(f"Dataset generation complete: {successful_runs}/{num_runs} successful")
    return stats
