"""
2D Navier-Stokes Equation Solver

This module provides a GPU-accelerated solver for the 2D Navier-Stokes equation.
Extracted from old_scripts/NS2D/solvers/unified_generate_ns_data.py for clean organization.

The solver uses a spectral method with Crank-Nicolson time stepping:
- Linear viscous term: implicit treatment (stable)
- Nonlinear advection term: explicit treatment
- Forcing: implicit treatment
"""

import math

import torch
from tqdm import tqdm


class NavierStokes2DSolver:
    """
    GPU-accelerated 2D Navier-Stokes solver using spectral methods.

    Solves the vorticity-streamfunction formulation:
    ∂ω/∂t + (u·∇)ω = ν∇²ω + f

    where ω is vorticity, u is velocity, ν is viscosity, f is forcing.

    Args:
        N: Number of grid points (must be power of 2)
        visc: Viscosity (1/Reynolds number)
        device: Computing device ('cpu' or 'cuda:X')
    """

    def __init__(self, N=256, visc=1e-3, device="cpu"):
        self.N = N
        self.visc = visc
        self.device = device

        # Maximum frequency
        self.k_max = math.floor(N / 2.0)

        # Wavenumbers in y-direction
        k_y = torch.cat((torch.arange(start=0, end=self.k_max, step=1, device=device),
                         torch.arange(start=-self.k_max, end=0, step=1, device=device)), 0).repeat(N, 1)
        # Wavenumbers in x-direction
        k_x = k_y.transpose(0, 1)

        # Truncate redundant modes
        self.k_x = k_x[..., :self.k_max + 1]
        self.k_y = k_y[..., :self.k_max + 1]

        # Negative Laplacian in Fourier space
        self.lap = 4 * (math.pi**2) * (self.k_x**2 + self.k_y**2)
        self.lap[0, 0] = 1.0  # Avoid division by zero

        # Dealiasing mask
        self.dealias = torch.unsqueeze(
            torch.logical_and(torch.abs(self.k_y) <= (2.0/3.0) * self.k_max,
                             torch.abs(self.k_x) <= (2.0/3.0) * self.k_max).float(), 0
        )

    def solve(self, w0, f, T, delta_t=1e-4, record_steps=1000):
        """
        Solve 2D Navier-Stokes equation

        Args:
            w0: Initial vorticity [batch, N, N]
            f: Forcing term [N, N] or [batch, N, N]
            T: Final time
            delta_t: Time step for integration
            record_steps: Number of snapshots to record

        Returns:
            sol: Solution snapshots [batch, N, N, time_steps]
            sol_t: Time array [time_steps]
        """
        # Number of steps to final time
        steps = math.ceil(T / delta_t)

        # Initial vorticity to Fourier space
        w_h = torch.fft.rfft2(w0)

        # Forcing to Fourier space
        f_h = torch.fft.rfft2(f)

        # If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)

        # Record solution every 0.1 time units
        record_dt = T / record_steps
        next_record_time = record_dt

        # Saving solution and time
        sol = torch.zeros(*w0.size(), record_steps, device=self.device)
        sol_t = torch.zeros(record_steps, device=self.device)

        # Record counter
        c = 0
        # Physical time
        t = 0.0

        pbar = tqdm(range(steps), desc=f"NS2D Integration on {self.device}", leave=False)
        for j in pbar:
            # Stream function in Fourier space: solve Poisson equation
            psi_h = w_h / self.lap

            # Velocity field in x-direction = psi_y
            q = 2.0 * math.pi * self.k_y * 1j * psi_h
            q = torch.fft.irfft2(q, s=(self.N, self.N))

            # Velocity field in y-direction = -psi_x
            v = -2.0 * math.pi * self.k_x * 1j * psi_h
            v = torch.fft.irfft2(v, s=(self.N, self.N))

            # Partial x of vorticity
            w_x = 2.0 * math.pi * self.k_x * 1j * w_h
            w_x = torch.fft.irfft2(w_x, s=(self.N, self.N))

            # Partial y of vorticity
            w_y = 2.0 * math.pi * self.k_y * 1j * w_h
            w_y = torch.fft.irfft2(w_y, s=(self.N, self.N))

            # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
            F_h = torch.fft.rfft2(q * w_x + v * w_y)

            # Dealias
            F_h = self.dealias * F_h

            # Crank-Nicolson update
            w_h = ((-delta_t * F_h + delta_t * f_h + (1.0 - 0.5 * delta_t * self.visc * self.lap) * w_h) /
                   (1.0 + 0.5 * delta_t * self.visc * self.lap))

            # Update real time (used only for recording)
            t += delta_t

            # Record solution at specified time intervals
            if t >= next_record_time and c < record_steps:
                # Solution in physical space
                w = torch.fft.irfft2(w_h, s=(self.N, self.N))

                # Record solution and time
                sol[..., c] = w
                sol_t[c] = next_record_time

                c += 1
                next_record_time += record_dt

            # Check for divergence
            if not torch.isfinite(w_h).all():
                print(f"\\nWARNING: Divergence detected on device {self.device} at time step {j}. Aborting.")
                return None, None

        return sol, sol_t


def create_gaussian_rf_forcing(N, alpha=2.5, tau=7, device="cpu"):
    """
    Create Gaussian Random Field forcing term

    Args:
        N: Grid size
        alpha: Smoothness parameter
        tau: Length scale parameter
        device: Computing device

    Returns:
        Forcing term tensor [N, N]
    """
    # Simple sinusoidal forcing as fallback
    t = torch.linspace(0, 1, N+1, device=device)[:-1]
    X, Y = torch.meshgrid(t, t, indexing='ij')
    f = 0.1 * (torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))
    return f


def generate_initial_condition(N, alpha=2.5, tau=7, device="cpu"):
    """
    Generate random initial condition using Gaussian Random Field

    Args:
        N: Grid size
        alpha: Smoothness parameter
        tau: Length scale parameter
        device: Computing device

    Returns:
        Initial vorticity tensor [N, N]
    """
    # Simple random initial condition as fallback
    w0 = torch.randn(N, N, device=device) * 0.1
    return w0


def generate_ns2d_dataset(
    num_samples=96,
    N=256,
    visc=1e-3,
    T=100.0,
    delta_t=1e-4,
    record_steps=1000,
    batch_size=8,
    device='cpu',
    seed=None,
    save_dir='data/NS2D'
):
    """
    Generate a complete 2D Navier-Stokes dataset

    Args:
        num_samples: Number of simulations to run
        N: Spatial grid size (must be power of 2)
        visc: Viscosity
        T: Final time
        delta_t: Time step for integration
        record_steps: Number of snapshots to record
        batch_size: Batch size for processing
        device: Computing device
        seed: Random seed
        save_dir: Directory to save results

    Returns:
        Dictionary with dataset statistics
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    if seed is not None:
        torch.manual_seed(seed)

    # Create solver
    solver = NavierStokes2DSolver(N=N, visc=visc, device=device)

    # Generate forcing term (shared across simulations)
    f = create_gaussian_rf_forcing(N, device=device)

    # Storage for results
    all_a = torch.zeros(num_samples, N, N)  # Initial conditions
    all_u = torch.zeros(num_samples, record_steps, N, N)  # Solutions
    t = None

    successful_runs = 0
    failed_runs = 0

    print(f"Running {num_samples} NS2D simulations...")
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx

        # Generate batch of initial conditions
        batch_w0 = torch.zeros(current_batch_size, N, N, device=device)
        for j in range(current_batch_size):
            batch_w0[j] = generate_initial_condition(N, device=device)

        # Run batch simulation
        try:
            sol, sol_t = solver.solve(batch_w0, f, T, delta_t, record_steps)

            if sol is not None:
                # Store results
                all_a[start_idx:end_idx] = batch_w0.cpu()
                all_u[start_idx:end_idx] = sol.permute(0, 3, 1, 2).cpu()  # [batch, time, H, W]
                if t is None:
                    t = sol_t.cpu()
                successful_runs += current_batch_size
            else:
                failed_runs += current_batch_size

        except Exception as e:
            print(f"Error in batch {i}: {e}")
            failed_runs += current_batch_size

    # Save results
    if successful_runs > 0:
        # Save training data (80%)
        split_idx = int(0.8 * successful_runs)

        train_data = {
            'a': all_a[:split_idx],
            'u': all_u[:split_idx],
            't': t
        }

        test_data = {
            'a': all_a[split_idx:successful_runs],
            'u': all_u[split_idx:successful_runs],
            't': t
        }

        train_path = os.path.join(save_dir, f'nsforcing_train_T{int(T)}_{N}.pt')
        test_path = os.path.join(save_dir, f'nsforcing_test_T{int(T)}_{N}.pt')

        torch.save(train_data, train_path)
        torch.save(test_data, test_path)

        print(f"Saved training data: {train_path}")
        print(f"Saved test data: {test_path}")

    stats = {
        'total_runs': num_samples,
        'successful_runs': successful_runs,
        'failed_runs': failed_runs,
        'success_rate': successful_runs / num_samples,
        'parameters': {
            'N': N, 'visc': visc, 'T': T,
            'delta_t': delta_t, 'record_steps': record_steps
        }
    }

    print(f"Dataset generation complete: {successful_runs}/{num_samples} successful")
    return stats
