import torch
import math
from timeit import default_timer
import os
import argparse
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm
from typing import Optional, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    _plotting_enabled = True
except ImportError:
    _plotting_enabled = False


class NavierStokesSolver:
    """
    GPU-accelerated 2D Navier-Stokes equation solver using a Fourier spectral method.
    Solves the vorticity formulation of the incompressible Navier-Stokes equations.
    """

    def __init__(self, N: int = 256, width: float = 2 * np.pi, viscosity: float = 1e-3, device: str = "cpu"):
        """
        Initialize the Navier-Stokes solver.

        Args:
            N: Number of spatial points in each dimension (NxN grid)
            width: Spatial domain width
            viscosity: Viscosity coefficient (nu)
            device: Device to run computations on
        """
        self.N = N
        self.width = width
        self.dx = width / N
        self.viscosity = viscosity
        self.device = device

        # Precompute wavenumbers for spectral differentiation
        k = 2 * np.pi * torch.fft.fftfreq(N, d=self.dx).to(device)
        self.k_max = math.floor(N/2.0)
        
        # Wavenumbers in y-direction
        self.k_y = torch.cat((
            torch.arange(start=0, end=self.k_max, step=1, device=device),
            torch.arange(start=-self.k_max, end=0, step=1, device=device)
        ), 0).repeat(N, 1)
        
        # Wavenumbers in x-direction
        self.k_x = self.k_y.transpose(0, 1)
        
        # Truncate redundant modes
        self.k_x = self.k_x[..., :self.k_max + 1]
        self.k_y = self.k_y[..., :self.k_max + 1]
        
        # Negative Laplacian in Fourier space
        self.lap = 4*(math.pi**2)*(self.k_x**2 + self.k_y**2)
        self.lap[0, 0] = 1.0
        
        # Dealiasing mask
        self.dealias = torch.unsqueeze(
            torch.logical_and(
                torch.abs(self.k_y) <= (2.0/3.0)*self.k_max,
                torch.abs(self.k_x) <= (2.0/3.0)*self.k_max
            ).float(), 0)

    def _navier_stokes_2d_step(self, w_h, f_h, delta_t):
        """
        A single step of the Navier-Stokes solver in Fourier space.
        
        Args:
            w_h: Vorticity in Fourier space
            f_h: Forcing term in Fourier space
            delta_t: Time step
            
        Returns:
            Updated vorticity in Fourier space
        """
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / self.lap

        # Velocity field in x-direction = psi_y
        q = 2. * math.pi * self.k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(self.N, self.N))

        # Velocity field in y-direction = -psi_x
        v = -2. * math.pi * self.k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(self.N, self.N))

        # Partial x of vorticity
        w_x = 2. * math.pi * self.k_x * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(self.N, self.N))

        # Partial y of vorticity
        w_y = 2. * math.pi * self.k_y * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(self.N, self.N))

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.rfft2(q*w_x + v*w_y)

        # Dealias
        F_h = self.dealias * F_h

        # Crank-Nicolson update
        w_h_next = (-delta_t*F_h + delta_t*f_h + (1.0 - 0.5*delta_t*self.viscosity*self.lap)*w_h) / (1.0 + 0.5*delta_t*self.viscosity*self.lap)
        
        return w_h_next

    def integrate(self, w0: torch.Tensor, f: torch.Tensor, T: float, delta_t: float = 1e-4) -> Optional[torch.Tensor]:
        """
        Integrates a batch of initial conditions.
        
        Args:
            w0: Initial vorticity conditions, shape (batch_size, N, N)
            f: Forcing term, shape (N, N) or (batch_size, N, N)
            T: Final time
            delta_t: Time step size
            
        Returns:
            Solution tensor of shape (batch_size, record_steps, N, N) or None if divergence occurs
        """
        # Record solution every 0.1 time units
        record_dt = 0.1
        next_record_time = record_dt
        
        # Number of steps to final time
        steps = math.ceil(T/delta_t)
        
        # Number of record steps
        record_steps = int(T / record_dt) + 1
        
        # Initial vorticity to Fourier space
        w_h = torch.fft.rfft2(w0)
        
        # Forcing to Fourier space
        f_h = torch.fft.rfft2(f)
        
        # If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)
            
        # Saving solution and time
        sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
        sol_t = torch.zeros(record_steps, device=w0.device)
        
        # Record counter
        c = 0
        # Physical time
        t = 0.0
        
        # Record initial condition
        sol[..., c] = w0
        sol_t[c] = t
        c += 1
        
        for j in range(steps):
            # Update vorticity
            w_h = self._navier_stokes_2d_step(w_h, f_h, delta_t)
            
            # Update real time
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
                if not torch.isfinite(w).all():
                    print(f"WARNING: Divergence detected at time step {j}")
                    return None

        # Permute to match expected output format (batch, time, N, N)
        return sol.permute(0, 3, 1, 2)  # (batch, record_steps, N, N)


class GaussianRF(object):

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real 

#w0: initial vorticity
#f: forcing term
#visc: viscosity (1/Re)
#T: final time
#delta_t: internal time-step for solve (descrease if blow-up)
#record_steps: number of in-time snapshots to record
def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):

    #Grid size - must be power of 2
    N = w0.size()[-1]

    #Maximum frequency
    k_max = math.floor(N/2.0)

    #Number of steps to final time
    steps = math.ceil(T/delta_t)

    #Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)

    #Forcing to Fourier space
    f_h = torch.fft.rfft2(f)

    #If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    #Record solution every 0.1 time units
    record_dt = 0.1
    next_record_time = record_dt

    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)

    #Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    #Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0
    #Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    #Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    #Record counter
    c = 0
    #Physical time
    t = 0.0
    for j in range(steps):
        #Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        #Velocity field in x-direction = psi_y
        q = 2. * math.pi * k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N, N))

        #Velocity field in y-direction = -psi_x
        v = -2. * math.pi * k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N, N))

        #Partial x of vorticity
        w_x = 2. * math.pi * k_x * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(N, N))

        #Partial y of vorticity
        w_y = 2. * math.pi * k_y * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(N, N))

        #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.rfft2(q*w_x + v*w_y)

        #Dealias
        F_h = dealias* F_h

        #Crank-Nicolson update
        w_h = (-delta_t*F_h + delta_t*f_h + (1.0 - 0.5*delta_t*visc*lap)*w_h)/(1.0 + 0.5*delta_t*visc*lap)

        #Update real time (used only for recording)
        t += delta_t

        # Record solution at specified time intervals (every 0.1 time units)
        if t >= next_record_time and c < record_steps:
            #Solution in physical space
            w = torch.fft.irfft2(w_h, s=(N, N))

            #Record solution and time
            sol[...,c] = w
            sol_t[c] = next_record_time

            c += 1
            next_record_time += record_dt
    return sol, sol_t

def generation_worker(gpu_id, num_samples, batch_size, data_dir, output_file_template, sim_time, time_step, record_steps):
    device = torch.device(f'cuda:{gpu_id}')
    print(f"Worker started on GPU {gpu_id} for {num_samples} samples.")

    s = 256
    N = num_samples

    if N <= 0:
        print(f"Worker on GPU {gpu_id} has no samples to generate. Exiting.")
        return

    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

    t = torch.linspace(0, 1, s+1, device=device)
    t = t[0:-1]
    X,Y = torch.meshgrid(t, t, indexing='ij')
    f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

    a = torch.zeros(N, s, s)
    u = torch.zeros(N, s, s, record_steps)

    bsize = batch_size
    sol_t = None 

    num_batches = (N + bsize - 1) // bsize
    
    with tqdm(range(num_batches), desc=f"GPU {gpu_id}", position=gpu_id, leave=False) as pbar:
        for j in pbar:
            start_idx = j * bsize
            end_idx = min(start_idx + bsize, N)
            current_bsize = end_idx - start_idx
            
            w0 = GRF.sample(current_bsize)
            t0 = default_timer()
            sol, sol_t_batch = navier_stokes_2d(w0, f, 1e-3, sim_time, time_step, record_steps)
            t1 = default_timer()
            
            if sol_t is None:
                sol_t = sol_t_batch

            a[start_idx:end_idx,...] = w0
            u[start_idx:end_idx,...] = sol
            
            pbar.set_postfix_str(f"batch time: {t1-t0:.2f}s")
    
    output_path = os.path.join(data_dir, output_file_template.format(gpu_id))
    u_permuted = u.cpu().permute(0, 3, 1, 2)
    
    if sol_t is None:
        print(f"Warning: sol_t was not generated for GPU {gpu_id}. Saving empty time tensor.")
        sol_t = torch.tensor([])

    torch.save({'a': a.cpu(), 'u': u_permuted, 't': sol_t.cpu()}, output_path)
    print(f"Worker on GPU {gpu_id} finished. Data saved to {output_path}")

def merge_and_split_data(data_dir, file_prefix, sim_time, train_split_ratio=0.8, cleanup=True):
    print("nStarting merge and split process...")
    parts = [f for f in os.listdir(data_dir) if f.startswith(file_prefix) and f.endswith('.pt')]
    parts.sort()

    if not parts:
        print("No data parts found to merge.")
        return

    print(f"Found {len(parts)} parts to merge.")

    all_a, all_u = [], []
    sol_t = None

    for part_file in tqdm(parts, desc="Merging files"):
        path = os.path.join(data_dir, part_file)
        data = torch.load(path, map_location='cpu')
        all_a.append(data['a'])
        all_u.append(data['u'])
        if sol_t is None: sol_t = data['t']
        
    a_combined = torch.cat(all_a, dim=0)
    u_combined = torch.cat(all_u, dim=0)

    resolution = u_combined.shape[-1]
    num_samples = a_combined.shape[0]
    num_train = int(num_samples * train_split_ratio)
    
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_indices, test_indices = indices[:num_train], indices[num_train:]
    
    train_data = {'a': a_combined[train_indices], 'u': u_combined[train_indices], 't': sol_t}
    test_data = {'a': a_combined[test_indices], 'u': u_combined[test_indices], 't': sol_t}

    train_output_path = os.path.join(data_dir, f"nsforcing_train_T{int(sim_time)}_{resolution}.pt")
    test_output_path = os.path.join(data_dir, f"nsforcing_test_T{int(sim_time)}_{resolution}.pt")
    
    torch.save(train_data, train_output_path)
    print(f"Train data saved to {train_output_path} (Samples: {len(train_indices)})")

    torch.save(test_data, test_output_path)
    print(f"Test data saved to {test_output_path} (Samples: {len(test_indices)})")

    if cleanup:
        print("Cleaning up temporary part files...")
        for part_file in parts:
            os.remove(os.path.join(data_dir, part_file))
        print("Cleanup complete.")


def generate_ns2d_data(
    n_train: int = 1000,
    n_test: int = 200,
    N: int = 64,
    T: float = 10.0,
    dt: float = 1e-4,
    visc: float = 0.001,
    device: str = "cpu",
    save_path: Optional[str] = None
) -> Tuple[dict, dict]:
    """
    Generate training and test datasets for 2D Navier-Stokes equation.
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        N: Number of spatial points in each dimension (NxN grid)
        T: Final time
        dt: Time step size
        visc: Viscosity coefficient
        device: Device to run on
        save_path: Path to save data (optional)
        
    Returns:
        Tuple of (train_data, test_data) dictionaries
    """
    print(f"Generating 2D Navier-Stokes equation data...")
    print(f"Parameters: N={N}, T={T}, dt={dt}, visc={visc}")
    print(f"Train samples: {n_train}, Test samples: {n_test}")
    
    # Initialize solver
    solver = NavierStokesSolver(N=N, viscosity=visc, device=device)
    
    # Create forcing term (same for all samples)
    t = torch.linspace(0, 1, N+1, device=device)
    t = t[0:-1]
    X, Y = torch.meshgrid(t, t, indexing='ij')
    f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))
    
    # Time evaluation points (record every 0.1 time units)
    record_dt = 0.1
    n_time = int(T / record_dt) + 1
    t_eval = torch.linspace(0, T, n_time, device=device)
    
    def generate_dataset(n_samples, desc):
        # Generate initial conditions using Gaussian Random Field
        GRF = GaussianRF(2, N, alpha=2.5, tau=7, device=device)
        u0 = GRF.sample(n_samples)
        
        # Solve in batches to manage memory
        batch_size = min(10, n_samples)  # Smaller batch size for memory management
        all_solutions = []
        
        for i in tqdm(range(0, n_samples, batch_size), desc=desc):
            end_idx = min(i + batch_size, n_samples)
            batch_u0 = u0[i:end_idx]
            
            # Integrate
            sol = solver.integrate(batch_u0, f, T, dt)
            if sol is None:
                raise RuntimeError(f"Integration failed for batch {i//batch_size}")
                
            all_solutions.append(sol.cpu())
        
        return {
            'u': torch.cat(all_solutions, dim=0),  # (n_samples, n_time, N, N)
            't': t_eval.cpu(),                     # (n_time,)
            'x': torch.linspace(0, 1, N),         # (N,) - placeholder
            'dt': dt,
            'visc': visc
        }
    
    # Generate datasets
    train_data = generate_dataset(n_train, "Generating train data")
    test_data = generate_dataset(n_test, "Generating test data")
    
    # Save if requested
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        torch.save(train_data, os.path.join(save_path, 'ns2d_train.pt'))
        torch.save(test_data, os.path.join(save_path, 'ns2d_test.pt'))
        print(f"Data saved to {save_path}")
    
    print("Data generation complete!")
    return train_data, test_data


def load_ns2d_data(data_path: str) -> Tuple[dict, dict]:
    """Load 2D Navier-Stokes equation data from saved files."""
    train_data = torch.load(os.path.join(data_path, 'ns2d_train.pt'))
    test_data = torch.load(os.path.join(data_path, 'ns2d_test.pt'))
    return train_data, test_data

def plot_data_rollout_gif(data_dir, filename, save_path, num_frames=None, trajectory_idx=0):
    """
    Generates a GIF of a single trajectory from the generated dataset.
    """
    if not _plotting_enabled:
        print("Plotting libraries (matplotlib, imageio) not found. Skipping GIF generation.")
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        print(f"Data file not found at {file_path}. Cannot generate GIF.")
        return

    print("Loading data for plotting...")
    data_dict = torch.load(file_path, map_location='cpu')
    
    if trajectory_idx >= len(data_dict['u']):
        print(f"Error: trajectory_idx {trajectory_idx} is out of bounds for dataset with {len(data_dict['u'])} samples.")
        return
        
    trajectory = data_dict['u'][trajectory_idx] # Shape [T, H, W]
    time_points = data_dict.get('t', torch.arange(trajectory.shape[0]))

    frames = []
    
    if num_frames is None:
        num_frames_to_plot = trajectory.shape[0]
    else:
        num_frames_to_plot = min(num_frames, trajectory.shape[0])

    print(f"Generating GIF with {num_frames_to_plot} frames from trajectory {trajectory_idx}...")
    print(f"Time points available: {len(time_points)}")
    print(f"Time range: {time_points[0]:.2f} to {time_points[-1]:.2f}")
    
    vmin = trajectory.min().item()
    vmax = trajectory.max().item()

    for i in range(num_frames_to_plot):
        # Create a new figure for each frame to avoid clearing issues
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.imshow(trajectory[i].cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        time = time_points[i] if i < len(time_points) else (i+1)
        ax.set_title(f"Time: {time:.2f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis('on')
        
        # Save the figure to a buffer
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)
        frames.append(frame)
        
        # Close the figure to free memory
        plt.close(fig)
        
        if i < 5 or i >= num_frames_to_plot - 5:  # Print first and last 5 frames
            print(f"Frame {i+1}: Time {time:.2f}")

    print(f"Generated {len(frames)} frames, saving GIF...")
    imageio.mimsave(save_path, frames, fps=10)
    print(f"Rollout GIF saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate, merge, and split Navier-Stokes 2D data using multiple GPUs.")
    parser.add_argument("--mode", type=str, default="generate", choices=['generate', 'plot-only'], help="Operation mode: 'generate' creates new data, 'plot-only' visualizes existing data.")
    parser.add_argument("--gpu_ids", type=str, default="4,5,6,7", help="Comma-separated list of GPU device IDs to use.")
    parser.add_argument("--total_samples", type=int, default=40, help="Total number of solutions to generate.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for generation on each worker.")
    parser.add_argument("--data_dir", type=str, default="/data2/lenz/moNS2D", help="Directory where data is stored.")
    parser.add_argument("--plot", action="store_true", help="When in 'generate' mode, also create a GIF of a sample rollout.")
    parser.add_argument("--plot_traj_idx", type=int, default=0, help="Index of the trajectory to plot for the GIF.")
    parser.add_argument("--plot_frames", type=int, default=None, help="Number of frames for the GIF. Defaults to all available.")
    parser.add_argument("--record_steps", type=int, default=1000, help="Number of in-time snapshots to record during generation.")
    parser.add_argument("--sim_time", type=float, default=100.0, help="Final simulation time (T).")
    parser.add_argument("--time_step", type=float, default=1e-4, help="Internal time-step for solver (delta_t).")
    args = parser.parse_args()

    if args.mode == 'generate':
        gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
        num_gpus = len(gpu_ids)
        
        samples_per_gpu = args.total_samples // num_gpus
        if args.total_samples % num_gpus != 0:
            print(f"Warning: Total samples ({args.total_samples}) is not evenly divisible by the number of GPUs ({num_gpus}).")
            print("Adjusting samples per GPU.")

        process_args = []
        for i, gpu_id in enumerate(gpu_ids):
            num_samples = samples_per_gpu + (1 if i < args.total_samples % num_gpus else 0)
            if num_samples > 0:
                process_args.append((gpu_id, num_samples, args.batch_size, args.data_dir, "ns_data_part_{}.pt", args.sim_time, args.time_step, args.record_steps))

        os.makedirs(args.data_dir, exist_ok=True)
        
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass # Start method already set

        with mp.Pool(processes=num_gpus) as pool:
            pool.starmap(generation_worker, process_args)
        
        merge_and_split_data(args.data_dir, "ns_data_part_", sim_time=args.sim_time)

        if args.plot:
            print("n--- Generation complete. Now plotting sample rollout. ---")
            resolution = 256 # This is hardcoded in the generation worker
            plot_filename = f"nsforcing_train_T{int(args.sim_time)}_{resolution}.pt"
            gif_save_path = os.path.join("plots/moNS2D", "sample_rollout.gif")
            plot_data_rollout_gif(args.data_dir, plot_filename, gif_save_path, num_frames=args.plot_frames, trajectory_idx=args.plot_traj_idx)

    elif args.mode == 'plot-only':
        print("n--- Running in plot-only mode. ---")
        resolution = 256 # Assuming 256 resolution for existing data
        
        # Try new filename format first, then fall back to old format
        plot_filename = f"nsforcing_train_T{int(args.sim_time)}_{resolution}.pt"
        file_path = os.path.join(args.data_dir, plot_filename)
        
        if not os.path.exists(file_path):
            print(f"File {plot_filename} not found. Trying old filename format...")
            plot_filename = f"nsforcing_train_{resolution}.pt"
            file_path = os.path.join(args.data_dir, plot_filename)
            
            if not os.path.exists(file_path):
                print(f"Error: Neither new nor old format data file found in {args.data_dir}")
                return
            else:
                print(f"Using existing data file: {plot_filename}")
        
        gif_save_path = os.path.join("plots/moNS2D", "sample_rollout.gif")
        plot_data_rollout_gif(args.data_dir, plot_filename, gif_save_path, num_frames=args.plot_frames, trajectory_idx=args.plot_traj_idx)


def generate_random_field(N: int, device: str = "cpu") -> torch.Tensor:
    """Generate a random 2D field using Gaussian random field."""
    grf = GaussianRF(dim=2, size=N, alpha=2.5, tau=7.0, device=device)
    return grf.sample(1).squeeze(0)


def generate_forcing_term(N: int, device: str = "cpu") -> torch.Tensor:
    """Generate a forcing term for 2D Navier-Stokes."""
    grf = GaussianRF(dim=2, size=N, alpha=1.5, tau=5.0, device=device)
    return grf.sample(1).squeeze(0)


if __name__ == "__main__":
    main()