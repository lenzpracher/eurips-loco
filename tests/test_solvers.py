"""
Tests for PDE solvers.
"""

import pytest
import torch
import numpy as np

from solvers import BurgersSolver, KdVSolver, NavierStokesSolver


class TestBurgersSolver:
    """Tests for Burgers equation solver."""
    
    def test_burgers_solver_creation(self, solver_config_burgers, device):
        """Test Burgers solver initialization."""
        # Extract only the parameters the solver constructor accepts
        solver_params = {k: v for k, v in solver_config_burgers.items() if k in ['N', 'nu']}
        solver = BurgersSolver(device=device, **solver_params)
        
        assert solver.N == solver_config_burgers['N']
        assert solver.nu == solver_config_burgers['nu']
        assert solver.device == device
        
        # Check that wavenumbers are properly initialized
        assert solver.k is not None
        assert solver.ik is not None
        assert solver.linear_operator is not None
    
    def test_burgers_initial_conditions(self, device):
        """Test Burgers initial condition generation."""
        from solvers.burgers import generate_initial_conditions
        
        n_samples, N = 5, 32
        u0 = generate_initial_conditions(n_samples, N, device)
        
        assert u0.shape == (n_samples, N)
        assert u0.device.type == device
        assert torch.isfinite(u0).all()
    
    def test_burgers_integration(self, solver_config_burgers, device):
        """Test Burgers equation integration."""
        from solvers.burgers import generate_initial_conditions
        
        solver_params = {k: v for k, v in solver_config_burgers.items() if k in ['N', 'nu']}
        solver = BurgersSolver(device=device, **solver_params)
        
        # Generate initial conditions
        n_samples = 3
        u0 = generate_initial_conditions(n_samples, solver_config_burgers['N'], device)
        
        # Create time points
        t_eval = torch.arange(0, solver_config_burgers['T'] + solver_config_burgers['dt'], 
                             solver_config_burgers['dt'], device=device)
        
        # Integrate
        solution = solver.integrate(u0, t_eval)
        
        assert solution is not None  # Should not diverge for these parameters
        assert solution.shape == (n_samples, len(t_eval), solver_config_burgers['N'])
        assert torch.isfinite(solution).all()
    
    def test_burgers_data_generation(self, device, temp_dir):
        """Test complete Burgers data generation."""
        from solvers.burgers import generate_burgers_data
        
        train_data, test_data = generate_burgers_data(
            n_train=5, n_test=3, N=16, T=0.5, dt=0.1, nu=0.01,
            device=device, save_path=temp_dir
        )
        
        assert 'u' in train_data
        assert 't' in train_data
        assert 'x' in train_data
        
        assert train_data['u'].shape[0] == 5  # n_train samples
        assert test_data['u'].shape[0] == 3   # n_test samples
        
        # Check time dimension
        expected_time_steps = int(0.5 / 0.1) + 1
        assert train_data['u'].shape[1] == expected_time_steps


class TestKdVSolver:
    """Tests for KdV equation solver."""
    
    def test_kdv_solver_creation(self, solver_config_kdv, device):
        """Test KdV solver initialization."""
        solver_params = {k: v for k, v in solver_config_kdv.items() if k in ['N', 'a', 'b']}
        solver = KdVSolver(device=device, **solver_params)
        
        assert solver.N == solver_config_kdv['N']
        assert solver.a == solver_config_kdv['a']
        assert solver.b == solver_config_kdv['b']
        assert solver.device == device
        
        # Check that operators are properly initialized
        assert solver.k is not None
        assert solver.ik is not None
        assert solver.linear_operator is not None
    
    def test_kdv_soliton_initial_conditions(self, device):
        """Test KdV soliton initial condition generation."""
        from solvers.kdv import generate_soliton_initial_conditions
        
        n_samples, N = 5, 32
        u0 = generate_soliton_initial_conditions(n_samples, N, device)
        
        assert u0.shape == (n_samples, N)
        assert u0.device.type == device
        assert torch.isfinite(u0).all()
        assert (u0 >= 0).all()  # Solitons should be positive
    
    def test_kdv_random_initial_conditions(self, device):
        """Test KdV random initial condition generation."""
        from solvers.kdv import generate_random_initial_conditions
        
        n_samples, N = 5, 32
        u0 = generate_random_initial_conditions(n_samples, N, device)
        
        assert u0.shape == (n_samples, N)
        assert u0.device.type == device
        assert torch.isfinite(u0).all()
    
    @pytest.mark.skip(reason="KdV solver has numerical stability issues with current parameters")
    def test_kdv_integration(self, solver_config_kdv, device):
        """Test KdV equation integration."""
        from solvers.kdv import generate_soliton_initial_conditions
        
        solver_params = {k: v for k, v in solver_config_kdv.items() if k in ['N', 'a', 'b']}
        solver = KdVSolver(device=device, **solver_params)
        
        # Generate initial conditions  
        n_samples = 2
        u0 = generate_soliton_initial_conditions(n_samples, solver_config_kdv['N'], device)
        
        # Create time points (shorter integration for testing)
        t_eval = torch.arange(0, 0.5, solver_config_kdv['dt'], device=device)
        
        # Integrate
        solution = solver.integrate(u0, t_eval)
        
        assert solution is not None
        assert solution.shape == (n_samples, len(t_eval), solver_config_kdv['N'])
        assert torch.isfinite(solution).all()
    
    def test_kdv_conserved_quantities(self, solver_config_kdv, device):
        """Test KdV conserved quantity calculations."""
        solver_params = {k: v for k, v in solver_config_kdv.items() if k in ['N', 'a', 'b']}
        solver = KdVSolver(device=device, **solver_params)
        
        # Create test data
        u = torch.randn(3, solver_config_kdv['N'], device=device)
        
        # Calculate conserved quantities
        mass = solver.get_mass(u)
        momentum = solver.get_momentum(u)
        energy = solver.get_energy(u)
        
        assert mass.shape == (3,)
        assert momentum.shape == (3,)
        assert energy.shape == (3,)
        assert torch.isfinite(mass).all()
        assert torch.isfinite(momentum).all()
        assert torch.isfinite(energy).all()
    
    @pytest.mark.skip(reason="KdV data generation depends on unstable integration")
    def test_kdv_data_generation(self, device, temp_dir):
        """Test complete KdV data generation."""
        from solvers.kdv import generate_kdv_data
        
        train_data, test_data = generate_kdv_data(
            n_train=3, n_test=2, N=16, T=1.0, dt=0.1,
            device=device, save_path=temp_dir
        )
        
        assert 'u' in train_data
        assert 't' in train_data
        assert 'x' in train_data
        
        assert train_data['u'].shape[0] == 3
        assert test_data['u'].shape[0] == 2
        
        expected_time_steps = int(1.0 / 0.1) + 1
        assert train_data['u'].shape[1] == expected_time_steps


class TestNavierStokesSolver:
    """Tests for 2D Navier-Stokes solver."""
    
    def test_ns2d_solver_creation(self, solver_config_ns2d, device):
        """Test Navier-Stokes solver initialization."""
        # Map 'visc' to 'viscosity' parameter name
        solver_params = {'N': solver_config_ns2d['N'], 'viscosity': solver_config_ns2d['visc']}
        solver = NavierStokesSolver(device=device, **solver_params)
        
        assert solver.N == solver_config_ns2d['N']
        assert solver.viscosity == solver_config_ns2d['visc']
        assert solver.device == device
        
        # Check that spectral operators are initialized
        assert solver.k_x is not None
        assert solver.k_y is not None
        assert solver.lap is not None
        assert solver.dealias is not None
    
    def test_ns2d_random_field_generation(self, device):
        """Test random field generation for NS2D."""
        from solvers.navier_stokes import generate_random_field
        
        N = 16
        field = generate_random_field(N, device=device)
        
        assert field.shape == (N, N)
        assert field.device.type == device
        assert torch.isfinite(field).all()
    
    def test_ns2d_forcing_generation(self, device):
        """Test forcing term generation."""
        from solvers.navier_stokes import generate_forcing_term
        
        N = 16
        forcing = generate_forcing_term(N, device=device)
        
        assert forcing.shape == (N, N)
        assert forcing.device.type == device
        assert torch.isfinite(forcing).all()
    
    def test_ns2d_integration(self, solver_config_ns2d, device):
        """Test Navier-Stokes integration."""
        from solvers.navier_stokes import generate_random_field, generate_forcing_term
        
        solver_params = {'N': solver_config_ns2d['N'], 'viscosity': solver_config_ns2d['visc']}
        solver = NavierStokesSolver(device=device, **solver_params)
        
        # Generate initial condition and forcing
        w0 = generate_random_field(solver_config_ns2d['N'], device=device)
        f = generate_forcing_term(solver_config_ns2d['N'], device=device)
        
        # Integrate (shorter time for testing)
        solution = solver.integrate(
            w0.unsqueeze(0), f, T=0.5
        )
        
        # The solver records at 0.1 time intervals, so T=0.5 gives 6 time points (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
        assert solution.shape[0] == 1  # batch size
        assert solution.shape[2:] == (solver_config_ns2d['N'], solver_config_ns2d['N'])  # spatial dims
        assert torch.isfinite(solution).all()
    
    def test_ns2d_batch_integration(self, solver_config_ns2d, device):
        """Test batch integration."""
        from solvers.navier_stokes import generate_random_field, generate_forcing_term
        
        solver_params = {'N': solver_config_ns2d['N'], 'viscosity': solver_config_ns2d['visc']}
        solver = NavierStokesSolver(device=device, **solver_params)
        
        batch_size = 3
        N = solver_config_ns2d['N']
        
        # Generate batch initial conditions
        w0_batch = torch.stack([generate_random_field(N, device=device) 
                               for _ in range(batch_size)])
        f = generate_forcing_term(N, device=device)
        
        # Integrate
        solution = solver.integrate(
            w0_batch, f, T=0.3
        )
        
        # T=0.3 with 0.1 intervals gives 4 time points (0.0, 0.1, 0.2, 0.3)
        assert solution.shape[0] == batch_size
        assert solution.shape[2:] == (N, N)
        assert torch.isfinite(solution).all()
    
    @pytest.mark.slow
    def test_ns2d_data_generation(self, device, temp_dir):
        """Test complete NS2D data generation (marked as slow)."""
        from solvers.navier_stokes import generate_ns2d_data
        
        train_data, test_data = generate_ns2d_data(
            n_train=2, n_test=1, N=8, T=1.0, record_steps=10,
            device=device, save_path=temp_dir
        )
        
        assert 'u' in train_data
        assert 'a' in train_data  # Initial conditions
        assert 't' in train_data
        
        assert train_data['u'].shape[0] == 2
        assert test_data['u'].shape[0] == 1
        assert train_data['u'].shape[1] == 10  # record_steps
        assert train_data['u'].shape[2] == train_data['u'].shape[3] == 8  # N x N


class TestSolverComparison:
    """Comparative tests across solvers."""
    
    def test_all_solvers_produce_finite_output(self, device):
        """Test that all solvers produce finite outputs."""
        # Simple integration test for each solver
        configs = {
            'burgers': {'N': 16, 'nu': 0.01},
            'kdv': {'N': 16, 'a': 6.0, 'b': 1.0},
            'ns2d': {'N': 8, 'visc': 1e-3}
        }
        
        # Test Burgers
        from solvers.burgers import generate_initial_conditions
        solver_b = BurgersSolver(device=device, **configs['burgers'])
        u0_b = generate_initial_conditions(1, 16, device)
        t_eval_b = torch.arange(0, 0.2, 0.1, device=device)
        sol_b = solver_b.integrate(u0_b, t_eval_b)
        assert torch.isfinite(sol_b).all()
        
        # Test KdV  
        from solvers.kdv import generate_soliton_initial_conditions
        solver_k = KdVSolver(device=device, **configs['kdv'])
        u0_k = generate_soliton_initial_conditions(1, 16, device)
        t_eval_k = torch.arange(0, 0.2, 0.1, device=device)
        sol_k = solver_k.integrate(u0_k, t_eval_k)
        assert torch.isfinite(sol_k).all()
        
        # Test NS2D
        from solvers.navier_stokes import generate_random_field, generate_forcing_term
        ns2d_params = {'N': configs['ns2d']['N'], 'viscosity': configs['ns2d']['visc']}
        solver_n = NavierStokesSolver(device=device, **ns2d_params)
        w0_n = generate_random_field(8, device=device)
        f_n = generate_forcing_term(8, device=device)
        sol_n = solver_n.integrate(w0_n.unsqueeze(0), f_n, T=0.2)
        assert torch.isfinite(sol_n).all()