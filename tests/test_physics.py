"""
Unit and Physics Validation Tests.
Validates Newtonian gravitation, softening, momentum conservation, energy drift,
and numerical stability.
"""
import numpy as np
from src.constants import SimulationConfig
from src.initial_conditions import generate_keplerian_disk
from src.cpu_solver import CPUNBodySolver
from src.gpu_direct_solver import GPUNBodyDirectSolver
from src.gpu_tiled_solver import GPUTiledNBodySolver
from src.integrators import compute_system_metrics

def test_initial_conditions():
    config = SimulationConfig(num_bodies=150)
    pos, vel, mass = generate_keplerian_disk(config, is_3d=True, seed=42)
    
    assert pos.shape == (150, 3)
    assert vel.shape == (150, 3)
    assert mass.shape == (150,)
    assert mass[0] == config.central_mass
    assert np.all(mass[1:] == config.orbiter_mass)
    
    # Check no NaNs or Infs
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(vel))
    
    # Check center-of-mass momentum balance
    total_momentum = np.sum(mass[:, None] * vel, axis=0)
    np.testing.assert_allclose(total_momentum, [0.0, 0.0, 0.0], atol=1e-10)

def test_newtonian_acceleration_and_softening():
    # 2 bodies separated by 1.0 along x-axis
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    vel = np.zeros_like(pos)
    mass = np.array([10.0, 5.0], dtype=np.float64)
    eps = 0.1
    G = 1.0
    
    config = SimulationConfig(G=G, softening=eps)
    solver = CPUNBodySolver(pos, vel, mass, config)
    acc = solver.compute_acceleration(pos)
    
    # Expected acceleration on body 0: pulled towards body 1 (+x)
    # r12 = 1.0, r_eff = sqrt(1.0^2 + 0.1^2) = sqrt(1.01)
    r_eff = np.sqrt(1.0**2 + eps**2)
    expected_a0_x = G * 5.0 * 1.0 / (r_eff**3)
    expected_a1_x = -G * 10.0 * 1.0 / (r_eff**3)
    
    np.testing.assert_allclose(acc[0, 0], expected_a0_x, rtol=1e-7)
    np.testing.assert_allclose(acc[1, 0], expected_a1_x, rtol=1e-7)
    np.testing.assert_allclose(acc[:, 1:], 0.0, atol=1e-12)

def test_energy_conservation_and_drift():
    # Poster Step 5: 2D simulation uses adaptive timestepping for orbital stability
    config = SimulationConfig(num_bodies=150)
    pos, vel, mass = generate_keplerian_disk(config, is_3d=False, seed=123)
    
    solver = CPUNBodySolver(pos, vel, mass, config)
    res = solver.simulate(num_steps=300, adaptive=True, sample_interval=50)
    
    metrics = res["metrics"]
    e_initial = metrics[0]["total_energy"]
    e_final = metrics[-1]["total_energy"]
    
    drift = abs(e_final - e_initial) / abs(e_initial)
    # Symplectic Velocity Verlet with adaptive stepping keeps drift << 0.5%
    assert drift < 0.005, f"Energy drift {drift * 100:.4f}% exceeds 0.5% tolerance!"

def test_reproducibility():
    config = SimulationConfig(num_bodies=50)
    p1, v1, m1 = generate_keplerian_disk(config, seed=999)
    p2, v2, m2 = generate_keplerian_disk(config, seed=999)
    
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(m1, m2)
