"""
Cross-Solver Equivalence and Parity Tests.
Compares CPU direct, GPU direct, GPU tiled, and Barnes-Hut solvers on identical states.
Validates the poster's claim of <0.5% numerical error between solvers.
"""
import numpy as np
from src.constants import SimulationConfig
from src.initial_conditions import generate_keplerian_disk
from src.cpu_solver import CPUNBodySolver
from src.gpu_direct_solver import GPUNBodyDirectSolver
from src.gpu_tiled_solver import GPUTiledNBodySolver
from src.barnes_hut import BarnesHutSolver

def test_cpu_gpu_direct_parity():
    config = SimulationConfig(num_bodies=150)
    pos, vel, mass = generate_keplerian_disk(config, is_3d=True, seed=42)
    
    cpu_solver = CPUNBodySolver(pos, vel, mass, config)
    gpu_solver = GPUNBodyDirectSolver(pos, vel, mass, config)
    
    acc_cpu = cpu_solver.compute_acceleration(pos)
    acc_gpu = gpu_solver.compute_acceleration(pos)
    
    # Check max relative error between CPU and GPU direct force
    diff = np.linalg.norm(acc_cpu - acc_gpu, axis=-1)
    norm = np.linalg.norm(acc_cpu, axis=-1) + 1e-12
    rel_error = np.max(diff / norm)
    
    assert rel_error < 1e-6, f"GPU Direct force differs from CPU by {rel_error:.2e}"

def test_cpu_gpu_tiled_parity():
    config = SimulationConfig(num_bodies=150)
    pos, vel, mass = generate_keplerian_disk(config, is_3d=True, seed=42)
    
    cpu_solver = CPUNBodySolver(pos, vel, mass, config)
    gpu_tiled = GPUTiledNBodySolver(pos, vel, mass, config)
    
    acc_cpu = cpu_solver.compute_acceleration(pos)
    
    # State from GPU tiled solver
    _, _, acc_tiled, _ = gpu_tiled.get_state()
    
    diff = np.linalg.norm(acc_cpu - acc_tiled, axis=-1)
    norm = np.linalg.norm(acc_cpu, axis=-1) + 1e-12
    rel_error = np.max(diff / norm)
    
    assert rel_error < 1e-6, f"GPU Tiled force differs from CPU by {rel_error:.2e}"

def test_barnes_hut_accuracy_bound():
    config = SimulationConfig(num_bodies=150, theta=0.5)
    pos, vel, mass = generate_keplerian_disk(config, is_3d=True, seed=42)
    
    cpu_solver = CPUNBodySolver(pos, vel, mass, config)
    bh_solver = BarnesHutSolver(pos, vel, mass, config)
    
    acc_cpu = cpu_solver.compute_acceleration(pos)
    acc_bh = bh_solver.compute_acceleration(pos)
    
    diff = np.linalg.norm(acc_cpu - acc_bh, axis=-1)
    norm = np.linalg.norm(acc_cpu, axis=-1) + 1e-12
    rel_error = diff / norm
    
    mean_rel_error = float(np.mean(rel_error))
    p95_rel_error = float(np.percentile(rel_error, 95))
    
    print(f"\nBarnes-Hut Mean Rel Error: {mean_rel_error * 100:.3f}%")
    print(f"Barnes-Hut 95th Percentile Rel Error: {p95_rel_error * 100:.3f}%")
    
    # Validate poster claim: "<0.5% error"
    assert mean_rel_error < 0.005, f"Barnes-Hut mean relative error ({mean_rel_error * 100:.3f}%) exceeds 0.5%!"
