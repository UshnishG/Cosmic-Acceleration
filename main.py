"""
Cosmic Acceleration: A CUDA-Powered N-Body Simulation.
Main entry point for running simulations, generating visualizations, and benchmarking.
"""
import os
import sys
import argparse
import numpy as np

from src.constants import SimulationConfig
from src.initial_conditions import generate_keplerian_disk
from src.cpu_solver import CPUNBodySolver
from src.gpu_direct_solver import GPUNBodyDirectSolver, get_available_device
from src.gpu_tiled_solver import GPUTiledNBodySolver
from src.visualizer import (
    plot_3d_physical_orbits,
    plot_3d_spacetime_paths,
    create_2d_fading_trails_animation
)

def run_3d_simulation(config: SimulationConfig, steps: int = 1000, output_dir: str = "output"):
    print(f"\n=======================================================")
    print(f"RUNNING 3D N-BODY SIMULATION (150 BODIES)")
    print(f"Device: {get_available_device()}")
    print(f"=======================================================")
    
    # 1. Generate 1 heavy central body + 149 orbiters
    pos, vel, mass = generate_keplerian_disk(config, is_3d=True, seed=42)
    
    # 2. Initialize GPU solver
    solver = GPUNBodyDirectSolver(pos, vel, mass, config)
    
    # 3. Simulate with fixed dt (Poster Step 5: For 3D simulations, fixed small time step is used)
    # Poster Step 6: Sample regular steps for 3D runs (1000 steps * 0.005 = 5.0s)
    print(f"Simulating {steps} steps (dt = {config.dt_fixed}, Total Time = {steps * config.dt_fixed:.1f})...")
    res = solver.simulate(
        num_steps=steps,
        dt=config.dt_fixed,
        adaptive=False,
        sample_interval=config.sample_interval_3d
    )
    
    trajectory = res["trajectory"]
    timestamps = res["timestamps"]
    print(f"Simulation completed in {res['total_runtime_s']:.3f} s ({res['mean_step_time_ms']:.2f} ms/step).")
    print(f"Captured {len(trajectory)} trajectory snapshots.")
    
    # 4. Generate Visualizations (Poster Step 7)
    # Output 1: 3D Physical orbits plot (plasma colors, dark background)
    p1 = os.path.join(output_dir, "150_body_3d_orbits_gpu.png")
    plot_3d_physical_orbits(trajectory, output_path=p1)
    
    # Output 2: 3D Space-time path plot (vertical axis = Time)
    p2 = os.path.join(output_dir, "150_body_3d_orbit_paths.png")
    plot_3d_spacetime_paths(trajectory, timestamps, output_path=p2)
    
    return res

def run_2d_simulation(config: SimulationConfig, steps: int = 500, output_dir: str = "output"):
    print(f"\n=======================================================")
    print(f"RUNNING 2D N-BODY SIMULATION (ADAPTIVE TIMESTEPPING)")
    print(f"Device: {get_available_device()}")
    print(f"=======================================================")
    
    pos, vel, mass = generate_keplerian_disk(config, is_3d=False, seed=42)
    solver = GPUNBodyDirectSolver(pos, vel, mass, config)
    
    # Poster Step 5: For 2D simulations, adaptive time step is used
    # Poster Step 6: Sample every regular steps for 2D runs
    print(f"Simulating {steps} steps with adaptive time-stepping...")
    res = solver.simulate(
        num_steps=steps,
        adaptive=True,
        sample_interval=config.sample_interval_2d
    )
    
    trajectory = res["trajectory"]
    print(f"Simulation completed in {res['total_runtime_s']:.3f} s ({res['mean_step_time_ms']:.2f} ms/step).")
    
    # Output 3: 2D Animated Fading Trails
    p3 = os.path.join(output_dir, "2d_fading_trails.gif")
    create_2d_fading_trails_animation(trajectory, output_path=p3, fps=25)
    return res

def main():
    parser = argparse.ArgumentParser(description="Cosmic Acceleration: A CUDA-Powered N-Body Simulation")
    parser.add_argument("--mode", choices=["3d", "2d", "all", "benchmark", "test"], default="all",
                        help="Execution mode (default: all)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of integration steps")
    parser.add_argument("--outdir", type=str, default="output", help="Directory for output plots")
    args = parser.parse_args()
    
    config = SimulationConfig()
    
    if args.mode == "test":
        import subprocess
        subprocess.run([sys.executable, "run_tests.py"], check=True)
    elif args.mode == "benchmark":
        from benchmarks.benchmark_runner import benchmark_solvers
        from benchmarks.plot_benchmarks import generate_benchmark_charts
        benchmark_solvers()
        generate_benchmark_charts()
    elif args.mode == "3d":
        run_3d_simulation(config, steps=args.steps, output_dir=args.outdir)
    elif args.mode == "2d":
        run_2d_simulation(config, steps=args.steps // 2, output_dir=args.outdir)
    elif args.mode == "all":
        run_3d_simulation(config, steps=args.steps, output_dir=args.outdir)
        run_2d_simulation(config, steps=500, output_dir=args.outdir)

if __name__ == "__main__":
    main()
