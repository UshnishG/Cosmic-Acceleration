"""
Comprehensive and Reproducible Benchmark Runner.
Evaluates CPU Direct, GPU Direct, GPU Tiled, and Barnes-Hut across varying particle counts (N=150 to 4096).
Measures runtime per step, end-to-end speedup, and numerical force error.
"""
import os
import sys
import time
import json
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.constants import SimulationConfig
from src.initial_conditions import generate_keplerian_disk
from src.cpu_solver import CPUNBodySolver
from src.gpu_direct_solver import GPUNBodyDirectSolver, get_available_device
from src.gpu_tiled_solver import GPUTiledNBodySolver
from src.barnes_hut import BarnesHutSolver

def benchmark_solvers(particle_counts=(150, 500, 1000, 2048), num_warmup=5, num_eval=20):
    results = {
        "hardware": get_available_device(),
        "precision": "float64",
        "benchmarks": []
    }
    
    print(f"==========================================================================")
    print(f"COSMIC ACCELERATION N-BODY BENCHMARK SUITE")
    print(f"Hardware Detected: {results['hardware']}")
    print(f"Precision: {results['precision']}")
    print(f"==========================================================================\n")
    
    for N in particle_counts:
        print(f"--- Benchmarking N = {N} Bodies ---")
        config = SimulationConfig(num_bodies=N)
        pos, vel, mass = generate_keplerian_disk(config, is_3d=True, seed=42)
        
        entry = {
            "N": N,
            "timings_ms": {},
            "speedups": {},
            "relative_force_errors": {}
        }
        
        # 1. CPU Direct Solver
        # For N >= 2048, run fewer eval steps to prevent excessive waiting
        eval_steps_cpu = 10 if N >= 2048 else num_eval
        cpu_solver = CPUNBodySolver(pos, vel, mass, config)
        
        # Warmup
        for _ in range(num_warmup):
            cpu_solver.compute_acceleration(cpu_solver.pos)
            
        t0 = time.perf_counter()
        for _ in range(eval_steps_cpu):
            acc_cpu = cpu_solver.compute_acceleration(cpu_solver.pos)
        t_cpu_ms = (time.perf_counter() - t0) / eval_steps_cpu * 1000.0
        entry["timings_ms"]["CPU Direct"] = round(t_cpu_ms, 3)
        print(f"  CPU Direct:      {t_cpu_ms:8.2f} ms/step")
        
        # 2. GPU Direct Solver
        gpu_direct = GPUNBodyDirectSolver(pos, vel, mass, config)
        for _ in range(num_warmup):
            gpu_direct.step(config.dt_fixed)
            
        t0 = time.perf_counter()
        for _ in range(num_eval):
            gpu_direct.step(config.dt_fixed)
        t_gpu_dir_ms = (time.perf_counter() - t0) / num_eval * 1000.0
        entry["timings_ms"]["GPU Direct"] = round(t_gpu_dir_ms, 3)
        speedup_gpu_dir = t_cpu_ms / t_gpu_dir_ms
        entry["speedups"]["GPU Direct"] = round(speedup_gpu_dir, 2)
        print(f"  GPU Direct:      {t_gpu_dir_ms:8.2f} ms/step | Speedup: {speedup_gpu_dir:6.2f}x")
        
        # 3. GPU Tiled Solver
        gpu_tiled = GPUTiledNBodySolver(pos, vel, mass, config, tile_size=256)
        for _ in range(num_warmup):
            gpu_tiled.step(config.dt_fixed)
            
        t0 = time.perf_counter()
        for _ in range(num_eval):
            gpu_tiled.step(config.dt_fixed)
        t_gpu_tiled_ms = (time.perf_counter() - t0) / num_eval * 1000.0
        entry["timings_ms"]["GPU Tiled"] = round(t_gpu_tiled_ms, 3)
        speedup_gpu_tiled = t_cpu_ms / t_gpu_tiled_ms
        entry["speedups"]["GPU Tiled"] = round(speedup_gpu_tiled, 2)
        print(f"  GPU Tiled:       {t_gpu_tiled_ms:8.2f} ms/step | Speedup: {speedup_gpu_tiled:6.2f}x")
        
        # 4. Barnes-Hut Tree Solver
        bh_solver = BarnesHutSolver(pos, vel, mass, config)
        eval_steps_bh = 5 if N >= 2048 else 10
        t0 = time.perf_counter()
        for _ in range(eval_steps_bh):
            acc_bh = bh_solver.compute_acceleration(pos)
        t_bh_ms = (time.perf_counter() - t0) / eval_steps_bh * 1000.0
        entry["timings_ms"]["Barnes-Hut (theta=0.5)"] = round(t_bh_ms, 3)
        
        # Compute relative force error of Barnes-Hut vs CPU Direct
        norm_cpu = np.linalg.norm(acc_cpu, axis=-1) + 1e-12
        bh_error = float(np.mean(np.linalg.norm(acc_cpu - acc_bh, axis=-1) / norm_cpu))
        entry["relative_force_errors"]["Barnes-Hut"] = f"{bh_error * 100:.3f}%"
        print(f"  Barnes-Hut:      {t_bh_ms:8.2f} ms/step | Mean Error: {bh_error * 100:.3f}%")
        
        results["benchmarks"].append(entry)
        print()
        
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark completed! Saved results to {out_file}")
    return results

if __name__ == "__main__":
    benchmark_solvers()
