"""
CPU Direct-Force N-Body Reference Solver.
Uses fully vectorized NumPy operations and symplectic Velocity Verlet integration.
"""
import numpy as np
import time
from typing import Dict, List, Optional
from src.constants import SimulationConfig
from src.integrators import velocity_verlet_step, compute_adaptive_dt, compute_system_metrics

class CPUNBodySolver:
    def __init__(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        mass: np.ndarray,
        config: Optional[SimulationConfig] = None
    ):
        self.pos = np.ascontiguousarray(pos, dtype=np.float64).copy()
        self.vel = np.ascontiguousarray(vel, dtype=np.float64).copy()
        self.mass = np.ascontiguousarray(mass, dtype=np.float64).copy()
        self.config = config or SimulationConfig()
        self.N = len(self.mass)
        self.G = self.config.G
        self.softening = self.config.softening
        
        # Precompute initial acceleration
        self.acc = self.compute_acceleration(self.pos)
        self.time = 0.0
        self.step_count = 0
        
    def compute_acceleration(self, pos: np.ndarray) -> np.ndarray:
        """
        Calculates gravitational acceleration for all N particles:
        a_i = sum_{j != i} G * m_j * (r_j - r_i) / (||r_j - r_i||^2 + eps^2)^(3/2)
        """
        # diff[i, j, :] = pos[j] - pos[i]
        diff = pos[None, :, :] - pos[:, None, :]  # shape: (N, N, 3)
        dist_sq = np.sum(diff**2, axis=-1) + self.softening**2  # shape: (N, N)
        
        # inv_dist_cubed = 1.0 / (dist_sq^(3/2))
        inv_dist_cubed = 1.0 / (dist_sq * np.sqrt(dist_sq))
        
        # Zero out self-interaction
        np.fill_diagonal(inv_dist_cubed, 0.0)
        
        # Scale by mass of body j: G * m_j * inv_dist_cubed
        weights = self.G * self.mass[None, :] * inv_dist_cubed  # shape: (N, N)
        
        # a_i = sum_j weights[i, j] * diff[i, j, :]
        accel = np.sum(weights[:, :, None] * diff, axis=1)  # shape: (N, 3)
        return accel

    def step(self, dt: float) -> float:
        """Advance system by dt using symplectic Velocity Verlet."""
        self.pos, self.vel, self.acc = velocity_verlet_step(
            self.pos, self.vel, self.acc, self.compute_acceleration, dt
        )
        self.time += dt
        self.step_count += 1
        return dt

    def simulate(
        self,
        num_steps: int,
        dt: Optional[float] = None,
        adaptive: bool = False,
        sample_interval: int = 10
    ) -> Dict:
        """
        Runs the simulation for num_steps and records trajectories and diagnostics.
        """
        trajectory = []
        timestamps = []
        metrics_history = []
        step_times = []
        
        # Record initial state
        trajectory.append(self.pos.copy())
        timestamps.append(self.time)
        metrics_history.append(compute_system_metrics(
            self.pos, self.vel, self.mass, self.G, self.softening
        ))
        
        for step_idx in range(1, num_steps + 1):
            t_start = time.perf_counter()
            if adaptive:
                current_dt = compute_adaptive_dt(
                    self.acc,
                    self.softening,
                    eta=self.config.eta_adaptive,
                    dt_min=self.config.dt_min,
                    dt_max=self.config.dt_max
                )
            else:
                current_dt = dt or self.config.dt_fixed
                
            self.step(current_dt)
            t_end = time.perf_counter()
            step_times.append(t_end - t_start)
            
            if step_idx % sample_interval == 0 or step_idx == num_steps:
                trajectory.append(self.pos.copy())
                timestamps.append(self.time)
                metrics_history.append(compute_system_metrics(
                    self.pos, self.vel, self.mass, self.G, self.softening
                ))
                
        return {
            "trajectory": np.array(trajectory),  # (num_samples, N, 3)
            "timestamps": np.array(timestamps),
            "metrics": metrics_history,
            "mean_step_time_ms": float(np.mean(step_times) * 1000.0),
            "total_runtime_s": float(np.sum(step_times)),
            "device": "CPU (NumPy Vectorized)"
        }
