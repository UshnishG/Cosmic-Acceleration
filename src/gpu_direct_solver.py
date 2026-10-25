"""
GPU Direct-Force N-Body Solver.
Supports both CuPy (if available) and PyTorch CUDA tensor acceleration on NVIDIA GPU,
with automatic fallback to CPU if no GPU device is present.
"""
import numpy as np
import time
from typing import Dict, Optional
from src.constants import SimulationConfig
from src.integrators import compute_system_metrics

# Detect GPU capability
_HAS_CUPY = False
try:
    import cupy as cp
    if cp.cuda.is_available():
        _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

_HAS_TORCH_CUDA = False
try:
    import torch
    if torch.cuda.is_available():
        _HAS_TORCH_CUDA = True
except Exception:
    _HAS_TORCH_CUDA = False

def get_available_device() -> str:
    if _HAS_CUPY:
        return "CUDA (CuPy)"
    elif _HAS_TORCH_CUDA:
        return f"CUDA (PyTorch GPU: {torch.cuda.get_device_name(0)})"
    return "CPU (NumPy Fallback)"

class GPUNBodyDirectSolver:
    def __init__(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        mass: np.ndarray,
        config: Optional[SimulationConfig] = None,
        force_backend: Optional[str] = None
    ):
        self.config = config or SimulationConfig()
        self.N = len(mass)
        self.G = self.config.G
        self.softening = self.config.softening
        self.softening_sq = self.softening ** 2
        
        # Decide backend: "cupy", "torch", or "cpu"
        if force_backend == "cupy" and _HAS_CUPY:
            self.backend = "cupy"
        elif force_backend == "torch" and _HAS_TORCH_CUDA:
            self.backend = "torch"
        elif _HAS_TORCH_CUDA:
            self.backend = "torch"
        elif _HAS_CUPY:
            self.backend = "cupy"
        else:
            self.backend = "cpu"
            
        self.time = 0.0
        self.step_count = 0
        
        # Initialize device memory
        if self.backend == "torch":
            self.device = torch.device("cuda:0")
            self.d_pos = torch.tensor(pos, dtype=torch.float64, device=self.device)
            self.d_vel = torch.tensor(vel, dtype=torch.float64, device=self.device)
            self.d_mass = torch.tensor(mass, dtype=torch.float64, device=self.device)
            self.d_acc = self._compute_accel_torch(self.d_pos)
        elif self.backend == "cupy":
            self.d_pos = cp.asarray(pos, dtype=cp.float64)
            self.d_vel = cp.asarray(vel, dtype=cp.float64)
            self.d_mass = cp.asarray(mass, dtype=cp.float64)
            self.d_acc = self._compute_accel_cupy(self.d_pos)
        else:
            from src.cpu_solver import CPUNBodySolver
            self._cpu_fallback = CPUNBodySolver(pos, vel, mass, self.config)

    def _compute_accel_torch(self, pos: "torch.Tensor") -> "torch.Tensor":
        """Vectorized GPU direct pairwise acceleration using PyTorch CUDA."""
        # diff[i, j, :] = pos[j] - pos[i]
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N, 3)
        dist_sq = torch.sum(diff ** 2, dim=-1) + self.softening_sq  # (N, N)
        inv_dist_cubed = 1.0 / (dist_sq * torch.sqrt(dist_sq))
        inv_dist_cubed.fill_diagonal_(0.0)
        
        weights = (self.G * self.d_mass.unsqueeze(0)) * inv_dist_cubed  # (N, N)
        accel = torch.sum(weights.unsqueeze(-1) * diff, dim=1)  # (N, 3)
        return accel

    def _compute_accel_cupy(self, pos: "cp.ndarray") -> "cp.ndarray":
        """Vectorized GPU direct pairwise acceleration using CuPy."""
        diff = pos[None, :, :] - pos[:, None, :]  # (N, N, 3)
        dist_sq = cp.sum(diff ** 2, axis=-1) + self.softening_sq
        inv_dist_cubed = 1.0 / (dist_sq * cp.sqrt(dist_sq))
        cp.fill_diagonal(inv_dist_cubed, 0.0)
        
        weights = (self.G * self.d_mass[None, :]) * inv_dist_cubed
        accel = cp.sum(weights[:, :, None] * diff, axis=1)
        return accel

    def compute_acceleration(self, pos_np: np.ndarray) -> np.ndarray:
        """Host-callable acceleration function returning NumPy array."""
        if self.backend == "torch":
            t_pos = torch.tensor(pos_np, dtype=torch.float64, device=self.device)
            acc = self._compute_accel_torch(t_pos)
            return acc.cpu().numpy()
        elif self.backend == "cupy":
            c_pos = cp.asarray(pos_np, dtype=cp.float64)
            acc = self._compute_accel_cupy(c_pos)
            return cp.asnumpy(acc)
        else:
            return self._cpu_fallback.compute_acceleration(pos_np)

    def step(self, dt: float) -> float:
        """Perform a single Velocity Verlet integration step on the GPU."""
        if self.backend == "torch":
            dt_f = float(dt)
            # x(t + dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
            self.d_pos = self.d_pos + self.d_vel * dt_f + 0.5 * self.d_acc * (dt_f ** 2)
            # a(t + dt)
            new_acc = self._compute_accel_torch(self.d_pos)
            # v(t + dt) = v(t) + 0.5*(a(t) + a(t + dt))*dt
            self.d_vel = self.d_vel + 0.5 * (self.d_acc + new_acc) * dt_f
            self.d_acc = new_acc
            torch.cuda.synchronize()
        elif self.backend == "cupy":
            dt_f = float(dt)
            self.d_pos += self.d_vel * dt_f + 0.5 * self.d_acc * (dt_f ** 2)
            new_acc = self._compute_accel_cupy(self.d_pos)
            self.d_vel += 0.5 * (self.d_acc + new_acc) * dt_f
            self.d_acc = new_acc
            cp.cuda.Stream.null.synchronize()
        else:
            self._cpu_fallback.step(dt)

        self.time += dt
        self.step_count += 1
        return dt

    def get_state(self):
        """Retrieve current pos, vel, acc, mass as NumPy arrays."""
        if self.backend == "torch":
            return (
                self.d_pos.cpu().numpy(),
                self.d_vel.cpu().numpy(),
                self.d_acc.cpu().numpy(),
                self.d_mass.cpu().numpy()
            )
        elif self.backend == "cupy":
            return (
                cp.asnumpy(self.d_pos),
                cp.asnumpy(self.d_vel),
                cp.asnumpy(self.d_acc),
                cp.asnumpy(self.d_mass)
            )
        else:
            return (
                self._cpu_fallback.pos.copy(),
                self._cpu_fallback.vel.copy(),
                self._cpu_fallback.acc.copy(),
                self._cpu_fallback.mass.copy()
            )

    def simulate(
        self,
        num_steps: int,
        dt: Optional[float] = None,
        adaptive: bool = False,
        sample_interval: int = 20
    ) -> Dict:
        """
        Runs the simulation entirely on the GPU, sampling host snapshots at sample_interval.
        """
        trajectory = []
        timestamps = []
        metrics_history = []
        step_times = []
        
        pos_np, vel_np, acc_np, mass_np = self.get_state()
        trajectory.append(pos_np.copy())
        timestamps.append(self.time)
        metrics_history.append(compute_system_metrics(
            pos_np, vel_np, mass_np, self.G, self.softening
        ))
        
        for step_idx in range(1, num_steps + 1):
            if adaptive:
                if self.backend == "torch":
                    a_mag = torch.norm(self.d_acc, dim=-1)
                    max_a = float(torch.max(a_mag).item())
                elif self.backend == "cupy":
                    a_mag = cp.linalg.norm(self.d_acc, axis=-1)
                    max_a = float(cp.max(a_mag).item())
                else:
                    max_a = float(np.max(np.linalg.norm(self._cpu_fallback.acc, axis=-1)))
                
                if max_a <= 1e-12:
                    current_dt = self.config.dt_max
                else:
                    current_dt = float(np.clip(
                        self.config.eta_adaptive * np.sqrt(self.softening / max_a),
                        self.config.dt_min,
                        self.config.dt_max
                    ))
            else:
                current_dt = dt or self.config.dt_fixed
                
            t_start = time.perf_counter()
            self.step(current_dt)
            t_end = time.perf_counter()
            step_times.append(t_end - t_start)
            
            if step_idx % sample_interval == 0 or step_idx == num_steps:
                p_np, v_np, a_np, m_np = self.get_state()
                trajectory.append(p_np.copy())
                timestamps.append(self.time)
                metrics_history.append(compute_system_metrics(
                    p_np, v_np, m_np, self.G, self.softening
                ))
                
        return {
            "trajectory": np.array(trajectory),
            "timestamps": np.array(timestamps),
            "metrics": metrics_history,
            "mean_step_time_ms": float(np.mean(step_times) * 1000.0),
            "total_runtime_s": float(np.sum(step_times)),
            "device": get_available_device()
        }
