"""
Optimized Tiled CUDA N-Body Solver using Shared Memory.
Reduces global memory bandwidth from O(N^2) to O(N^2 / B) by caching particle tiles in shared memory.
Supports CuPy (with native runtime NVRTC CUDA compilation) and PyTorch CUDA.
"""
import numpy as np
import time
from typing import Dict, Optional
from src.constants import SimulationConfig
from src.integrators import compute_system_metrics

# Try importing cupy
_HAS_CUPY = False
try:
    import cupy as cp
    if cp.cuda.is_available():
        _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

# Try importing torch with CUDA
_HAS_TORCH_CUDA = False
try:
    import torch
    if torch.cuda.is_available():
        _HAS_TORCH_CUDA = True
except Exception:
    _HAS_TORCH_CUDA = False

# Raw CUDA C++ Source Code for Shared-Memory Tiling
CUDA_TILED_RAW_SRC = r"""
extern "C" __global__
void compute_accel_tiled_kernel(
    const double* __restrict__ pos,      // (N, 3)
    const double* __restrict__ mass,     // (N)
    double* __restrict__ acc,            // (N, 3)
    int N,
    double G,
    double softening_sq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double my_px = 0.0, my_py = 0.0, my_pz = 0.0;
    if (i < N) {
        my_px = pos[i * 3 + 0];
        my_py = pos[i * 3 + 1];
        my_pz = pos[i * 3 + 2];
    }
    
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;
    
    const int BLOCK_SIZE = 256;
    __shared__ double s_px[BLOCK_SIZE];
    __shared__ double s_py[BLOCK_SIZE];
    __shared__ double s_pz[BLOCK_SIZE];
    __shared__ double s_m[BLOCK_SIZE];
    
    int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int tile = 0; tile < num_tiles; ++tile) {
        int tile_idx = tile * BLOCK_SIZE + threadIdx.x;
        if (tile_idx < N) {
            s_px[threadIdx.x] = pos[tile_idx * 3 + 0];
            s_py[threadIdx.x] = pos[tile_idx * 3 + 1];
            s_pz[threadIdx.x] = pos[tile_idx * 3 + 2];
            s_m[threadIdx.x]  = mass[tile_idx];
        } else {
            s_px[threadIdx.x] = 0.0;
            s_py[threadIdx.x] = 0.0;
            s_pz[threadIdx.x] = 0.0;
            s_m[threadIdx.x]  = 0.0;
        }
        __syncthreads();
        
        if (i < N) {
            #pragma unroll 16
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                int global_j = tile * BLOCK_SIZE + j;
                if (global_j >= N || global_j == i) continue;
                
                double dx = s_px[j] - my_px;
                double dy = s_py[j] - my_py;
                double dz = s_pz[j] - my_pz;
                
                double dist_sq = dx * dx + dy * dy + dz * dz + softening_sq;
                double inv_dist = rsqrt(dist_sq);
                double inv_dist3 = inv_dist * inv_dist * inv_dist;
                
                double w = G * s_m[j] * inv_dist3;
                ax += w * dx;
                ay += w * dy;
                az += w * dz;
            }
        }
        __syncthreads();
    }
    
    if (i < N) {
        acc[i * 3 + 0] = ax;
        acc[i * 3 + 1] = ay;
        acc[i * 3 + 2] = az;
    }
}
"""

_CUPY_TILED_KERNEL = None

def _get_cupy_tiled_kernel():
    global _CUPY_TILED_KERNEL
    if _CUPY_TILED_KERNEL is None and _HAS_CUPY:
        _CUPY_TILED_KERNEL = cp.RawKernel(CUDA_TILED_RAW_SRC, "compute_accel_tiled_kernel")
    return _CUPY_TILED_KERNEL

class GPUTiledNBodySolver:
    """
    N-Body Solver utilizing block-tiled force computation.
    Uses CuPy RawKernel shared-memory CUDA kernel, PyTorch CUDA tensor chunking, or CPU fallback.
    """
    def __init__(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        mass: np.ndarray,
        config: Optional[SimulationConfig] = None,
        tile_size: int = 256
    ):
        self.config = config or SimulationConfig()
        self.N = len(mass)
        self.tile_size = tile_size
        self.G = float(self.config.G)
        self.softening = float(self.config.softening)
        self.softening_sq = self.softening ** 2
        
        self.time = 0.0
        self.step_count = 0
        
        if _HAS_CUPY:
            self.backend = "cupy"
            self.d_pos = cp.asarray(pos, dtype=cp.float64)
            self.d_vel = cp.asarray(vel, dtype=cp.float64)
            self.d_mass = cp.asarray(mass, dtype=cp.float64)
            self.d_acc = cp.zeros((self.N, 3), dtype=cp.float64)
            self.cupy_kernel = _get_cupy_tiled_kernel()
            self._compute_accel_cupy(self.d_pos, self.d_acc)
        elif _HAS_TORCH_CUDA:
            self.backend = "torch"
            self.device = torch.device("cuda:0")
            self.d_pos = torch.tensor(pos, dtype=torch.float64, device=self.device)
            self.d_vel = torch.tensor(vel, dtype=torch.float64, device=self.device)
            self.d_mass = torch.tensor(mass, dtype=torch.float64, device=self.device)
            self.d_acc = self._compute_accel_torch(self.d_pos)
        else:
            self.backend = "cpu"
            from src.cpu_solver import CPUNBodySolver
            self._cpu_fallback = CPUNBodySolver(pos, vel, mass, self.config)

    def _compute_accel_cupy(self, pos: "cp.ndarray", out_acc: "cp.ndarray"):
        threads = 256
        blocks = (self.N + threads - 1) // threads
        self.cupy_kernel(
            (blocks,),
            (threads,),
            (pos, self.d_mass, out_acc, cp.int32(self.N), cp.float64(self.G), cp.float64(self.softening_sq))
        )

    def _compute_accel_torch(self, pos: "torch.Tensor") -> "torch.Tensor":
        N = pos.size(0)
        accel = torch.zeros_like(pos)
        
        for j_start in range(0, N, self.tile_size):
            j_end = min(j_start + self.tile_size, N)
            tile_pos = pos[j_start:j_end]
            tile_mass = self.d_mass[j_start:j_end]
            
            diff = tile_pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, Tile, 3)
            dist_sq = torch.sum(diff ** 2, dim=-1) + self.softening_sq  # (N, Tile)
            inv_dist_cubed = 1.0 / (dist_sq * torch.sqrt(dist_sq))
            
            idx_i = torch.arange(N, device=self.device).unsqueeze(1)
            idx_j = torch.arange(j_start, j_end, device=self.device).unsqueeze(0)
            inv_dist_cubed[idx_i == idx_j] = 0.0
            
            weights = (self.G * tile_mass.unsqueeze(0)) * inv_dist_cubed
            accel += torch.sum(weights.unsqueeze(-1) * diff, dim=1)
            
        return accel

    def compute_acceleration(self, pos_np: np.ndarray) -> np.ndarray:
        if self.backend == "cupy":
            c_pos = cp.asarray(pos_np, dtype=cp.float64)
            c_acc = cp.zeros((len(pos_np), 3), dtype=cp.float64)
            self._compute_accel_cupy(c_pos, c_acc)
            return cp.asnumpy(c_acc)
        elif self.backend == "torch":
            t_pos = torch.tensor(pos_np, dtype=torch.float64, device=self.device)
            acc = self._compute_accel_torch(t_pos)
            return acc.cpu().numpy()
        else:
            return self._cpu_fallback.compute_acceleration(pos_np)

    def step(self, dt: float) -> float:
        """Perform a single Velocity Verlet integration step."""
        dt_f = float(dt)
        if self.backend == "cupy":
            self.d_pos += self.d_vel * dt_f + 0.5 * self.d_acc * (dt_f ** 2)
            new_acc = cp.zeros_like(self.d_acc)
            self._compute_accel_cupy(self.d_pos, new_acc)
            self.d_vel += 0.5 * (self.d_acc + new_acc) * dt_f
            self.d_acc = new_acc
            cp.cuda.Stream.null.synchronize()
        elif self.backend == "torch":
            self.d_pos = self.d_pos + self.d_vel * dt_f + 0.5 * self.d_acc * (dt_f ** 2)
            new_acc = self._compute_accel_torch(self.d_pos)
            self.d_vel = self.d_vel + 0.5 * (self.d_acc + new_acc) * dt_f
            self.d_acc = new_acc
            torch.cuda.synchronize()
        else:
            self._cpu_fallback.step(dt)
            
        self.time += dt
        self.step_count += 1
        return dt

    def get_state(self):
        if self.backend == "cupy":
            return (
                cp.asnumpy(self.d_pos),
                cp.asnumpy(self.d_vel),
                cp.asnumpy(self.d_acc),
                cp.asnumpy(self.d_mass)
            )
        elif self.backend == "torch":
            return (
                self.d_pos.cpu().numpy(),
                self.d_vel.cpu().numpy(),
                self.d_acc.cpu().numpy(),
                self.d_mass.cpu().numpy()
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
                if self.backend == "cupy":
                    a_mag = cp.linalg.norm(self.d_acc, axis=-1)
                    max_a = float(cp.max(a_mag).item())
                elif self.backend == "torch":
                    a_mag = torch.norm(self.d_acc, dim=-1)
                    max_a = float(torch.max(a_mag).item())
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
                
        backend_name = "CUDA Tiled (CuPy Shared-Memory Kernel)" if self.backend == "cupy" else "CUDA Tiled (PyTorch)"
        return {
            "trajectory": np.array(trajectory),
            "timestamps": np.array(timestamps),
            "metrics": metrics_history,
            "mean_step_time_ms": float(np.mean(step_times) * 1000.0),
            "total_runtime_s": float(np.sum(step_times)),
            "device": backend_name
        }
