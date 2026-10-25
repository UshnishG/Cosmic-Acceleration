"""
Symplectic Integrators and Physical Conservation Diagnostic Tools.
Implements Velocity Verlet, Leapfrog, Energy/Momentum metrics, and Adaptive Timestepping.
"""
import numpy as np
from typing import Tuple, Callable

def compute_system_metrics(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    G: float = 1.0,
    softening: float = 0.02
) -> dict:
    """
    Computes total kinetic, potential, and mechanical energy, plus linear and angular momentum.
    """
    N = len(mass)
    # Kinetic Energy: K = 0.5 * sum(m * v^2)
    v_sq = np.sum(vel**2, axis=1)
    kinetic = 0.5 * np.sum(mass * v_sq)
    
    # Gravitational Potential Energy: U = -0.5 * G * sum_{i != j} (m_i * m_j / sqrt(r_ij^2 + eps^2))
    diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff**2, axis=-1) + softening**2
    inv_dist = 1.0 / np.sqrt(dist_sq)
    np.fill_diagonal(inv_dist, 0.0)
    
    potential = -0.5 * G * np.sum(mass[:, None] * mass[None, :] * inv_dist)
    total_energy = kinetic + potential
    
    # Linear Momentum: P = sum(m * v)
    linear_momentum = np.sum(mass[:, None] * vel, axis=0)
    
    # Angular Momentum: L = sum(m * (r x v))
    angular_momentum = np.sum(mass[:, None] * np.cross(pos, vel), axis=0)
    
    return {
        "kinetic_energy": float(kinetic),
        "potential_energy": float(potential),
        "total_energy": float(total_energy),
        "linear_momentum": linear_momentum,
        "angular_momentum": angular_momentum,
        "linear_momentum_mag": float(np.linalg.norm(linear_momentum)),
        "angular_momentum_mag": float(np.linalg.norm(angular_momentum))
    }

def compute_adaptive_dt(
    accel: np.ndarray,
    softening: float,
    eta: float = 0.02,
    dt_min: float = 1e-4,
    dt_max: float = 0.02
) -> float:
    """
    Dynamical adaptive time step selection based on maximum gravitational acceleration:
    dt = eta * min_i( sqrt(eps / ||a_i||) )
    Ensures stability during close encounters while taking larger steps in quiescent regions.
    """
    a_mag = np.linalg.norm(accel, axis=-1)
    max_a = np.max(a_mag)
    if max_a <= 1e-12:
        return dt_max
    dt = eta * np.sqrt(softening / max_a)
    return float(np.clip(dt, dt_min, dt_max))

def velocity_verlet_step(
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    force_func: Callable[[np.ndarray], np.ndarray],
    dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Symplectic 2nd-order Velocity Verlet integration step.
    
    x(t + dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    a(t + dt) = Force(x(t + dt))
    v(t + dt) = v(t) + 0.5*(a(t) + a(t + dt))*dt
    """
    new_pos = pos + vel * dt + 0.5 * acc * (dt ** 2)
    new_acc = force_func(new_pos)
    new_vel = vel + 0.5 * (acc + new_acc) * dt
    return new_pos, new_vel, new_acc
