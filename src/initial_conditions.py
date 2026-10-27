"""
Initial Conditions Generator for Cosmic Acceleration N-Body System.
Creates a heavy central body and N-1 orbiting bodies in stable circular orbits,
including mathematically exact inclined Keplerian orbits for 3D simulations.
"""
import numpy as np
from typing import Tuple
from src.constants import SimulationConfig

def generate_keplerian_disk(
    config: SimulationConfig = None,
    is_3d: bool = True,
    seed: int = 42,
    moving_center: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate initial positions, velocities, and masses.
    
    Args:
        config: SimulationConfig instance
        is_3d: Whether to generate 3D inclined disk or flat 2D disk
        seed: Random seed for reproducibility
        moving_center: If True, the central body executes its own 3D orbit around the origin
    
    Returns:
        pos: (N, 3) float64 array of positions [x, y, z]
        vel: (N, 3) float64 array of velocities [vx, vy, vz]
        mass: (N,) float64 array of particle masses
    """
    if config is None:
        config = SimulationConfig()
        
    rng = np.random.default_rng(seed)
    N = config.num_bodies
    
    pos = np.zeros((N, 3), dtype=np.float64)
    vel = np.zeros((N, 3), dtype=np.float64)
    mass = np.full(N, config.orbiter_mass, dtype=np.float64)
    
    # Body 0 is the heavy central body
    mass[0] = config.central_mass
    num_orbiters = N - 1

    is_moving = moving_center or config.moving_center
    if is_moving:
        r0 = config.central_orbit_radius
        omega0 = config.central_omega
        z0 = config.central_orbit_tilt if is_3d else 0.0
        v0 = omega0 * r0
        
        pos[0] = [r0, 0.0, z0]
        vel[0] = [0.0, v0, 0.0]
        
        # Radii relative to moving center: [0.15, 0.65]
        rel_radii = rng.uniform(0.15, 0.65, size=num_orbiters)
        angles = rng.uniform(0.0, 2.0 * np.pi, size=num_orbiters)
        r_eff = np.sqrt(rel_radii**2 + config.softening**2)
        v_circ = np.sqrt(config.G * config.central_mass * rel_radii / (r_eff**1.5))
        
        if is_3d:
            inc = rng.uniform(-config.z_tilt_max, config.z_tilt_max, size=num_orbiters)
            pos[1:, 0] = pos[0, 0] + rel_radii * np.cos(angles)
            pos[1:, 1] = pos[0, 1] + rel_radii * np.sin(angles)
            pos[1:, 2] = pos[0, 2] + inc
            
            vel[1:, 0] = vel[0, 0] - v_circ * np.sin(angles)
            vel[1:, 1] = vel[0, 1] + v_circ * np.cos(angles)
            vel[1:, 2] = rng.uniform(-0.02, 0.02, size=num_orbiters)
        else:
            pos[1:, 0] = pos[0, 0] + rel_radii * np.cos(angles)
            pos[1:, 1] = pos[0, 1] + rel_radii * np.sin(angles)
            pos[1:, 2] = 0.0
            
            vel[1:, 0] = vel[0, 0] - v_circ * np.sin(angles)
            vel[1:, 1] = vel[0, 1] + v_circ * np.cos(angles)
            vel[1:, 2] = 0.0
            
        return pos, vel, mass
    
    pos[0] = [0.0, 0.0, 0.0]
    vel[0] = [0.0, 0.0, 0.0]
    
    # Orbiters: 1 to N-1
    # Distribute radii in range [r_min, r_max]
    radii = rng.uniform(config.r_min, config.r_max, size=num_orbiters)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=num_orbiters)
    
    # Effective circular orbital speed accounting for Plummer softening
    r_effective = np.sqrt(radii**2 + config.softening**2)
    v_circ = np.sqrt(config.G * config.central_mass * radii / (r_effective**1.5))
    
    if is_3d:
        # Mathematically exact inclined Keplerian orbits
        # Inclination i in radians: small tilt for visual depth
        inc_max = config.z_tilt_max / config.r_max  # ~0.05 rad (~2.8 degrees)
        inc = rng.uniform(-inc_max, inc_max, size=num_orbiters)
        Omega = rng.uniform(0.0, 2.0 * np.pi, size=num_orbiters)  # Ascending node angle
        
        # Orbital plane basis vectors P (ascending node) and Q (semi-latus rectum)
        Px = np.cos(Omega)
        Py = np.sin(Omega)
        Pz = np.zeros_like(Omega)
        
        Qx = -np.sin(Omega) * np.cos(inc)
        Qy = np.cos(Omega) * np.cos(inc)
        Qz = np.sin(inc)
        
        # Positions: r = radii * (cos(theta) * P + sin(theta) * Q)
        pos[1:, 0] = radii * (np.cos(angles) * Px + np.sin(angles) * Qx)
        pos[1:, 1] = radii * (np.cos(angles) * Py + np.sin(angles) * Qy)
        pos[1:, 2] = radii * (np.cos(angles) * Pz + np.sin(angles) * Qz)
        
        # Velocities: v = v_circ * (-sin(theta) * P + cos(theta) * Q)
        vel[1:, 0] = v_circ * (-np.sin(angles) * Px + np.cos(angles) * Qx)
        vel[1:, 1] = v_circ * (-np.sin(angles) * Py + np.cos(angles) * Qy)
        vel[1:, 2] = v_circ * (-np.sin(angles) * Pz + np.cos(angles) * Qz)
    else:
        # Pure 2D orbital plane
        pos[1:, 0] = radii * np.cos(angles)
        pos[1:, 1] = radii * np.sin(angles)
        pos[1:, 2] = 0.0
        
        vel[1:, 0] = -v_circ * np.sin(angles)
        vel[1:, 1] = v_circ * np.cos(angles)
        vel[1:, 2] = 0.0
        
    # Balance total system linear momentum so center-of-mass velocity is zero
    total_momentum = np.sum(mass[:, None] * vel, axis=0)
    vel[0] = -total_momentum / mass[0]
    
    return pos, vel, mass
