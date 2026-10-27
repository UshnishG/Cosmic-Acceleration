"""
Physical and Simulation Constants for Cosmic Acceleration.
"""
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    # System Setup (Poster Step 1)
    num_bodies: int = 150              # 1 central body + 149 orbiting bodies
    central_mass: float = 1.2          # Mass of the heavy central body
    orbiter_mass: float = 1e-5         # Mass of each orbiting body
    
    # Orbit boundaries
    r_min: float = 0.20                # Inner orbital radius
    r_max: float = 1.10                # Outer orbital radius
    z_tilt_max: float = 0.05           # Small vertical perturbation for 3D depth (Z in [-0.06, 0.06])
    
    # Gravitational Physics (Poster Steps 3 & 4)
    G: float = 1.0                     # Gravitational constant in normalized cosmic units
    softening: float = 0.02            # Softening factor (epsilon) to avoid singularities
    
    # Integration & Timestepping (Poster Step 5)
    dt_fixed: float = 0.005            # Fixed time step for 3D simulations (T = 5.0 at 1000 steps)
    eta_adaptive: float = 0.05         # Adaptive timestep safety factor: dt = eta * min(sqrt(softening / ||a||))
    dt_min: float = 0.0005             # Minimum allowed adaptive dt
    dt_max: float = 0.015              # Maximum allowed adaptive dt
    
    # Sampling & Logging (Poster Step 6)
    sample_interval_2d: int = 5        # Save every 5 steps for smooth 2D animations
    sample_interval_3d: int = 5        # Save every 5 steps for smooth 3D trajectories
    
    # Barnes-Hut & Monte Carlo
    theta: float = 0.5                 # Barnes-Hut opening threshold (multipole acceptance criterion)
    mc_sample_ratio: float = 0.15      # Fraction of particles for Monte Carlo control-variate sampling
    
    # Moving Central Body Orbit Configuration
    moving_center: bool = False        # If True, central heavy body moves in its own stable 3D orbit
    central_orbit_radius: float = 0.25 # Radius of central body's orbit around system barycenter
    central_omega: float = 1.20        # Angular velocity of central body's orbit
    central_orbit_tilt: float = 0.02   # 3D inclination tilt of central body's orbit
