"""
Statistical Validation Tests for Monte Carlo Control Variates.
Verifies estimator unbiasedness and statistical variance reduction factor.
"""
import numpy as np
from src.constants import SimulationConfig
from src.initial_conditions import generate_keplerian_disk
from src.cpu_solver import CPUNBodySolver
from src.monte_carlo_cv import MonteCarloControlVariateEstimator

def test_control_variates_unbiasedness_and_variance_reduction():
    config = SimulationConfig(num_bodies=100)
    pos, vel, mass = generate_keplerian_disk(config, is_3d=True, seed=42)
    
    cpu_solver = CPUNBodySolver(pos, vel, mass, config)
    exact_accels = cpu_solver.compute_acceleration(pos)
    
    cv_estimator = MonteCarloControlVariateEstimator(pos, mass, config)
    stats = cv_estimator.statistical_validation(
        exact_accels, sample_ratio=0.20, num_trials=40, seed=123
    )
    
    print("\nControl Variates Validation Results:")
    print(f"  Mean CV Bias: {stats['mean_cv_bias']:.4e}")
    print(f"  Mean MC Bias: {stats['mean_mc_bias']:.4e}")
    print(f"  t-statistic: {stats['t_statistic']:.4f}, p-value: {stats['p_value']:.4f}")
    print(f"  Variance Reduction Factor (VRF): {stats['variance_reduction_factor']:.2f}x")
    
    assert stats["is_unbiased"], "Control variate estimator shows statistically significant bias!"
    assert stats["variance_reduction_factor"] >= 1.0, "Control variate failed to achieve variance reduction!"
