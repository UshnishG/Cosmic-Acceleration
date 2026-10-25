"""
Automated Test Runner for Cosmic Acceleration N-Body Simulation.
Runs all physics, equivalence, and statistical validation tests using Python's built-in unittest framework.
"""
import sys
import unittest
import numpy as np

# Add project root to sys.path
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tests.test_physics import (
    test_initial_conditions,
    test_newtonian_acceleration_and_softening,
    test_energy_conservation_and_drift,
    test_reproducibility
)
from tests.test_solvers_equivalence import (
    test_cpu_gpu_direct_parity,
    test_cpu_gpu_tiled_parity,
    test_barnes_hut_accuracy_bound
)
from tests.test_control_variates import (
    test_control_variates_unbiasedness_and_variance_reduction
)

class TestPhysicsSuite(unittest.TestCase):
    def test_01_initial_conditions(self):
        test_initial_conditions()

    def test_02_newtonian_acceleration_and_softening(self):
        test_newtonian_acceleration_and_softening()

    def test_03_energy_conservation_and_drift(self):
        test_energy_conservation_and_drift()

    def test_04_reproducibility(self):
        test_reproducibility()

class TestEquivalenceSuite(unittest.TestCase):
    def test_05_cpu_gpu_direct_parity(self):
        test_cpu_gpu_direct_parity()

    def test_06_cpu_gpu_tiled_parity(self):
        test_cpu_gpu_tiled_parity()

    def test_07_barnes_hut_accuracy_bound(self):
        test_barnes_hut_accuracy_bound()

class TestStatisticsSuite(unittest.TestCase):
    def test_08_control_variates(self):
        test_control_variates_unbiasedness_and_variance_reduction()

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
