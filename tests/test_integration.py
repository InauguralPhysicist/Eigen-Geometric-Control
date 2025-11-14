import pytest
"""
Integration tests for complete Eigen simulation pipeline

Tests full workflow: imports → simulation → convergence
"""

import sys

sys.path.insert(0, ".")  # noqa: E402

import numpy as np

from src import run_arm_simulation, run_xor_simulation


class TestArmSimulation:
    """Test complete arm control simulation"""

    def test_simulation_runs(self):
        """Test that simulation completes without errors"""
        results = run_arm_simulation(n_ticks=10)
        assert len(results) == 10, f"Expected 10 ticks, got {len(results)}"

    def test_simulation_columns(self):
        """Test that results contain all expected columns"""
        results = run_arm_simulation(n_ticks=5)

        expected_columns = [
            "tick",
            "theta1_rad",
            "theta2_rad",
            "x",
            "y",
            "ds2_total",
            "target_term",
            "obs_term",
            "reg_term",
            "grad_norm",
            "C",
            "S",
            "ds2_CS",
            "d_obs",
            "delta_theta1",
            "delta_theta2",
            "delta_theta_sq",
        ]

        for col in expected_columns:
            assert col in results.columns, f"Missing column: {col}"

    def test_ds2_decreases(self):
        """Test that ds² decreases over time (convergence)"""
        results = run_arm_simulation(n_ticks=50)

        initial_ds2 = results["ds2_total"].iloc[0]
        final_ds2 = results["ds2_total"].iloc[-1]

        assert final_ds2 < initial_ds2, f"ds² should decrease: {initial_ds2:.4f} → {final_ds2:.4f}"

    def test_gradient_decreases(self):
        """Test that gradient magnitude decreases (convergence)"""
        results = run_arm_simulation(n_ticks=50)

        initial_grad = results["grad_norm"].iloc[0]
        final_grad = results["grad_norm"].iloc[-1]

        assert (
            final_grad < initial_grad
        ), f"Gradient should decrease: {initial_grad:.4f} → {final_grad:.4f}"

    def test_obstacle_avoidance(self):
        """Test that arm maintains safe distance from obstacle"""
        results = run_arm_simulation(obstacle_radius=0.25, n_ticks=100)

        min_distance = results["d_obs"].min()
        # Should maintain distance > obstacle radius
        assert min_distance > 0.25, f"Violated obstacle boundary: min_distance={min_distance:.4f}m"

    def test_custom_parameters(self):
        """Test simulation with custom parameters"""
        results = run_arm_simulation(
            theta_init=(-1.0, 1.0), target=(1.5, 0.5), eta=0.15, n_ticks=20
        )

        assert len(results) == 20
        assert results["theta1_rad"].iloc[0] == -1.0
        assert results["theta2_rad"].iloc[0] == 1.0


class TestXORSimulation:
    """Test XOR rotation simulation"""

    def test_xor_runs(self):
        """Test that XOR simulation completes"""
        results = run_xor_simulation(n_ticks=10)
        assert len(results) == 10

    def test_xor_columns(self):
        """Test XOR results have expected columns"""
        results = run_xor_simulation(n_ticks=5)

        expected_columns = [
            "tick",
            "state_before_hex",
            "state_after_hex",
            "axis_hex",
            "C",
            "S",
            "ds2_CS",
        ]

        for col in expected_columns:
            assert col in results.columns, f"Missing column: {col}"

    def test_xor_period_2(self):
        """Test XOR exhibits period-2 oscillation"""
        results = run_xor_simulation(n_ticks=4, seed=42)

        state0 = results["state_before_hex"].iloc[0]
        state1 = results["state_after_hex"].iloc[0]
        state2 = results["state_before_hex"].iloc[1]
        state3 = results["state_after_hex"].iloc[1]

        # Period-2: state0 → state1 → state0 → state1
        assert state1 == state2, "XOR should oscillate between two states"
        assert state0 == state3, "XOR should return to initial state"

    def test_xor_constant_metrics(self):
        """Test XOR maintains constant C, S, ds² values"""
        results = run_xor_simulation(n_ticks=10, seed=42)

        # All C values should be the same
        C_values = results["C"].unique()
        assert len(C_values) == 1, f"C should be constant, got {C_values}"

        # All S values should be the same
        S_values = results["S"].unique()
        assert len(S_values) == 1, f"S should be constant, got {S_values}"

        # All ds² values should be the same
        ds2_values = results["ds2_CS"].unique()
        assert len(ds2_values) == 1, f"ds² should be constant, got {ds2_values}"

    def test_xor_space_like(self):
        """Test XOR stays in space-like regime (ds² < 0)"""
        results = run_xor_simulation(n_ticks=10)

        ds2 = results["ds2_CS"].iloc[0]
        assert ds2 < 0, f"XOR should be space-like (ds²<0), got ds²={ds2}"


class TestReproducibility:
    """Test that simulations are deterministic"""

    def test_arm_reproducibility(self):
        """Test arm simulation produces identical results"""
        results1 = run_arm_simulation(n_ticks=20)
        results2 = run_arm_simulation(n_ticks=20)

        # Should be identical (same initial conditions)
        np.testing.assert_array_almost_equal(
            results1["ds2_total"].values,
            results2["ds2_total"].values,
            decimal=10,
            err_msg="Arm simulation not reproducible",
        )

    def test_xor_reproducibility(self):
        """Test XOR simulation produces identical results with same seed"""
        results1 = run_xor_simulation(n_ticks=10, seed=42)
        results2 = run_xor_simulation(n_ticks=10, seed=42)

        # Should be identical with same seed
        assert results1["state_before_hex"].equals(
            results2["state_before_hex"]
        ), "XOR simulation not reproducible with same seed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
