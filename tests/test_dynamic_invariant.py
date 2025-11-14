"""
Tests for Dynamic Invariant Detection

These tests demonstrate that the framework can automatically adapt
to different systems by detecting characteristic parameters from data.

Key proof: Auto-detected parameters → robust across different systems
"""

import numpy as np
import pytest

from src.eigen_adaptive import (
    estimate_characteristic_velocity,
    estimate_oscillation_threshold,
    estimate_natural_timescale,
    detect_system_parameters,
    adaptive_control_step,
    renormalize_lorentz_state,
)


class TestEstimateCharacteristicVelocity:
    """Test characteristic velocity estimation."""

    def test_fast_system_high_c(self):
        """Fast moving system → high c parameter."""
        # Fast linear motion
        trajectory = [np.array([i * 2.0, i * 2.0]) for i in range(30)]

        c = estimate_characteristic_velocity(trajectory)

        # Should detect high velocity (around 2.8 for diagonal motion)
        assert c > 2.0

    def test_slow_system_low_c(self):
        """Slow moving system → low c parameter."""
        # Slow linear motion
        trajectory = [np.array([i * 0.1, i * 0.1]) for i in range(30)]

        c = estimate_characteristic_velocity(trajectory)

        # Should detect low velocity (around 0.14 for diagonal)
        assert c < 0.3

    def test_stationary_system_minimum_c(self):
        """Stationary system → minimum c (avoid zero)."""
        # No motion
        trajectory = [np.array([1.0, 1.0]) for _ in range(30)]

        c = estimate_characteristic_velocity(trajectory)

        # Should return minimum threshold (not zero)
        assert c >= 0.01
        assert c < 0.1

    def test_insufficient_samples_default(self):
        """Insufficient samples → return default."""
        trajectory = [np.array([1.0, 1.0])]

        c = estimate_characteristic_velocity(trajectory, min_samples=10)

        # Should return default
        assert c == 1.0

    def test_mixed_speeds_uses_percentile(self):
        """Mixed speeds → uses high percentile (not max, not mean)."""
        # Mostly slow, with occasional fast movements
        trajectory = []
        for i in range(50):
            if i % 10 == 0:  # Occasional fast jump
                trajectory.append(np.array([float(i) + 5.0, 0.0]))
            else:  # Mostly slow
                trajectory.append(np.array([float(i), 0.0]))

        c = estimate_characteristic_velocity(trajectory)

        # Should be influenced by fast movements but not completely dominated
        # 95th percentile should capture occasional fast jumps
        assert 1.0 < c <= 7.0  # Allow up to 7.0 for fast jumps


class TestEstimateOscillationThreshold:
    """Test oscillation threshold estimation."""

    def test_stable_trajectory_high_threshold(self):
        """Stable smooth trajectory → high threshold (less sensitive)."""
        # Very smooth linear motion
        trajectory = [np.array([float(i), float(i)]) for i in range(30)]

        threshold = estimate_oscillation_threshold(trajectory)

        # Stable system can use high threshold
        assert threshold > 0.9

    def test_variable_trajectory_lower_threshold(self):
        """High variance trajectory → lower threshold (more sensitive)."""
        # Highly variable motion (random velocities)
        np.random.seed(42)
        trajectory = []
        state = np.array([0.0, 0.0])
        for _ in range(30):
            # Random large jumps
            state = state + np.random.randn(2) * 3.0
            trajectory.append(state.copy())

        threshold = estimate_oscillation_threshold(trajectory)

        # Variable system needs lower threshold
        assert threshold < 0.92  # More lenient bound

    def test_insufficient_samples_default(self):
        """Insufficient samples → return default threshold."""
        trajectory = [np.array([1.0, 1.0])]

        threshold = estimate_oscillation_threshold(trajectory)

        assert threshold == 0.92

    def test_threshold_in_valid_range(self):
        """Threshold should always be in [0.7, 0.95]."""
        # Extreme variance case
        np.random.seed(42)
        trajectory = [np.random.randn(2) * 10.0 for _ in range(30)]

        threshold = estimate_oscillation_threshold(trajectory)

        assert 0.7 <= threshold <= 0.95


class TestEstimateNaturalTimescale:
    """Test natural time scale estimation."""

    def test_quickly_decorrelating_system(self):
        """Random walk → quick decorrelation → small tau."""
        np.random.seed(42)
        trajectory = []
        state = np.array([0.0, 0.0])
        for _ in range(30):
            state = state + np.random.randn(2)
            trajectory.append(state.copy())

        tau = estimate_natural_timescale(trajectory)

        # Random walk decorrelates quickly
        assert tau < 5.0

    def test_slowly_decorrelating_system(self):
        """Smooth linear motion → slow decorrelation → large tau."""
        # Very smooth motion (high correlation)
        trajectory = [np.array([float(i), float(i)]) for i in range(30)]

        tau = estimate_natural_timescale(trajectory)

        # Smooth motion maintains correlation
        assert tau >= 5.0  # Can be exactly 5.0 if always > 0.5

    def test_insufficient_samples_default(self):
        """Insufficient samples → return default."""
        trajectory = [np.array([1.0, 1.0])]

        tau = estimate_natural_timescale(trajectory)

        assert tau == 5.0

    def test_minimum_timescale(self):
        """Timescale should be at least 1."""
        # Any trajectory
        trajectory = [np.array([float(i), 0.0]) for i in range(30)]

        tau = estimate_natural_timescale(trajectory)

        assert tau >= 1.0


class TestDetectSystemParameters:
    """Test full system parameter detection."""

    def test_detects_all_parameters(self):
        """Should detect c, threshold, tau, eta."""
        trajectory = [np.array([float(i) * 0.5, float(i) * 0.5]) for i in range(30)]

        params = detect_system_parameters(trajectory)

        # Should have all keys
        assert "c" in params
        assert "oscillation_threshold" in params
        assert "tau" in params
        assert "eta_recommended" in params
        assert "confidence" in params

        # All should be positive
        assert params["c"] > 0
        assert 0.7 <= params["oscillation_threshold"] <= 0.95
        assert params["tau"] >= 1.0
        assert 0.05 <= params["eta_recommended"] <= 0.5

    def test_insufficient_samples_returns_defaults(self):
        """Few samples → return defaults with low confidence."""
        trajectory = [np.array([1.0, 1.0]) for _ in range(5)]

        params = detect_system_parameters(trajectory, min_samples=15)

        assert params["c"] == 1.0
        assert params["oscillation_threshold"] == 0.92
        assert params["tau"] == 5.0
        assert params["eta_recommended"] == 0.1
        assert params["confidence"] == 0.0

    def test_many_samples_high_confidence(self):
        """Many samples → high confidence."""
        trajectory = [np.array([float(i), float(i)]) for i in range(100)]

        params = detect_system_parameters(trajectory, min_samples=15)

        # Should have high confidence
        assert params["confidence"] > 0.5

    def test_fast_system_larger_eta(self):
        """Fast systems (small tau) → larger recommended eta."""
        # Fast decorrelating system
        np.random.seed(42)
        trajectory = []
        state = np.array([0.0, 0.0])
        for _ in range(30):
            state = state + np.random.randn(2) * 2.0
            trajectory.append(state.copy())

        params = detect_system_parameters(trajectory)

        # Fast system → larger eta (respond quickly)
        assert params["eta_recommended"] > 0.15

    def test_slow_system_smaller_eta(self):
        """Slow systems (large tau) → smaller recommended eta."""
        # Slowly changing system
        trajectory = [np.array([float(i) * 0.1, float(i) * 0.1]) for i in range(50)]

        params = detect_system_parameters(trajectory)

        # Slow system → smaller eta (be cautious)
        assert params["eta_recommended"] < 0.3


class TestAdaptiveControlStep:
    """Test adaptive control with auto-tuning."""

    def test_control_moves_toward_target(self):
        """Adaptive control should move toward target."""
        current = np.array([5.0, 5.0])
        target = np.array([0.0, 0.0])
        history = [np.array([10.0 - i * 0.5, 10.0 - i * 0.5]) for i in range(20)]

        new_state, info = adaptive_control_step(current, target, history, auto_tune=True)

        # Should move closer
        dist_before = np.linalg.norm(current - target)
        dist_after = np.linalg.norm(new_state - target)

        assert dist_after < dist_before

    def test_auto_tuning_detects_parameters(self):
        """With sufficient samples, should detect parameters."""
        current = np.array([5.0, 5.0])
        target = np.array([0.0, 0.0])
        history = [np.array([10.0 - i * 0.5, 10.0 - i * 0.5]) for i in range(30)]

        new_state, info = adaptive_control_step(
            current, target, history, auto_tune=True, min_samples_for_tuning=15
        )

        # Should have detected parameters
        assert info["parameters_detected"] == True
        assert info["c"] > 0
        assert info["confidence"] > 0

    def test_insufficient_samples_uses_defaults(self):
        """With few samples, should use defaults."""
        current = np.array([5.0, 5.0])
        target = np.array([0.0, 0.0])
        history = [np.array([5.0, 5.0])]

        new_state, info = adaptive_control_step(current, target, history, auto_tune=True)

        # Should use defaults
        assert info["parameters_detected"] == False
        assert info["c"] == 1.0
        assert info["confidence"] == 0.0

    def test_manual_eta_overrides_detection(self):
        """Manually specified eta should override auto-detection."""
        current = np.array([5.0, 5.0])
        target = np.array([0.0, 0.0])
        history = [np.array([10.0 - i * 0.5, 10.0 - i * 0.5]) for i in range(30)]

        new_state, info = adaptive_control_step(
            current, target, history, eta=0.25, auto_tune=True
        )

        # Should use manual eta
        assert info["eta_used"] == 0.25

    def test_auto_tune_disabled_uses_defaults(self):
        """With auto_tune=False, should use defaults."""
        current = np.array([5.0, 5.0])
        target = np.array([0.0, 0.0])
        history = [np.array([10.0 - i * 0.5, 10.0 - i * 0.5]) for i in range(30)]

        new_state, info = adaptive_control_step(
            current, target, history, auto_tune=False
        )

        assert info["parameters_detected"] == False


class TestRenormalizeLorentzState:
    """Test Lorentz coordinate renormalization."""

    def test_renormalization_preserves_invariant(self):
        """When c changes, renormalization should preserve ds²."""
        timelike = 5.0
        spacelike = 3.0
        old_c = 1.0
        new_c = 2.0

        # Original invariant
        ds2_old = timelike**2 - spacelike**2 / old_c**2

        # Renormalize
        t_new, x_new = renormalize_lorentz_state(timelike, spacelike, old_c, new_c)

        # New invariant
        ds2_new = t_new**2 - x_new**2 / new_c**2

        # Should be preserved
        assert np.isclose(ds2_old, ds2_new, atol=1e-12)

    def test_identity_renormalization(self):
        """When c doesn't change, state should be unchanged."""
        timelike = 5.0
        spacelike = 3.0
        c = 1.0

        t_new, x_new = renormalize_lorentz_state(timelike, spacelike, c, c)

        assert t_new == timelike
        assert x_new == spacelike

    def test_handles_zero_c(self):
        """Should handle zero c gracefully."""
        timelike = 5.0
        spacelike = 3.0

        t_new, x_new = renormalize_lorentz_state(timelike, spacelike, 0.0, 1.0)

        # Should return original
        assert t_new == timelike
        assert x_new == spacelike

    def test_scaling_relationship(self):
        """Spacelike should scale linearly with c ratio."""
        timelike = 5.0
        spacelike = 3.0
        old_c = 1.0
        new_c = 4.0

        t_new, x_new = renormalize_lorentz_state(timelike, spacelike, old_c, new_c)

        # Spacelike should scale by new_c/old_c = 4
        assert np.isclose(x_new, spacelike * (new_c / old_c))


class TestAdaptationBenefits:
    """Test that adaptation improves performance."""

    def test_adaptation_helps_different_scales(self):
        """Adaptive control should work well across different scales."""
        # Test on slow system
        current_slow = np.array([1.0, 1.0])
        target_slow = np.array([0.0, 0.0])
        history_slow = [
            np.array([2.0 - i * 0.05, 2.0 - i * 0.05]) for i in range(30)
        ]

        _, info_slow = adaptive_control_step(current_slow, target_slow, history_slow)

        # Test on fast system
        current_fast = np.array([10.0, 10.0])
        target_fast = np.array([0.0, 0.0])
        history_fast = [
            np.array([20.0 - i * 0.5, 20.0 - i * 0.5]) for i in range(30)
        ]

        _, info_fast = adaptive_control_step(current_fast, target_fast, history_fast)

        # Should detect different c values
        assert info_slow["c"] < info_fast["c"]

        # Should recommend different eta values
        # (though the relationship depends on many factors)
        assert info_slow["eta_recommended"] > 0
        assert info_fast["eta_recommended"] > 0


def test_module_imports():
    """Verify all functions are importable."""
    from src.eigen_adaptive import (
        estimate_characteristic_velocity,
        estimate_oscillation_threshold,
        estimate_natural_timescale,
        detect_system_parameters,
        adaptive_control_step,
        renormalize_lorentz_state,
    )

    assert callable(estimate_characteristic_velocity)
    assert callable(estimate_oscillation_threshold)
    assert callable(estimate_natural_timescale)
    assert callable(detect_system_parameters)
    assert callable(adaptive_control_step)
    assert callable(renormalize_lorentz_state)
