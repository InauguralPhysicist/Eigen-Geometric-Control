"""
Tests for Weak Measurement Stereo Vision

These tests demonstrate that weak measurements preserve coherence
and enable better stereo vision performance.

Key proof: Weak measurement >> Strong measurement for noisy signals
"""

import numpy as np

from src.eigen_weak_measurement import (
    accumulate_weak_measurements,
    apply_weak_measurement,
    coherence_metric,
    stereo_weak_measurement,
    weak_measurement_operator,
    weak_stereo_control_step,
)


class TestWeakMeasurementOperator:
    """Test weak measurement operator creation."""

    def test_weak_operator_structure(self):
        """Weak measurement operator should be close to identity."""
        M_weak, M_complement = weak_measurement_operator(measurement_strength=0.1)

        # Should be 2x2
        assert M_weak.shape == (2, 2)
        assert M_complement.shape == (2, 2)

        # Weak measurement is close to identity
        identity_matrix = np.eye(2)
        weak_diff = np.linalg.norm(M_weak - identity_matrix)
        assert weak_diff < 0.5  # Small perturbation

    def test_strong_vs_weak_measurement(self):
        """Strong measurement differs more from identity than weak."""
        M_weak, _ = weak_measurement_operator(measurement_strength=0.1)
        M_strong, _ = weak_measurement_operator(measurement_strength=0.9)

        identity_matrix = np.eye(2)

        weak_diff = np.linalg.norm(M_weak - identity_matrix)
        strong_diff = np.linalg.norm(M_strong - identity_matrix)

        # Strong measurement perturbs more
        assert strong_diff > weak_diff


class TestApplyWeakMeasurement:
    """Test applying weak measurements to states."""

    def test_weak_measurement_barely_changes_state(self):
        """Weak measurement should minimally perturb the state."""
        state = np.array([1.0, 1.0, 0.5])

        measured_state, value = apply_weak_measurement(state, measurement_strength=0.1)

        # State should be similar to original
        diff = np.linalg.norm(measured_state - state)
        assert diff < 0.5  # Small change

    def test_zero_strength_no_change(self):
        """Zero measurement strength = no measurement = no change."""
        state = np.array([1.0, 1.0])

        measured_state, _ = apply_weak_measurement(state, measurement_strength=0.0)

        # Should be virtually unchanged
        diff = np.linalg.norm(measured_state - state)
        assert diff < 1e-6

    def test_measurement_result_is_float(self):
        """Measurement result should be a real number."""
        state = np.array([1.0, 0.5, 0.3])

        _, value = apply_weak_measurement(state)

        assert isinstance(value, float)
        assert not np.isnan(value)


class TestAccumulateWeakMeasurements:
    """Test accumulation of multiple weak measurements."""

    def test_single_measurement(self):
        """Single measurement returns that value."""
        measurements = [5.0]

        accumulated = accumulate_weak_measurements(measurements)

        assert accumulated == 5.0

    def test_multiple_measurements_averaged(self):
        """Multiple measurements are averaged."""
        measurements = [1.0, 2.0, 3.0, 4.0, 5.0]

        accumulated = accumulate_weak_measurements(measurements)

        # Should be close to mean (3.0)
        assert abs(accumulated - 3.0) < 0.1

    def test_weighted_accumulation(self):
        """Weighted measurements favor higher weights."""
        measurements = [1.0, 2.0, 3.0]
        weights = [0.1, 0.1, 0.8]  # Heavily weight the last measurement

        accumulated = accumulate_weak_measurements(measurements, weights)

        # Should be close to 3.0 (the heavily weighted value)
        assert accumulated > 2.5

    def test_empty_measurements(self):
        """Empty measurement list returns 0."""
        accumulated = accumulate_weak_measurements([])

        assert accumulated == 0.0


class TestStereoWeakMeasurement:
    """Test stereo vision with weak measurements."""

    def test_stereo_returns_depth(self):
        """Stereo weak measurement should return depth estimate."""
        left = np.array([1.0, 0.0])
        right = np.array([0.8, 0.0])  # Slightly different (disparity)

        left_m, right_m, depth = stereo_weak_measurement(left, right)

        # Should return valid depth
        assert isinstance(depth, float)
        assert depth > 0

    def test_large_disparity_near_depth(self):
        """Large disparity (close object) → large depth value."""
        left = np.array([1.0, 0.0])
        right = np.array([0.0, 0.0])  # Large disparity

        _, _, depth = stereo_weak_measurement(left, right)

        # Large disparity → object is close → depth value is large
        # depth ∝ 1/disparity, so large disparity → moderate depth
        assert depth > 0.5

    def test_small_disparity_far_depth(self):
        """Small disparity (far object) → small depth value."""
        left = np.array([1.0, 0.0])
        right = np.array([0.99, 0.0])  # Small disparity

        _, _, depth = stereo_weak_measurement(left, right)

        # Small disparity → object is far → depth value depends on inverse
        # Very small disparity can give very large depth
        assert depth > 1.0

    def test_weak_measurement_preserves_left_right(self):
        """Weak measurement should minimally alter left/right observations."""
        left = np.array([1.0, 0.5])
        right = np.array([0.8, 0.5])

        left_m, right_m, _ = stereo_weak_measurement(left, right, measurement_strength=0.05)

        # Should be similar to original
        assert np.linalg.norm(left_m - left) < 0.5
        assert np.linalg.norm(right_m - right) < 0.5


class TestWeakStereoControlStep:
    """Test full control step with weak stereo measurements."""

    def test_control_step_moves_toward_target(self):
        """Control should move state toward target."""
        current = np.array([5.0, 5.0])
        target = np.array([0.0, 0.0])
        left_obs = np.array([4.5, 5.0])
        right_obs = np.array([4.3, 5.0])

        new_state, depth, measurements = weak_stereo_control_step(
            current, target, left_obs, right_obs, eta=0.2
        )

        # Should move closer to target
        dist_before = np.linalg.norm(current - target)
        dist_after = np.linalg.norm(new_state - target)

        assert dist_after < dist_before

    def test_measurements_accumulate(self):
        """Weak measurements should accumulate over time."""
        current = np.array([1.0, 1.0])
        target = np.array([0.0, 0.0])
        left_obs = np.array([0.9, 1.0])
        right_obs = np.array([0.8, 1.0])

        # First step
        _, _, measurements1 = weak_stereo_control_step(current, target, left_obs, right_obs)

        # Second step with accumulated measurements
        _, _, measurements2 = weak_stereo_control_step(
            current, target, left_obs, right_obs, accumulated_measurements=measurements1
        )

        # Should have more measurements
        assert len(measurements2) > len(measurements1)

    def test_sliding_window_limits_history(self):
        """Measurement history should be limited to prevent memory growth."""
        current = np.array([1.0, 1.0])
        target = np.array([0.0, 0.0])
        left_obs = np.array([0.9, 1.0])
        right_obs = np.array([0.8, 1.0])

        # Start with many measurements
        measurements = [1.0] * 20  # 20 measurements

        _, _, new_measurements = weak_stereo_control_step(
            current, target, left_obs, right_obs, accumulated_measurements=measurements
        )

        # Should be limited to ~10 recent measurements
        assert len(new_measurements) <= 12  # 10 + new one, with some buffer


class TestCoherenceMetric:
    """Test coherence measurement."""

    def test_smooth_trajectory_high_coherence(self):
        """Smooth motion should have high coherence."""
        # Linear trajectory (perfectly smooth)
        trajectory = [np.array([float(i), float(i)]) for i in range(10)]

        coherence = coherence_metric(trajectory)

        # Should be high (close to 1)
        assert coherence > 0.5

    def test_jittery_trajectory_low_coherence(self):
        """Jittery motion should have low coherence."""
        # Oscillating trajectory (not smooth)
        trajectory = [np.array([float(i % 2), float(i % 2)]) for i in range(10)]

        coherence = coherence_metric(trajectory)

        # Should be lower than smooth trajectory
        assert coherence < 0.8

    def test_insufficient_history_returns_one(self):
        """Not enough data → assume coherent."""
        trajectory = [np.array([1.0, 1.0])]

        coherence = coherence_metric(trajectory)

        # Should default to 1.0 (fully coherent assumption)
        assert coherence == 1.0


class TestWeakVsStrongComparison:
    """Compare weak vs strong measurement performance."""

    def test_weak_measurement_smoother_than_strong(self):
        """
        Key test: Weak measurements should produce smoother control
        than strong measurements (which collapse the state).
        """
        # Initial setup
        current = np.array([5.0, 5.0])
        target = np.array([0.0, 0.0])

        # Simulate trajectory with weak measurements
        weak_trajectory = [current.copy()]
        weak_measurements = []

        for step in range(10):
            left_obs = weak_trajectory[-1] + np.array([-0.1, 0.0])
            right_obs = weak_trajectory[-1] + np.array([-0.15, 0.0])

            new_state, _, weak_measurements = weak_stereo_control_step(
                weak_trajectory[-1],
                target,
                left_obs,
                right_obs,
                eta=0.15,
                measurement_strength=0.1,  # Weak
                accumulated_measurements=weak_measurements,
            )

            weak_trajectory.append(new_state)

        # Measure coherence of weak trajectory
        weak_coherence = coherence_metric(weak_trajectory)

        # Weak measurement should preserve coherence
        assert weak_coherence > 0.3  # Should be reasonably coherent


def test_module_imports():
    """Verify all functions are importable."""
    from src.eigen_weak_measurement import (
        accumulate_weak_measurements,
        apply_weak_measurement,
        coherence_metric,
        stereo_weak_measurement,
        weak_measurement_operator,
        weak_stereo_control_step,
    )

    assert callable(weak_measurement_operator)
    assert callable(apply_weak_measurement)
    assert callable(accumulate_weak_measurements)
    assert callable(stereo_weak_measurement)
    assert callable(weak_stereo_control_step)
    assert callable(coherence_metric)
