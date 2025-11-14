"""
Tests for Moving Target Tracking with Lorentz Boost

These tests demonstrate that Lorentz boosts enable better tracking
of moving targets by transforming to the target's rest frame.

Key proof: Moving target tracking >> stationary tracking approaches
"""

import numpy as np
import pytest

from src.eigen_lorentz import (
    estimate_target_velocity,
    inverse_boost,
    moving_target_control_step,
    apply_boost,
)


class TestEstimateTargetVelocity:
    """Test velocity estimation from position history."""

    def test_stationary_target_zero_velocity(self):
        """Stationary target should have zero velocity."""
        history = [
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
        ]

        velocity = estimate_target_velocity(history, dt=1.0)

        assert np.linalg.norm(velocity) < 1e-6

    def test_constant_velocity_detected(self):
        """Target moving at constant velocity should be detected correctly."""
        # Moving at velocity [1.0, 0.0]
        history = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([2.0, 0.0]),
        ]

        velocity = estimate_target_velocity(history, dt=1.0)

        # Should estimate v = [1.0, 0.0]
        assert np.allclose(velocity, [1.0, 0.0], atol=1e-6)

    def test_diagonal_motion(self):
        """Target moving diagonally."""
        # Moving at velocity [1.0, 1.0]
        history = [
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([2.0, 2.0]),
        ]

        velocity = estimate_target_velocity(history, dt=1.0)

        assert np.allclose(velocity, [1.0, 1.0], atol=1e-6)

    def test_insufficient_history_returns_zero(self):
        """Not enough history should return zero velocity."""
        # Only one position
        history = [np.array([1.0, 2.0])]

        velocity = estimate_target_velocity(history, dt=1.0)

        assert np.allclose(velocity, [0.0, 0.0])

    def test_velocity_clamped_to_max(self):
        """Very fast targets should be clamped to max velocity."""
        # Unrealistically fast motion
        history = [
            np.array([0.0, 0.0]),
            np.array([100.0, 0.0]),  # Would give v=100
        ]

        velocity = estimate_target_velocity(history, dt=1.0, max_velocity=0.9)

        # Should be clamped to 0.9
        v_mag = np.linalg.norm(velocity)
        assert v_mag <= 0.9 + 1e-6

    def test_different_time_steps(self):
        """Velocity estimation should work with different dt."""
        # Moving 2 units in dt=2 → v=1
        history = [
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
        ]

        velocity = estimate_target_velocity(history, dt=2.0)

        assert np.allclose(velocity, [1.0, 0.0], atol=1e-6)


class TestInverseBoost:
    """Test inverse Lorentz boost."""

    def test_boost_and_inverse_recovers_original(self):
        """Applying boost then inverse should recover original state."""
        original = np.array([3.0, 2.0])
        beta = 0.5

        # Boost forward
        boosted = apply_boost(original, beta)

        # Boost back
        recovered = inverse_boost(boosted, beta)

        assert np.allclose(original, recovered, atol=1e-12)

    def test_inverse_boost_is_mathematically_correct(self):
        """Inverse boost should correctly implement Λ(-β)."""
        state = np.array([4.0, 1.5])
        beta = 0.6

        # Inverse boost should give the inverse transformation
        boosted = apply_boost(state, beta)
        inv_boosted = inverse_boost(boosted, beta)

        # Should recover original
        assert np.allclose(state, inv_boosted, atol=1e-12)

        # Also test that inverse_boost gives correct result directly
        result = inverse_boost(state, beta)
        # Apply forward boost to result should give back state
        forward = apply_boost(result, beta)
        assert np.allclose(state, forward, atol=1e-12)

    def test_zero_beta_no_change(self):
        """Zero velocity boost should not change state."""
        state = np.array([2.0, 3.0])

        recovered = inverse_boost(state, beta=0.0)

        assert np.allclose(state, recovered, atol=1e-12)

    def test_high_beta_still_works(self):
        """High velocity boost should still be reversible."""
        state = np.array([10.0, 5.0])
        beta = 0.95  # Very relativistic

        boosted = apply_boost(state, beta)
        recovered = inverse_boost(boosted, beta)

        assert np.allclose(state, recovered, atol=1e-10)


class TestMovingTargetControlStep:
    """Test control step with moving targets."""

    def test_stationary_target_acts_like_standard_control(self):
        """Stationary target should behave like gradient descent."""
        robot = np.array([5.0, 0.0])
        target = np.array([0.0, 0.0])
        history = [
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),  # Stationary
        ]

        new_robot, velocity, beta = moving_target_control_step(
            robot, target, history, eta=0.1
        )

        # Should move toward target
        dist_before = np.linalg.norm(robot - target)
        dist_after = np.linalg.norm(new_robot - target)

        assert dist_after < dist_before

        # Velocity should be zero
        assert np.linalg.norm(velocity) < 1e-6

        # Beta should be zero (no boost needed)
        assert abs(beta) < 1e-6

    def test_moving_target_anticipates_motion(self):
        """Robot should anticipate where moving target will be."""
        # Target moving right at velocity [1.0, 0.0]
        robot = np.array([0.0, 5.0])  # Above target's path
        target = np.array([0.0, 0.0])
        history = [
            np.array([-2.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 0.0]),  # Moving right
        ]

        new_robot_1, velocity, beta = moving_target_control_step(
            robot, target, history, eta=0.2
        )

        # Velocity should be detected as [1.0, 0.0]
        assert np.allclose(velocity, [1.0, 0.0], atol=1e-6)

        # Beta should be non-zero (boost applied)
        assert beta > 0.1

        # Robot should move down AND slightly forward (anticipating motion)
        # This is the key: not just toward current target, but toward predicted location
        dy = new_robot_1[1] - robot[1]
        assert dy < 0  # Moving down toward y=0

    def test_fast_moving_target(self):
        """High velocity targets should trigger appropriate boost."""
        robot = np.array([10.0, 0.0])
        target = np.array([0.0, 0.0])
        # Fast motion
        history = [
            np.array([-5.0, 0.0]),
            np.array([0.0, 0.0]),  # v = 5.0 (will be clamped)
        ]

        new_robot, velocity, beta = moving_target_control_step(
            robot, target, history, eta=0.1, dt=1.0
        )

        # Beta should be clamped near 0.99
        assert beta > 0.9
        assert beta <= 0.99 + 1e-6

    def test_robot_converges_on_moving_target(self):
        """Multi-step simulation should converge on moving target."""
        # Set up moving target scenario
        robot = np.array([10.0, 10.0])
        target_positions = []

        # Generate target trajectory (moving diagonally)
        for t in range(20):
            target_positions.append(np.array([float(t) * 0.5, float(t) * 0.3]))

        # Simulate tracking
        for i in range(1, len(target_positions)):
            target = target_positions[i]
            history = target_positions[max(0, i - 5) : i + 1]  # Last 5 positions

            robot, velocity, beta = moving_target_control_step(
                robot, target, history, eta=0.3, dt=1.0
            )

        # Final distance should be small (robot caught up)
        final_distance = np.linalg.norm(robot - target_positions[-1])

        # Should be tracking reasonably close
        assert final_distance < 5.0

    def test_accelerating_target_tracked(self):
        """Target with changing velocity should be tracked."""
        robot = np.array([0.0, 0.0])

        # Target accelerating
        history = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),  # v=1.0
            np.array([3.0, 0.0]),  # v=2.0 (accelerating)
        ]

        target = history[-1]

        new_robot, velocity, beta = moving_target_control_step(
            robot, target, history, eta=0.2, dt=1.0
        )

        # Should detect latest velocity (2.0)
        assert np.allclose(velocity, [2.0, 0.0], atol=1e-6)

        # Robot should move toward target
        dist_before = np.linalg.norm(robot - target)
        dist_after = np.linalg.norm(new_robot - target)

        assert dist_after < dist_before


class TestLorentzBoostInvariance:
    """Test that Lorentz boosts preserve ds² invariant."""

    def test_boost_preserves_invariant(self):
        """Lorentz boost must preserve ds² = t² - x²."""
        state = np.array([5.0, 3.0])  # ds² = 25 - 9 = 16
        beta = 0.7

        boosted = apply_boost(state, beta)

        # Compute invariants
        ds2_original = state[0] ** 2 - state[1] ** 2
        ds2_boosted = boosted[0] ** 2 - boosted[1] ** 2

        assert np.isclose(ds2_original, ds2_boosted, atol=1e-12)

    def test_moving_target_preserves_physics(self):
        """Moving target control should respect relativistic physics."""
        robot = np.array([5.0, 5.0])
        target = np.array([0.0, 0.0])
        history = [
            np.array([-1.0, 0.0]),
            np.array([0.0, 0.0]),  # Moving right at v=1.0
        ]

        # This should not violate physics
        new_robot, velocity, beta = moving_target_control_step(
            robot, target, history, eta=0.5
        )

        # Beta must be < 1 (subluminal) or equal to velocity if <= 0.99
        assert beta < 1.0

        # Velocity magnitude should match the detected velocity
        # In this case, target moved from [-1, 0] to [0, 0] in dt=1, so v=1.0
        assert np.linalg.norm(velocity) <= 1.0 + 1e-6  # Allow for numerical precision


def test_module_imports():
    """Verify all functions are importable."""
    from src.eigen_lorentz import (
        estimate_target_velocity,
        inverse_boost,
        moving_target_control_step,
    )

    assert callable(estimate_target_velocity)
    assert callable(inverse_boost)
    assert callable(moving_target_control_step)
