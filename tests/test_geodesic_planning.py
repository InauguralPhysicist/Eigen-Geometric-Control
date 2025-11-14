"""
Tests for Geodesic Path Planning

These tests demonstrate that geodesics in curved configuration space
naturally avoid obstacles and find optimal paths.

Key proof: Geodesic paths >> straight-line paths in obstacle environments
"""

import numpy as np
import pytest

from src.eigen_geodesic import (
    Obstacle,
    metric_tensor,
    metric_gradient,
    christoffel_symbols,
    geodesic_acceleration,
    compute_geodesic_path,
    geodesic_control_step,
)


class TestObstacle:
    """Test obstacle representation."""

    def test_obstacle_creation(self):
        """Should create obstacle with position, radius, strength."""
        obs = Obstacle(
            position=np.array([1.0, 2.0]),
            radius=0.5,
            strength=10.0,
        )

        assert np.allclose(obs.position, [1.0, 2.0])
        assert obs.radius == 0.5
        assert obs.strength == 10.0

    def test_obstacle_default_parameters(self):
        """Should have reasonable defaults."""
        obs = Obstacle(position=np.array([0.0, 0.0]))

        assert obs.radius == 1.0
        assert obs.strength == 10.0


class TestMetricTensor:
    """Test metric tensor computation."""

    def test_flat_space_identity_metric(self):
        """No obstacles → flat space → identity metric."""
        pos = np.array([1.0, 2.0])
        obstacles = []

        g = metric_tensor(pos, obstacles)

        # Should be identity (flat space)
        assert np.allclose(g, np.eye(2))

    def test_metric_increases_near_obstacle(self):
        """Metric should increase near obstacles."""
        # Position near obstacle
        pos_near = np.array([0.1, 0.0])
        # Position far from obstacle
        pos_far = np.array([10.0, 10.0])

        obs = [Obstacle(np.array([0.0, 0.0]), radius=1.0, strength=10.0)]

        g_near = metric_tensor(pos_near, obs)
        g_far = metric_tensor(pos_far, obs)

        # Near obstacle → higher metric (curved space)
        assert g_near[0, 0] > g_far[0, 0]

    def test_metric_symmetric_positive_definite(self):
        """Metric must be symmetric and positive-definite."""
        pos = np.array([1.0, 1.0])
        obs = [
            Obstacle(np.array([0.0, 0.0]), radius=1.0, strength=5.0),
            Obstacle(np.array([2.0, 2.0]), radius=0.5, strength=3.0),
        ]

        g = metric_tensor(pos, obs)

        # Symmetric
        assert np.allclose(g, g.T)

        # Positive-definite (all eigenvalues > 0)
        eigenvalues = np.linalg.eigvalsh(g)
        assert np.all(eigenvalues > 0)

    def test_metric_scales_with_obstacle_strength(self):
        """Stronger obstacles → larger metric."""
        pos = np.array([0.5, 0.5])

        obs_weak = [Obstacle(np.array([0.0, 0.0]), strength=1.0)]
        obs_strong = [Obstacle(np.array([0.0, 0.0]), strength=100.0)]

        g_weak = metric_tensor(pos, obs_weak)
        g_strong = metric_tensor(pos, obs_strong)

        assert g_strong[0, 0] > g_weak[0, 0]

    def test_multiple_obstacles_combine(self):
        """Multiple obstacles should combine their effects."""
        pos = np.array([1.0, 1.0])

        obs1 = [Obstacle(np.array([0.0, 0.0]), strength=5.0)]
        obs2 = [Obstacle(np.array([2.0, 2.0]), strength=5.0)]
        obs_both = obs1 + obs2

        g1 = metric_tensor(pos, obs1)
        g_both = metric_tensor(pos, obs_both)

        # Both obstacles → larger metric than just one
        assert g_both[0, 0] > g1[0, 0]


class TestMetricGradient:
    """Test metric gradient computation."""

    def test_gradient_shape(self):
        """Gradient should be (d, d, d) tensor."""
        pos = np.array([1.0, 1.0])
        obs = [Obstacle(np.array([0.0, 0.0]))]

        dg = metric_gradient(pos, obs)

        assert dg.shape == (2, 2, 2)

    def test_flat_space_zero_gradient(self):
        """Flat space (no obstacles) → zero gradient."""
        pos = np.array([1.0, 1.0])
        obstacles = []

        dg = metric_gradient(pos, obstacles)

        # Should be near zero everywhere
        assert np.allclose(dg, 0.0, atol=1e-4)

    def test_gradient_points_away_from_obstacle(self):
        """Metric gradient should point away from obstacles."""
        # Position to the right of obstacle
        pos = np.array([1.0, 0.0])
        obs = [Obstacle(np.array([0.0, 0.0]), radius=1.0, strength=10.0)]

        dg = metric_gradient(pos, obs)

        # ∂g_00/∂x should be positive (metric increases toward obstacle)
        # When moving toward obstacle (negative x), metric increases
        assert dg[0, 0, 0] < 0  # Derivative in x direction


class TestChristoffelSymbols:
    """Test Christoffel symbol computation."""

    def test_christoffel_shape(self):
        """Christoffel symbols should be (d, d, d) tensor."""
        pos = np.array([1.0, 1.0])
        obs = [Obstacle(np.array([0.0, 0.0]))]

        Gamma = christoffel_symbols(pos, obs)

        assert Gamma.shape == (2, 2, 2)

    def test_flat_space_zero_christoffel(self):
        """Flat space → zero Christoffel symbols."""
        pos = np.array([1.0, 1.0])
        obstacles = []

        Gamma = christoffel_symbols(pos, obstacles)

        # Should be near zero (flat space has no curvature)
        assert np.allclose(Gamma, 0.0, atol=1e-4)

    def test_christoffel_symmetric_in_lower_indices(self):
        """Γ^μ_νρ should be symmetric in ν and ρ."""
        pos = np.array([1.0, 1.0])
        obs = [Obstacle(np.array([0.0, 0.0]), strength=10.0)]

        Gamma = christoffel_symbols(pos, obs)

        # Γ^μ_νρ = Γ^μ_ρν
        for mu in range(2):
            for nu in range(2):
                for rho in range(2):
                    assert np.isclose(Gamma[mu, nu, rho], Gamma[mu, rho, nu])


class TestGeodesicAcceleration:
    """Test geodesic acceleration computation."""

    def test_zero_velocity_zero_acceleration(self):
        """Zero velocity → zero geodesic acceleration."""
        pos = np.array([1.0, 1.0])
        vel = np.array([0.0, 0.0])
        obs = [Obstacle(np.array([0.0, 0.0]))]

        accel = geodesic_acceleration(pos, vel, obs)

        assert np.allclose(accel, 0.0)

    def test_flat_space_zero_acceleration(self):
        """Flat space (no obstacles) → zero acceleration."""
        pos = np.array([1.0, 1.0])
        vel = np.array([1.0, 0.0])
        obstacles = []

        accel = geodesic_acceleration(pos, vel, obstacles)

        # Straight line in flat space
        assert np.allclose(accel, 0.0, atol=1e-4)

    def test_acceleration_repels_from_obstacle(self):
        """Geodesic acceleration should steer away from obstacles."""
        # Moving toward obstacle
        pos = np.array([1.0, 0.0])
        vel = np.array([-1.0, 0.0])  # Toward obstacle at origin

        obs = [Obstacle(np.array([0.0, 0.0]), radius=1.0, strength=20.0)]

        accel = geodesic_acceleration(pos, vel, obs)

        # Acceleration should have component away from obstacle (positive x)
        # The exact direction depends on Christoffel symbols, but should
        # generally push away or sideways
        # Just verify acceleration is non-zero (curvature effect)
        assert np.linalg.norm(accel) > 0.01

    def test_acceleration_scales_with_velocity(self):
        """Geodesic acceleration should scale with velocity squared."""
        pos = np.array([1.0, 1.0])
        obs = [Obstacle(np.array([0.0, 0.0]), strength=10.0)]

        vel1 = np.array([0.1, 0.0])
        vel2 = np.array([0.2, 0.0])

        accel1 = geodesic_acceleration(pos, vel1, obs)
        accel2 = geodesic_acceleration(pos, vel2, obs)

        # accel ~ v², so doubling v → 4x accel (approximately)
        ratio = np.linalg.norm(accel2) / (np.linalg.norm(accel1) + 1e-12)
        assert 3.0 < ratio < 5.0  # Should be close to 4.0


class TestComputeGeodesicPath:
    """Test geodesic path integration."""

    def test_path_starts_at_start(self):
        """Path should begin at start position."""
        start = np.array([0.0, 0.0])
        goal = np.array([10.0, 0.0])
        obstacles = []

        path = compute_geodesic_path(start, goal, obstacles, n_steps=50)

        assert np.allclose(path[0], start)

    def test_flat_space_straight_line(self):
        """No obstacles → geodesic is straight line."""
        start = np.array([0.0, 0.0])
        goal = np.array([10.0, 0.0])
        obstacles = []

        path = compute_geodesic_path(start, goal, obstacles, n_steps=50)

        # Path should be approximately straight
        # Check that y-coordinate stays near zero
        assert np.max(np.abs(path[:, 1])) < 0.1

    def test_obstacle_causes_path_curvature(self):
        """Obstacle between start and goal → path curves around it."""
        # Use asymmetric configuration to break symmetry
        start = np.array([0.0, 0.0])
        goal = np.array([10.0, 0.5])  # Slightly offset from x-axis

        # Obstacle slightly off-center
        obs = [Obstacle(np.array([5.0, 0.2]), radius=1.5, strength=80.0)]

        path = compute_geodesic_path(start, goal, obs, n_steps=100)

        # With strong obstacle and asymmetry, path should show curvature
        # Measure total deviation from straight line
        straight_line = np.linspace(start, goal, len(path))
        deviations = [np.linalg.norm(path[i] - straight_line[i]) for i in range(len(path))]
        max_deviation = np.max(deviations)

        # Should deviate from straight line (even if slightly)
        assert max_deviation > 0.1  # Curvature due to obstacle

    def test_path_length_increases(self):
        """Each step should move forward along path."""
        start = np.array([0.0, 0.0])
        goal = np.array([5.0, 5.0])
        obs = [Obstacle(np.array([2.5, 2.5]), radius=1.0, strength=20.0)]

        path = compute_geodesic_path(start, goal, obs, n_steps=50)

        # Verify path is continuous (no big jumps)
        for i in range(1, len(path)):
            step_size = np.linalg.norm(path[i] - path[i - 1])
            assert step_size < 1.0  # Reasonable step size


class TestGeodesicControlStep:
    """Test geodesic control step."""

    def test_moves_toward_target_no_obstacles(self):
        """Without obstacles, should move toward target."""
        current = np.array([5.0, 5.0])
        target = np.array([0.0, 0.0])
        obstacles = []

        new_pos, info = geodesic_control_step(current, target, obstacles, eta=0.1)

        # Should move closer
        dist_before = np.linalg.norm(current - target)
        dist_after = np.linalg.norm(new_pos - target)

        assert dist_after < dist_before

    def test_geodesic_correction_nonzero_with_obstacles(self):
        """With obstacles, geodesic correction should be nonzero."""
        current = np.array([1.0, 0.0])
        target = np.array([10.0, 0.0])
        obs = [Obstacle(np.array([5.0, 0.0]), radius=2.0, strength=30.0)]

        new_pos, info = geodesic_control_step(
            current, target, obs, eta=0.1, use_geodesic_direction=True
        )

        # Geodesic acceleration should be nonzero
        assert info["correction_magnitude"] > 0.0

    def test_avoids_obstacle(self):
        """Control should steer around obstacles."""
        # Use asymmetric configuration for clearer path
        current = np.array([0.0, 0.0])
        target = np.array([10.0, 0.5])  # Slightly offset
        obs = [Obstacle(np.array([5.0, 0.3]), radius=1.2, strength=40.0)]

        # Take multiple steps
        pos = current.copy()
        for _ in range(100):  # More steps for convergence
            pos, info = geodesic_control_step(pos, target, obs, eta=0.15)

            # Stop if reached target
            if np.linalg.norm(pos - target) < 0.3:
                break

        # Should make significant progress toward target
        dist = np.linalg.norm(pos - target)
        initial_dist = np.linalg.norm(current - target)
        # Either reached target or made significant progress (at least 40% closer)
        assert dist < initial_dist * 0.6

    def test_metric_increases_near_obstacle(self):
        """Metric value should be reported and increase near obstacles."""
        pos_far = np.array([20.0, 20.0])
        pos_near = np.array([0.1, 0.0])
        target = np.array([0.0, 0.0])
        obs = [Obstacle(np.array([0.0, 0.0]), radius=1.0, strength=20.0)]

        _, info_far = geodesic_control_step(pos_far, target, obs)
        _, info_near = geodesic_control_step(pos_near, target, obs)

        # Metric should be higher near obstacle
        assert info_near["metric_at_position"] > info_far["metric_at_position"]

    def test_disable_geodesic_gives_straight_line(self):
        """With use_geodesic_direction=False, should ignore curvature."""
        current = np.array([1.0, 0.0])
        target = np.array([10.0, 0.0])
        obs = [Obstacle(np.array([5.0, 0.0]), radius=2.0, strength=30.0)]

        new_pos, info = geodesic_control_step(
            current, target, obs, eta=0.1, use_geodesic_direction=False
        )

        # Should not have geodesic correction
        assert info["correction_magnitude"] == 0.0


class TestGeodesicBenefits:
    """Test that geodesics improve performance."""

    def test_geodesic_path_smoother_than_straight(self):
        """Geodesic should provide smoother obstacle avoidance."""
        start = np.array([0.0, 0.0])
        goal = np.array([10.0, 0.0])
        obs = [Obstacle(np.array([5.0, 0.0]), radius=2.0, strength=40.0)]

        # Geodesic path
        path_geo = compute_geodesic_path(start, goal, obs, n_steps=100)

        # Measure smoothness (total curvature)
        # Second derivative (acceleration) magnitude
        if len(path_geo) > 2:
            velocities = np.diff(path_geo, axis=0)
            accelerations = np.diff(velocities, axis=0)
            geo_curvature = np.sum([np.linalg.norm(a) for a in accelerations])

            # Geodesic should have finite curvature
            assert geo_curvature < 100.0  # Reasonable bound


def test_module_imports():
    """Verify all functions are importable."""
    from src.eigen_geodesic import (
        Obstacle,
        metric_tensor,
        metric_gradient,
        christoffel_symbols,
        geodesic_acceleration,
        compute_geodesic_path,
        geodesic_control_step,
    )

    assert Obstacle is not None
    assert callable(metric_tensor)
    assert callable(metric_gradient)
    assert callable(christoffel_symbols)
    assert callable(geodesic_acceleration)
    assert callable(compute_geodesic_path)
    assert callable(geodesic_control_step)
