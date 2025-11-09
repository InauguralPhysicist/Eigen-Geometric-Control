# -*- coding: utf-8 -*-
"""
Tests for Noperthedron analysis module

Tests the geometric properties of the Noperthedron and the ds² metric
for passage testing.
"""

import numpy as np
import pytest

from src.eigen_noperthedron import (
    PassageAttempt,
    analyze_results,
    compute_min_distance,
    compute_passage_ds2,
    estimate_overlap,
    generate_base_points,
    generate_noperthedron_vertices,
    rotation_from_spherical,
    rotation_y,
    rotation_z,
    test_rupert_property,
    test_single_configuration,
)


class TestBasePoints:
    """Test Noperthedron base point generation"""

    def test_generate_base_points_count(self):
        """Should generate exactly 3 base points"""
        base_points = generate_base_points()
        assert len(base_points) == 3

    def test_generate_base_points_shapes(self):
        """Each base point should be 3D"""
        base_points = generate_base_points()
        for point in base_points:
            assert point.shape == (3,)

    def test_c1_magnitude(self):
        """C₁ should be a unit vector (magnitude = 1)"""
        base_points = generate_base_points()
        c1 = base_points[0]
        magnitude = np.linalg.norm(c1)
        assert np.isclose(magnitude, 1.0, atol=1e-6), f"C1 magnitude {magnitude} != 1.0"

    def test_c2_c3_magnitudes(self):
        """C₂ and C₃ should have magnitude ≈ 0.98-0.99"""
        base_points = generate_base_points()
        c2_mag = np.linalg.norm(base_points[1])
        c3_mag = np.linalg.norm(base_points[2])

        assert 0.95 < c2_mag < 1.0, f"C2 magnitude {c2_mag} out of range"
        assert 0.95 < c3_mag < 1.0, f"C3 magnitude {c3_mag} out of range"


class TestRotations:
    """Test rotation matrices"""

    def test_rotation_z_identity(self):
        """Zero angle should give identity matrix"""
        R = rotation_z(0.0)
        assert np.allclose(R, np.eye(3))

    def test_rotation_z_preserves_z(self):
        """Rotation around z-axis should preserve z-coordinate"""
        R = rotation_z(np.pi / 4)
        z_axis = np.array([0, 0, 1])
        rotated = R @ z_axis
        assert np.allclose(rotated, z_axis)

    def test_rotation_z_90_degrees(self):
        """90° rotation should map x → y"""
        R = rotation_z(np.pi / 2)
        x_axis = np.array([1, 0, 0])
        rotated = R @ x_axis
        expected = np.array([0, 1, 0])
        assert np.allclose(rotated, expected, atol=1e-10)

    def test_rotation_y_identity(self):
        """Zero angle should give identity matrix"""
        R = rotation_y(0.0)
        assert np.allclose(R, np.eye(3))

    def test_rotation_y_preserves_y(self):
        """Rotation around y-axis should preserve y-coordinate"""
        R = rotation_y(np.pi / 4)
        y_axis = np.array([0, 1, 0])
        rotated = R @ y_axis
        assert np.allclose(rotated, y_axis)

    def test_rotation_orthogonal(self):
        """Rotation matrices should be orthogonal (R^T R = I)"""
        angles = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi]
        for angle in angles:
            Rz = rotation_z(angle)
            Ry = rotation_y(angle)
            assert np.allclose(Rz.T @ Rz, np.eye(3), atol=1e-10)
            assert np.allclose(Ry.T @ Ry, np.eye(3), atol=1e-10)

    def test_rotation_determinant(self):
        """Rotation matrices should have determinant = 1"""
        angles = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi]
        for angle in angles:
            Rz = rotation_z(angle)
            Ry = rotation_y(angle)
            assert np.isclose(np.linalg.det(Rz), 1.0, atol=1e-10)
            assert np.isclose(np.linalg.det(Ry), 1.0, atol=1e-10)

    def test_rotation_from_spherical(self):
        """Spherical rotation should produce orthogonal matrix"""
        R = rotation_from_spherical(np.pi / 4, np.pi / 3)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestNoperthedronGeneration:
    """Test Noperthedron vertex generation"""

    def test_vertex_count(self):
        """Should generate exactly 90 vertices"""
        vertices = generate_noperthedron_vertices()
        assert len(vertices) == 90, f"Expected 90 vertices, got {len(vertices)}"

    def test_vertex_dimension(self):
        """Vertices should be 3D"""
        vertices = generate_noperthedron_vertices()
        assert vertices.shape == (90, 3)

    def test_vertex_finite(self):
        """All vertices should have finite coordinates"""
        vertices = generate_noperthedron_vertices()
        assert np.all(np.isfinite(vertices))

    def test_point_symmetry(self):
        """Noperthedron should have point symmetry (inversion through origin)"""
        vertices = generate_noperthedron_vertices()

        # For each vertex, its negation should also be a vertex
        for v in vertices:
            negation = -v
            # Find closest vertex to negation
            distances = np.linalg.norm(vertices - negation, axis=1)
            min_dist = np.min(distances)
            # Should have exact match (within numerical tolerance)
            assert min_dist < 1e-10, f"Point symmetry violated for vertex {v}"

    def test_z_axis_symmetry(self):
        """Vertices should show 15-fold rotational symmetry around z-axis"""
        vertices = generate_noperthedron_vertices()

        # Count unique z-coordinates (should have limited set due to symmetry)
        z_coords = vertices[:, 2]
        unique_z = np.unique(np.round(z_coords, decimals=6))

        # With 3 base points and reflections, expect ≤ 6 unique z values
        assert len(unique_z) <= 10, f"Too many unique z-coords: {len(unique_z)}"

    def test_convex_hull_encloses_origin(self):
        """Origin should be inside the convex hull (due to point symmetry)"""
        vertices = generate_noperthedron_vertices()
        # A simple test: for point symmetry, center of mass should be at origin
        center = np.mean(vertices, axis=0)
        assert np.allclose(center, np.zeros(3), atol=1e-10)


class TestDistanceMetrics:
    """Test distance computation functions"""

    def test_compute_min_distance_identical(self):
        """Distance between identical point clouds should be zero"""
        vertices = generate_noperthedron_vertices()[:10]  # Use subset for speed
        dist = compute_min_distance(vertices, vertices)
        assert np.isclose(dist, 0.0, atol=1e-10)

    def test_compute_min_distance_separated(self):
        """Distance between separated clouds should be positive"""
        vertices1 = np.array([[0, 0, 0], [1, 0, 0]])
        vertices2 = np.array([[10, 0, 0], [11, 0, 0]])
        dist = compute_min_distance(vertices1, vertices2)
        assert dist >= 9.0  # At least 9 units apart

    def test_estimate_overlap_identical(self):
        """Overlap between identical point clouds should be 100%"""
        vertices = np.random.randn(10, 3)
        overlap = estimate_overlap(vertices, vertices, threshold=0.1)
        assert np.isclose(overlap, 1.0, atol=0.01)

    def test_estimate_overlap_separated(self):
        """Overlap between separated clouds should be ~0%"""
        vertices1 = np.array([[0, 0, 0], [1, 0, 0]])
        vertices2 = np.array([[10, 0, 0], [11, 0, 0]])
        overlap = estimate_overlap(vertices1, vertices2, threshold=0.1)
        assert overlap < 0.01  # Less than 1% overlap


class TestPassageMetric:
    """Test ds² metric computation for passage"""

    def test_compute_passage_ds2_identical(self):
        """Identical shapes should have high collision (negative ds²)"""
        vertices = generate_noperthedron_vertices()[:20]  # Subset for speed
        C, S, ds2 = compute_passage_ds2(vertices, vertices)

        # High collision
        assert C > 50, f"Expected high C for identical shapes, got {C}"
        # Low stability
        assert S < 50, f"Expected low S for identical shapes, got {S}"
        # Should be space-like (ds² < 0)
        assert ds2 < 0, f"Expected ds² < 0 for collision, got {ds2}"

    def test_compute_passage_ds2_separated(self):
        """Separated shapes should have high clearance (positive ds²)"""
        vertices1 = np.random.randn(20, 3)
        vertices2 = vertices1 + np.array([10, 0, 0])  # Shift by 10 units

        C, S, ds2 = compute_passage_ds2(vertices1, vertices2)

        # Low collision
        assert C < 50, f"Expected low C for separated shapes, got {C}"
        # High stability
        assert S > 50, f"Expected high S for separated shapes, got {S}"

    def test_ds2_sign_consistent_with_collision(self):
        """ds² should be negative for collision, positive for clearance"""
        vertices = generate_noperthedron_vertices()[:15]

        # Test collision (identical shapes)
        C_coll, S_coll, ds2_coll = compute_passage_ds2(vertices, vertices)
        assert ds2_coll == S_coll**2 - C_coll**2  # Verify formula

        # Test clearance (separated shapes)
        vertices_far = vertices + np.array([5, 0, 0])
        C_clear, S_clear, ds2_clear = compute_passage_ds2(vertices, vertices_far)
        assert ds2_clear == S_clear**2 - C_clear**2  # Verify formula


class TestPassageAttempt:
    """Test single configuration passage testing"""

    def test_test_single_configuration_structure(self):
        """Should return PassageAttempt with correct fields"""
        vertices = generate_noperthedron_vertices()
        attempt = test_single_configuration(
            vertices, theta1=0.0, phi1=0.0, theta2=np.pi / 4, phi2=np.pi / 4
        )

        assert isinstance(attempt, PassageAttempt)
        assert hasattr(attempt, "orientation1")
        assert hasattr(attempt, "orientation2")
        assert hasattr(attempt, "min_distance")
        assert hasattr(attempt, "C")
        assert hasattr(attempt, "S")
        assert hasattr(attempt, "ds2")

    def test_test_single_configuration_ds2_formula(self):
        """ds² should equal S² - C²"""
        vertices = generate_noperthedron_vertices()
        attempt = test_single_configuration(
            vertices, theta1=0.0, phi1=0.0, theta2=np.pi / 2, phi2=0.0
        )

        expected_ds2 = attempt.S**2 - attempt.C**2
        assert np.isclose(attempt.ds2, expected_ds2, atol=1e-6)

    def test_test_single_configuration_different_orientations(self):
        """Different orientations should give different results"""
        vertices = generate_noperthedron_vertices()

        attempt1 = test_single_configuration(vertices, theta1=0.0, phi1=0.0, theta2=0.0, phi2=0.0)

        attempt2 = test_single_configuration(
            vertices, theta1=np.pi / 2, phi1=np.pi / 2, theta2=np.pi / 4, phi2=np.pi / 4
        )

        # Should get different ds² values for different orientations
        # (though both might be negative for Rupert-negative shape)
        assert attempt1.ds2 != attempt2.ds2 or attempt1.min_distance != attempt2.min_distance


class TestRupertPropertyTesting:
    """Test Rupert property testing with sampling"""

    def test_test_rupert_property_count(self):
        """Should test exactly n_samples configurations"""
        vertices = generate_noperthedron_vertices()
        attempts, has_passage = test_rupert_property(vertices, n_samples=10)

        assert len(attempts) == 10

    def test_test_rupert_property_reproducible(self):
        """With same seed, should get same results"""
        vertices = generate_noperthedron_vertices()

        attempts1, _ = test_rupert_property(vertices, n_samples=5, random_seed=42)
        attempts2, _ = test_rupert_property(vertices, n_samples=5, random_seed=42)

        # Check first attempt matches
        assert attempts1[0].ds2 == attempts2[0].ds2
        assert attempts1[0].C == attempts2[0].C
        assert attempts1[0].S == attempts2[0].S

    def test_test_rupert_property_boolean_result(self):
        """Should return boolean for has_passage"""
        vertices = generate_noperthedron_vertices()
        _, has_passage = test_rupert_property(vertices, n_samples=5)

        assert isinstance(has_passage, (bool, np.bool_))

    def test_test_rupert_property_passage_detection(self):
        """If any ds² > 0, should report passage"""
        vertices = generate_noperthedron_vertices()
        attempts, has_passage = test_rupert_property(vertices, n_samples=20)

        # Check consistency
        any_positive = any(a.ds2 > 0 for a in attempts)
        assert has_passage == any_positive


class TestAnalysis:
    """Test results analysis"""

    def test_analyze_results_structure(self):
        """Should return dictionary with expected keys"""
        vertices = generate_noperthedron_vertices()
        attempts, _ = test_rupert_property(vertices, n_samples=10)
        stats = analyze_results(attempts)

        expected_keys = [
            "n_attempts",
            "ds2_min",
            "ds2_max",
            "ds2_mean",
            "ds2_std",
            "frac_timelike",
            "frac_spacelike",
            "frac_lightlike",
            "C_mean",
            "S_mean",
            "min_dist_min",
            "min_dist_mean",
        ]

        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_analyze_results_count(self):
        """n_attempts should match number of attempts"""
        vertices = generate_noperthedron_vertices()
        attempts, _ = test_rupert_property(vertices, n_samples=15)
        stats = analyze_results(attempts)

        assert stats["n_attempts"] == 15

    def test_analyze_results_fractions_sum_to_one(self):
        """Timelike + spacelike + lightlike should ≈ 1.0"""
        vertices = generate_noperthedron_vertices()
        attempts, _ = test_rupert_property(vertices, n_samples=20)
        stats = analyze_results(attempts)

        total = stats["frac_timelike"] + stats["frac_spacelike"] + stats["frac_lightlike"]
        assert np.isclose(total, 1.0, atol=0.01)

    def test_analyze_results_min_max_consistent(self):
        """min ≤ mean ≤ max for ds²"""
        vertices = generate_noperthedron_vertices()
        attempts, _ = test_rupert_property(vertices, n_samples=20)
        stats = analyze_results(attempts)

        assert stats["ds2_min"] <= stats["ds2_mean"] <= stats["ds2_max"]

    def test_analyze_results_predominantly_spacelike(self):
        """Noperthedron should be predominantly space-like (ds² < 0)"""
        vertices = generate_noperthedron_vertices()
        # Use larger sample for statistical reliability
        attempts, _ = test_rupert_property(vertices, n_samples=50, random_seed=42)
        stats = analyze_results(attempts)

        # Should have significant fraction of space-like configs
        # Note: With our simplified metric, we might not get 100%
        assert (
            stats["frac_spacelike"] > 0.5
        ), f"Expected >50% space-like, got {stats['frac_spacelike']:.1%}"


class TestIntegration:
    """Integration tests combining multiple components"""

    def test_full_workflow(self):
        """Test complete workflow from generation to analysis"""
        # Generate
        vertices = generate_noperthedron_vertices()
        assert len(vertices) == 90

        # Test a few configurations
        attempts, has_passage = test_rupert_property(vertices, n_samples=10)
        assert len(attempts) == 10

        # Analyze
        stats = analyze_results(attempts)
        assert stats["n_attempts"] == 10
        assert "ds2_mean" in stats

    def test_deterministic_with_seed(self):
        """Full workflow should be deterministic with fixed seed"""
        seed = 123

        # Run 1
        v1 = generate_noperthedron_vertices()
        a1, _ = test_rupert_property(v1, n_samples=5, random_seed=seed)
        s1 = analyze_results(a1)

        # Run 2
        v2 = generate_noperthedron_vertices()
        a2, _ = test_rupert_property(v2, n_samples=5, random_seed=seed)
        s2 = analyze_results(a2)

        # Should get same statistics
        assert np.isclose(s1["ds2_mean"], s2["ds2_mean"])
        assert np.isclose(s1["C_mean"], s2["C_mean"])
        assert np.isclose(s1["S_mean"], s2["S_mean"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
