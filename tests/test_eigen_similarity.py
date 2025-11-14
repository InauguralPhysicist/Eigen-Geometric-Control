# -*- coding: utf-8 -*-
"""
Tests for eigen_similarity module

Validates the Lorentz-invariant similarity from EigenFunction.
"""

import numpy as np
import pytest

from src.eigen_similarity import (
    compare_self_similarity,
    detect_oscillation,
    lightlike_damping_factor,
    lorentz_similarity,
    regime_from_similarity,
    standard_cosine_similarity,
)


class TestLorentzSimilarity:
    """Test Lorentz-invariant similarity computation"""

    def test_self_similarity_is_zero(self):
        """Self-similarity should be 0.0 (lightlike)"""
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([3.0, 4.0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.random.randn(10),
        ]

        for v in vectors:
            sim = lorentz_similarity(v, v)
            assert np.isclose(sim, 0.0, atol=1e-10), f"Self-similarity should be 0, got {sim}"

    def test_parallel_vectors_lightlike(self):
        """Parallel vectors should have similarity 0.0 (on light cone)"""
        u = np.array([1.0, 2.0, 3.0])
        v = 2.5 * u  # Parallel

        sim = lorentz_similarity(u, v)
        assert np.isclose(sim, 0.0, atol=1e-10), "Parallel vectors should be lightlike"

    def test_non_parallel_vectors_spacelike(self):
        """Non-parallel vectors should have negative inner product"""
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])  # Perpendicular

        # Lorentz inner product
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        inner_prod = np.dot(u, v) - norm_u * norm_v

        assert inner_prod < 0, "Non-parallel should have negative inner product (spacelike)"

    def test_similarity_range(self):
        """Similarity should be in [-1, 1]"""
        u = np.random.randn(5)
        v = np.random.randn(5)

        sim = lorentz_similarity(u, v)
        assert -1.0 <= sim <= 1.0, f"Similarity {sim} out of range [-1, 1]"

    def test_dimension_mismatch_raises(self):
        """Different dimension vectors should raise ValueError"""
        u = np.array([1.0, 2.0])
        v = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="same shape"):
            lorentz_similarity(u, v)

    def test_zero_vector_handling(self):
        """Zero vectors should be handled gracefully"""
        u = np.zeros(3)
        v = np.array([1.0, 2.0, 3.0])

        sim = lorentz_similarity(u, v)
        assert np.isclose(sim, 0.0), "Zero vector should give 0 similarity"


class TestStandardCosineSimilarity:
    """Test standard cosine similarity for comparison"""

    def test_self_similarity_is_one(self):
        """Standard self-similarity should be 1.0 (enables loops!)"""
        v = np.array([3.0, 4.0])
        sim = standard_cosine_similarity(v, v)
        assert np.isclose(sim, 1.0), "Standard self-similarity should be 1.0"

    def test_perpendicular_vectors_zero(self):
        """Perpendicular vectors should have similarity 0.0"""
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])

        sim = standard_cosine_similarity(u, v)
        assert np.isclose(sim, 0.0), "Perpendicular vectors should have 0 similarity"

    def test_opposite_vectors_negative_one(self):
        """Opposite vectors should have similarity -1.0"""
        u = np.array([1.0, 0.0])
        v = np.array([-1.0, 0.0])

        sim = standard_cosine_similarity(u, v)
        assert np.isclose(sim, -1.0), "Opposite vectors should have -1.0 similarity"


class TestOscillationDetection:
    """Test oscillation detection"""

    def test_no_oscillation_stable(self):
        """Stable sequence with low similarity should not detect oscillation"""
        # States that are distinct and don't return to previous states
        history = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0]),
        ]

        osc, strength = detect_oscillation(history, window=2, threshold=0.95)
        assert not osc, "Non-repeating sequence should not oscillate"
        assert strength < 0.95, f"Oscillation strength {strength} too high for non-repeating sequence"

    def test_period_2_oscillation_detected(self):
        """Period-2 oscillation should be detected"""
        state_A = np.array([1.0, 0.0])
        state_B = np.array([0.0, 1.0])

        history = [state_A, state_B, state_A, state_B, state_A]

        osc, strength = detect_oscillation(history, window=2, threshold=0.95)
        assert osc, "Period-2 oscillation should be detected"
        assert strength > 0.95, f"Oscillation strength {strength} should be high"

    def test_short_history_no_detection(self):
        """Short history should not detect oscillation"""
        history = [np.array([1.0, 0.0])]

        osc, strength = detect_oscillation(history)
        assert not osc, "Single state cannot oscillate"
        assert strength == 0.0, "Strength should be 0 for single state"


class TestLightlikeDamping:
    """Test lightlike damping factor computation"""

    def test_zero_strength_minimal_damping(self):
        """Zero oscillation should give minimal damping"""
        damping = lightlike_damping_factor(0.0)
        assert damping < 0.1, f"Damping {damping} too high for zero oscillation"

    def test_full_strength_maximum_damping(self):
        """Full oscillation should give maximum damping"""
        damping = lightlike_damping_factor(1.0, max_damping=0.5)
        assert damping > 0.45, f"Damping {damping} should be near max_damping=0.5"

    def test_damping_monotonic(self):
        """Damping should increase monotonically with strength"""
        strengths = np.linspace(0.0, 1.0, 10)
        dampings = [lightlike_damping_factor(s) for s in strengths]

        for i in range(len(dampings) - 1):
            assert (
                dampings[i] <= dampings[i + 1]
            ), f"Damping not monotonic: {dampings[i]} > {dampings[i+1]}"

    def test_damping_range(self):
        """Damping should be in [0, max_damping]"""
        max_damp = 0.7
        strengths = np.linspace(0.0, 1.0, 20)

        for s in strengths:
            damping = lightlike_damping_factor(s, max_damping=max_damp)
            assert 0.0 <= damping <= max_damp, f"Damping {damping} out of range [0, {max_damp}]"


class TestCompareSelfSimilarity:
    """Test self-similarity comparison utility"""

    def test_returns_dict_with_keys(self):
        """Should return dict with expected keys"""
        v = np.array([3.0, 4.0])
        result = compare_self_similarity(v)

        assert "standard" in result
        assert "lorentz" in result
        assert "vector_norm" in result
        assert "interpretation" in result

    def test_standard_is_one(self):
        """Standard should be 1.0"""
        v = np.array([1.0, 2.0, 3.0])
        result = compare_self_similarity(v)

        assert np.isclose(result["standard"], 1.0), "Standard should be 1.0"

    def test_lorentz_is_zero(self):
        """Lorentz should be 0.0"""
        v = np.array([1.0, 2.0, 3.0])
        result = compare_self_similarity(v)

        assert np.isclose(result["lorentz"], 0.0), "Lorentz should be 0.0"

    def test_norm_correct(self):
        """Norm should match np.linalg.norm"""
        v = np.array([3.0, 4.0])
        result = compare_self_similarity(v)

        expected_norm = np.linalg.norm(v)
        assert np.isclose(result["vector_norm"], expected_norm), "Norm mismatch"


class TestRegimeClassification:
    """Test regime classification from similarity"""

    def test_parallel_is_lightlike(self):
        """Parallel vectors should be lightlike"""
        u = np.array([1.0, 2.0, 3.0])
        v = 2.0 * u

        regime = regime_from_similarity(u, v)
        assert regime == "lightlike", "Parallel should be lightlike"

    def test_non_parallel_is_spacelike(self):
        """Non-parallel vectors should be spacelike"""
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])

        regime = regime_from_similarity(u, v)
        assert regime == "spacelike", "Non-parallel should be spacelike"


class TestNumericalStability:
    """Test numerical stability"""

    def test_very_small_vectors(self):
        """Very small vectors should be handled"""
        u = np.array([1e-15, 1e-15])
        v = np.array([1e-15, 1e-15])

        sim = lorentz_similarity(u, v)
        assert np.isfinite(sim), "Should handle very small vectors"

    def test_very_large_vectors(self):
        """Very large vectors should be handled"""
        u = np.array([1e10, 1e10])
        v = np.array([1e10, -1e10])

        sim = lorentz_similarity(u, v)
        assert np.isfinite(sim), "Should handle very large vectors"
        assert -1.0 <= sim <= 1.0, "Should stay in valid range"


class TestIntegration:
    """Integration tests"""

    def test_workflow_oscillation_damping(self):
        """Full workflow: detect oscillation → compute damping"""
        # Create oscillating sequence
        state_A = np.array([1.0, 0.0, 1.0, 0.0])
        state_B = np.array([0.0, 1.0, 0.0, 1.0])

        history = [state_A, state_B, state_A, state_B, state_A]

        # Detect
        osc, strength = detect_oscillation(history)
        assert osc, "Should detect oscillation"

        # Compute damping
        damping = lightlike_damping_factor(strength)
        assert damping > 0.0, "Should compute non-zero damping"

        # Gradient scaling
        scaling = 1.0 - damping
        assert 0.0 < scaling < 1.0, "Scaling should reduce gradient"

    def test_lightlike_observer_property(self):
        """Verify lightlike observer property: ⟨u,u⟩_L = 0"""
        # Random vectors
        for _ in range(10):
            v = np.random.randn(np.random.randint(2, 20))

            # Self-similarity should be 0
            sim = lorentz_similarity(v, v)
            assert np.isclose(sim, 0.0, atol=1e-10), "Lightlike observer property violated"

            # Standard should be 1
            std_sim = standard_cosine_similarity(v, v)
            assert np.isclose(std_sim, 1.0), "Standard should be 1.0"
