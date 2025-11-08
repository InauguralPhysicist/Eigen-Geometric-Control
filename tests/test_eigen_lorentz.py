# -*- coding: utf-8 -*-
"""
Tests for Lorentz transformation framework

Validates:
1. Lorentz boost preserves ds² invariant
2. Composition of boosts follows group structure
3. Integration with C/S metrics from eigen_core
4. Stereo vision mapping to Lorentz coordinates
5. Regime classification (spacelike/timelike/lightlike)
"""

import numpy as np
import pytest

from src.eigen_lorentz import (
    LorentzState,
    apply_boost,
    beta_from_rapidity,
    boost_lorentz_state,
    change_stability_to_lorentz,
    create_lorentz_state,
    disparity_to_rapidity,
    lorentz_boost_from_rapidity,
    lorentz_boost_matrix,
    lorentz_factor,
    lorentz_to_stereo,
    proper_distance,
    proper_time,
    rapidity_from_beta,
    regime_classification,
    stereo_to_lorentz,
    verify_lorentz_invariance,
)


class TestLorentzFactor:
    """Test Lorentz factor computation"""

    def test_gamma_at_rest(self):
        """γ(0) = 1"""
        assert np.isclose(lorentz_factor(0.0), 1.0)

    def test_gamma_increases_with_velocity(self):
        """γ increases monotonically with β"""
        gammas = [lorentz_factor(beta) for beta in [0.0, 0.3, 0.6, 0.9]]
        assert all(g1 < g2 for g1, g2 in zip(gammas[:-1], gammas[1:]))

    def test_gamma_half_light_speed(self):
        """γ(0.5) = 1/√(1-0.25) ≈ 1.1547"""
        assert np.isclose(lorentz_factor(0.5), 1.1547005383792515)

    def test_gamma_rejects_superluminal(self):
        """γ(β≥1) raises ValueError"""
        with pytest.raises(ValueError, match="beta must be in"):
            lorentz_factor(1.0)

        with pytest.raises(ValueError):
            lorentz_factor(1.5)

    def test_gamma_high_velocity(self):
        """γ(0.99) ≈ 7.09"""
        assert np.isclose(lorentz_factor(0.99), 7.0888, rtol=1e-4)


class TestRapidityConversion:
    """Test rapidity ↔ velocity conversions"""

    def test_rapidity_at_rest(self):
        """θ(0) = 0"""
        assert rapidity_from_beta(0.0) == 0.0

    def test_roundtrip_conversion(self):
        """β → θ → β recovers original"""
        for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
            theta = rapidity_from_beta(beta)
            beta_recovered = beta_from_rapidity(theta)
            assert np.isclose(beta, beta_recovered)

    def test_rapidity_additive(self):
        """Rapidities add: θ₁ + θ₂"""
        # Two successive boosts
        beta1 = 0.3
        beta2 = 0.4

        theta1 = rapidity_from_beta(beta1)
        theta2 = rapidity_from_beta(beta2)

        # Combined rapidity
        theta_total = theta1 + theta2
        beta_total = beta_from_rapidity(theta_total)

        # Verify β_total < 1 (velocities don't simply add)
        assert beta_total < 1.0
        # And β_total ≠ β₁ + β₂ (relativistic composition)
        assert not np.isclose(beta_total, beta1 + beta2)


class TestLorentzBoostMatrix:
    """Test Lorentz transformation matrix construction"""

    def test_identity_at_rest(self):
        """Λ(0) = I"""
        Lambda = lorentz_boost_matrix(0.0)
        assert np.allclose(Lambda, np.eye(2))

    def test_matrix_structure(self):
        """Λ has correct hyperbolic structure"""
        beta = 0.6
        Lambda = lorentz_boost_matrix(beta)
        gamma = lorentz_factor(beta)

        # Check diagonal elements
        assert np.isclose(Lambda[0, 0], gamma)
        assert np.isclose(Lambda[1, 1], gamma)

        # Check off-diagonal (symmetric)
        assert np.isclose(Lambda[0, 1], Lambda[1, 0])
        assert np.isclose(Lambda[0, 1], -gamma * beta)

    def test_determinant_unity(self):
        """det(Λ) = 1 (unimodular)"""
        for beta in [0.2, 0.5, 0.8]:
            Lambda = lorentz_boost_matrix(beta)
            assert np.isclose(np.linalg.det(Lambda), 1.0)

    def test_not_orthogonal(self):
        """Λ is NOT orthogonal (Λᵀ ≠ Λ⁻¹)"""
        Lambda = lorentz_boost_matrix(0.5)
        # Orthogonal would satisfy: Λᵀ Λ = I
        product = Lambda.T @ Lambda
        # But Lorentz group uses different metric
        assert not np.allclose(product, np.eye(2))

    def test_rapidity_form_equivalent(self):
        """Λ(β) = Λ(θ) for θ = arctanh(β)"""
        beta = 0.7
        theta = rapidity_from_beta(beta)

        Lambda_beta = lorentz_boost_matrix(beta)
        Lambda_theta = lorentz_boost_from_rapidity(theta)

        assert np.allclose(Lambda_beta, Lambda_theta)


class TestInvariancePreservation:
    """Test that Lorentz boosts preserve ds² invariant"""

    def test_invariance_timelike(self):
        """ds² > 0 preserved under boost"""
        state = np.array([5.0, 3.0])  # ds² = 25 - 9 = 16
        assert verify_lorentz_invariance(state, beta=0.5)

    def test_invariance_spacelike(self):
        """ds² < 0 preserved under boost"""
        state = np.array([3.0, 5.0])  # ds² = 9 - 25 = -16
        assert verify_lorentz_invariance(state, beta=0.7)

    def test_invariance_lightlike(self):
        """ds² = 0 preserved under boost"""
        state = np.array([4.0, 4.0])  # ds² = 16 - 16 = 0
        assert verify_lorentz_invariance(state, beta=0.6)

    def test_invariance_multiple_boosts(self):
        """ds² preserved through multiple boosts"""
        state = np.array([7.0, 2.0])
        ds2_original = state[0] ** 2 - state[1] ** 2

        # Apply three successive boosts
        state = apply_boost(state, 0.3)
        state = apply_boost(state, 0.4)
        state = apply_boost(state, 0.2)

        ds2_final = state[0] ** 2 - state[1] ** 2

        assert np.isclose(ds2_original, ds2_final)

    def test_invariance_extreme_boost(self):
        """ds² preserved even at β = 0.99"""
        state = np.array([10.0, 6.0])
        assert verify_lorentz_invariance(state, beta=0.99, rtol=1e-9)


class TestLorentzStateClass:
    """Test LorentzState dataclass"""

    def test_creation(self):
        """Create LorentzState with correct invariant"""
        state = create_lorentz_state(timelike=5.0, spacelike=3.0)
        assert state.timelike == 5.0
        assert state.spacelike == 3.0
        assert state.ds2 == 16.0

    def test_invariant_validation(self):
        """LorentzState rejects inconsistent invariant"""
        with pytest.raises(ValueError, match="Invariant violation"):
            LorentzState(timelike=5.0, spacelike=3.0, ds2=100.0)

    def test_boost_preserves_invariant(self):
        """Boosting LorentzState preserves ds²"""
        state = create_lorentz_state(8.0, 4.0)
        boosted = boost_lorentz_state(state, beta=0.6)

        assert np.isclose(state.ds2, boosted.ds2)

    def test_boost_changes_components(self):
        """Boosting changes t, x but not ds²"""
        state = create_lorentz_state(10.0, 5.0)
        boosted = boost_lorentz_state(state, beta=0.5)

        # Components change
        assert not np.isclose(state.timelike, boosted.timelike)
        assert not np.isclose(state.spacelike, boosted.spacelike)

        # Invariant preserved
        assert np.isclose(state.ds2, boosted.ds2)


class TestStereoVisionMapping:
    """Test stereo vision ↔ Lorentz coordinate mappings"""

    def test_stereo_to_lorentz_roundtrip(self):
        """L, R → t, x → L', R' recovers original"""
        left = 12.0
        right = 8.0

        t, x, ds2 = stereo_to_lorentz(left, right)
        left_recovered, right_recovered = lorentz_to_stereo(t, x)

        assert np.isclose(left, left_recovered)
        assert np.isclose(right, right_recovered)

    def test_stereo_invariant_is_product(self):
        """ds² = L · R for stereo views"""
        left = 15.0
        right = 10.0

        t, x, ds2 = stereo_to_lorentz(left, right)

        assert np.isclose(ds2, left * right)

    def test_equal_views_gives_pure_timelike(self):
        """L = R → x = 0 (pure timelike, no disparity)"""
        left = right = 20.0

        t, x, ds2 = stereo_to_lorentz(left, right)

        assert np.isclose(x, 0.0)  # No disparity
        assert np.isclose(t, left)
        assert ds2 > 0  # Timelike

    def test_disparity_maps_to_spacelike(self):
        """Large disparity → large spacelike component"""
        # Small disparity
        t1, x1, _ = stereo_to_lorentz(10.0, 9.0)

        # Large disparity
        t2, x2, _ = stereo_to_lorentz(10.0, 2.0)

        assert abs(x2) > abs(x1)


class TestChangeStabilityMapping:
    """Test (C, S) metrics → Lorentz framework"""

    def test_cs_to_lorentz_structure(self):
        """(C, S) maps correctly to Lorentz state"""
        C, S = 5, 12

        state = change_stability_to_lorentz(C, S)

        assert state.spacelike == float(C)
        assert state.timelike == float(S)
        assert state.ds2 == float(S * S - C * C)

    def test_cs_regime_classification(self):
        """C > S → spacelike, C < S → timelike"""
        # Spacelike (change dominates)
        state_space = change_stability_to_lorentz(C=10, S=3)
        assert regime_classification(state_space.ds2) == "spacelike"

        # Timelike (stability dominates)
        state_time = change_stability_to_lorentz(C=3, S=10)
        assert regime_classification(state_time.ds2) == "timelike"

        # Lightlike (boundary)
        state_light = change_stability_to_lorentz(C=7, S=7)
        assert regime_classification(state_light.ds2) == "lightlike"

    def test_xor_rotation_maps_to_spacelike(self):
        """XOR rotation (C=33, S=31) → spacelike regime"""
        # From eigen_xor_rotation.py: constant C=33, S=31
        state = change_stability_to_lorentz(C=33, S=31)

        assert regime_classification(state.ds2) == "spacelike"
        assert state.ds2 == -128  # S² - C² = 961 - 1089 = -128


class TestDisparityToRapidity:
    """Test stereo disparity → Lorentz rapidity conversion"""

    def test_zero_disparity(self):
        """Zero disparity (infinite depth) → zero rapidity"""
        theta = disparity_to_rapidity(0, max_disparity=100)
        assert np.isclose(theta, 0.0)

    def test_small_disparity(self):
        """Small disparity (far) → small rapidity"""
        theta_small = disparity_to_rapidity(5, max_disparity=100)
        assert 0 < theta_small < 0.1

    def test_large_disparity(self):
        """Large disparity (near) → large rapidity"""
        theta_large = disparity_to_rapidity(90, max_disparity=100)
        assert theta_large > 1.0

    def test_monotonic_increase(self):
        """Rapidity increases monotonically with disparity"""
        rapidities = [disparity_to_rapidity(d, 100) for d in [10, 30, 50, 70, 90]]
        assert all(r1 < r2 for r1, r2 in zip(rapidities[:-1], rapidities[1:]))

    def test_max_disparity_rejection(self):
        """Disparity ≥ max_disparity raises ValueError"""
        with pytest.raises(ValueError, match="Disparity must be"):
            disparity_to_rapidity(100, max_disparity=100)

        with pytest.raises(ValueError):
            disparity_to_rapidity(150, max_disparity=100)


class TestRegimeClassification:
    """Test spacelike/timelike/lightlike regime detection"""

    def test_spacelike_regime(self):
        """ds² < 0 → spacelike (exploring, motion)"""
        assert regime_classification(-10.0) == "spacelike"
        assert regime_classification(-0.01) == "spacelike"

    def test_timelike_regime(self):
        """ds² > 0 → timelike (settled, stable)"""
        assert regime_classification(10.0) == "timelike"
        assert regime_classification(0.01) == "timelike"

    def test_lightlike_boundary(self):
        """ds² ≈ 0 → lightlike (transition)"""
        assert regime_classification(0.0) == "lightlike"
        assert regime_classification(1e-12) == "lightlike"

    def test_convergence_trajectory(self):
        """Robot arm convergence: spacelike → lightlike → timelike"""
        # Early (exploring)
        assert regime_classification(-50.0) == "spacelike"

        # Transition
        assert regime_classification(0.0) == "lightlike"

        # Converged (settled)
        assert regime_classification(50.0) == "timelike"


class TestProperTimeDistance:
    """Test proper time (timelike) and proper distance (spacelike)"""

    def test_proper_time_timelike(self):
        """τ = √(ds²) for ds² > 0"""
        ds2 = 25.0
        tau = proper_time(ds2)
        assert np.isclose(tau, 5.0)

    def test_proper_time_undefined_spacelike(self):
        """τ undefined for ds² < 0"""
        assert proper_time(-10.0) is None

    def test_proper_distance_spacelike(self):
        """σ = √(-ds²) for ds² < 0"""
        ds2 = -16.0
        sigma = proper_distance(ds2)
        assert np.isclose(sigma, 4.0)

    def test_proper_distance_undefined_timelike(self):
        """σ undefined for ds² > 0"""
        assert proper_distance(10.0) is None

    def test_lightlike_neither_defined(self):
        """Both τ and σ undefined at ds² = 0"""
        assert proper_time(0.0) is None
        assert proper_distance(0.0) is None


class TestBoostComposition:
    """Test composition of Lorentz boosts"""

    def test_boost_composition_associative(self):
        """(Λ₁ Λ₂) Λ₃ = Λ₁ (Λ₂ Λ₃)"""
        Lambda1 = lorentz_boost_matrix(0.3)
        Lambda2 = lorentz_boost_matrix(0.4)
        Lambda3 = lorentz_boost_matrix(0.2)

        left_assoc = (Lambda1 @ Lambda2) @ Lambda3
        right_assoc = Lambda1 @ (Lambda2 @ Lambda3)

        assert np.allclose(left_assoc, right_assoc)

    def test_inverse_boost(self):
        """Λ(β)⁻¹ Λ(β) = I"""
        beta = 0.6
        Lambda = lorentz_boost_matrix(beta)
        Lambda_inv = np.linalg.inv(Lambda)

        # Forward then inverse = identity
        product = Lambda_inv @ Lambda
        assert np.allclose(product, np.eye(2))

        # Inverse is also a Lorentz boost (with -β rapidity)
        # Verify det(Λ⁻¹) = 1
        assert np.isclose(np.linalg.det(Lambda_inv), 1.0)

    def test_boost_then_inverse_recovers_state(self):
        """Λ(β)⁻¹ Λ(β) x = x"""
        state = np.array([10.0, 6.0])
        beta = 0.7

        # Apply boost
        boosted = apply_boost(state, beta)

        # Apply inverse boost (Λ⁻¹ = Λᵀ for Lorentz in this metric)
        Lambda = lorentz_boost_matrix(beta)
        recovered = np.linalg.inv(Lambda) @ boosted

        assert np.allclose(state, recovered)


class TestNumericalStability:
    """Test numerical stability at edge cases"""

    def test_near_light_speed(self):
        """β → 1 remains numerically stable"""
        state = np.array([10.0, 5.0])

        for beta in [0.9, 0.99, 0.999]:
            boosted = apply_boost(state, beta)
            # Should not have NaN or Inf
            assert np.all(np.isfinite(boosted))

            # Invariant still preserved
            assert verify_lorentz_invariance(state, beta, rtol=1e-8)

    def test_small_rapidity(self):
        """Small θ approximates non-relativistic limit"""
        theta = 0.01  # Small rapidity
        Lambda = lorentz_boost_from_rapidity(theta)

        # Should be close to identity + small correction
        deviation = np.linalg.norm(Lambda - np.eye(2))
        assert deviation < 0.1

    def test_large_rapidity(self):
        """Large θ remains well-behaved"""
        theta = 3.0  # Large rapidity (β ≈ 0.995)
        Lambda = lorentz_boost_from_rapidity(theta)

        # Should not overflow
        assert np.all(np.isfinite(Lambda))

        # Still has det = 1
        assert np.isclose(np.linalg.det(Lambda), 1.0)


class TestIntegrationWithEigenCore:
    """Test integration with existing eigen_core.py metrics"""

    def test_arm_convergence_trajectory(self):
        """Simulate arm convergence: C→0, S→max → timelike"""
        # Early: high change (exploring)
        C_early, S_early = 40, 24
        state_early = change_stability_to_lorentz(C_early, S_early)
        assert regime_classification(state_early.ds2) == "spacelike"

        # Mid: balanced
        C_mid, S_mid = 32, 32
        state_mid = change_stability_to_lorentz(C_mid, S_mid)
        assert regime_classification(state_mid.ds2) == "lightlike"

        # Late: low change (converged)
        C_late, S_late = 5, 59
        state_late = change_stability_to_lorentz(C_late, S_late)
        assert regime_classification(state_late.ds2) == "timelike"

    def test_xor_stays_spacelike(self):
        """XOR rotation maintains constant spacelike regime"""
        # From src/eigen_xor_rotation.py: C=33, S=31 constant
        C, S = 33, 31

        state = change_stability_to_lorentz(C, S)

        assert regime_classification(state.ds2) == "spacelike"
        assert state.ds2 == -128

        # After boost, should remain spacelike
        boosted = boost_lorentz_state(state, beta=0.5)
        assert regime_classification(boosted.ds2) == "spacelike"
        assert boosted.ds2 == -128  # Invariant!


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
