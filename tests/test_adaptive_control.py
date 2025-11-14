# -*- coding: utf-8 -*-
"""
Tests for adaptive two-phase control parameters
"""

import numpy as np
import pytest

from src import adaptive_control_parameters


def test_far_from_target_uses_fast_phase():
    """When far from target, should use fast approach"""
    distance = 1.0  # Far (1 meter)
    eta, use_lightlike, phase = adaptive_control_parameters(distance)

    assert phase == "FAST_APPROACH"
    assert use_lightlike is False
    assert eta == pytest.approx(0.12, abs=0.01)  # Should be close to fast_eta


def test_close_to_target_uses_terminal_phase():
    """When close to target, should use terminal descent"""
    distance = 0.05  # Close (5 cm)
    eta, use_lightlike, phase = adaptive_control_parameters(distance)

    assert phase == "TERMINAL_DESCENT"
    assert use_lightlike is True
    # Sigmoid transition means eta approaches but doesn't exactly reach terminal_eta
    assert eta < 0.05  # Should be much closer to terminal (0.015) than fast (0.12)


def test_at_threshold_transitions_smoothly():
    """At threshold boundary, should be transitioning"""
    threshold = 0.15
    eta, use_lightlike, phase = adaptive_control_parameters(threshold)

    # Should be transitioning (eta between fast and terminal)
    assert 0.015 < eta < 0.12


def test_smooth_transition_region():
    """Verify smooth sigmoid transition without discontinuities"""
    distances = np.linspace(0.01, 0.3, 100)
    etas = []

    for d in distances:
        eta, _, _ = adaptive_control_parameters(d)
        etas.append(eta)

    etas = np.array(etas)

    # Check monotonic increase (eta increases with distance)
    assert np.all(np.diff(etas) >= -1e-6), "Eta should increase monotonically with distance"

    # Check no sudden jumps (smooth transition)
    deta = np.diff(etas)
    max_jump = np.max(np.abs(deta))
    assert max_jump < 0.01, f"Maximum eta jump {max_jump} too large (not smooth)"


def test_extreme_distances():
    """Test behavior at extreme distances"""
    # Very far
    eta_far, use_ll_far, phase_far = adaptive_control_parameters(10.0)
    assert eta_far == pytest.approx(0.12, abs=0.001)
    assert phase_far == "FAST_APPROACH"
    assert use_ll_far is False

    # Very close (sigmoid means we approach but don't exactly reach terminal_eta)
    eta_close, use_ll_close, phase_close = adaptive_control_parameters(0.001)
    assert 0.015 <= eta_close <= 0.025  # Close to terminal_eta within sigmoid range
    assert phase_close == "TERMINAL_DESCENT"
    assert use_ll_close is True


def test_custom_parameters():
    """Test with custom threshold and eta values"""
    distance = 0.5
    threshold = 0.4
    fast = 0.2
    terminal = 0.01

    eta, use_ll, phase = adaptive_control_parameters(
        distance,
        terminal_threshold=threshold,
        fast_eta=fast,
        terminal_eta=terminal,
    )

    # Distance > threshold, so should be fast phase
    assert phase == "FAST_APPROACH"
    assert use_ll is False
    assert eta > terminal  # Should be above terminal_eta


def test_transition_width_affects_smoothness():
    """Wider transition should give more gradual eta changes"""
    # Test away from threshold to see difference
    distance = 0.18  # Slightly above threshold (0.15)

    # Narrow transition (sharp change at threshold)
    eta_narrow, _, _ = adaptive_control_parameters(distance, transition_width=0.01)

    # Wide transition (gradual change)
    eta_wide, _, _ = adaptive_control_parameters(distance, transition_width=0.1)

    # With narrow transition, should be closer to fast_eta
    # With wide transition, should still be transitioning
    assert eta_narrow > eta_wide, "Narrow transition should reach fast eta quicker"

    # Both should be in valid range
    assert 0.015 < eta_narrow < 0.12
    assert 0.015 < eta_wide < 0.12


def test_lightlike_only_enabled_in_terminal_phase():
    """Lightlike observer should only be enabled in terminal descent"""
    # Test many distances
    distances = np.linspace(0.01, 0.5, 50)

    for d in distances:
        _, use_ll, phase = adaptive_control_parameters(d, terminal_threshold=0.15)

        if phase == "FAST_APPROACH":
            assert use_ll is False, f"Lightlike should be OFF in fast approach (d={d})"
        elif phase == "TERMINAL_DESCENT":
            assert use_ll is True, f"Lightlike should be ON in terminal descent (d={d})"


def test_returns_correct_types():
    """Verify return types are correct"""
    eta, use_ll, phase = adaptive_control_parameters(0.1)

    assert isinstance(eta, float)
    assert isinstance(use_ll, bool)
    assert isinstance(phase, str)
    assert phase in ["FAST_APPROACH", "TERMINAL_DESCENT"]


def test_adaptive_vs_fixed_fast():
    """Adaptive should match fixed fast phase when far from target"""
    distance = 1.0  # Far

    # Adaptive
    eta_adaptive, _, _ = adaptive_control_parameters(distance, fast_eta=0.12)

    # Should be essentially identical to fixed fast
    assert eta_adaptive == pytest.approx(0.12, abs=0.001)


def test_adaptive_vs_fixed_terminal():
    """Adaptive should approach fixed terminal phase when close to target"""
    distance = 0.01  # Very close

    # Adaptive
    eta_adaptive, _, _ = adaptive_control_parameters(distance, terminal_eta=0.015)

    # Sigmoid means we approach but don't exactly reach terminal_eta
    # Should be much closer to terminal (0.015) than fast (0.12)
    assert 0.015 <= eta_adaptive <= 0.03


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
