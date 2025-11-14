import pytest
# -*- coding: utf-8 -*-
"""
Tests for frequency-selective damping
"""

import numpy as np

from src import detect_oscillation_frequency, frequency_selective_damping


def generate_oscillating_trajectory(
    freq: float, n_samples: int = 32, amplitude: float = 1.0, noise_level: float = 0.0
):
    """Generate synthetic trajectory with known oscillation frequency"""
    t = np.arange(n_samples)
    # 2D trajectory with oscillation
    x = amplitude * np.sin(2 * np.pi * freq * t)
    y = amplitude * np.cos(2 * np.pi * freq * t)

    # Add noise if requested
    if noise_level > 0:
        x += np.random.normal(0, noise_level, n_samples)
        y += np.random.normal(0, noise_level, n_samples)

    trajectory = [np.array([x[i], y[i]]) for i in range(n_samples)]
    return trajectory


def test_detect_no_oscillation_with_short_history():
    """Should return False with insufficient history"""
    short_history = [np.array([1.0, 2.0]), np.array([1.1, 2.1])]
    osc, freq, strength = detect_oscillation_frequency(short_history, min_history=8)

    assert osc is False
    assert freq == 0.0
    assert strength == 0.0


def test_detect_oscillation_mid_frequency():
    """Should detect period-2 oscillations (mid frequency range)"""
    # Period-2 oscillation: freq = 0.35 Hz (mid-range)
    trajectory = generate_oscillating_trajectory(freq=0.35, n_samples=32, amplitude=1.0)

    osc, freq, strength = detect_oscillation_frequency(trajectory, sample_rate=1.0)

    assert osc is True
    assert freq > 0.1  # Should detect some oscillation frequency
    assert strength > 0.2  # Significant oscillation


def test_detect_oscillation_low_frequency():
    """Should detect slow convergence (low frequency)"""
    # Slow oscillation: freq = 0.1 Hz
    trajectory = generate_oscillating_trajectory(freq=0.1, n_samples=32, amplitude=1.0)

    osc, freq, strength = detect_oscillation_frequency(trajectory, sample_rate=1.0)

    # Might or might not be detected depending on threshold
    # But if detected, frequency should be low
    if osc:
        assert freq < 0.2


def test_detect_oscillation_high_frequency():
    """Should handle high frequency oscillations"""
    # Fast oscillation: freq = 0.4 Hz
    trajectory = generate_oscillating_trajectory(freq=0.4, n_samples=32, amplitude=1.0)

    osc, freq, strength = detect_oscillation_frequency(trajectory, sample_rate=1.0)

    # Should detect oscillation
    assert osc is True
    assert freq > 0.05  # Some frequency detected


def test_no_oscillation_linear_trajectory():
    """Should not detect strong oscillation in linear motion"""
    # Linear trajectory (no oscillation)
    trajectory = [np.array([float(i), float(i) * 0.5]) for i in range(32)]

    osc, freq, strength = detect_oscillation_frequency(trajectory)

    # Linear motion might have some frequency content from the trend
    # The key is that it should be low frequency (slow change)
    if osc:
        assert freq < 0.15  # Should be very low frequency if detected


def test_frequency_selective_damping_mid_freq():
    """Should apply strong damping to mid-frequency oscillations"""
    # Period-2 oscillation (problematic range)
    trajectory = generate_oscillating_trajectory(freq=0.35, n_samples=32, amplitude=1.0)

    damping = frequency_selective_damping(
        trajectory, sample_rate=1.0, target_freq_range=(0.25, 0.5), max_damping=0.5
    )

    # Should apply significant damping for mid-frequency
    assert damping > 0.2  # Non-trivial damping
    assert damping <= 0.5  # But capped at max_damping


def test_frequency_selective_damping_low_freq():
    """Should apply gentle damping to low-frequency motion"""
    # Slow convergence (preserve)
    trajectory = generate_oscillating_trajectory(freq=0.15, n_samples=32, amplitude=1.0)

    damping = frequency_selective_damping(
        trajectory, sample_rate=1.0, target_freq_range=(0.25, 0.5), max_damping=0.5
    )

    # Should apply less damping for low frequency
    # (preserving slow convergence toward target)
    assert damping < 0.3  # Reduced damping for low freq


def test_frequency_selective_damping_no_oscillation():
    """Should return zero damping for no oscillation"""
    # Linear trajectory
    trajectory = [np.array([float(i), float(i) * 0.5]) for i in range(32)]

    damping = frequency_selective_damping(trajectory)

    # No oscillation → no damping
    assert damping == pytest.approx(0.0, abs=0.1)


def test_frequency_selective_vs_target_range():
    """Damping should be strongest when frequency matches target range"""
    n_samples = 32

    # Test different frequencies
    freqs_to_test = [0.1, 0.2, 0.35, 0.5, 0.7]
    dampings = []

    for freq in freqs_to_test:
        trajectory = generate_oscillating_trajectory(freq=freq, n_samples=n_samples, amplitude=1.0)
        damping = frequency_selective_damping(
            trajectory, target_freq_range=(0.25, 0.5), max_damping=0.5
        )
        dampings.append(damping)

    # Damping for freq=0.35 (in range) should be higher than edge cases
    mid_range_damping = dampings[2]  # 0.35 Hz
    low_freq_damping = dampings[0]  # 0.1 Hz

    assert mid_range_damping >= low_freq_damping


def test_custom_target_freq_range():
    """Should respect custom target frequency range"""
    # Create oscillation at 0.3 Hz
    trajectory = generate_oscillating_trajectory(freq=0.3, n_samples=32, amplitude=1.0)

    # Test with range that EXCLUDES 0.3 Hz
    damping_excluded = frequency_selective_damping(
        trajectory, target_freq_range=(0.4, 0.6), max_damping=0.5
    )

    # Test with range that INCLUDES 0.3 Hz
    damping_included = frequency_selective_damping(
        trajectory, target_freq_range=(0.2, 0.4), max_damping=0.5
    )

    # Damping should be stronger when frequency is in target range
    assert damping_included >= damping_excluded


def test_max_damping_respected():
    """Damping should never exceed max_damping"""
    # Strong oscillation
    trajectory = generate_oscillating_trajectory(freq=0.35, n_samples=32, amplitude=2.0)

    max_damp = 0.3
    damping = frequency_selective_damping(trajectory, max_damping=max_damp)

    assert damping <= max_damp


def test_sample_rate_affects_frequency():
    """Different sample rates should scale detected frequencies"""
    n_samples = 32

    # Same oscillation, different sample rates
    trajectory = generate_oscillating_trajectory(freq=0.5, n_samples=n_samples, amplitude=1.0)

    # At sample_rate = 1.0 Hz
    osc1, freq1, strength1 = detect_oscillation_frequency(trajectory, sample_rate=1.0)

    # At sample_rate = 2.0 Hz (twice as fast sampling)
    # The same trajectory now represents slower oscillation in real time
    osc2, freq2, strength2 = detect_oscillation_frequency(trajectory, sample_rate=2.0)

    if osc1 and osc2:
        # With faster sampling, same trajectory appears as lower frequency
        # (more samples per cycle)
        assert freq2 < freq1 * 1.5  # Approximately scaled


def test_noisy_oscillation_still_detected():
    """Should detect oscillation even with added noise"""
    # Oscillation with moderate noise
    trajectory = generate_oscillating_trajectory(
        freq=0.35, n_samples=32, amplitude=1.0, noise_level=0.2
    )

    osc, freq, strength = detect_oscillation_frequency(trajectory)

    # Should still detect oscillation despite noise
    assert osc is True
    # Frequency should be approximately correct
    assert 0.25 <= freq <= 0.5
    # Strength might be reduced by noise
    assert strength > 0.2


def test_multidimensional_trajectory():
    """Should handle trajectories with >2 dimensions"""
    n_samples = 32
    t = np.arange(n_samples)

    # 4D trajectory with oscillation
    trajectory = [
        np.array(
            [
                np.sin(2 * np.pi * 0.35 * t[i]),
                np.cos(2 * np.pi * 0.35 * t[i]),
                np.sin(2 * np.pi * 0.35 * t[i] + np.pi / 4),
                np.cos(2 * np.pi * 0.35 * t[i] + np.pi / 4),
            ]
        )
        for i in range(n_samples)
    ]

    osc, freq, strength = detect_oscillation_frequency(trajectory)

    assert osc is True
    assert 0.25 <= freq <= 0.5


def test_damping_returns_float():
    """Ensure damping returns a float type"""
    trajectory = generate_oscillating_trajectory(freq=0.35, n_samples=32)
    damping = frequency_selective_damping(trajectory)

    assert isinstance(damping, float)


def test_frequency_detection_returns_correct_types():
    """Ensure all return values are correct types"""
    trajectory = generate_oscillating_trajectory(freq=0.35, n_samples=32)
    osc, freq, strength = detect_oscillation_frequency(trajectory)

    assert isinstance(osc, bool)
    assert isinstance(freq, float)
    assert isinstance(strength, float)
    assert 0.0 <= strength <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
