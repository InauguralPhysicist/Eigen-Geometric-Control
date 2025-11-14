"""
Tests for Quantum-Inspired Error Correction

These tests demonstrate that redundant encoding with parity checks
enables robust sensor fusion despite measurement errors and failures.

Key proof: QEC >> No error correction for faulty sensors
"""

import numpy as np
import pytest

from src.eigen_qec import (
    apply_measurement_errors,
    compute_parity_checks,
    correct_measurement_errors,
    detect_error_location,
    encode_state_redundant,
    qec_control_step,
    simulate_sensor_array,
)


class TestEncodeStateRedundant:
    """Test redundant state encoding."""

    def test_creates_n_copies(self):
        """Should create n redundant copies."""
        state = np.array([1.0, 2.0])
        n_copies = 5

        encoded = encode_state_redundant(state, n_copies=n_copies)

        assert len(encoded) == n_copies

    def test_all_copies_identical(self):
        """All copies should be identical to original."""
        state = np.array([1.0, 2.0, 3.0])

        encoded = encode_state_redundant(state, n_copies=3)

        for copy in encoded:
            assert np.allclose(copy, state)

    def test_copies_are_independent(self):
        """Modifying one copy shouldn't affect others."""
        state = np.array([1.0, 2.0])

        encoded = encode_state_redundant(state, n_copies=3)

        # Modify first copy
        encoded[0][0] = 999.0

        # Others should be unchanged
        assert encoded[1][0] == 1.0
        assert encoded[2][0] == 1.0


class TestApplyMeasurementErrors:
    """Test measurement error simulation."""

    def test_returns_same_length(self):
        """Should return same number of measurements."""
        encoded = [np.array([1.0, 2.0]) for _ in range(3)]

        noisy = apply_measurement_errors(encoded, error_rate=0.5)

        assert len(noisy) == len(encoded)

    def test_high_error_rate_causes_differences(self):
        """High error rate should cause measurements to differ."""
        np.random.seed(42)
        encoded = [np.array([1.0, 2.0]) for _ in range(10)]

        noisy = apply_measurement_errors(encoded, error_rate=0.8, error_magnitude=5.0)

        # At least some should have large errors
        differences = [np.linalg.norm(noisy[i] - encoded[i]) for i in range(10)]
        assert any(d > 1.0 for d in differences)

    def test_zero_error_rate_small_noise(self):
        """Zero error rate should only add small noise."""
        np.random.seed(42)
        encoded = [np.array([1.0, 2.0]) for _ in range(10)]

        noisy = apply_measurement_errors(encoded, error_rate=0.0, error_magnitude=1.0)

        # All should be close to original (only small noise)
        for i in range(10):
            assert np.linalg.norm(noisy[i] - encoded[i]) < 0.5


class TestComputeParityChecks:
    """Test parity check computation."""

    def test_identical_measurements_zero_syndromes(self):
        """Identical measurements → zero syndromes (no error)."""
        measurements = [np.array([1.0, 2.0]) for _ in range(3)]

        syndromes = compute_parity_checks(measurements)

        assert np.allclose(syndromes, 0.0)

    def test_one_different_nonzero_syndromes(self):
        """One different measurement → nonzero syndromes."""
        measurements = [
            np.array([1.0, 2.0]),
            np.array([5.0, 6.0]),  # Different
            np.array([1.0, 2.0]),
        ]

        syndromes = compute_parity_checks(measurements)

        # Should detect inconsistency
        assert np.max(syndromes) > 0.5

    def test_syndrome_length_correct(self):
        """Should return n syndromes for n measurements."""
        # 3 measurements → 3 checks (01, 12, 02)
        measurements = [np.array([1.0]) for _ in range(3)]

        syndromes = compute_parity_checks(measurements)

        assert len(syndromes) == 3

    def test_two_measurements_one_check(self):
        """Two measurements → one check."""
        measurements = [np.array([1.0, 2.0]), np.array([1.0, 2.0])]

        syndromes = compute_parity_checks(measurements)

        assert len(syndromes) == 1

    def test_empty_measurements_empty_syndromes(self):
        """Empty or single measurement → empty syndromes."""
        measurements = []

        syndromes = compute_parity_checks(measurements)

        assert len(syndromes) == 0


class TestDetectErrorLocation:
    """Test error location detection."""

    def test_no_error_returns_none(self):
        """All syndromes zero → no error detected."""
        syndromes = np.array([0.0, 0.0, 0.0])

        error_loc = detect_error_location(syndromes, threshold=0.5)

        assert error_loc is None

    def test_detects_sensor_0_error(self):
        """Pattern [1, 0, 1] → sensor 0 has error."""
        syndromes = np.array([2.0, 0.1, 2.0])  # Sensor 0 differs from 1 and 2

        error_loc = detect_error_location(syndromes, threshold=0.5)

        assert error_loc == 0

    def test_detects_sensor_1_error(self):
        """Pattern [1, 1, 0] → sensor 1 has error."""
        syndromes = np.array([2.0, 2.0, 0.1])  # Sensor 1 differs from 0 and 2

        error_loc = detect_error_location(syndromes, threshold=0.5)

        assert error_loc == 1

    def test_detects_sensor_2_error(self):
        """Pattern [0, 1, 1] → sensor 2 has error."""
        syndromes = np.array([0.1, 2.0, 2.0])  # Sensor 2 differs from 0 and 1

        error_loc = detect_error_location(syndromes, threshold=0.5)

        assert error_loc == 2

    def test_empty_syndromes_none(self):
        """Empty syndromes → None."""
        syndromes = np.array([])

        error_loc = detect_error_location(syndromes, threshold=0.5)

        assert error_loc is None


class TestCorrectMeasurementErrors:
    """Test error correction."""

    def test_no_error_averages_all(self):
        """No error → average all measurements."""
        measurements = [
            np.array([1.0, 2.0]),
            np.array([1.1, 2.1]),
            np.array([0.9, 1.9]),
        ]

        corrected, info = correct_measurement_errors(measurements, threshold=0.5)

        # Should average all three
        expected = np.mean(measurements, axis=0)
        assert np.allclose(corrected, expected)
        assert info["error_detected"] == False

    def test_detects_and_corrects_error(self):
        """One faulty sensor → detect and exclude it."""
        true_state = np.array([1.0, 2.0])
        measurements = [
            true_state.copy(),
            np.array([10.0, 20.0]),  # Large error
            true_state.copy(),
        ]

        corrected, info = correct_measurement_errors(measurements, threshold=0.5)

        # Should detect error at sensor 1
        assert info["error_detected"] == True
        assert info["error_location"] == 1
        assert info["correction_applied"] == True

        # Should exclude faulty sensor and average the other two
        assert np.allclose(corrected, true_state, atol=0.1)

    def test_single_measurement_no_correction(self):
        """Single measurement → return as is."""
        measurement = np.array([1.0, 2.0])

        corrected, info = correct_measurement_errors([measurement])

        assert np.allclose(corrected, measurement)
        assert info["error_detected"] == False

    def test_reports_syndromes(self):
        """Should report syndrome values."""
        measurements = [
            np.array([1.0]),
            np.array([5.0]),  # Error
            np.array([1.0]),
        ]

        corrected, info = correct_measurement_errors(measurements)

        assert "syndromes" in info
        assert len(info["syndromes"]) == 3
        assert info["max_syndrome"] > 0

    def test_two_measurements_simple_average(self):
        """Two measurements → simple average."""
        measurements = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

        corrected, info = correct_measurement_errors(measurements)

        expected = np.array([2.0, 3.0])
        assert np.allclose(corrected, expected)


class TestQECControlStep:
    """Test QEC-based control."""

    def test_moves_toward_target(self):
        """Should move toward target."""
        current = np.array([5.0, 5.0])
        target = np.array([0.0, 0.0])
        measurements = [current.copy() for _ in range(3)]  # All good

        new_state, info = qec_control_step(current, target, measurements, eta=0.2)

        # Should move closer
        dist_before = np.linalg.norm(current - target)
        dist_after = np.linalg.norm(new_state - target)

        assert dist_after < dist_before

    def test_robust_to_measurement_error(self):
        """Should handle measurement errors gracefully."""
        current = np.array([5.0, 5.0])
        target = np.array([0.0, 0.0])
        measurements = [
            np.array([5.0, 5.0]),
            np.array([20.0, 20.0]),  # Large error
            np.array([5.0, 5.0]),
        ]

        new_state, info = qec_control_step(current, target, measurements, eta=0.2)

        # Should detect error
        assert info["error_detected"] == True
        assert info["error_location"] == 1

        # Should still move toward target (not misled by error)
        dist_after = np.linalg.norm(new_state - target)
        dist_before = np.linalg.norm(current - target)

        # Progress toward target should be reasonable
        # (not affected by the large error)
        assert dist_after < dist_before
        # New state should not be near the erroneous measurement
        assert np.linalg.norm(new_state - np.array([20.0, 20.0])) > 5.0

    def test_reports_measurement_quality(self):
        """Should report measurement quality metric."""
        current = np.array([5.0, 5.0])
        target = np.array([0.0, 0.0])

        # Good measurements
        good_measurements = [current.copy() for _ in range(3)]
        _, info_good = qec_control_step(current, target, good_measurements)

        # Bad measurements
        bad_measurements = [
            current.copy(),
            current + np.array([10.0, 10.0]),
            current.copy(),
        ]
        _, info_bad = qec_control_step(current, target, bad_measurements)

        # Good should have higher quality
        assert info_good["measurement_quality"] > info_bad["measurement_quality"]

    def test_handles_all_faulty_sensors(self):
        """Should handle case where all sensors are noisy."""
        current = np.array([5.0, 5.0])
        target = np.array([0.0, 0.0])

        # All measurements slightly different
        measurements = [
            np.array([5.1, 5.1]),
            np.array([4.9, 4.9]),
            np.array([5.0, 5.0]),
        ]

        new_state, info = qec_control_step(current, target, measurements, eta=0.2)

        # Should still work (average all and move toward target)
        dist_after = np.linalg.norm(new_state - target)
        dist_before = np.linalg.norm(current - target)

        assert dist_after < dist_before


class TestSimulateSensorArray:
    """Test sensor array simulation."""

    def test_returns_n_measurements(self):
        """Should return n measurements."""
        true_state = np.array([1.0, 2.0])
        n_sensors = 5

        measurements = simulate_sensor_array(true_state, n_sensors=n_sensors)

        assert len(measurements) == n_sensors

    def test_high_failure_rate_causes_errors(self):
        """High failure rate should cause large errors."""
        np.random.seed(42)
        true_state = np.array([1.0, 2.0])

        measurements = simulate_sensor_array(
            true_state, n_sensors=20, failure_rate=0.8, failure_magnitude=10.0
        )

        # Should have many large errors
        errors = [np.linalg.norm(m - true_state) for m in measurements]
        large_errors = [e for e in errors if e > 5.0]

        assert len(large_errors) > 10  # Most should have large errors

    def test_zero_failure_rate_small_noise(self):
        """Zero failure rate → only small noise."""
        np.random.seed(42)
        true_state = np.array([1.0, 2.0])

        measurements = simulate_sensor_array(
            true_state, n_sensors=10, failure_rate=0.0, noise_level=0.1
        )

        # All should be close to true state
        for m in measurements:
            assert np.linalg.norm(m - true_state) < 0.5


class TestQECBenefits:
    """Test that QEC improves robustness."""

    def test_qec_recovers_from_single_failure(self):
        """QEC should recover true state despite one sensor failure."""
        true_state = np.array([1.0, 2.0])

        # Measurements with one failure
        measurements = [
            true_state + np.random.randn(2) * 0.05,  # Good
            true_state + np.random.randn(2) * 10.0,  # Failed
            true_state + np.random.randn(2) * 0.05,  # Good
        ]

        corrected, info = correct_measurement_errors(measurements, threshold=0.5)

        # Should be close to true state
        assert np.linalg.norm(corrected - true_state) < 0.5

        # Should detect the error
        assert info["error_detected"] == True

    def test_naive_average_fails_with_outlier(self):
        """Naive averaging performs poorly with outliers."""
        true_state = np.array([1.0, 2.0])

        measurements = [
            true_state.copy(),
            np.array([100.0, 200.0]),  # Large outlier
            true_state.copy(),
        ]

        # Naive average
        naive_avg = np.mean(measurements, axis=0)

        # QEC correction
        qec_corrected, _ = correct_measurement_errors(measurements)

        # QEC should be much closer to true state
        naive_error = np.linalg.norm(naive_avg - true_state)
        qec_error = np.linalg.norm(qec_corrected - true_state)

        assert qec_error < naive_error * 0.1  # 10x better


def test_module_imports():
    """Verify all functions are importable."""
    from src.eigen_qec import (
        apply_measurement_errors,
        compute_parity_checks,
        correct_measurement_errors,
        detect_error_location,
        encode_state_redundant,
        qec_control_step,
        simulate_sensor_array,
    )

    assert callable(encode_state_redundant)
    assert callable(apply_measurement_errors)
    assert callable(compute_parity_checks)
    assert callable(detect_error_location)
    assert callable(correct_measurement_errors)
    assert callable(qec_control_step)
    assert callable(simulate_sensor_array)
