"""
Quantum-Inspired Error Correction for Sensor Robustness

Uses redundancy and parity checks (inspired by quantum error correction)
to detect and correct sensor measurement errors.

Key Insight from Quantum Computing:
- Quantum error correction encodes logical qubits redundantly
- Parity checks detect errors without measuring the state directly
- Syndrome extraction identifies error locations
- Error correction recovers the original state

Applied to Robotics:
- Encode robot state using multiple redundant sensors
- Parity checks detect which sensor has an error
- Majority voting or syndrome-based correction recovers true state
- Robust to individual sensor failures

Classical Error Correction Codes:
- Repetition code (3-bit): |0⟩ → |000⟩, |1⟩ → |111⟩
- Detects and corrects single bit flips via majority vote
- Extended to continuous measurements via thresholding
"""

from typing import List, Optional, Tuple

import numpy as np


def encode_state_redundant(
    state: np.ndarray,
    n_copies: int = 3,
) -> List[np.ndarray]:
    """
    Encode state redundantly across multiple virtual sensors.

    Inspired by quantum repetition codes: |ψ⟩ → |ψψψ⟩

    Parameters
    ----------
    state : ndarray
        True state to encode
    n_copies : int
        Number of redundant copies (typically 3 for basic error correction)

    Returns
    -------
    encoded : list of ndarray
        Redundant copies of the state

    Notes
    -----
    In quantum error correction, information is encoded redundantly so that
    errors can be detected and corrected. Here we create multiple virtual
    "sensor readings" of the same state.

    Examples
    --------
    >>> state = np.array([1.0, 2.0])
    >>> encoded = encode_state_redundant(state, n_copies=3)
    >>> len(encoded)
    3
    >>> all(np.allclose(s, state) for s in encoded)
    True
    """
    return [state.copy() for _ in range(n_copies)]


def apply_measurement_errors(
    encoded_states: List[np.ndarray],
    error_rate: float = 0.1,
    error_magnitude: float = 1.0,
) -> List[np.ndarray]:
    """
    Simulate measurement errors on encoded states.

    Parameters
    ----------
    encoded_states : list of ndarray
        Redundant state copies
    error_rate : float
        Probability of error per sensor
    error_magnitude : float
        Magnitude of measurement errors

    Returns
    -------
    noisy_states : list of ndarray
        States with simulated measurement errors

    Notes
    -----
    In quantum systems, decoherence causes bit flips and phase errors.
    Here we simulate sensor failures or noise.
    """
    noisy_states = []
    for state in encoded_states:
        if np.random.rand() < error_rate:
            # Add large error (sensor malfunction)
            error = np.random.randn(*state.shape) * error_magnitude
            noisy_states.append(state + error)
        else:
            # Small noise (normal operation)
            noise = np.random.randn(*state.shape) * (error_magnitude * 0.1)
            noisy_states.append(state + noise)

    return noisy_states


def compute_parity_checks(
    measurements: List[np.ndarray],
) -> np.ndarray:
    """
    Compute parity checks to detect errors.

    Inspired by stabilizer formalism in quantum error correction.

    Parameters
    ----------
    measurements : list of ndarray
        Sensor measurements (should be identical if no errors)

    Returns
    -------
    syndromes : ndarray, shape (n_checks,)
        Parity check results (syndromes)
        - 0 = no error detected
        - >0 = inconsistency detected

    Notes
    -----
    For 3-sensor encoding:
    - Check 1: sensors 0 vs 1
    - Check 2: sensors 1 vs 2
    - Check 3: sensors 0 vs 2

    Syndrome pattern identifies which sensor has error:
    - [0, 0, 0] → no error
    - [1, 0, 1] → sensor 0 has error
    - [1, 1, 0] → sensor 1 has error
    - [0, 1, 1] → sensor 2 has error

    Examples
    --------
    >>> # No errors
    >>> states = [np.array([1.0]), np.array([1.0]), np.array([1.0])]
    >>> syndromes = compute_parity_checks(states)
    >>> np.allclose(syndromes, 0.0)
    True

    >>> # Sensor 1 has error
    >>> states = [np.array([1.0]), np.array([5.0]), np.array([1.0])]
    >>> syndromes = compute_parity_checks(states)
    >>> syndromes[0] > 0  # Check 0 vs 1 fails
    True
    >>> syndromes[1] > 0  # Check 1 vs 2 fails
    True
    """
    if len(measurements) < 2:
        return np.array([])

    n = len(measurements)
    syndromes = []

    # Pairwise checks
    for i in range(n - 1):
        # Distance between adjacent sensors
        diff = np.linalg.norm(measurements[i] - measurements[i + 1])
        syndromes.append(diff)

    # Also check first and last (wrap around)
    if n >= 3:
        diff = np.linalg.norm(measurements[0] - measurements[-1])
        syndromes.append(diff)

    return np.array(syndromes)


def detect_error_location(
    syndromes: np.ndarray,
    threshold: float = 0.5,
) -> Optional[int]:
    """
    Identify which sensor has an error based on syndrome pattern.

    Parameters
    ----------
    syndromes : ndarray
        Parity check results
    threshold : float
        Threshold for detecting inconsistency

    Returns
    -------
    error_index : int or None
        Index of sensor with error, or None if no error detected

    Notes
    -----
    For 3-sensor system:
    - syndromes = [check_01, check_12, check_02]

    Error patterns:
    - No error: [0, 0, 0]
    - Error at 0: [1, 0, 1] (differs from 1 and 2)
    - Error at 1: [1, 1, 0] (differs from 0 and 2)
    - Error at 2: [0, 1, 1] (differs from 1 and 0)

    Examples
    --------
    >>> # Sensor 1 has error
    >>> syndromes = np.array([2.0, 2.0, 0.1])  # 0≈2, but 1 differs
    >>> detect_error_location(syndromes, threshold=0.5)
    1
    """
    if len(syndromes) == 0:
        return None

    # Threshold syndromes (binary error pattern)
    error_pattern = (syndromes > threshold).astype(int)

    if len(syndromes) == 1:
        # 2-sensor case: can't determine which sensor is wrong
        # Need at least 3 sensors for error correction
        # Return None - just average both measurements
        return None

    elif len(syndromes) == 3:
        # 3-sensor case: syndrome lookup
        # Patterns: [check_01, check_12, check_02]

        if np.array_equal(error_pattern, [0, 0, 0]):
            # No error
            return None
        elif np.array_equal(error_pattern, [1, 0, 1]):
            # Sensor 0 differs from 1 and 2
            return 0
        elif np.array_equal(error_pattern, [1, 1, 0]):
            # Sensor 1 differs from 0 and 2
            return 1
        elif np.array_equal(error_pattern, [0, 1, 1]):
            # Sensor 2 differs from 0 and 1
            return 2
        else:
            # Multiple errors or ambiguous
            # Return sensor with largest deviation
            return None

    else:
        # More than 3 sensors: return sensor with most check failures
        # Count how many checks each sensor fails
        n_sensors = len(syndromes)
        error_counts = np.zeros(n_sensors)

        for i, syndrome_val in enumerate(syndromes):
            if syndrome_val > threshold:
                # This check failed - both involved sensors suspicious
                sensor1 = i
                sensor2 = (i + 1) % n_sensors
                error_counts[sensor1] += 1
                error_counts[sensor2] += 1

        if np.max(error_counts) == 0:
            return None

        # Return sensor with most failures
        return int(np.argmax(error_counts))


def correct_measurement_errors(
    measurements: List[np.ndarray],
    threshold: float = 0.5,
) -> Tuple[np.ndarray, dict]:
    """
    Detect and correct sensor measurement errors.

    Uses redundancy and parity checks to identify and fix errors.

    Parameters
    ----------
    measurements : list of ndarray
        Redundant sensor measurements
    threshold : float
        Threshold for error detection

    Returns
    -------
    corrected_state : ndarray
        Best estimate of true state (error-corrected)
    diagnostics : dict
        Error correction information

    Algorithm
    ---------
    1. Compute parity checks (syndromes)
    2. Identify error location from syndrome pattern
    3. Correct by majority voting (exclude faulty sensor)

    Examples
    --------
    >>> # Correct measurement with one error
    >>> true_state = np.array([1.0, 2.0])
    >>> measurements = [
    ...     np.array([1.0, 2.0]),  # Good
    ...     np.array([5.0, 8.0]),  # Error
    ...     np.array([1.0, 2.0]),  # Good
    ... ]
    >>> corrected, info = correct_measurement_errors(measurements)
    >>> info['error_detected']
    True
    >>> info['error_location']
    1
    >>> np.allclose(corrected, true_state, atol=0.1)
    True
    """
    if len(measurements) == 0:
        raise ValueError("No measurements provided")

    if len(measurements) == 1:
        # No redundancy - return as is
        return measurements[0], {
            "error_detected": False,
            "error_location": None,
            "syndromes": np.array([]),
            "correction_applied": False,
        }

    # Compute parity checks
    syndromes = compute_parity_checks(measurements)

    # Detect error location
    error_location = detect_error_location(syndromes, threshold=threshold)

    # Correct measurement
    if error_location is not None:
        # Error detected - exclude faulty sensor
        valid_measurements = [m for i, m in enumerate(measurements) if i != error_location]
        corrected_state = np.mean(valid_measurements, axis=0)
        correction_applied = True
        error_detected = True
    else:
        # No error detected - average all measurements
        corrected_state = np.mean(measurements, axis=0)
        correction_applied = False
        error_detected = False

    diagnostics = {
        "error_detected": error_detected,
        "error_location": error_location,
        "syndromes": syndromes,
        "correction_applied": correction_applied,
        "max_syndrome": float(np.max(syndromes)) if len(syndromes) > 0 else 0.0,
        "n_sensors": len(measurements),
    }

    return corrected_state, diagnostics


def qec_control_step(
    current_state: np.ndarray,
    target: np.ndarray,
    sensor_measurements: List[np.ndarray],
    eta: float = 0.1,
    error_threshold: float = 0.5,
) -> Tuple[np.ndarray, dict]:
    """
    Control step with quantum-inspired error correction.

    Uses redundant sensor measurements with error correction to
    robustly estimate state and move toward target.

    Parameters
    ----------
    current_state : ndarray
        Current state estimate
    target : ndarray
        Target state
    sensor_measurements : list of ndarray
        Redundant sensor measurements of current position
    eta : float
        Learning rate / step size
    error_threshold : float
        Threshold for error detection

    Returns
    -------
    new_state : ndarray
        Updated state after control step
    diagnostics : dict
        Error correction and control information

    Algorithm
    ---------
    1. Detect and correct measurement errors
    2. Update state estimate with corrected measurement
    3. Move toward target

    Examples
    --------
    >>> current = np.array([5.0, 5.0])
    >>> target = np.array([0.0, 0.0])
    >>> # Measurements with one error
    >>> measurements = [
    ...     np.array([5.0, 5.0]),
    ...     np.array([10.0, 10.0]),  # Error
    ...     np.array([5.0, 5.0]),
    ... ]
    >>> new_state, info = qec_control_step(current, target, measurements)
    >>> info['error_detected']
    True
    >>> # Should still move toward target despite error
    >>> np.linalg.norm(new_state - target) < np.linalg.norm(current - target)
    True
    """
    # Error correction
    corrected_measurement, qec_info = correct_measurement_errors(
        sensor_measurements, threshold=error_threshold
    )

    # Update state estimate (blend current state with corrected measurement)
    blend_factor = 0.5  # Weight for measurement vs state estimate
    updated_state = blend_factor * corrected_measurement + (1 - blend_factor) * current_state

    # Control: move toward target
    direction = target - updated_state
    new_state = updated_state + eta * direction

    # Diagnostics
    diagnostics = {
        **qec_info,
        "corrected_measurement": corrected_measurement,
        "measurement_quality": 1.0 - qec_info["max_syndrome"],  # 1 = perfect, 0 = bad
    }

    return new_state, diagnostics


def simulate_sensor_array(
    true_state: np.ndarray,
    n_sensors: int = 3,
    noise_level: float = 0.05,
    failure_rate: float = 0.1,
    failure_magnitude: float = 2.0,
) -> List[np.ndarray]:
    """
    Simulate redundant sensor array with failures.

    Parameters
    ----------
    true_state : ndarray
        True state being measured
    n_sensors : int
        Number of redundant sensors
    noise_level : float
        Normal measurement noise (small)
    failure_rate : float
        Probability of sensor failure
    failure_magnitude : float
        Magnitude of failure-induced errors (large)

    Returns
    -------
    measurements : list of ndarray
        Sensor measurements (with potential failures)

    Notes
    -----
    Simulates:
    - Normal operation: small Gaussian noise
    - Sensor failure: large errors with probability failure_rate
    """
    measurements = []

    for _ in range(n_sensors):
        if np.random.rand() < failure_rate:
            # Sensor failure - large error
            error = np.random.randn(*true_state.shape) * failure_magnitude
            measurements.append(true_state + error)
        else:
            # Normal operation - small noise
            noise = np.random.randn(*true_state.shape) * noise_level
            measurements.append(true_state + noise)

    return measurements
