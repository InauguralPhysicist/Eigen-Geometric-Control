"""
Weak Measurement for Stereo Vision

Applies quantum weak measurement principles to stereo vision:
- Strong measurement: Collapses wave function, destroys coherence
- Weak measurement: Gentle observation, preserves coherence, accumulates info

Key insight from your calculus:
- Left eye = timelike measurement (a=b, looking at convergence)
- Right eye = spacelike measurement (b≠a, looking at divergence)
- Weak observer = doesn't collapse, accumulates both views
- Depth emerges from interference pattern of timelike + spacelike

This is how the lightlike observer can measure WITHOUT forcing collapse.
"""

from typing import List, Tuple

import numpy as np


def weak_measurement_operator(
    measurement_strength: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create weak measurement operators.

    Strong measurement: Projects fully onto eigenstates (strength=1.0)
    Weak measurement: Gentle perturbation (strength << 1.0)

    Args:
        measurement_strength: How "hard" to measure, in [0, 1]
                            0 = no measurement (no collapse)
                            1 = strong measurement (full collapse)
                            0.1 = weak measurement (gentle, typical)

    Returns:
        - M_weak: Weak measurement operator (perturbs slightly)
        - M_complement: Complement operator (preserves coherence)
    """
    # Weak measurement operator
    # For position measurement: M = I + ε * σ_z
    # Where ε = measurement_strength
    epsilon = measurement_strength

    # 2x2 weak measurement operator (for 2D state space)
    M_weak = np.array([[1.0 + epsilon, 0.0], [0.0, 1.0 - epsilon]])

    # Complement operator (preserves what wasn't measured)
    M_complement = np.array([[1.0 - epsilon, 0.0], [0.0, 1.0 + epsilon]])

    return M_weak, M_complement


def apply_weak_measurement(
    state: np.ndarray,
    measurement_strength: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """
    Apply weak measurement to state without full collapse.

    Args:
        state: Current state vector (any dimension)
        measurement_strength: How strongly to measure

    Returns:
        - measured_state: State after weak measurement (barely perturbed)
        - measurement_result: Weak measurement value
    """
    # For general N-dimensional state, we measure along first two dimensions
    # This is the "left/right eye" distinction

    if len(state) < 2:
        # 1D state - just return it
        return state.copy(), float(state[0]) if len(state) > 0 else 0.0

    # Extract first two components (left/right eye analogy)
    state_2d = state[:2].copy()

    # Normalize to unit vector for measurement
    norm = np.linalg.norm(state_2d)
    if norm < 1e-9:
        return state.copy(), 0.0

    state_normalized = state_2d / norm

    # Apply weak measurement operator
    M_weak, M_complement = weak_measurement_operator(measurement_strength)

    # Weak measurement doesn't fully project
    # Instead, it slightly perturbs the state
    measured_2d = M_weak @ state_normalized

    # Renormalize (measurement changes norm slightly)
    measured_2d = measured_2d / np.linalg.norm(measured_2d)

    # Measurement result is the weak value
    # This is the expectation value under weak perturbation
    measurement_result = float(np.dot(state_normalized, measured_2d))

    # Reconstruct full state with measured components
    measured_state = state.copy()
    measured_state[:2] = measured_2d * norm

    return measured_state, measurement_result


def accumulate_weak_measurements(
    measurements: List[float],
    weights: List[float] = None,
) -> float:
    """
    Accumulate multiple weak measurements to extract strong signal.

    Weak measurements are noisy individually, but accumulation
    over time reveals the true value (like quantum weak values).

    Args:
        measurements: List of weak measurement results
        weights: Optional weights for each measurement (recency, confidence)

    Returns:
        accumulated_value: Strong estimate from weak measurements
    """
    if len(measurements) == 0:
        return 0.0

    if weights is None:
        # Equal weights
        weights = [1.0] * len(measurements)

    # Weighted average
    total_weight = sum(weights)
    if total_weight < 1e-9:
        return 0.0

    accumulated = sum(m * w for m, w in zip(measurements, weights)) / total_weight

    return float(accumulated)


def stereo_weak_measurement(
    left_state: np.ndarray,
    right_state: np.ndarray,
    measurement_strength: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Weak measurement for stereo vision.

    Your framework insight:
    - Left eye: Timelike view (a=b, convergence measurement)
    - Right eye: Spacelike view (b≠a, divergence measurement)
    - Weak lightlike observer: Measures both WITHOUT collapsing
    - Depth: Emerges from interference of timelike + spacelike

    Args:
        left_state: Left eye observation
        right_state: Right eye observation
        measurement_strength: How weakly to measure (preserves coherence)

    Returns:
        - left_measured: Left state after weak measurement
        - right_measured: Right state after weak measurement
        - depth_estimate: Weak depth value from interference
    """
    # Apply weak measurement to each eye separately
    left_measured, left_value = apply_weak_measurement(left_state, measurement_strength)

    right_measured, right_value = apply_weak_measurement(right_state, measurement_strength)

    # Depth emerges from INTERFERENCE of the two measurements
    # This is the key: timelike (left) XOR spacelike (right)
    # Weak measurement preserves the interference pattern

    # Compute disparity (difference between left/right)
    # This is the geometric manifestation of the XOR
    if len(left_state) >= 2 and len(right_state) >= 2:
        # Disparity in position
        left_pos = left_state[:2]
        right_pos = right_state[:2]

        disparity_vector = left_pos - right_pos
        disparity = np.linalg.norm(disparity_vector)

        # Depth is inversely related to disparity (parallax)
        # But computed weakly, preserving coherence
        if disparity > 1e-6:
            # Weak depth estimate
            # Strong measurement would force discrete depth
            # Weak measurement allows continuous depth
            depth_estimate = 1.0 / disparity
        else:
            depth_estimate = float("inf")  # Very far
    else:
        depth_estimate = 1.0  # Default depth

    return left_measured, right_measured, float(depth_estimate)


def weak_stereo_control_step(
    current_state: np.ndarray,
    target_state: np.ndarray,
    left_observation: np.ndarray,
    right_observation: np.ndarray,
    eta: float = 0.1,
    measurement_strength: float = 0.1,
    accumulated_measurements: List[float] = None,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Control step using weak stereo measurements.

    This is where weak measurement improves performance:
    - Strong measurement: Discrete depth → jittery control
    - Weak measurement: Continuous depth → smooth control

    Args:
        current_state: Current robot state
        target_state: Target state
        left_observation: Left eye observation (timelike)
        right_observation: Right eye observation (spacelike)
        eta: Learning rate
        measurement_strength: Weak measurement strength
        accumulated_measurements: History of weak measurements

    Returns:
        - new_state: Updated state
        - depth: Current depth estimate
        - updated_measurements: New measurement history
    """
    # Perform weak stereo measurement
    left_measured, right_measured, depth = stereo_weak_measurement(
        left_observation, right_observation, measurement_strength
    )

    # Accumulate weak measurements over time
    # Make a copy to avoid modifying the caller's list
    if accumulated_measurements is None:
        accumulated_measurements = []
    else:
        accumulated_measurements = accumulated_measurements.copy()

    accumulated_measurements.append(depth)

    # Keep only recent measurements (sliding window)
    if len(accumulated_measurements) > 10:
        accumulated_measurements = accumulated_measurements[-10:]

    # Strong depth estimate from accumulated weak measurements
    # Recency-weighted: more recent measurements weighted higher
    weights = [
        0.5 ** (len(accumulated_measurements) - 1 - i) for i in range(len(accumulated_measurements))
    ]

    strong_depth = accumulate_weak_measurements(accumulated_measurements, weights)

    # Use depth to inform control
    # Depth affects how we weigh the visual servoing

    # Standard gradient toward target
    direction = target_state - current_state

    # Visual servoing correction (weighted by depth confidence)
    # Closer objects (larger depth value) → stronger correction
    depth_confidence = 1.0 / (1.0 + strong_depth) if strong_depth < float("inf") else 0.0

    # Combine visual information
    if len(left_measured) == len(current_state):
        visual_direction = left_measured - current_state
    else:
        visual_direction = np.zeros_like(current_state)

    # Weighted combination
    combined_direction = (1 - depth_confidence) * direction + depth_confidence * visual_direction

    # Gradient step
    new_state = current_state + eta * combined_direction

    return new_state, strong_depth, accumulated_measurements


def coherence_metric(
    state_history: List[np.ndarray],
) -> float:
    """
    Measure quantum coherence of state trajectory.

    Coherent motion: Smooth, wave-like, predictable
    Decoherent motion: Choppy, particle-like, jittery

    Weak measurement preserves coherence.
    Strong measurement destroys coherence.

    Args:
        state_history: Recent state trajectory

    Returns:
        coherence: Measure in [0, 1], 1 = fully coherent
    """
    if len(state_history) < 3:
        return 1.0  # Assume coherent if not enough data

    # Coherence = smoothness of trajectory
    # Compute second derivative (acceleration/jitter)
    velocities = [state_history[i] - state_history[i - 1] for i in range(1, len(state_history))]

    accelerations = [velocities[i] - velocities[i - 1] for i in range(1, len(velocities))]

    # Low acceleration = high coherence (smooth motion)
    # High acceleration = low coherence (jittery motion)
    mean_acceleration = np.mean([np.linalg.norm(a) for a in accelerations])

    # Coherence score (inverse of jitter)
    coherence = 1.0 / (1.0 + mean_acceleration)

    return float(coherence)
