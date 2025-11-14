"""
Dynamic Invariant Detection for Auto-Adaptive Control

Automatically detects system-specific parameters from trajectory data:
- Characteristic velocity scale (c parameter for Lorentz transforms)
- Oscillation threshold (for lightlike observer)
- Natural time scale (for damping)

This enables the framework to adapt to:
- Different robot types (slow vs fast)
- Different environments (large vs small workspace)
- Different tasks (precise vs coarse control)

Key insight: The framework should be scale-invariant.
The "c" parameter and thresholds should emerge from data, not be hardcoded.
"""

from typing import List, Tuple, Dict

import numpy as np


def estimate_characteristic_velocity(
    trajectory_history: List[np.ndarray],
    min_samples: int = 10,
) -> float:
    """
    Estimate characteristic velocity scale from trajectory.

    This determines the natural "speed limit" of the system,
    which becomes the "c" parameter in Lorentz transformations.

    Parameters
    ----------
    trajectory_history : list of ndarray
        Recent state trajectory
    min_samples : int
        Minimum samples needed for reliable estimate

    Returns
    -------
    c_estimate : float
        Characteristic velocity (speed of light analog)

    Algorithm
    ---------
    1. Compute velocities: v[t] = (x[t] - x[t-1]) / dt
    2. Characteristic velocity = 95th percentile of |v|
       (High percentile to capture "fast" motion, but not outliers)

    Examples
    --------
    >>> # Fast robot
    >>> trajectory = [np.array([i * 2.0, 0.0]) for i in range(20)]
    >>> c = estimate_characteristic_velocity(trajectory)
    >>> c  # Should be around 2.0
    2.0

    >>> # Slow robot
    >>> trajectory = [np.array([i * 0.1, 0.0]) for i in range(20)]
    >>> c = estimate_characteristic_velocity(trajectory)
    >>> c  # Should be around 0.1
    0.1
    """
    if len(trajectory_history) < min_samples:
        return 1.0  # Default fallback

    # Compute velocities
    velocities = [
        trajectory_history[i] - trajectory_history[i - 1]
        for i in range(1, len(trajectory_history))
    ]

    # Velocity magnitudes
    velocity_mags = [np.linalg.norm(v) for v in velocities]

    if len(velocity_mags) == 0:
        return 1.0

    # Characteristic velocity = 95th percentile (captures typical fast motion)
    c_estimate = np.percentile(velocity_mags, 95)

    # Ensure minimum value (avoid division by zero)
    c_estimate = max(c_estimate, 0.01)

    return float(c_estimate)


def estimate_oscillation_threshold(
    trajectory_history: List[np.ndarray],
    min_samples: int = 10,
) -> float:
    """
    Estimate oscillation detection threshold from trajectory variance.

    Systems with high velocity variance are more likely to oscillate,
    so we need a lower threshold. Stable systems can use higher threshold.

    Parameters
    ----------
    trajectory_history : list of ndarray
        Recent state trajectory
    min_samples : int
        Minimum samples for reliable estimate

    Returns
    -------
    threshold : float
        Oscillation detection threshold in [0, 1]

    Algorithm
    ---------
    1. Compute velocity variance: Var(v)
    2. High variance → lower threshold (more sensitive)
    3. Low variance → higher threshold (less sensitive)
    4. Threshold = 0.9 - clip(Var(v) / max_var, 0, 0.3)

    Notes
    -----
    Default threshold is 0.92 for most systems.
    This adapts it based on observed variability.
    """
    if len(trajectory_history) < min_samples:
        return 0.92  # Default

    # Compute velocities
    velocities = [
        trajectory_history[i] - trajectory_history[i - 1]
        for i in range(1, len(trajectory_history))
    ]

    if len(velocities) < 2:
        return 0.92

    # Velocity magnitudes
    velocity_mags = np.array([np.linalg.norm(v) for v in velocities])

    # Variance of velocity magnitude
    velocity_variance = np.var(velocity_mags)

    # Normalize by mean (coefficient of variation)
    mean_velocity = np.mean(velocity_mags)
    if mean_velocity > 1e-6:
        normalized_variance = velocity_variance / (mean_velocity**2)
    else:
        normalized_variance = 0.0

    # Map to threshold
    # High variance → lower threshold (0.7-0.9)
    # Low variance → higher threshold (0.9-0.95)
    variability_factor = np.clip(normalized_variance, 0.0, 1.0)
    threshold = 0.95 - 0.25 * variability_factor

    return float(np.clip(threshold, 0.7, 0.95))


def estimate_natural_timescale(
    trajectory_history: List[np.ndarray],
    min_samples: int = 10,
) -> float:
    """
    Estimate natural time scale of the system.

    This determines how quickly the system responds to changes,
    which affects damping rates and control gains.

    Parameters
    ----------
    trajectory_history : list of ndarray
        Recent state trajectory
    min_samples : int
        Minimum samples needed

    Returns
    -------
    tau : float
        Natural time scale (in steps)

    Algorithm
    ---------
    Compute autocorrelation decay time:
    - Fast systems: Quick decorrelation → small tau
    - Slow systems: Slow decorrelation → large tau
    """
    if len(trajectory_history) < min_samples:
        return 5.0  # Default

    # Compute autocorrelation
    from .eigen_decomposition import compute_autocorrelation

    # Convert to velocity signal
    velocities = [
        trajectory_history[i] - trajectory_history[i - 1]
        for i in range(1, len(trajectory_history))
    ]

    if len(velocities) < 3:
        return 5.0

    signal = np.array(velocities)
    autocorr = compute_autocorrelation(signal, max_lag=min(10, len(velocities) - 1))

    # Find where autocorrelation drops below 0.5 (half-life)
    tau = 5.0  # Default
    for lag in range(1, len(autocorr)):
        if autocorr[lag] < 0.5:
            tau = float(lag)
            break

    return max(tau, 1.0)  # At least 1 step


def detect_system_parameters(
    trajectory_history: List[np.ndarray],
    min_samples: int = 15,
) -> Dict[str, float]:
    """
    Detect all system parameters from trajectory.

    Convenience function that estimates all parameters at once.

    Parameters
    ----------
    trajectory_history : list of ndarray
        Recent state trajectory
    min_samples : int
        Minimum samples for reliable detection

    Returns
    -------
    params : dict
        Dictionary containing:
        - 'c': Characteristic velocity (Lorentz parameter)
        - 'oscillation_threshold': Threshold for oscillation detection
        - 'tau': Natural time scale
        - 'eta_recommended': Recommended learning rate

    Examples
    --------
    >>> trajectory = [np.array([i * 0.5, 0.0]) for i in range(30)]
    >>> params = detect_system_parameters(trajectory)
    >>> params['c']  # Characteristic velocity
    >>> params['oscillation_threshold']  # Adaptive threshold
    >>> params['eta_recommended']  # Suggested learning rate
    """
    if len(trajectory_history) < min_samples:
        # Return defaults
        return {
            "c": 1.0,
            "oscillation_threshold": 0.92,
            "tau": 5.0,
            "eta_recommended": 0.1,
            "confidence": 0.0,  # Low confidence with few samples
        }

    # Detect parameters
    c = estimate_characteristic_velocity(trajectory_history)
    threshold = estimate_oscillation_threshold(trajectory_history)
    tau = estimate_natural_timescale(trajectory_history)

    # Recommended learning rate based on time scale
    # Faster systems → larger eta (respond quickly)
    # Slower systems → smaller eta (be cautious)
    eta_recommended = 1.0 / max(tau, 1.0)
    eta_recommended = np.clip(eta_recommended, 0.05, 0.5)

    # Confidence based on number of samples
    confidence = min(1.0, (len(trajectory_history) - min_samples) / 50.0)

    return {
        "c": float(c),
        "oscillation_threshold": float(threshold),
        "tau": float(tau),
        "eta_recommended": float(eta_recommended),
        "confidence": float(confidence),
    }


def adaptive_control_step(
    current_state: np.ndarray,
    target: np.ndarray,
    trajectory_history: List[np.ndarray],
    eta: float = None,
    auto_tune: bool = True,
    min_samples_for_tuning: int = 15,
) -> Tuple[np.ndarray, Dict]:
    """
    Control step with automatic parameter adaptation.

    Analyzes trajectory to detect system parameters and adjusts
    control accordingly.

    Parameters
    ----------
    current_state : ndarray
        Current state
    target : ndarray
        Target state
    trajectory_history : list of ndarray
        Recent trajectory (for parameter detection)
    eta : float, optional
        Learning rate (if None, auto-detect)
    auto_tune : bool
        Whether to auto-tune parameters
    min_samples_for_tuning : int
        Minimum samples before auto-tuning

    Returns
    -------
    new_state : ndarray
        Updated state
    diagnostics : dict
        Diagnostic information including detected parameters

    Examples
    --------
    >>> current = np.array([5.0, 5.0])
    >>> target = np.array([0.0, 0.0])
    >>> history = [np.array([10.0 - i, 10.0 - i]) for i in range(20)]
    >>> new_state, info = adaptive_control_step(current, target, history)
    >>> info['c']  # Auto-detected characteristic velocity
    >>> info['eta_used']  # Learning rate used
    """
    # Detect system parameters
    if auto_tune and len(trajectory_history) >= min_samples_for_tuning:
        params = detect_system_parameters(trajectory_history)

        # Use detected eta if not provided
        if eta is None:
            eta = params["eta_recommended"]

        detected = True
    else:
        # Use defaults
        params = {
            "c": 1.0,
            "oscillation_threshold": 0.92,
            "tau": 5.0,
            "eta_recommended": 0.1,
            "confidence": 0.0,
        }
        if eta is None:
            eta = 0.1

        detected = False

    # Simple gradient descent toward target
    direction = target - current_state
    new_state = current_state + eta * direction

    # Diagnostics
    diagnostics = {
        **params,
        "eta_used": float(eta),
        "parameters_detected": detected,
        "n_samples": len(trajectory_history),
    }

    return new_state, diagnostics


def renormalize_lorentz_state(
    timelike: float,
    spacelike: float,
    old_c: float,
    new_c: float,
) -> Tuple[float, float]:
    """
    Renormalize Lorentz coordinates when c parameter changes.

    When we detect a new characteristic velocity, we need to
    rescale the Lorentz coordinates to maintain consistency.

    Parameters
    ----------
    timelike : float
        Original timelike component
    spacelike : float
        Original spacelike component
    old_c : float
        Previous c parameter
    new_c : float
        New detected c parameter

    Returns
    -------
    timelike_new : float
        Renormalized timelike component
    spacelike_new : float
        Renormalized spacelike component

    Notes
    -----
    Invariant preserved: ds² = t² - x²/c²

    When c changes: c' = α·c
    Need to rescale: x' = α·x, t' = t (keep timelike fixed)
    Then: ds'² = t'² - x'²/c'² = t² - (α·x)²/(α·c)² = t² - x²/c² = ds²
    """
    if abs(old_c) < 1e-9 or abs(new_c) < 1e-9:
        return timelike, spacelike

    # Scaling factor
    alpha = new_c / old_c

    # Rescale spacelike to maintain invariant
    # ds² = t² - x²/c² should remain constant
    # With new c: ds² = t² - x'²/c'²
    # So: x'²/c'² = x²/c² → x' = x·(c'/c) = x·alpha
    spacelike_new = spacelike * alpha

    return timelike, spacelike_new
