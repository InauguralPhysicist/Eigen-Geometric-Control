"""
Coherent/Incoherent Signal Decomposition

Separates observations into:
- Coherent component: Predictable, smooth, correlated over time
- Incoherent component: Random noise, uncorrelated

Key insight from quantum mechanics:
- Coherent states: Preserve phase relationships (waves)
- Incoherent states: Random phases (particles, noise)

Algorithm uses autocorrelation to detect coherence:
- High autocorrelation → coherent signal
- Low autocorrelation → incoherent noise

This enables:
- Selective processing (weak measurements on coherent part only)
- Noise filtering (ignore incoherent part)
- Computational savings (don't process noise)
"""

from typing import List, Tuple

import numpy as np


def compute_autocorrelation(
    signal: np.ndarray,
    max_lag: int = 5,
) -> np.ndarray:
    """
    Compute autocorrelation function of a signal.

    Autocorrelation measures how similar a signal is to a time-shifted
    version of itself. High autocorrelation → coherent (predictable).

    Parameters
    ----------
    signal : ndarray, shape (N,) or (N, d)
        Time series signal (1D or multi-dimensional)
    max_lag : int
        Maximum time lag to compute

    Returns
    -------
    autocorr : ndarray, shape (max_lag+1,)
        Autocorrelation at each lag (normalized to [0, 1])

    Notes
    -----
    For a signal x[t], autocorrelation at lag k is:
        R(k) = E[x[t] · x[t-k]] / E[x[t]²]

    Examples
    --------
    >>> # Perfectly coherent signal (sine wave)
    >>> t = np.linspace(0, 10, 100)
    >>> signal = np.sin(t)
    >>> autocorr = compute_autocorrelation(signal, max_lag=10)
    >>> autocorr[0]  # Always 1.0 at lag 0
    1.0
    >>> autocorr[1]  # High for coherent signal
    > 0.9
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    N, d = signal.shape
    autocorr = np.zeros(max_lag + 1)

    # Normalize signal (zero mean)
    signal_centered = signal - np.mean(signal, axis=0)

    # Variance at lag 0
    var = np.mean(np.sum(signal_centered**2, axis=1))

    if var < 1e-12:
        # Signal is constant - perfect autocorrelation
        return np.ones(max_lag + 1)

    # Compute autocorrelation at each lag
    for lag in range(max_lag + 1):
        if lag >= N:
            autocorr[lag] = 0.0
            continue

        # Correlation between x[t] and x[t-lag]
        corr = np.mean(np.sum(signal_centered[lag:] * signal_centered[: N - lag], axis=1))

        autocorr[lag] = corr / var

    return autocorr


def coherence_score(
    trajectory: List[np.ndarray],
    threshold_lag: int = 3,
) -> float:
    """
    Compute coherence score from trajectory autocorrelation.

    High score → signal is coherent (predictable, wave-like)
    Low score → signal is incoherent (random, noise-like)

    Parameters
    ----------
    trajectory : list of ndarray
        Recent state trajectory
    threshold_lag : int
        How many steps ahead we want to predict (default 3)

    Returns
    -------
    coherence : float
        Coherence score in [0, 1], where:
        - 1.0 = perfectly coherent (predictable)
        - 0.0 = perfectly incoherent (random noise)

    Notes
    -----
    Uses average autocorrelation over first few lags as coherence measure.
    """
    if len(trajectory) < 3:
        return 0.5  # Not enough data - assume neutral

    # Convert trajectory to signal (use differences for stationarity)
    if len(trajectory) < 2:
        return 0.5

    velocities = [trajectory[i] - trajectory[i - 1] for i in range(1, len(trajectory))]
    signal = np.array(velocities)

    # Compute autocorrelation
    autocorr = compute_autocorrelation(signal, max_lag=threshold_lag)

    # Coherence = average autocorrelation (excluding lag 0)
    if len(autocorr) > 1:
        coherence = np.mean(np.abs(autocorr[1 : threshold_lag + 1]))
    else:
        coherence = 0.0

    return float(np.clip(coherence, 0.0, 1.0))


def decompose_signal(
    observation: np.ndarray,
    trajectory_history: List[np.ndarray],
    coherence_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Decompose observation into coherent and incoherent components.

    Uses trajectory history to determine what's predictable (coherent)
    vs what's random noise (incoherent).

    Parameters
    ----------
    observation : ndarray
        Current observation (potentially noisy)
    trajectory_history : list of ndarray
        Recent trajectory for coherence analysis
    coherence_threshold : float
        Threshold for coherent/incoherent classification

    Returns
    -------
    coherent_component : ndarray
        Predictable part of observation
    incoherent_component : ndarray
        Random noise part of observation
    coherence_score : float
        Overall coherence of the trajectory

    Algorithm
    ---------
    1. Compute coherence from trajectory history
    2. If coherent (score > threshold):
       - Predict next state from trajectory
       - Coherent = predicted state
       - Incoherent = observation - predicted
    3. If incoherent (score < threshold):
       - Coherent = 0
       - Incoherent = observation

    Examples
    --------
    >>> # Smooth trajectory → observation is coherent
    >>> history = [np.array([0.0]), np.array([1.0]), np.array([2.0])]
    >>> obs = np.array([3.0])  # Continues trend
    >>> coh, incoh, score = decompose_signal(obs, history)
    >>> score > 0.5  # Should be coherent
    True
    """
    if len(trajectory_history) < 2:
        # Not enough history - treat as incoherent
        return np.zeros_like(observation), observation.copy(), 0.0

    # 1. Compute coherence score
    coh_score = coherence_score(trajectory_history)

    # 2. Predict next state (linear extrapolation)
    recent_states = trajectory_history[-3:]  # Use last 3 states
    if len(recent_states) >= 2:
        # Linear prediction: x[t+1] = x[t] + (x[t] - x[t-1])
        velocity = recent_states[-1] - recent_states[-2]
        predicted = recent_states[-1] + velocity
    else:
        predicted = recent_states[-1]

    # 3. Decompose based on coherence
    if coh_score > coherence_threshold:
        # Signal is coherent
        # Coherent component = predicted state (smooth trend)
        # Incoherent component = deviation from prediction (noise)
        coherent = predicted
        incoherent = observation - predicted
    else:
        # Signal is incoherent (random)
        # Treat entire observation as noise
        coherent = np.zeros_like(observation)
        incoherent = observation.copy()

    return coherent, incoherent, float(coh_score)


def filtered_observation(
    observation: np.ndarray,
    trajectory_history: List[np.ndarray],
    coherence_threshold: float = 0.5,
    noise_suppression: float = 0.8,
) -> Tuple[np.ndarray, dict]:
    """
    Get filtered observation with noise suppression.

    Decomposes signal and suppresses incoherent (noise) component.

    Parameters
    ----------
    observation : ndarray
        Current noisy observation
    trajectory_history : list of ndarray
        Recent trajectory
    coherence_threshold : float
        Threshold for coherence classification
    noise_suppression : float
        How much to suppress incoherent component (0=none, 1=complete)

    Returns
    -------
    filtered_obs : ndarray
        Cleaned observation with noise suppressed
    diagnostics : dict
        Diagnostic information about decomposition

    Examples
    --------
    >>> # Noisy observation on smooth trajectory
    >>> history = [np.array([i * 1.0]) for i in range(5)]
    >>> noisy_obs = np.array([5.0 + 0.5])  # 5.0 + noise
    >>> filtered, info = filtered_observation(noisy_obs, history)
    >>> info['coherence_score'] > 0.5  # Should detect coherence
    True
    >>> np.linalg.norm(filtered - 5.0) < np.linalg.norm(noisy_obs - 5.0)  # Noise reduced
    True
    """
    # Decompose
    coherent, incoherent, coh_score = decompose_signal(
        observation, trajectory_history, coherence_threshold
    )

    # Suppress incoherent component
    filtered = coherent + incoherent * (1.0 - noise_suppression)

    # Diagnostics
    noise_reduction = float(np.linalg.norm(observation - filtered))

    diagnostics = {
        "coherence_score": coh_score,
        "coherent_magnitude": float(np.linalg.norm(coherent)),
        "incoherent_magnitude": float(np.linalg.norm(incoherent)),
        "noise_suppression": noise_suppression,
        "noise_reduction": noise_reduction,
        "is_coherent": coh_score > coherence_threshold,
    }

    return filtered, diagnostics


def coherent_control_step(
    current_state: np.ndarray,
    target: np.ndarray,
    observation: np.ndarray,
    trajectory_history: List[np.ndarray],
    eta: float = 0.1,
    coherence_threshold: float = 0.5,
    noise_suppression: float = 0.8,
) -> Tuple[np.ndarray, dict]:
    """
    Control step with coherent/incoherent decomposition.

    Uses decomposition to filter noise before computing control.

    Parameters
    ----------
    current_state : ndarray
        Current state estimate
    target : ndarray
        Target state
    observation : ndarray
        Current observation (potentially noisy)
    trajectory_history : list of ndarray
        Recent trajectory
    eta : float
        Learning rate
    coherence_threshold : float
        Coherence classification threshold
    noise_suppression : float
        How much to suppress noise

    Returns
    -------
    new_state : ndarray
        Updated state after control step
    diagnostics : dict
        Diagnostic information

    Algorithm
    ---------
    1. Decompose observation into coherent + incoherent
    2. Suppress incoherent (noise) component
    3. Use filtered observation for state update
    4. Move toward target based on filtered state
    """
    # Filter observation
    filtered_obs, decomp_info = filtered_observation(
        observation,
        trajectory_history,
        coherence_threshold,
        noise_suppression,
    )

    # Update state estimate with filtered observation
    # Blend current state with filtered observation
    blend = 0.5  # Weight for observation vs state
    updated_state = blend * filtered_obs + (1 - blend) * current_state

    # Control: move toward target
    direction = target - updated_state
    new_state = updated_state + eta * direction

    # Diagnostics
    diagnostics = {
        **decomp_info,
        "filtered_obs": filtered_obs,
        "noise_reduction": float(np.linalg.norm(observation - filtered_obs)),
    }

    return new_state, diagnostics
