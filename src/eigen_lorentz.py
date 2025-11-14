# -*- coding: utf-8 -*-
"""
Eigen: Lorentz Transformation Framework

Implements Lorentz boosts for geometric control and stereo vision fusion.

Key concepts:
- ds² = S² - C² has Minkowski signature (timelike/spacelike)
- Lorentz boosts preserve ds² invariant
- Unifies discrete (XOR) and continuous (gradient) approaches
- Enables multi-frame sensor fusion

Mathematical foundation:
    Λ(β) = [[γ, -γβ],      where γ = 1/√(1 - β²)
            [-γβ, γ]]

    Invariant: ds² = t² - x² (preserved under boosts)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class LorentzState:
    """
    State in Lorentz framework

    Attributes
    ----------
    timelike : float
        Timelike component (stability, settled behavior)
    spacelike : float
        Spacelike component (change, motion)
    ds2 : float
        Invariant interval: timelike² - spacelike²
    """

    timelike: float
    spacelike: float
    ds2: float

    def __post_init__(self):
        """Verify invariant"""
        computed_ds2 = self.timelike**2 - self.spacelike**2
        if not np.isclose(self.ds2, computed_ds2, rtol=1e-10):
            raise ValueError(
                f"Invariant violation: ds²={self.ds2:.6f} " f"but t²-x²={computed_ds2:.6f}"
            )


def rapidity_from_beta(beta: float) -> float:
    """
    Convert velocity parameter to rapidity

    Parameters
    ----------
    beta : float
        Velocity parameter (0 ≤ β < 1)

    Returns
    -------
    theta : float
        Rapidity (θ = arctanh(β))

    Raises
    ------
    ValueError
        If beta is outside valid range [0, 1)
    """
    if not 0 <= beta < 1:
        raise ValueError(f"beta must be in [0, 1), got {beta}")

    if beta == 0:
        return 0.0

    return np.arctanh(beta)


def beta_from_rapidity(theta: float) -> float:
    """
    Convert rapidity to velocity parameter

    Parameters
    ----------
    theta : float
        Rapidity

    Returns
    -------
    beta : float
        Velocity parameter (β = tanh(θ))
    """
    return np.tanh(theta)


def lorentz_factor(beta: float) -> float:
    """
    Compute Lorentz factor γ

    Parameters
    ----------
    beta : float
        Velocity parameter (0 ≤ β < 1)

    Returns
    -------
    gamma : float
        Lorentz factor γ = 1/√(1 - β²)

    Raises
    ------
    ValueError
        If beta is outside valid range [0, 1)
    """
    if not 0 <= beta < 1:
        raise ValueError(f"beta must be in [0, 1), got {beta}")

    if beta == 0:
        return 1.0

    return 1.0 / np.sqrt(1.0 - beta**2)


def lorentz_boost_matrix(beta: float) -> np.ndarray:
    """
    Construct Lorentz boost transformation matrix

    Parameters
    ----------
    beta : float
        Velocity parameter (0 ≤ β < 1)

    Returns
    -------
    Lambda : ndarray, shape (2, 2)
        Lorentz transformation matrix:
        [[γ, -γβ],
         [-γβ, γ]]

    Examples
    --------
    >>> Λ = lorentz_boost_matrix(0.6)
    >>> np.allclose(Λ @ Λ.T, np.eye(2))  # Not orthogonal!
    False
    >>> # But preserves metric with signature (+, -)
    """
    gamma = lorentz_factor(beta)
    beta_gamma = beta * gamma

    return np.array([[gamma, -beta_gamma], [-beta_gamma, gamma]])


def lorentz_boost_from_rapidity(theta: float) -> np.ndarray:
    """
    Construct Lorentz boost from rapidity (more numerically stable)

    Parameters
    ----------
    theta : float
        Rapidity parameter

    Returns
    -------
    Lambda : ndarray, shape (2, 2)
        Lorentz transformation matrix:
        [[cosh(θ), -sinh(θ)],
         [-sinh(θ), cosh(θ)]]
    """
    cosh_theta = np.cosh(theta)
    sinh_theta = np.sinh(theta)

    return np.array([[cosh_theta, -sinh_theta], [-sinh_theta, cosh_theta]])


def apply_boost(state_vector: np.ndarray, beta: float) -> np.ndarray:
    """
    Apply Lorentz boost to state vector

    Parameters
    ----------
    state_vector : ndarray, shape (2,)
        State in original frame [timelike, spacelike]
    beta : float
        Boost velocity (0 ≤ β < 1)

    Returns
    -------
    boosted_state : ndarray, shape (2,)
        State in boosted frame

    Examples
    --------
    >>> state = np.array([3.0, 2.0])  # ds² = 9 - 4 = 5
    >>> boosted = apply_boost(state, beta=0.5)
    >>> ds2_orig = state[0]**2 - state[1]**2
    >>> ds2_boost = boosted[0]**2 - boosted[1]**2
    >>> np.isclose(ds2_orig, ds2_boost)  # Invariant preserved
    True
    """
    if len(state_vector) != 2:
        raise ValueError(f"State vector must have length 2, got {len(state_vector)}")

    Lambda = lorentz_boost_matrix(beta)
    return Lambda @ state_vector


def create_lorentz_state(timelike: float, spacelike: float) -> LorentzState:
    """
    Create LorentzState from components

    Parameters
    ----------
    timelike : float
        Timelike component (stability)
    spacelike : float
        Spacelike component (change)

    Returns
    -------
    state : LorentzState
        State with computed invariant
    """
    ds2 = timelike**2 - spacelike**2
    return LorentzState(timelike=timelike, spacelike=spacelike, ds2=ds2)


def boost_lorentz_state(state: LorentzState, beta: float) -> LorentzState:
    """
    Apply Lorentz boost to LorentzState

    Parameters
    ----------
    state : LorentzState
        Original state
    beta : float
        Boost velocity

    Returns
    -------
    boosted_state : LorentzState
        Transformed state (with same ds² invariant)
    """
    vec = np.array([state.timelike, state.spacelike])
    boosted_vec = apply_boost(vec, beta)

    return LorentzState(
        timelike=boosted_vec[0], spacelike=boosted_vec[1], ds2=state.ds2  # Invariant preserved
    )


def stereo_to_lorentz(left_view: float, right_view: float) -> Tuple[float, float, float]:
    """
    Convert stereo views to Lorentz coordinates

    Maps stereo vision to Lorentz framework:
    - Timelike: Average of views (stable, agreed-upon)
    - Spacelike: Disparity between views (change, disagreement)

    Parameters
    ----------
    left_view : float
        Left eye measurement
    right_view : float
        Right eye measurement

    Returns
    -------
    timelike : float
        Timelike component (L + R) / 2
    spacelike : float
        Spacelike component (L - R) / 2
    ds2 : float
        Invariant L·R

    Notes
    -----
    Preserves information: (L, R) ↔ (t, x)
    - L = t + x
    - R = t - x
    - ds² = t² - x² = L·R
    """
    timelike = (left_view + right_view) / 2.0
    spacelike = (left_view - right_view) / 2.0
    ds2 = left_view * right_view

    return timelike, spacelike, ds2


def lorentz_to_stereo(timelike: float, spacelike: float) -> Tuple[float, float]:
    """
    Convert Lorentz coordinates back to stereo views

    Parameters
    ----------
    timelike : float
        Timelike component
    spacelike : float
        Spacelike component

    Returns
    -------
    left_view : float
        Reconstructed left view = t + x
    right_view : float
        Reconstructed right view = t - x
    """
    left_view = timelike + spacelike
    right_view = timelike - spacelike

    return left_view, right_view


def change_stability_to_lorentz(C: int, S: int) -> LorentzState:
    """
    Convert (C, S) metrics to Lorentz state

    Maps geometric control metrics to relativistic framework:
    - C (Change) → spacelike (motion, exploring)
    - S (Stability) → timelike (rest, settled)
    - ds² = S² - C² → Lorentz invariant

    Parameters
    ----------
    C : int
        Change count (components in motion)
    S : int
        Stability count (components at rest)

    Returns
    -------
    state : LorentzState
        Equivalent Lorentz representation
    """
    return LorentzState(timelike=float(S), spacelike=float(C), ds2=float(S * S - C * C))


def disparity_to_rapidity(disparity: float, max_disparity: float) -> float:
    """
    Convert stereo disparity to Lorentz rapidity

    Maps depth perception to relativistic velocity:
    - Small disparity (far object) → small rapidity → minimal boost
    - Large disparity (near object) → large rapidity → significant boost

    Parameters
    ----------
    disparity : float
        Horizontal disparity in pixels/units (0 ≤ d ≤ max_disparity)
    max_disparity : float
        Maximum possible disparity (defines β = 1 limit)

    Returns
    -------
    theta : float
        Rapidity parameter

    Examples
    --------
    >>> # Far object (small disparity)
    >>> rapidity_to_rapidity(5, 100)
    0.05004...

    >>> # Near object (large disparity)
    >>> disparity_to_rapidity(90, 100)
    1.472...
    """
    if disparity < 0 or disparity >= max_disparity:
        raise ValueError(
            f"Disparity must be in [0, max_disparity), " f"got {disparity} with max {max_disparity}"
        )

    beta = disparity / max_disparity

    if beta >= 0.999:  # Numerical stability near β = 1
        beta = 0.999

    return rapidity_from_beta(beta)


def verify_lorentz_invariance(state: np.ndarray, beta: float, rtol: float = 1e-10) -> bool:
    """
    Verify that Lorentz boost preserves ds² invariant

    Parameters
    ----------
    state : ndarray, shape (2,)
        Original state [timelike, spacelike]
    beta : float
        Boost velocity
    rtol : float
        Relative tolerance for invariance check

    Returns
    -------
    is_invariant : bool
        True if ds² is preserved within tolerance

    Examples
    --------
    >>> state = np.array([5.0, 3.0])
    >>> verify_lorentz_invariance(state, beta=0.8)
    True
    """
    ds2_original = state[0] ** 2 - state[1] ** 2

    boosted = apply_boost(state, beta)
    ds2_boosted = boosted[0] ** 2 - boosted[1] ** 2

    return np.isclose(ds2_original, ds2_boosted, rtol=rtol)


def regime_classification(ds2: float, eps: float = 1e-10) -> str:
    """
    Classify regime based on invariant

    Parameters
    ----------
    ds2 : float
        Lorentz invariant
    eps : float
        Tolerance for light-like boundary

    Returns
    -------
    regime : str
        'spacelike', 'lightlike', or 'timelike'

    Notes
    -----
    - Spacelike (ds² < 0): Change dominates, exploring, motion
    - Lightlike (ds² = 0): Boundary, critical transition
    - Timelike (ds² > 0): Stability dominates, settled, at rest
    """
    if ds2 < -eps:
        return "spacelike"
    elif ds2 > eps:
        return "timelike"
    else:
        return "lightlike"


def proper_time(ds2: float) -> Optional[float]:
    """
    Compute proper time for timelike trajectories

    Parameters
    ----------
    ds2 : float
        Lorentz invariant

    Returns
    -------
    tau : float or None
        Proper time τ = √(ds²) if timelike, else None

    Notes
    -----
    Proper time measures "aging" along trajectory.
    Only defined for timelike paths (ds² > 0).
    """
    if ds2 > 0:
        return np.sqrt(ds2)
    else:
        return None


def proper_distance(ds2: float) -> Optional[float]:
    """
    Compute proper distance for spacelike separations

    Parameters
    ----------
    ds2 : float
        Lorentz invariant

    Returns
    -------
    sigma : float or None
        Proper distance σ = √(-ds²) if spacelike, else None

    Notes
    -----
    Proper distance measures spatial separation.
    Only defined for spacelike paths (ds² < 0).
    """
    if ds2 < 0:
        return np.sqrt(-ds2)
    else:
        return None


def estimate_target_velocity(
    target_history: list,
    dt: float = 1.0,
    max_velocity: float = 10.0,
) -> np.ndarray:
    """
    Estimate target velocity from position history.

    Uses recent trajectory to estimate current velocity vector.
    This is needed for Lorentz boost to target's rest frame.

    Parameters
    ----------
    target_history : list of ndarray
        Recent target positions [oldest, ..., newest]
    dt : float
        Time step between observations (default 1.0)
    max_velocity : float
        Maximum allowed velocity magnitude (default 10.0 for realistic tracking)

    Returns
    -------
    velocity : ndarray
        Estimated velocity vector

    Notes
    -----
    Uses finite differences: v ≈ (x[t] - x[t-1]) / dt
    If history is insufficient, returns zero velocity.
    Velocities are clamped to max_velocity to avoid numerical issues.

    Examples
    --------
    >>> history = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
    >>> v = estimate_target_velocity(history, dt=1.0)
    >>> np.allclose(v, [1.0, 0.0])
    True
    """
    if len(target_history) < 2:
        # Not enough history - assume stationary
        if len(target_history) == 1:
            return np.zeros_like(target_history[0])
        else:
            return np.zeros(2)  # Default to 2D

    # Use last two positions for velocity estimate
    x_current = target_history[-1]
    x_previous = target_history[-2]

    # Finite difference
    velocity = (x_current - x_previous) / dt

    # Clamp to max velocity (avoid numerical issues)
    v_mag = np.linalg.norm(velocity)
    if v_mag > max_velocity:
        velocity = velocity * (max_velocity / v_mag)

    return velocity


def inverse_boost(
    state_vector: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Apply inverse Lorentz boost (transform from boosted frame back to lab frame).

    If Λ(β) transforms lab → boosted, then Λ(-β) transforms boosted → lab.

    Parameters
    ----------
    state_vector : ndarray, shape (2,)
        State in boosted frame [timelike, spacelike]
    beta : float
        Boost velocity that was originally applied

    Returns
    -------
    original_state : ndarray, shape (2,)
        State in original lab frame

    Notes
    -----
    Lorentz boosts have the property: Λ(β)⁻¹ = Λ(-β)
    We implement this directly to avoid negative beta validation issues.

    Examples
    --------
    >>> state = np.array([3.0, 2.0])
    >>> boosted = apply_boost(state, beta=0.5)
    >>> recovered = inverse_boost(boosted, beta=0.5)
    >>> np.allclose(state, recovered)
    True
    """
    if len(state_vector) != 2:
        raise ValueError(f"State vector must have length 2, got {len(state_vector)}")

    # Inverse boost: Λ(-β) with same gamma
    gamma = lorentz_factor(beta)
    beta_gamma = beta * gamma

    # Inverse matrix (negate beta)
    Lambda_inv = np.array([[gamma, beta_gamma], [beta_gamma, gamma]])

    return Lambda_inv @ state_vector


def moving_target_control_step(
    robot_state: np.ndarray,
    target_position: np.ndarray,
    target_history: list,
    eta: float = 0.1,
    dt: float = 1.0,
    lookahead_time: float = 1.0,
) -> tuple:
    """
    Control step with predictive tracking for moving targets.

    Algorithm:
    1. Estimate target velocity from history
    2. Predict where target will be after lookahead_time
    3. Move toward predicted position
    4. For very fast targets, clamp beta for Lorentz boost diagnostics

    This approach automatically handles:
    - Moving targets (predicts future position)
    - Stationary targets (prediction = current position)
    - Accelerating targets (velocity estimate updates each step)

    Parameters
    ----------
    robot_state : ndarray
        Current robot position (N-dimensional)
    target_position : ndarray
        Current target position (N-dimensional)
    target_history : list of ndarray
        Recent target positions for velocity estimation
    eta : float
        Learning rate / control gain
    dt : float
        Time step for velocity estimation
    lookahead_time : float
        How far ahead to predict target position (default 1.0)

    Returns
    -------
    new_robot_state : ndarray
        Updated robot position after control step
    velocity_estimate : ndarray
        Estimated target velocity (for diagnostics)
    boost_beta : float
        Velocity parameter for Lorentz boost diagnostics

    Examples
    --------
    >>> robot = np.array([5.0, 0.0])
    >>> target = np.array([2.0, 0.0])
    >>> history = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
    >>> new_state, velocity, beta = moving_target_control_step(robot, target, history)
    >>> # Robot should move toward where target will be
    """
    # 1. Estimate target velocity
    target_velocity = estimate_target_velocity(target_history, dt=dt)

    # 2. Predict future target position
    predicted_target = target_position + target_velocity * lookahead_time

    # 3. Move toward predicted position (gradient descent)
    direction = predicted_target - robot_state
    new_state = robot_state + eta * direction

    # 4. Compute beta for diagnostics (Lorentz boost parameter)
    v_mag = np.linalg.norm(target_velocity)
    beta = min(v_mag, 0.99)  # Clamp to subluminal

    return new_state, target_velocity, beta
