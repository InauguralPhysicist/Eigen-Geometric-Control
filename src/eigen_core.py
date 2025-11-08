# -*- coding: utf-8 -*-
"""
Eigen: Geometric Control Framework - Core Functions

Provides basic geometric computations for 2-DOF planar arm with
comprehensive error handling and documentation.
"""

from typing import Dict, Tuple, Union

import numpy as np


def forward_kinematics(
    theta1: float, theta2: float, L1: float = 0.9, L2: float = 0.9
) -> Tuple[float, float]:
    """
    Compute end-effector position from joint angles.

    Uses standard 2D forward kinematics for planar manipulator:
    x = L1*cos(θ₁) + L2*cos(θ₁ + θ₂)
    y = L1*sin(θ₁) + L2*sin(θ₁ + θ₂)

    Parameters
    ----------
    theta1 : float
        Shoulder angle in radians
    theta2 : float
        Elbow angle in radians
    L1 : float, optional
        First link length in meters (default: 0.9)
    L2 : float, optional
        Second link length in meters (default: 0.9)

    Returns
    -------
    x : float
        End-effector x-coordinate in meters
    y : float
        End-effector y-coordinate in meters

    Raises
    ------
    ValueError
        If link lengths are non-positive
    TypeError
        If inputs are not numeric

    Examples
    --------
    >>> x, y = forward_kinematics(0, 0, L1=1.0, L2=1.0)
    >>> print(f"Position: ({x:.2f}, {y:.2f})")
    Position: (2.00, 0.00)
    """
    # Type validation
    if not all(isinstance(x, (int, float, np.number)) for x in [theta1, theta2, L1, L2]):
        raise TypeError("All inputs must be numeric")

    # Physical constraint validation
    if L1 <= 0 or L2 <= 0:
        raise ValueError(f"Link lengths must be positive (got L1={L1}, L2={L2})")

    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)

    return float(x), float(y)


def jacobian(theta1: float, theta2: float, L1: float = 0.9, L2: float = 0.9) -> np.ndarray:
    """
    Compute Jacobian matrix mapping joint velocities to end-effector velocities.

    The Jacobian J relates joint-space velocities to task-space velocities:
    [ẋ]   [∂x/∂θ₁  ∂x/∂θ₂] [θ̇₁]
    [ẏ] = [∂y/∂θ₁  ∂y/∂θ₂] [θ̇₂]

    Parameters
    ----------
    theta1 : float
        Shoulder angle in radians
    theta2 : float
        Elbow angle in radians
    L1 : float, optional
        First link length in meters (default: 0.9)
    L2 : float, optional
        Second link length in meters (default: 0.9)

    Returns
    -------
    J : np.ndarray
        2×2 Jacobian matrix [[∂x/∂θ₁, ∂x/∂θ₂], [∂y/∂θ₁, ∂y/∂θ₂]]

    Raises
    ------
    ValueError
        If link lengths are non-positive
    TypeError
        If inputs are not numeric

    Notes
    -----
    The Jacobian is singular at fully extended or folded positions.
    """
    # Type validation
    if not all(isinstance(x, (int, float, np.number)) for x in [theta1, theta2, L1, L2]):
        raise TypeError("All inputs must be numeric")

    # Physical constraint validation
    if L1 <= 0 or L2 <= 0:
        raise ValueError(f"Link lengths must be positive (got L1={L1}, L2={L2})")

    return np.array(
        [
            [-L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2), -L2 * np.sin(theta1 + theta2)],
            [L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2), L2 * np.cos(theta1 + theta2)],
        ]
    )


def compute_ds2(
    theta1: float,
    theta2: float,
    target: Union[np.ndarray, Tuple[float, float]],
    obstacle_center: Union[np.ndarray, Tuple[float, float]],
    obstacle_radius: float,
    Go: float = 4.0,
    lam: float = 0.02,
    L1: float = 0.9,
    L2: float = 0.9,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute geometric objective function ds².

    The objective combines three terms:
    ds² = ||p - p_target||² + G₀·max(0, r - d_obs)² + λ||θ||²

    Parameters
    ----------
    theta1 : float
        Shoulder angle in radians
    theta2 : float
        Elbow angle in radians
    target : array-like of shape (2,)
        Goal position [x_target, y_target] in meters
    obstacle_center : array-like of shape (2,)
        Obstacle position [x_obs, y_obs] in meters
    obstacle_radius : float
        Obstacle radius in meters
    Go : float, optional
        Obstacle repulsion strength (default: 4.0)
    lam : float, optional
        Regularization weight (default: 0.02)
    L1 : float, optional
        First link length in meters (default: 0.9)
    L2 : float, optional
        Second link length in meters (default: 0.9)

    Returns
    -------
    ds2_total : float
        Total objective value
    components : dict
        Dictionary with 'target_term', 'obs_term', 'reg_term', 'd_obs'

    Raises
    ------
    ValueError
        If obstacle_radius, Go, or lam are negative
        If target or obstacle_center don't have 2 elements
    """
    # Convert to numpy arrays
    target = np.asarray(target, dtype=float)
    obstacle_center = np.asarray(obstacle_center, dtype=float)

    # Shape validation
    if target.shape != (2,):
        raise ValueError(f"target must have shape (2,), got {target.shape}")
    if obstacle_center.shape != (2,):
        raise ValueError(f"obstacle_center must have shape (2,), got {obstacle_center.shape}")

    # Physical constraint validation
    if obstacle_radius < 0:
        raise ValueError(f"obstacle_radius must be non-negative, got {obstacle_radius}")
    if Go < 0:
        raise ValueError(f"Go must be non-negative, got {Go}")
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")

    # Compute current position
    x, y = forward_kinematics(theta1, theta2, L1, L2)
    pos = np.array([x, y])

    # Target term
    target_term = float(np.sum((pos - target) ** 2))

    # Obstacle term
    d_obs = float(np.linalg.norm(pos - obstacle_center))
    obs_term = float(Go * max(0.0, obstacle_radius - d_obs) ** 2)

    # Regularization term
    reg_term = float(lam * (theta1**2 + theta2**2))

    ds2_total = target_term + obs_term + reg_term

    return ds2_total, {
        "target_term": target_term,
        "obs_term": obs_term,
        "reg_term": reg_term,
        "d_obs": d_obs,
    }


def compute_gradient(
    theta1: float,
    theta2: float,
    target: Union[np.ndarray, Tuple[float, float]],
    obstacle_center: Union[np.ndarray, Tuple[float, float]],
    obstacle_radius: float,
    Go: float = 4.0,
    lam: float = 0.02,
    L1: float = 0.9,
    L2: float = 0.9,
) -> Tuple[np.ndarray, float]:
    """
    Compute gradient ∇ds² for current configuration.

    Parameters
    ----------
    theta1 : float
        Shoulder angle in radians
    theta2 : float
        Elbow angle in radians
    target : array-like of shape (2,)
        Goal position
    obstacle_center : array-like of shape (2,)
        Obstacle position
    obstacle_radius : float
        Obstacle radius
    Go : float, optional
        Obstacle repulsion strength (default: 4.0)
    lam : float, optional
        Regularization weight (default: 0.02)
    L1 : float, optional
        First link length (default: 0.9)
    L2 : float, optional
        Second link length (default: 0.9)

    Returns
    -------
    grad : np.ndarray of shape (2,)
        Gradient vector [∂ds²/∂θ₁, ∂ds²/∂θ₂]
    grad_norm : float
        Magnitude ||∇ds²||

    Notes
    -----
    The gradient vanishes at local minima. Convergence achieved when ||∇ds²|| < ε.
    """
    target = np.asarray(target, dtype=float)
    obstacle_center = np.asarray(obstacle_center, dtype=float)

    x, y = forward_kinematics(theta1, theta2, L1, L2)
    pos = np.array([x, y])
    J = jacobian(theta1, theta2, L1, L2)

    # Target gradient
    grad_target = 2.0 * J.T @ (pos - target)

    # Obstacle gradient
    d_obs = float(np.linalg.norm(pos - obstacle_center))
    if obstacle_radius - d_obs > 0:
        grad_obs = (
            Go * 2.0 * (obstacle_radius - d_obs) * (-1.0 / d_obs) * (J.T @ (pos - obstacle_center))
        )
    else:
        grad_obs = np.zeros(2)

    # Regularization gradient
    grad_reg = 2.0 * lam * np.array([theta1, theta2])

    grad = grad_target + grad_obs + grad_reg
    grad_norm = float(np.linalg.norm(grad))

    return grad, grad_norm


def compute_change_stability(
    delta: Union[np.ndarray, Tuple[float, ...]], eps_change: float = 1e-3
) -> Tuple[int, int, int]:
    """
    Compute change (C) and stability (S) counts from configuration delta.

    Parameters
    ----------
    delta : array-like
        Configuration changes
    eps_change : float, optional
        Threshold for detecting change (default: 1e-3)

    Returns
    -------
    C : int
        Number of changing components
    S : int
        Number of stable components
    ds2_CS : int
        Geometric invariant S² - C²

    Raises
    ------
    ValueError
        If delta is empty or eps_change is negative

    Notes
    -----
    ds² < 0: Space-like (motion dominates)
    ds² = 0: Light-like (boundary)
    ds² > 0: Time-like (stability dominates)
    """
    try:
        delta = np.asarray(delta, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"delta must be array-like: {e}")

    if delta.size == 0:
        raise ValueError("delta must be non-empty")
    if eps_change < 0:
        raise ValueError(f"eps_change must be non-negative, got {eps_change}")

    changing = np.abs(delta) > eps_change
    C = int(np.sum(changing))
    S = int(delta.size - C)
    ds2_CS = S * S - C * C

    return C, S, ds2_CS
