# -*- coding: utf-8 -*-
"""
Eigen: 3-DOF Spatial Arm - Core Geometric Functions

3-DOF arm operating in 3D workspace (x, y, z).
Configuration space: (θ₁, θ₂, θ₃) — 3 independent orientation constraints.

Joint layout (ZYY convention):
  θ₁: base rotation about z-axis
  θ₂: shoulder elevation about local y-axis
  θ₃: elbow flexion about local y-axis

This is the minimal arm where:
  - The Jacobian is 3×3 (square)
  - Singularity manifolds are 1D curves, not isolated points
  - The workspace has 8 octants (closure in the π-halving sense)
  - ds² operates on a configuration space where topology matters

ds² = ‖p - p_target‖² + G₀·max(0, r - d_obs)² + λ‖θ‖²

Same invariant, now in 3D.
"""

from typing import Dict, List, Tuple, Union

import numpy as np


def forward_kinematics_3d(
    theta1: float,
    theta2: float,
    theta3: float,
    L1: float = 0.9,
    L2: float = 0.9,
    L3: float = 0.9,
) -> Tuple[float, float, float]:
    """
    Compute end-effector position in 3D from joint angles.

    ZYY convention:
      Joint 1 (θ₁): rotates about z-axis at base
      Joint 2 (θ₂): rotates about local y-axis (shoulder elevation)
      Joint 3 (θ₃): rotates about local y-axis (elbow flexion)

    Returns (x, y, z) of end-effector.
    """
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)
    c23, s23 = np.cos(theta2 + theta3), np.sin(theta2 + theta3)

    # Projection onto xz-plane after shoulder and elbow
    r = L2 * c2 + L3 * c23  # radial distance from z-axis (in shoulder frame)
    z = L1 + L2 * s2 + L3 * s23  # height

    x = float(r * c1)
    y = float(r * s1)
    z = float(z)

    return x, y, z


def jacobian_3d(
    theta1: float,
    theta2: float,
    theta3: float,
    L1: float = 0.9,
    L2: float = 0.9,
    L3: float = 0.9,
) -> np.ndarray:
    """
    Compute 3×3 Jacobian mapping joint velocities to task-space velocities.

    J = [∂x/∂θ₁  ∂x/∂θ₂  ∂x/∂θ₃]
        [∂y/∂θ₁  ∂y/∂θ₂  ∂y/∂θ₃]
        [∂z/∂θ₁  ∂z/∂θ₂  ∂z/∂θ₃]
    """
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)
    c23, s23 = np.cos(theta2 + theta3), np.sin(theta2 + theta3)

    r = L2 * c2 + L3 * c23

    # ∂r/∂θ₂ = -L2*s2 - L3*s23
    # ∂r/∂θ₃ = -L3*s23
    dr_dth2 = -L2 * s2 - L3 * s23
    dr_dth3 = -L3 * s23

    # ∂z/∂θ₂ = L2*c2 + L3*c23
    # ∂z/∂θ₃ = L3*c23
    dz_dth2 = L2 * c2 + L3 * c23
    dz_dth3 = L3 * c23

    # x = r*c1, y = r*s1
    # ∂x/∂θ₁ = -r*s1,  ∂x/∂θ₂ = dr/dθ₂ * c1,  ∂x/∂θ₃ = dr/dθ₃ * c1
    # ∂y/∂θ₁ =  r*c1,  ∂y/∂θ₂ = dr/dθ₂ * s1,  ∂y/∂θ₃ = dr/dθ₃ * s1
    # ∂z/∂θ₁ = 0,       ∂z/∂θ₂ = dz/dθ₂,        ∂z/∂θ₃ = dz/dθ₃

    return np.array(
        [
            [-r * s1, dr_dth2 * c1, dr_dth3 * c1],
            [r * c1, dr_dth2 * s1, dr_dth3 * s1],
            [0.0, dz_dth2, dz_dth3],
        ]
    )


def compute_ds2_3d(
    theta: Union[np.ndarray, Tuple[float, float, float]],
    target: Union[np.ndarray, Tuple[float, float, float]],
    obstacles: List[Tuple[np.ndarray, float]],
    Go: float = 4.0,
    lam: float = 0.02,
    L1: float = 0.9,
    L2: float = 0.9,
    L3: float = 0.9,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute ds² for 3-DOF arm.

    ds² = ‖p - p_target‖² + Σ G₀·max(0, rᵢ - dᵢ)² + λ‖θ‖²

    Parameters
    ----------
    theta : array-like of shape (3,)
        Joint angles [θ₁, θ₂, θ₃]
    target : array-like of shape (3,)
        Goal position [x, y, z]
    obstacles : list of (center, radius) pairs
        Each obstacle is (np.ndarray of shape (3,), float)
    Go : float
        Obstacle repulsion strength
    lam : float
        Regularization weight
    L1, L2, L3 : float
        Link lengths

    Returns
    -------
    ds2_total : float
    components : dict with 'target_term', 'obs_term', 'reg_term', 'd_obs_min'
    """
    theta = np.asarray(theta, dtype=float)
    target = np.asarray(target, dtype=float)

    x, y, z = forward_kinematics_3d(theta[0], theta[1], theta[2], L1, L2, L3)
    pos = np.array([x, y, z])

    # Target term
    target_term = float(np.sum((pos - target) ** 2))

    # Obstacle term (sum over all obstacles)
    obs_term = 0.0
    d_obs_min = float("inf")
    for obs_center, obs_radius in obstacles:
        obs_center = np.asarray(obs_center, dtype=float)
        d = float(np.linalg.norm(pos - obs_center))
        d_obs_min = min(d_obs_min, d)
        penetration = max(0.0, obs_radius - d)
        obs_term += Go * penetration**2

    # Regularization term
    reg_term = float(lam * np.sum(theta**2))

    ds2_total = target_term + obs_term + reg_term

    return ds2_total, {
        "target_term": target_term,
        "obs_term": obs_term,
        "reg_term": reg_term,
        "d_obs_min": d_obs_min if obstacles else float("inf"),
    }


def compute_gradient_3d(
    theta: Union[np.ndarray, Tuple[float, float, float]],
    target: Union[np.ndarray, Tuple[float, float, float]],
    obstacles: List[Tuple[np.ndarray, float]],
    Go: float = 4.0,
    lam: float = 0.02,
    L1: float = 0.9,
    L2: float = 0.9,
    L3: float = 0.9,
) -> Tuple[np.ndarray, float]:
    """
    Compute ∇ds² for 3-DOF arm.

    Returns
    -------
    grad : np.ndarray of shape (3,)
    grad_norm : float
    """
    theta = np.asarray(theta, dtype=float)
    target = np.asarray(target, dtype=float)

    x, y, z = forward_kinematics_3d(theta[0], theta[1], theta[2], L1, L2, L3)
    pos = np.array([x, y, z])
    J = jacobian_3d(theta[0], theta[1], theta[2], L1, L2, L3)

    # Target gradient: ∂/∂θ ‖p - t‖² = 2 J^T (p - t)
    grad_target = 2.0 * J.T @ (pos - target)

    # Obstacle gradient: Σ ∂/∂θ [Go * max(0, r - d)²]
    grad_obs = np.zeros(3)
    for obs_center, obs_radius in obstacles:
        obs_center = np.asarray(obs_center, dtype=float)
        d = float(np.linalg.norm(pos - obs_center))
        penetration = obs_radius - d
        if penetration > 0 and d > 1e-12:
            # ∂/∂θ [Go*(r-d)²] = Go * 2*(r-d) * (-1/d) * J^T (p - c)
            grad_obs += Go * 2.0 * penetration * (-1.0 / d) * (J.T @ (pos - obs_center))

    # Regularization gradient: ∂/∂θ [λ‖θ‖²] = 2λθ
    grad_reg = 2.0 * lam * theta

    grad = grad_target + grad_obs + grad_reg
    grad_norm = float(np.linalg.norm(grad))

    return grad, grad_norm


def run_3dof_simulation(
    theta_init=(0.0, 0.5, -0.5),
    target=(0.8, 0.5, 1.5),
    obstacles=None,
    n_ticks=200,
    eta=0.08,
    Go=4.0,
    lam=0.02,
    L1=0.9,
    L2=0.9,
    L3=0.9,
):
    """
    Run gradient descent on ds² for the 3-DOF spatial arm.

    Returns list of dicts with full diagnostic trace.
    """
    if obstacles is None:
        obstacles = []

    theta = np.array(theta_init, dtype=float)
    trace = []

    for t in range(n_ticks):
        ds2, comp = compute_ds2_3d(theta, target, obstacles, Go, lam, L1, L2, L3)
        grad, gnorm = compute_gradient_3d(theta, target, obstacles, Go, lam, L1, L2, L3)

        x, y, z = forward_kinematics_3d(theta[0], theta[1], theta[2], L1, L2, L3)
        pos = np.array([x, y, z])
        J = jacobian_3d(theta[0], theta[1], theta[2], L1, L2, L3)

        # Per-term gradient norms
        target_arr = np.asarray(target, dtype=float)
        g_target = float(np.linalg.norm(2.0 * J.T @ (pos - target_arr)))

        g_obs = 0.0
        for obs_center, obs_radius in obstacles:
            obs_center = np.asarray(obs_center, dtype=float)
            d = float(np.linalg.norm(pos - obs_center))
            penetration = obs_radius - d
            if penetration > 0 and d > 1e-12:
                g_obs += float(
                    np.linalg.norm(Go * 2.0 * penetration * (-1.0 / d) * (J.T @ (pos - obs_center)))
                )

        g_reg = float(np.linalg.norm(2.0 * lam * theta))

        delta = -eta * grad
        theta_new = theta + delta

        ds2_next, _ = compute_ds2_3d(theta_new, target, obstacles, Go, lam, L1, L2, L3)

        trace.append(
            {
                "t": t,
                "theta": theta.copy(),
                "pos": pos.copy(),
                "ds2": ds2,
                "ds2_next": ds2_next,
                "delta_ds2": ds2_next - ds2,
                "grad_norm": gnorm,
                "grad": grad.copy(),
                "g_target": g_target,
                "g_obs": g_obs,
                "g_reg": g_reg,
                "d_obs_min": comp["d_obs_min"],
                "det_J": float(np.linalg.det(J)),
                "cond_J": (
                    float(np.linalg.cond(J)) if abs(np.linalg.det(J)) > 1e-15 else float("inf")
                ),
            }
        )

        theta = theta_new

    return trace
