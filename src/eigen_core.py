# -*- coding: utf-8 -*-
"""
Eigen: Geometric Control Framework - Core Functions

Provides basic geometric computations for 2-DOF planar arm:
- Forward kinematics
- Jacobian computation  
- Objective function (ds²)
- Gradient computation
- Change/Stability metrics
"""

import numpy as np

def forward_kinematics(theta1, theta2, L1=0.9, L2=0.9):
    """
    Compute end-effector position from joint angles
    
    Args:
        theta1: Shoulder angle (radians)
        theta2: Elbow angle (radians)
        L1, L2: Link lengths (meters)
    
    Returns:
        x, y: End-effector coordinates (meters)
    """
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return x, y


def jacobian(theta1, theta2, L1=0.9, L2=0.9):
    """
    Compute Jacobian matrix mapping joint velocities to end-effector velocities
    
    Returns:
        J: 2×2 matrix [[∂x/∂θ₁, ∂x/∂θ₂], [∂y/∂θ₁, ∂y/∂θ₂]]
    """
    return np.array([
        [-L1*np.sin(theta1) - L2*np.sin(theta1 + theta2), -L2*np.sin(theta1 + theta2)],
        [ L1*np.cos(theta1) + L2*np.cos(theta1 + theta2),  L2*np.cos(theta1 + theta2)]
    ])


def compute_ds2(theta1, theta2, target, obstacle_center, obstacle_radius,
                Go=4.0, lam=0.02, L1=0.9, L2=0.9):
    """
    Compute geometric objective function ds²
    
    ds² = ||p - p_target||² + G·max(0, r - d_obs)² + λ||θ||²
    
    Args:
        theta1, theta2: Joint angles
        target: Goal position [x_target, y_target]
        obstacle_center: Obstacle position [x_obs, y_obs]
        obstacle_radius: Obstacle radius (meters)
        Go: Obstacle repulsion strength
        lam: Regularization weight
        L1, L2: Link lengths
    
    Returns:
        ds2_total: Total objective value
        components: Dict with breakdown of terms
    """
    x, y = forward_kinematics(theta1, theta2, L1, L2)
    pos = np.array([x, y])
    
    # Distance to target
    target_term = float(np.sum((pos - target)**2))
    
    # Distance to obstacle
    d_obs = float(np.linalg.norm(pos - obstacle_center))
    obs_term = float(Go * max(0.0, obstacle_radius - d_obs)**2)
    
    # Joint space regularization
    reg_term = float(lam * (theta1**2 + theta2**2))
    
    ds2_total = target_term + obs_term + reg_term
    
    return ds2_total, {
        'target_term': target_term,
        'obs_term': obs_term,
        'reg_term': reg_term,
        'd_obs': d_obs
    }


def compute_gradient(theta1, theta2, target, obstacle_center, obstacle_radius,
                     Go=4.0, lam=0.02, L1=0.9, L2=0.9):
    """
    Compute gradient ∇ds² for current configuration
    
    Returns:
        grad: Gradient vector [∂ds²/∂θ₁, ∂ds²/∂θ₂]
        grad_norm: Magnitude ||∇ds²||
    """
    x, y = forward_kinematics(theta1, theta2, L1, L2)
    pos = np.array([x, y])
    J = jacobian(theta1, theta2, L1, L2)
    
    # Target gradient: 2(p - p_target)ᵀ J
    grad_target = 2.0 * J.T @ (pos - target)
    
    # Obstacle gradient (active only inside repulsion zone)
    d_obs = float(np.linalg.norm(pos - obstacle_center))
    if obstacle_radius - d_obs > 0:
        grad_obs = (Go * 2.0 * (obstacle_radius - d_obs) * (-1.0 / d_obs) *
                    (J.T @ (pos - obstacle_center)))
    else:
        grad_obs = np.zeros(2)
    
    # Regularization gradient
    grad_reg = 2.0 * lam * np.array([theta1, theta2])
    
    grad = grad_target + grad_obs + grad_reg
    grad_norm = float(np.linalg.norm(grad))
    
    return grad, grad_norm


def compute_change_stability(delta, eps_change=1e-3):
    """
    Compute change (C) and stability (S) counts from joint angle deltas
    
    Args:
        delta: Array of joint angle changes [Δθ₁, Δθ₂]
        eps_change: Threshold for detecting change
    
    Returns:
        C: Number of changing joints
        S: Number of stable joints
        ds2_CS: Geometric invariant S² - C²
    """
    changing = np.abs(delta) > eps_change
    C = int(np.sum(changing))
    S = len(delta) - C
    ds2_CS = S*S - C*C
    
    return C, S, ds2_CS
