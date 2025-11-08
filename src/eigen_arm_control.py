# -*- coding: utf-8 -*-
"""
Eigen: 2-DOF Robot Arm Control

Demonstrates geometric control on planar 2-joint arm with:
- Target reaching
- Obstacle avoidance
- Smooth convergence
"""

from typing import Optional

import numpy as np
import pandas as pd

from .config import ArmConfig
from .eigen_core import compute_change_stability, compute_ds2, compute_gradient, forward_kinematics


def run_arm_simulation(
    config: Optional[ArmConfig] = None,
    theta_init=(-1.4, 1.2),
    target=(1.2, 0.3),
    obstacle_center=(0.6, 0.1),
    obstacle_radius=0.25,
    n_ticks=140,
    eta=0.12,
    Go=4.0,
    lam=0.02,
    eps_change=1e-3,
    L1=0.9,
    L2=0.9,
):
    """
    Run complete arm control simulation

    Args:
        config: Optional ArmConfig instance. If provided, overrides other parameters.
        theta_init: Initial joint angles (θ₁, θ₂) in radians
        target: Goal position (x, y) in meters
        obstacle_center: Obstacle position (x, y)
        obstacle_radius: Obstacle radius in meters
        n_ticks: Number of simulation steps
        eta: Step size (learning rate)
        Go: Obstacle repulsion strength
        lam: Regularization weight
        eps_change: Threshold for change detection
        L1, L2: Link lengths

    Returns:
        DataFrame with complete trajectory data

    Examples:
        >>> # Using ArmConfig
        >>> config = ArmConfig.from_yaml('configs/default.yaml')
        >>> results = run_arm_simulation(config=config)

        >>> # Using individual parameters (backward compatible)
        >>> results = run_arm_simulation(eta=0.15, n_ticks=100)
    """
    # If config provided, use its values
    if config is not None:
        theta_init = config.theta_init
        target = config.target
        obstacle_center = config.obstacle_center
        obstacle_radius = config.obstacle_radius
        n_ticks = config.n_ticks
        eta = config.eta
        Go = config.Go
        lam = config.lam
        eps_change = config.eps_change
        L1 = config.L1
        L2 = config.L2
    theta1, theta2 = theta_init
    target = np.array(target)
    obstacle_center = np.array(obstacle_center)

    rows = []

    for t in range(n_ticks):
        # Compute current state
        x, y = forward_kinematics(theta1, theta2, L1, L2)

        ds2_total, components = compute_ds2(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        grad, grad_norm = compute_gradient(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        # Update configuration: Q_{t+1} = Q_t - η∇ds²
        delta = -eta * grad
        theta1_new = theta1 + delta[0]
        theta2_new = theta2 + delta[1]

        # Compute change/stability metrics
        C, S, ds2_CS = compute_change_stability(delta, eps_change)

        # Record state
        rows.append(
            {
                "tick": t,
                "theta1_rad": theta1,
                "theta2_rad": theta2,
                "x": x,
                "y": y,
                "ds2_total": ds2_total,
                "target_term": components["target_term"],
                "obs_term": components["obs_term"],
                "reg_term": components["reg_term"],
                "grad_norm": grad_norm,
                "C": C,
                "S": S,
                "ds2_CS": ds2_CS,
                "d_obs": components["d_obs"],
                "delta_theta1": delta[0],
                "delta_theta2": delta[1],
                "delta_theta_sq": float(np.sum(delta**2)),
            }
        )

        # Update for next iteration
        theta1, theta2 = theta1_new, theta2_new

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Example 1: Using ArmConfig from YAML
    print("=== Using ArmConfig from YAML ===")
    config = ArmConfig.from_yaml("configs/default.yaml")
    results = run_arm_simulation(config=config)

    print(f"Initial ds²: {results['ds2_total'].iloc[0]:.4f}")
    print(f"Final ds²: {results['ds2_total'].iloc[-1]:.4f}")
    print(f"Gradient: {results['grad_norm'].iloc[0]:.2e} → {results['grad_norm'].iloc[-1]:.2e}")
    print(f"Min obstacle distance: {results['d_obs'].min():.4f}m")

    # Example 2: Using individual parameters (backward compatible)
    print("\n=== Using individual parameters ===")
    results2 = run_arm_simulation(eta=0.15, n_ticks=100)
    print(f"Final ds²: {results2['ds2_total'].iloc[-1]:.4f}")
