#!/usr/bin/env python3
"""
Accuracy Benchmark: v1.0.0 vs Latest (with optional lightlike observer)

Quantitatively compares:
1. Convergence accuracy (final distance to target)
2. Convergence speed (steps to threshold)
3. Oscillation behavior
4. Obstacle clearance
"""

import sys

import numpy as np

sys.path.insert(0, "/home/user/Eigen-Geometric-Control")  # noqa: E402

import pandas as pd  # noqa: E402

from src import (  # noqa: E402
    compute_change_stability,
    compute_ds2,
    compute_gradient,
    detect_oscillation,
    forward_kinematics,
    lightlike_damping_factor,
    run_arm_simulation,
)


def run_arm_with_lightlike_damping(
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
    window=3,
):
    """Run arm simulation WITH lightlike observer damping"""
    theta1, theta2 = theta_init
    target = np.array(target)
    obstacle_center = np.array(obstacle_center)

    rows = []
    state_history = []

    for t in range(n_ticks):
        # Compute current state
        x, y = forward_kinematics(theta1, theta2, L1, L2)
        state = np.array([theta1, theta2])
        state_history.append(state)

        ds2_total, components = compute_ds2(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        grad, grad_norm = compute_gradient(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        # LIGHTLIKE OBSERVER: Detect oscillation
        oscillating = False
        damping = 0.0
        osc_strength = 0.0

        if len(state_history) >= window:
            oscillating, osc_strength = detect_oscillation(state_history, window=window)
            if oscillating:
                damping = lightlike_damping_factor(osc_strength)

        # Apply damped gradient update
        delta = -(1.0 - damping) * eta * grad
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
                "oscillating": oscillating,
                "osc_strength": osc_strength,
                "damping": damping,
            }
        )

        # Update for next iteration
        theta1, theta2 = theta1_new, theta2_new

    return pd.DataFrame(rows)


def compute_metrics(df, target, name=""):
    """Compute accuracy metrics from simulation results"""
    final_x = df["x"].iloc[-1]
    final_y = df["y"].iloc[-1]
    final_dist = np.sqrt((final_x - target[0]) ** 2 + (final_y - target[1]) ** 2)

    # Convergence: first tick where distance < 0.1m
    target_dist = np.sqrt((df["x"] - target[0]) ** 2 + (df["y"] - target[1]) ** 2)
    converged_ticks = target_dist[target_dist < 0.1]
    ticks_to_converge = converged_ticks.index[0] if len(converged_ticks) > 0 else len(df)

    # Oscillation metrics
    grad_norms = df["grad_norm"].values
    osc_count = 0
    for i in range(10, len(grad_norms) - 1):
        # Check if gradient reverses (sign of oscillation)
        if grad_norms[i] > grad_norms[i - 1] * 1.5:  # Gradient increased significantly
            osc_count += 1

    metrics = {
        "name": name,
        "final_distance_to_target_m": final_dist,
        "ticks_to_converge": ticks_to_converge,
        "final_ds2": df["ds2_total"].iloc[-1],
        "initial_ds2": df["ds2_total"].iloc[0],
        "ds2_reduction": df["ds2_total"].iloc[0] - df["ds2_total"].iloc[-1],
        "min_obstacle_dist_m": df["d_obs"].min(),
        "final_grad_norm": df["grad_norm"].iloc[-1],
        "oscillation_events": osc_count,
        "mean_step_size": df["delta_theta_sq"].mean(),
    }

    return metrics


def main():
    print("=" * 80)
    print("ACCURACY BENCHMARK: v1.0.0 vs Latest (with Lightlike Observer)".center(80))
    print("=" * 80)
    print()

    # Test parameters
    target = np.array([1.2, 0.3])

    print("Running simulations...")
    print()

    # Run baseline (v1.0.0 algorithm - no lightlike observer)
    print("1. Running baseline (v1.0.0 algorithm)...")
    df_baseline = run_arm_simulation(
        theta_init=(-1.4, 1.2), target=tuple(target), n_ticks=140, eta=0.12
    )
    metrics_baseline = compute_metrics(df_baseline, target, "v1.0.0 Baseline")

    # Run with lightlike observer
    print("2. Running with lightlike observer damping...")
    df_lightlike = run_arm_with_lightlike_damping(
        theta_init=(-1.4, 1.2), target=tuple(target), n_ticks=140, eta=0.12, window=3
    )
    metrics_lightlike = compute_metrics(df_lightlike, target, "With Lightlike Observer")

    # Display results
    print()
    print("=" * 80)
    print("RESULTS COMPARISON".center(80))
    print("=" * 80)
    print()

    # Create comparison table
    print(f"{'Metric':<40} {'Baseline':<20} {'Lightlike':<20} {'Change':<10}")
    print("-" * 90)

    # Final accuracy
    baseline_dist = metrics_baseline["final_distance_to_target_m"]
    lightlike_dist = metrics_lightlike["final_distance_to_target_m"]
    change_dist = (
        ((lightlike_dist - baseline_dist) / baseline_dist * 100) if baseline_dist > 0 else 0
    )
    print(
        f"{'Final distance to target (m)':<40} {baseline_dist:<20.4f} {lightlike_dist:<20.4f} {change_dist:+.1f}%"
    )

    # Convergence speed
    baseline_conv = metrics_baseline["ticks_to_converge"]
    lightlike_conv = metrics_lightlike["ticks_to_converge"]
    change_conv = (
        ((lightlike_conv - baseline_conv) / baseline_conv * 100) if baseline_conv > 0 else 0
    )
    print(
        f"{'Ticks to converge (<0.1m)':<40} {baseline_conv:<20} {lightlike_conv:<20} {change_conv:+.1f}%"
    )

    # ds2 reduction
    baseline_ds2 = metrics_baseline["final_ds2"]
    lightlike_ds2 = metrics_lightlike["final_ds2"]
    change_ds2 = (
        ((lightlike_ds2 - baseline_ds2) / abs(baseline_ds2) * 100) if baseline_ds2 != 0 else 0
    )
    print(f"{'Final ds²':<40} {baseline_ds2:<20.4f} {lightlike_ds2:<20.4f} {change_ds2:+.1f}%")

    # Obstacle clearance
    baseline_obs = metrics_baseline["min_obstacle_dist_m"]
    lightlike_obs = metrics_lightlike["min_obstacle_dist_m"]
    change_obs = ((lightlike_obs - baseline_obs) / baseline_obs * 100) if baseline_obs > 0 else 0
    print(
        f"{'Min obstacle distance (m)':<40} {baseline_obs:<20.4f} {lightlike_obs:<20.4f} {change_obs:+.1f}%"
    )

    # Oscillation
    baseline_osc = metrics_baseline["oscillation_events"]
    lightlike_osc = metrics_lightlike["oscillation_events"]
    change_osc = lightlike_osc - baseline_osc
    print(f"{'Oscillation events':<40} {baseline_osc:<20} {lightlike_osc:<20} {change_osc:+d}")

    # Final gradient
    baseline_grad = metrics_baseline["final_grad_norm"]
    lightlike_grad = metrics_lightlike["final_grad_norm"]
    change_grad = (
        ((lightlike_grad - baseline_grad) / baseline_grad * 100) if baseline_grad > 0 else 0
    )
    print(
        f"{'Final gradient norm':<40} {baseline_grad:<20.6f} {lightlike_grad:<20.6f} {change_grad:+.1f}%"
    )

    print()
    print("=" * 80)
    print("INTERPRETATION".center(80))
    print("=" * 80)
    print()

    # Analyze if there's improvement
    if abs(change_dist) < 1.0 and abs(change_conv) < 5.0:
        print("RESULT: No significant accuracy improvement detected")
        print()
        print("The lightlike observer does not improve performance for this task because:")
        print("  • The baseline algorithm already converges smoothly without oscillation")
        print("  • This is a well-conditioned problem (clear gradient to target)")
        print("  • The XOR rotation problem (period-2 loop) is NOT present here")
        print()
        print("The lightlike observer is designed for:")
        print("  • Period-2 or period-N oscillations (like XOR rotation)")
        print("  • Pathological feedback loops")
        print("  • Self-reinforcement problems")
        print()
        print("For standard robot arm control with smooth convergence,")
        print("the baseline v1.0.0 algorithm is already optimal.")
    else:
        print("RESULT: Performance difference detected")
        print()
        print(f"Distance to target: {change_dist:+.1f}%")
        print(f"Convergence speed: {change_conv:+.1f}%")
        print(f"Oscillation events: {change_osc:+d}")

    print()
    print("=" * 80)

    # Check for lightlike observer activations
    if "damping" in df_lightlike.columns:
        damping_used = (df_lightlike["damping"] > 0).sum()
        max_damping = df_lightlike["damping"].max()
        print(f"\nLightlike observer activations: {damping_used} / {len(df_lightlike)} ticks")
        print(f"Maximum damping applied: {max_damping:.4f}")

        if damping_used == 0:
            print("\n→ Lightlike observer was NEVER activated (no oscillation detected)")


if __name__ == "__main__":
    main()
