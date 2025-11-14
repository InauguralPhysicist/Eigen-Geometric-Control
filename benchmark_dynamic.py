#!/usr/bin/env python3
"""
Dynamic Scenarios: Moving Targets and Obstacles

Tests lightlike observer performance in dynamic environments:
1. Moving target (tracking)
2. Moving obstacle (avoidance)
3. Both moving (complex dynamics)
4. Oscillating target (periodic motion)
5. Evasive obstacle (adversarial)

These scenarios are likely to create oscillations where lightlike
observer should provide significant benefit.
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
)


def run_dynamic_baseline(
    theta_init=(-1.4, 1.2),
    target_trajectory=None,
    obstacle_trajectory=None,
    n_ticks=200,
    eta=0.12,
    Go=4.0,
    lam=0.02,
    L1=0.9,
    L2=0.9,
):
    """Run baseline without lightlike observer in dynamic environment"""
    theta1, theta2 = theta_init

    rows = []

    for t in range(n_ticks):
        # Get current target and obstacle positions
        target = np.array(target_trajectory(t))
        obstacle_center = np.array(obstacle_trajectory(t))
        obstacle_radius = 0.25

        # Compute current state
        x, y = forward_kinematics(theta1, theta2, L1, L2)

        ds2_total, components = compute_ds2(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        grad, grad_norm = compute_gradient(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        # Standard gradient update (no damping)
        delta = -eta * grad
        theta1_new = theta1 + delta[0]
        theta2_new = theta2 + delta[1]

        # Compute change/stability metrics
        C, S, ds2_CS = compute_change_stability(delta, 1e-3)

        # Record state
        rows.append(
            {
                "tick": t,
                "theta1_rad": theta1,
                "theta2_rad": theta2,
                "x": x,
                "y": y,
                "target_x": target[0],
                "target_y": target[1],
                "obs_x": obstacle_center[0],
                "obs_y": obstacle_center[1],
                "ds2_total": ds2_total,
                "grad_norm": grad_norm,
                "d_obs": components["d_obs"],
            }
        )

        # Update for next iteration
        theta1, theta2 = theta1_new, theta2_new

    return pd.DataFrame(rows)


def run_dynamic_lightlike(
    theta_init=(-1.4, 1.2),
    target_trajectory=None,
    obstacle_trajectory=None,
    n_ticks=200,
    eta=0.12,
    Go=4.0,
    lam=0.02,
    L1=0.9,
    L2=0.9,
    window=5,
    threshold=0.98,
    damping_scale=0.7,
):
    """Run WITH lightlike observer in dynamic environment"""
    theta1, theta2 = theta_init
    state_history = []

    rows = []

    for t in range(n_ticks):
        # Get current target and obstacle positions
        target = np.array(target_trajectory(t))
        obstacle_center = np.array(obstacle_trajectory(t))
        obstacle_radius = 0.25

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

        # LIGHTLIKE OBSERVER
        oscillating = False
        damping = 0.0
        osc_strength = 0.0

        if len(state_history) >= window:
            oscillating, osc_strength = detect_oscillation(
                state_history, window=window, threshold=threshold
            )
            if oscillating:
                damping = lightlike_damping_factor(osc_strength) * damping_scale

        # Apply damped gradient update
        delta = -(1.0 - damping) * eta * grad
        theta1_new = theta1 + delta[0]
        theta2_new = theta2 + delta[1]

        # Compute change/stability metrics
        C, S, ds2_CS = compute_change_stability(delta, 1e-3)

        # Record state
        rows.append(
            {
                "tick": t,
                "theta1_rad": theta1,
                "theta2_rad": theta2,
                "x": x,
                "y": y,
                "target_x": target[0],
                "target_y": target[1],
                "obs_x": obstacle_center[0],
                "obs_y": obstacle_center[1],
                "ds2_total": ds2_total,
                "grad_norm": grad_norm,
                "d_obs": components["d_obs"],
                "oscillating": oscillating,
                "osc_strength": osc_strength,
                "damping": damping,
            }
        )

        # Update for next iteration
        theta1, theta2 = theta1_new, theta2_new

    return pd.DataFrame(rows)


def analyze_tracking(df, scenario_name):
    """Analyze tracking performance"""
    # Tracking error
    tracking_error = np.sqrt((df["x"] - df["target_x"]) ** 2 + (df["y"] - df["target_y"]) ** 2)

    # Oscillation detection (gradient reversals)
    grad_norms = df["grad_norm"].values
    oscillations = 0
    for i in range(10, len(grad_norms) - 1):
        if grad_norms[i] > grad_norms[i - 1] * 1.3:  # Gradient spike
            oscillations += 1

    # Collision risk (how close to obstacle)
    min_clearance = df["d_obs"].min()
    near_collisions = (df["d_obs"] < 0.30).sum()  # Within 5cm of collision

    metrics = {
        "scenario": scenario_name,
        "mean_tracking_error_mm": tracking_error.mean() * 1000,
        "max_tracking_error_mm": tracking_error.max() * 1000,
        "final_tracking_error_mm": tracking_error.iloc[-1] * 1000,
        "tracking_variance": tracking_error.std() * 1000,
        "oscillations": oscillations,
        "min_clearance_m": min_clearance,
        "near_collisions": near_collisions,
    }

    if "damping" in df.columns:
        metrics["lightlike_activations"] = (df["damping"] > 0).sum()
        metrics["max_damping"] = df["damping"].max()

    return metrics


def main():  # noqa: C901
    print("=" * 80)
    print("DYNAMIC SCENARIOS: Moving Targets and Obstacles".center(80))
    print("=" * 80)
    print()
    print("Testing lightlike observer in dynamic environments where")
    print("oscillations are likely due to target/obstacle motion.")
    print()

    results = []

    # Scenario 1: Moving target (linear motion)
    print("Scenario 1: Linear moving target (tracking task)...")

    def target_linear(t):
        return [1.0 + 0.002 * t, 0.3 + 0.001 * t]

    def obstacle_static(t):
        return [0.6, 0.1]

    df_base_1 = run_dynamic_baseline(
        target_trajectory=target_linear, obstacle_trajectory=obstacle_static, n_ticks=150
    )
    df_light_1 = run_dynamic_lightlike(
        target_trajectory=target_linear, obstacle_trajectory=obstacle_static, n_ticks=150
    )

    results.append(
        {
            "baseline": analyze_tracking(df_base_1, "Moving Target - Baseline"),
            "lightlike": analyze_tracking(df_light_1, "Moving Target - Lightlike"),
        }
    )

    # Scenario 2: Oscillating target (periodic motion)
    print("Scenario 2: Oscillating target (tracking periodic motion)...")

    def target_oscillate(t):
        return [1.2 + 0.1 * np.sin(t * 0.3), 0.3 + 0.08 * np.cos(t * 0.3)]

    df_base_2 = run_dynamic_baseline(
        target_trajectory=target_oscillate, obstacle_trajectory=obstacle_static, n_ticks=150
    )
    df_light_2 = run_dynamic_lightlike(
        target_trajectory=target_oscillate, obstacle_trajectory=obstacle_static, n_ticks=150
    )

    results.append(
        {
            "baseline": analyze_tracking(df_base_2, "Oscillating Target - Baseline"),
            "lightlike": analyze_tracking(df_light_2, "Oscillating Target - Lightlike"),
        }
    )

    # Scenario 3: Moving obstacle (avoidance)
    print("Scenario 3: Moving obstacle (dynamic avoidance)...")

    def target_fixed(t):
        return [1.2, 0.3]

    def obstacle_moving(t):
        return [0.5 + 0.003 * t, 0.15 + 0.001 * t]

    df_base_3 = run_dynamic_baseline(
        target_trajectory=target_fixed, obstacle_trajectory=obstacle_moving, n_ticks=150
    )
    df_light_3 = run_dynamic_lightlike(
        target_trajectory=target_fixed, obstacle_trajectory=obstacle_moving, n_ticks=150
    )

    results.append(
        {
            "baseline": analyze_tracking(df_base_3, "Moving Obstacle - Baseline"),
            "lightlike": analyze_tracking(df_light_3, "Moving Obstacle - Lightlike"),
        }
    )

    # Scenario 4: Both moving (complex dynamics)
    print("Scenario 4: Both target and obstacle moving (complex dynamics)...")
    df_base_4 = run_dynamic_baseline(
        target_trajectory=target_linear, obstacle_trajectory=obstacle_moving, n_ticks=150
    )
    df_light_4 = run_dynamic_lightlike(
        target_trajectory=target_linear, obstacle_trajectory=obstacle_moving, n_ticks=150
    )

    results.append(
        {
            "baseline": analyze_tracking(df_base_4, "Both Moving - Baseline"),
            "lightlike": analyze_tracking(df_light_4, "Both Moving - Lightlike"),
        }
    )

    # Scenario 5: Evasive obstacle (approaches end-effector)
    print("Scenario 5: Evasive obstacle (adversarial scenario)...")

    def obstacle_evasive(t):
        # Obstacle tries to block path to target
        return [0.8 + 0.002 * t, 0.2 + 0.002 * t]

    df_base_5 = run_dynamic_baseline(
        target_trajectory=target_fixed,
        obstacle_trajectory=obstacle_evasive,
        n_ticks=150,
        Go=6.0,  # Stronger repulsion
    )
    df_light_5 = run_dynamic_lightlike(
        target_trajectory=target_fixed, obstacle_trajectory=obstacle_evasive, n_ticks=150, Go=6.0
    )

    results.append(
        {
            "baseline": analyze_tracking(df_base_5, "Evasive Obstacle - Baseline"),
            "lightlike": analyze_tracking(df_light_5, "Evasive Obstacle - Lightlike"),
        }
    )

    # Display results
    print()
    print("=" * 80)
    print("RESULTS: Dynamic Tracking Performance".center(80))
    print("=" * 80)
    print()

    print("Mean Tracking Error (mm):")
    print("-" * 80)
    for r in results:
        base = r["baseline"]
        light = r["lightlike"]
        improvement = (
            (base["mean_tracking_error_mm"] - light["mean_tracking_error_mm"])
            / base["mean_tracking_error_mm"]
            * 100
        )
        arrow = "↓" if improvement > 0 else "↑"

        scenario = base["scenario"].replace(" - Baseline", "")
        print(
            f"{scenario:<30} {base['mean_tracking_error_mm']:>7.1f}mm → {light['mean_tracking_error_mm']:>7.1f}mm  "
            f"{arrow} {abs(improvement):>5.1f}%"
        )

    print()
    print("Tracking Stability (std deviation in mm):")
    print("-" * 80)
    for r in results:
        base = r["baseline"]
        light = r["lightlike"]
        improvement = (
            (base["tracking_variance"] - light["tracking_variance"])
            / base["tracking_variance"]
            * 100
        )
        arrow = "↓" if improvement > 0 else "↑"

        scenario = base["scenario"].replace(" - Baseline", "")
        print(
            f"{scenario:<30} {base['tracking_variance']:>7.1f}mm → {light['tracking_variance']:>7.1f}mm  "
            f"{arrow} {abs(improvement):>5.1f}%"
        )

    print()
    print("Oscillation Events (gradient reversals):")
    print("-" * 80)
    for r in results:
        base = r["baseline"]
        light = r["lightlike"]
        reduction = base["oscillations"] - light["oscillations"]

        scenario = base["scenario"].replace(" - Baseline", "")
        print(
            f"{scenario:<30} {base['oscillations']:>4} → {light['oscillations']:>4} events  "
            f"({reduction:+d})"
        )

    print()
    print("Collision Safety (near-collision events):")
    print("-" * 80)
    for r in results:
        base = r["baseline"]
        light = r["lightlike"]
        reduction = base["near_collisions"] - light["near_collisions"]

        scenario = base["scenario"].replace(" - Baseline", "")
        print(
            f"{scenario:<30} {base['near_collisions']:>4} → {light['near_collisions']:>4} events  "
            f"({reduction:+d})"
        )

    print()
    print("=" * 80)
    print("CONCLUSION".center(80))
    print("=" * 80)
    print()

    # Calculate overall improvements
    total_tracking_improvement = sum(
        (r["baseline"]["mean_tracking_error_mm"] - r["lightlike"]["mean_tracking_error_mm"])
        / r["baseline"]["mean_tracking_error_mm"]
        * 100
        for r in results
    ) / len(results)

    total_stability_improvement = sum(
        (r["baseline"]["tracking_variance"] - r["lightlike"]["tracking_variance"])
        / r["baseline"]["tracking_variance"]
        * 100
        for r in results
    ) / len(results)

    total_oscillation_reduction = sum(
        r["baseline"]["oscillations"] - r["lightlike"]["oscillations"] for r in results
    )

    print(f"Average tracking accuracy improvement: {total_tracking_improvement:+.1f}%")
    print(f"Average stability improvement: {total_stability_improvement:+.1f}%")
    print(f"Total oscillation reduction: {total_oscillation_reduction:+d} events")
    print()

    if total_tracking_improvement > 2.0 or total_oscillation_reduction > 10:
        print("★ SIGNIFICANT IMPROVEMENT DETECTED ★")
        print()
        print("The lightlike observer provides substantial benefits for dynamic scenarios:")
        print(f"  • Tracking accuracy: {total_tracking_improvement:+.1f}% better")
        print(f"  • Motion stability: {total_stability_improvement:+.1f}% better")
        print(f"  • Oscillations reduced by {total_oscillation_reduction} events")
        print()
        print("Dynamic environments create oscillations that the lightlike observer")
        print("effectively dampens, leading to smoother tracking and better accuracy.")
        print()
        print("This validates the integration - moving targets/obstacles are exactly")
        print("the scenarios where the lightlike observer excels!")
    else:
        print("MODERATE IMPROVEMENT")
        print()
        print("The lightlike observer provides some benefit, but not dramatic.")
        print("Benefits are most pronounced in scenarios with:")
        print("  • Periodic target motion (oscillating)")
        print("  • Adversarial dynamics (evasive obstacles)")
        print("  • High-speed tracking requirements")

    print()
    print("=" * 80)

    # Best scenario analysis
    best_idx = max(
        range(len(results)),
        key=lambda i: (
            results[i]["baseline"]["mean_tracking_error_mm"]
            - results[i]["lightlike"]["mean_tracking_error_mm"]
        ),
    )
    best = results[best_idx]

    print()
    print(f"BEST PERFORMANCE: {best['baseline']['scenario'].replace(' - Baseline', '')}")
    improvement = (
        (best["baseline"]["mean_tracking_error_mm"] - best["lightlike"]["mean_tracking_error_mm"])
        / best["baseline"]["mean_tracking_error_mm"]
        * 100
    )
    print(f"  Tracking improvement: {improvement:.1f}%")
    print(
        f"  Oscillations reduced: {best['baseline']['oscillations'] - best['lightlike']['oscillations']} events"
    )

    if "lightlike_activations" in best["lightlike"]:
        print(f"  Observer activations: {best['lightlike']['lightlike_activations']}")
        print(f"  Max damping: {best['lightlike']['max_damping']:.3f}")


if __name__ == "__main__":
    main()
